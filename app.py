# app.py
import os
import io
import json
import base64
import shutil
import subprocess
import zipfile
import re
import difflib
import sys
from typing import List, Tuple

import gradio as gr
import folium
import html as html_lib
import pandas as pd
import numpy as np
import rasterio
import ee

from matplotlib import pyplot as plt, colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from folium import Element
from dotenv import load_dotenv
from rasterio.crs import CRS as RioCRS
from rasterio.warp import transform as rio_transform

# Optional HF fallback
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

print("Starting SpatChat SDM (Together → optional HF Serverless fallback)")

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
HF_TOKEN         = os.getenv("HF_TOKEN", "")  # optional fallback
HF_FALLBACK_MODEL = os.getenv("HF_FALLBACK_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# Together client (lazy import)
Together = None
if TOGETHER_API_KEY:
    try:
        from together import Together as TogetherClient
        Together = TogetherClient(api_key=TOGETHER_API_KEY)
    except Exception as e:
        print("⚠️ Together client not available:", e)

# Predictors we expose
PREDICTOR_CHOICES = (
    [f"bio{i}" for i in range(1, 20)]
    + ["elevation", "slope", "aspect", "ndvi", "landcover"]
)
VALID_LAYERS = {p.lower() for p in PREDICTOR_CHOICES}

LANDCOVER_CLASSES = {
    c.lower() for c in (
        "water", "evergreen_needleleaf_forest", "evergreen_broadleaf_forest",
        "deciduous_needleleaf_forest", "deciduous_broadleaf_forest", "mixed_forest",
        "closed_shrublands", "open_shrublands", "woody_savannas", "savannas",
        "grasslands", "permanent_wetlands", "croplands", "urban_and_built_up",
        "cropland_natural_vegetation_mosaic", "snow_and_ice", "barren_or_sparsely_vegetated"
    )
}

LAYER_DOCS = {
    "bio": "WorldClim v1 Bioclimatic variables (WORLDCLIM/V1/BIO, ~1 km). bio1=Annual Mean Temp, bio12=Annual Precip, etc.",
    "elevation": "USGS SRTM GL1 v003 elevation (USGS/SRTMGL1_003, 30 m).",
    "slope": "Terrain slope derived from SRTM GL1 (30 m).",
    "aspect": "Terrain aspect derived from SRTM GL1 (30 m).",
    "ndvi": "MODIS NDVI mean (MODIS/061/MOD13Q1, 2022-01-01 to 2024-01-01), 250 m.",
    "landcover": "MODIS IGBP Land Cover Type 1 (MODIS/061/MCD12Q1, 500 m).",
}

# Pre-render colorbar → base64
fig, ax = plt.subplots(figsize=(4, 0.5))
norm = Normalize(vmin=0, vmax=1)
plt.colorbar(ScalarMappable(norm=norm, cmap="viridis"), cax=ax, orientation="horizontal").set_ticks([])
ax.set_xlabel("Low    High")
fig.tight_layout(pad=0)
_buf = io.BytesIO()
fig.savefig(_buf, format="png", dpi=100)
plt.close(fig)
_buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(_buf.read()).decode()

# Earth Engine auth
svc_json = os.environ.get("GEE_SERVICE_ACCOUNT", "")
if not svc_json:
    print("⚠️ GEE_SERVICE_ACCOUNT is not set. Earth Engine calls will fail.")
else:
    try:
        svc = json.loads(svc_json)
        creds = ee.ServiceAccountCredentials(svc.get("client_email"), key_data=json.dumps(svc))
        ee.Initialize(creds)
        print("✅ Earth Engine authenticated.")
    except Exception as e:
        print("❌ Failed EE auth:", e)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: filesystem, map, zips
# ──────────────────────────────────────────────────────────────────────────────
def clear_all():
    for d in ("predictor_rasters", "outputs", "inputs"):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs("inputs", exist_ok=True)
    if os.path.exists("spatchat_results.zip"):
        os.remove("spatchat_results.zip")
    os.environ.pop("SELECTED_LAYERS", None)
    os.environ.pop("SELECTED_LANDCOVER_CLASSES", None)

clear_all()

def create_map():
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)

    ppath = "inputs/presence_points.csv"
    if os.path.exists(ppath):
        try:
            df = pd.read_csv(ppath)
            lat_col, lon_col = detect_coords(df)
            if lat_col and lon_col:
                pts = df[[lat_col, lon_col]].dropna().values.tolist()
                if pts:
                    fg = folium.FeatureGroup(name="🟦 Presence Points")
                    for lat, lon in pts:
                        folium.CircleMarker(location=[lat, lon], radius=5, color="blue", fill=True, fill_opacity=0.8).add_to(fg)
                    fg.add_to(m)
                    m.fit_bounds(pts)
        except Exception as e:
            print("Warn reading CSV for map:", e)

    # predictor overlays
    rasdir = "predictor_rasters/wgs84"
    if os.path.isdir(rasdir):
        for fn in sorted(os.listdir(rasdir)):
            if not fn.endswith(".tif"):
                continue
            try:
                with rasterio.open(os.path.join(rasdir, fn)) as src:
                    arr = src.read(1)
                    bnd = src.bounds
                vmin, vmax = np.nanmin(arr), np.nanmax(arr)
                if not np.isnan(vmin) and vmin != vmax:
                    rgba = colormaps["viridis"]((arr - vmin) / (vmax - vmin))
                    folium.raster_layers.ImageOverlay(
                        rgba,
                        bounds=[[bnd.bottom, bnd.left], [bnd.top, bnd.right]],
                        opacity=1.0,  # fully opaque
                        name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})"
                    ).add_to(m)
            except Exception as e:
                print(f"Overlay warn for {fn}:", e)

    # suitability (opaque)
    sf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        try:
            with rasterio.open(sf) as src:
                arr = src.read(1)
                bnd = src.bounds
            vmin, vmax = np.nanmin(arr), np.nanmax(arr)
            rgba = colormaps["viridis"]((arr - vmin) / (vmax - vmin))
            folium.raster_layers.ImageOverlay(
                rgba, bounds=[[bnd.bottom, bnd.left], [bnd.top, bnd.right]],
                opacity=1.0,
                name="🎯 Suitability"
            ).add_to(m)
        except Exception as e:
            print("Suitability overlay warn:", e)

    folium.LayerControl(collapsed=False).add_to(m)
    img_html = f'<img src="data:image/png;base64,{COLORBAR_BASE64}" style="position:absolute; bottom:20px; right:10px; width:200px; height:30px; z-index:1000;"/>'
    m.get_root().html.add_child(Element(img_html))
    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" style="width:100%; height:450px; border:none;"></iframe>'

def zip_results():
    archive = "spatchat_results.zip"
    if os.path.exists(archive): os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters", "outputs"):
            for root, _, files in os.walk(fld):
                for fn in files:
                    full = os.path.join(root, fn)
                    zf.write(full, arcname=os.path.relpath(full, "."))
    return archive

# ──────────────────────────────────────────────────────────────────────────────
# Column detection & CRS helpers
# ──────────────────────────────────────────────────────────────────────────────
def detect_coords(df: pd.DataFrame, fuzz_threshold=80) -> Tuple[str, str]:
    cols = list(df.columns)
    low = [c.lower().strip() for c in cols]
    LAT = {'lat','latitude','y','y_coordinate','decilatitude','dec_latitude','dec lat','decimallatitude','decimal latitude'}
    LON = {'lon','long','longitude','x','x_coordinate','decilongitude','dec_longitude','dec longitude','decimallongitude','decimal longitude'}

    lat_idx = next((i for i, n in enumerate(low) if n in LAT), None)
    lon_idx = next((i for i, n in enumerate(low) if n in LON), None)
    if lat_idx is not None and lon_idx is not None:
        return cols[lat_idx], cols[lon_idx]

    lat_fz = difflib.get_close_matches("latitude", low, n=1, cutoff=fuzz_threshold/100)
    lon_fz = difflib.get_close_matches("longitude", low, n=1, cutoff=fuzz_threshold/100)
    if lat_fz and lon_fz:
        return cols[low.index(lat_fz[0])], cols[low.index(lon_fz[0])]

    numerics = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    lat_opts = [c for c in numerics if df[c].between(-90, 90).mean() > 0.98]
    lon_opts = [c for c in numerics if df[c].between(-180, 180).mean() > 0.98]
    if len(lat_opts) == 1 and len(lon_opts) == 1:
        return lat_opts[0], lon_opts[0]
    return None, None

def resolve_crs(raw: str) -> int:
    raw = (raw or "").strip()
    if not raw:
        return 4326
    m = re.match(r"^(\d{4,5})$", raw)
    if m:
        return int(m.group(1))
    m = re.search(r"utm\s*zone\s*(\d+)\s*([NS])", raw, re.I)
    if m:
        zone, hemi = int(m.group(1)), m.group(2).upper()
        return (32600 if hemi == 'N' else 32700) + zone

    # LLM parse (optional)
    if Together:
        try:
            resp = Together.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[
                    {"role":"system","content":"You're a GIS expert. Answer only as JSON: {\"epsg\": #### or null}."},
                    {"role":"user","content":f"CRS: '{raw}'"}
                ],
                temperature=0.0
            ).choices[0].message.content
            code = json.loads(resp).get("epsg")
            return int(code) if code else 4326
        except Exception:
            return 4326
    return 4326

# ──────────────────────────────────────────────────────────────────────────────
# LLM Router (Together → HF fallback)
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are SpatChat, a helpful SDM assistant. When the user requests one of these intents,
respond with **JSON only** (no prose) using the exact schema below.

Schemas:
{"tool":"fetch","layers":["bio1","ndvi",...],"landcover":["water","urban_and_built_up",...]}
{"tool":"run_model"}
{"tool":"explain_stats"}
{"tool":"download"}

Rules:
- Map natural phrasings like "download", "get", "grab", "fetch", "add", "i want" to {"tool":"fetch",...}
- If the user specifies just names (e.g., "bio1, bio12"), put them into "layers".
- Landcover subclasses go into "landcover".
- If the user asks to run the model, return {"tool":"run_model"}.
- If the user asks to download outputs/zip, return {"tool":"download"}.
- If the user asks to "explain/interpret/understand the stats, numbers, results, metrics",
  return {"tool":"explain_stats"}.
- Otherwise, reply with a short helpful sentence (not JSON).
""".strip()

FALLBACK_PROMPT = "You are SpatChat (SDM). Be brief and helpful."

def call_llm(messages: List[dict]) -> str:
    if Together:
        try:
            out = Together.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=messages,
                temperature=0.0
            ).choices[0].message.content
            return out
        except Exception as e:
            print("Together error:", e)

    if HF_AVAILABLE and HF_TOKEN:
        try:
            client = InferenceClient(model=HF_FALLBACK_MODEL, token=HF_TOKEN)
            prompt = ""
            for m in messages:
                prompt += f"{m.get('role','user').upper()}: {m.get('content','')}\n"
            prompt += "ASSISTANT: "
            out = client.text_generation(prompt, max_new_tokens=256, temperature=0.2, do_sample=False)
            return out
        except Exception as e:
            print("HF fallback error:", e)

    return "(LLM unavailable) If you want to fetch layers, say e.g. 'I want bio1, ndvi'."

def llm_route(user_msg: str, history: List[dict]) -> dict:
    messages = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":user_msg}]
    raw = call_llm(messages)
    try:
        call = json.loads(raw)
        if isinstance(call, dict) and "tool" in call:
            return call
    except Exception:
        pass
    return {"tool":"text","text": raw}

# ──────────────────────────────────────────────────────────────────────────────
# Local interceptors (no LLM)
# ──────────────────────────────────────────────────────────────────────────────
EXPLAIN_STATS_RE = re.compile(
    r"(explain|interpret|help me understand|make sense of|what do (these|those) (numbers|stats|results|metrics) mean|"
    r"explain (these|those)? (stats|numbers|results|metrics))",
    re.I,
)
def is_explain_stats_query(text: str) -> bool:
    return bool(EXPLAIN_STATS_RE.search(text or ""))

LAYER_QUESTION_RE = re.compile(
    r"\b(what\s+is|where\s+does|source\s+of|which\s+dataset\s+is)\b.*\b(bio\d{1,2}|ndvi|elevation|slope|aspect|landcover)\b",
    re.I,
)
def maybe_answer_layer_info(user_msg: str) -> str | None:
    m = LAYER_QUESTION_RE.search(user_msg or "")
    if not m:
        return None
    token = re.search(r"(bio\d{1,2}|ndvi|elevation|slope|aspect|landcover)", user_msg, re.I)
    if not token:
        return None
    key = token.group(1).lower()
    if key.startswith("bio"):
        return f"**{key.upper()}** — {LAYER_DOCS['bio']}"
    return f"**{key}** — {LAYER_DOCS.get(key, 'No description available.')}"

# ──────────────────────────────────────────────────────────────────────────────
# Actions
# ──────────────────────────────────────────────────────────────────────────────
def run_fetch(sl: List[str], lc: List[str]) -> Tuple[str, str]:
    layers = [l.strip().lower() for l in sl or [] if l and l.strip()]
    landc = [c.strip().lower() for c in lc or [] if c and c.strip()]
    if landc:
        layers.append("landcover")

    if not layers and not landc:
        return create_map(), "⚠️ Please specify at least one layer (e.g., 'I want bio1, ndvi')."

    bad_layers = [l for l in layers if l not in VALID_LAYERS]
    if bad_layers:
        sug = []
        for b in bad_layers:
            match = difflib.get_close_matches(b, VALID_LAYERS, n=1, cutoff=0.6)
            if match: sug.append(f"Did you mean '{match[0]}' instead of '{b}'?")
        return create_map(), "⚠️ " + (" ".join(sug) if sug else f"Unknown: {', '.join(bad_layers)}.")

    bad_codes = [c for c in landc if c not in LANDCOVER_CLASSES]
    if bad_codes:
        sug = []
        for b in bad_codes:
            match = difflib.get_close_matches(b, LANDCOVER_CLASSES, n=1, cutoff=0.6)
            if match: sug.append(f"Did you mean landcover '{match[0]}' instead of '{b}'?")
        return create_map(), "⚠️ " + (" ".join(sug) if sug else f"Unknown landcover: {', '.join(bad_codes)}.")

    os.environ["SELECTED_LAYERS"] = ",".join([l for l in layers if l != "landcover"] + (["landcover"] if "landcover" in layers else []))
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(landc)

    cmd = [sys.executable, "-u", os.path.join("scripts", "fetch_predictors.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return create_map(), f"❌ Fetch failed:\n```\n{logs}\n```"
    return create_map(), f"✅ Predictors fetched.\n\n```bash\n{logs}\n```"

def explain_latest_stats() -> str:
    perf_fp = "outputs/performance_metrics.csv"
    coef_fp = "outputs/coefficients.csv"

    have_any = False
    out = ["**Summary of recent model results**"]
    if os.path.exists(perf_fp):
        try:
            perf = pd.read_csv(perf_fp)
            have_any = True
            out.append("\n**Performance metrics**")
            out.append(perf.to_markdown(index=False))
            # helpful highlights if present
            for col in ("AUC", "Accuracy", "F1", "Precision", "Recall"):
                if col in perf.columns:
                    try:
                        val = float(perf[col].iloc[0])
                        out.append(f"- {col}: **{val:.3f}**")
                    except Exception:
                        pass
        except Exception as e:
            out.append(f"(Couldn't read performance: {e})")

    if os.path.exists(coef_fp):
        try:
            coef = pd.read_csv(coef_fp).dropna(axis=1, how='all')
            have_any = True
            out.append("\n**Coefficients**")
            out.append(coef.to_markdown(index=False))
        except Exception as e:
            out.append(f"(Couldn't read coefficients: {e})")

    if not have_any:
        return "I couldn't find any saved results yet. Try **run model**, then ask me to explain the stats."
    return "\n".join(out)

def run_model() -> Tuple[str, str]:
    """
    Train model, then return map + a rich chat message that includes the stats summary.
    """
    proc = subprocess.run(["python", "scripts/run_logistic_sdm.py"], capture_output=True, text=True)
    if proc.returncode != 0:
        return create_map(), f"❌ Model run failed:\n```\n{proc.stderr}\n```"

    # Package results
    zip_results()

    # Compose stats into the success message
    stats_md = explain_latest_stats()
    msg = "✅ Model trained. Download results below (ZIP).\n\n" + stats_md + "\n\n_Tip: you can also type **\"explain those stats\"** any time to reprint this summary._"
    return create_map(), msg

# ──────────────────────────────────────────────────────────────────────────────
# Chat loop
# ──────────────────────────────────────────────────────────────────────────────
LONG_HELP = (
    "Now say things like **\"I want bio1, ndvi, elevation\"** to fetch layers, or **\"run model\"** to train & predict.\n\n"
    "### 📦 Available layers (exact datasets)\n"
    "- **Bioclim:** bio1–bio19 — WorldClim v1 bioclim normals (WORLDCLIM/V1/BIO, ~1 km)\n"
    "- **Topography:** elevation, slope, aspect — USGS/SRTMGL1_003 (30 m)\n"
    "- **Remote sensing:** ndvi — MODIS/061/MOD13Q1 mean (2022-01-01→2024-01-01) (250 m)\n"
    "- **Landcover:** MODIS IGBP classes — MODIS/061/MCD12Q1 LC_Type1 (500 m)\n\n"
    "### 💬 Say it like this\n"
    "• \"I want bio1, ndvi, elevation\" (fetch layers)\n"
    "• \"Fetch bio5, bio12 and slope\"\n"
    "• \"Add landcover water and urban_and_built_up\"\n"
    "• \"Run model\" (train & predict)\n"
    "• \"What is bio5?\" / \"Where does ndvi come from?\"\n"
    "• \"Explain those stats\" (summarize latest performance & coefficients)\n"
)

def chat_step(file, user_msg, history, state):
    if not os.path.exists("inputs/presence_points.csv"):
        reply = "Please upload your presence CSV to begin."
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":reply}])
        return history, create_map(), state

    # Local intercepts first
    ans = maybe_answer_layer_info(user_msg)
    if ans:
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":ans}])
        return history, create_map(), state

    if is_explain_stats_query(user_msg):
        ans = explain_latest_stats()
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":ans}])
        return history, create_map(), state

    # Route with LLM
    call = llm_route(user_msg, history)
    tool = call.get("tool")

    if tool == "fetch":
        m_out, status = run_fetch(call.get("layers", []), call.get("landcover", []))
        assistant_txt = status

    elif tool == "run_model":
        m_out, status = run_model()
        assistant_txt = status  # includes stats summary now

    elif tool == "download":
        m_out, _ = create_map(), zip_results()
        assistant_txt = "✅ Preparing ZIP… Click **Download Results**."

    elif tool == "explain_stats":
        m_out, assistant_txt = create_map(), explain_latest_stats()

    elif tool == "text":
        m_out, assistant_txt = create_map(), call.get("text", "I'm here to help!")

    else:
        m_out, assistant_txt = create_map(), "I can fetch layers (e.g., 'I want bio1, ndvi') or **run model**."

    history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":assistant_txt}])
    return history, m_out, state

# ──────────────────────────────────────────────────────────────────────────────
# Upload & CRS confirm
# ──────────────────────────────────────────────────────────────────────────────
def on_upload(f, history, state):
    history2 = history.copy()
    clear_all()

    if f and hasattr(f, "name"):
        shutil.copy(f.name, "inputs/presence_points.csv")
        df = pd.read_csv("inputs/presence_points.csv")
        lat, lon = detect_coords(df)
        if lat and lon:
            df = df.rename(columns={lat: "latitude", lon: "longitude"})
            df.to_csv("inputs/presence_points.csv", index=False)
            history2.append({"role":"assistant","content":(
                "✅ I found your **latitude** and **longitude** columns.\n\n" + LONG_HELP
            )})
            return history2, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            history2.append({"role":"assistant","content":"I couldn't detect coordinate columns. Please select them and enter CRS below."})
            cols = list(df.columns)
            return history2, create_map(), state, gr.update(choices=cols, visible=True), gr.update(choices=cols, visible=True), gr.update(visible=True), gr.update(visible=True)

    history2.append({"role":"assistant","content":"Please upload a presence CSV to begin."})
    return history2, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def confirm_coords(lat_col, lon_col, crs_raw, history, state):
    df = pd.read_csv("inputs/presence_points.csv")
    try:
        src_epsg = resolve_crs(crs_raw) if crs_raw else 4326
    except Exception:
        history.append({"role":"assistant","content":"Sorry, I couldn't recognize that CRS. Please try another format (e.g., 32610, 'UTM zone 10N')."})
        return history, create_map(), state, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    src_crs = RioCRS.from_epsg(int(src_epsg))
    dst_crs = RioCRS.from_epsg(4326)
    lon_vals, lat_vals = rio_transform(src_crs, dst_crs, df[lon_col].tolist(), df[lat_col].tolist())
    df['latitude'], df['longitude'] = lat_vals, lon_vals
    df.to_csv("inputs/presence_points.csv", index=False)

    history.append({"role":"assistant","content":"✅ Coordinates set!\n\n" + LONG_HELP})
    return history, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="SpatChat: SDM") as demo:
    gr.Image(value="logo_long1.png", show_label=False, show_download_button=False, show_share_button=False, type="filepath", elem_id="logo-img")
    gr.HTML("""
    <style>
    #logo-img img { height: 90px; margin: 10px 50px 10px 10px; border-radius: 6px; }
    </style>
    """)
    gr.Markdown("## 🗺️ SpatChat: Species Distribution Model {sdm}  🐢🐍🦅🦋🦉🦊🐞 ")

    state = gr.State({"stage":"await_upload"})

    with gr.Row():
        with gr.Column(scale=1):
            map_out = gr.HTML(create_map(), label="🗺️ Map Preview")
            download_btn = gr.DownloadButton("📥 Download Results", zip_results)
        with gr.Column(scale=1):
            chat = gr.Chatbot(
                value=[{"role":"assistant","content":"👋 Hello, I'm SpatChat (SDM)! Upload a presence CSV to begin."}],
                type="messages", label="💬 Chat", height=430
            )
            user_in = gr.Textbox(label="Ask SpatChat", placeholder='e.g., "I want bio1, ndvi, elevation" or "run model" or "explain those stats"')

            file_input = gr.File(label="📄 Upload Presence CSV", type="filepath", file_types=[".csv"])
            lat_dropdown = gr.Dropdown(choices=[], label="Latitude column", visible=False)
            lon_dropdown = gr.Dropdown(choices=[], label="Longitude column", visible=False)
            crs_input = gr.Textbox(label="Input CRS (code, zone, or name)", placeholder="e.g. 32610, UTM zone 10N", visible=False)
            confirm_btn = gr.Button("Confirm Coordinates", visible=False)

    file_input.change(on_upload, inputs=[file_input, chat, state],
                      outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn])
    confirm_btn.click(confirm_coords, inputs=[lat_dropdown, lon_dropdown, crs_input, chat, state],
                      outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn])

    user_in.submit(chat_step, inputs=[file_input, user_in, chat, state],
                   outputs=[chat, map_out, state])
    user_in.submit(lambda: "", None, user_in)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
