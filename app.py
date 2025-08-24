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
import time
import random

import gradio as gr
import geemap.foliumap as foliumap  # noqa: F401 (kept for parity with other rooms)
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

# --- Optional LLM clients ---
from together import Together
from together.error import RateLimitError as TogetherRateLimitError  # for precise catch
from huggingface_hub import InferenceClient

from rasterio.crs import CRS as RioCRS
from rasterio.warp import transform as rio_transform

print("Starting SpatChat SDM (with Together → HF Serverless fallback)")

# =========================================
# Globals & constants
# =========================================
# Predictor menu (lowercase)
PREDICTOR_CHOICES = [f"bio{i}" for i in range(1, 20)] + ["elevation", "slope", "aspect", "ndvi", "landcover"]
VALID_LAYERS = {p.lower() for p in PREDICTOR_CHOICES}

LANDCOVER_CLASSES = {
    c.lower()
    for c in (
        "water",
        "evergreen_needleleaf_forest",
        "evergreen_broadleaf_forest",
        "deciduous_needleleaf_forest",
        "deciduous_broadleaf_forest",
        "mixed_forest",
        "closed_shrublands",
        "open_shrublands",
        "woody_savannas",
        "savannas",
        "grasslands",
        "permanent_wetlands",
        "croplands",
        "urban_and_built_up",
        "cropland_natural_vegetation_mosaic",
        "snow_and_ice",
        "barren_or_sparsely_vegetated",
    )
}

# Basic duplicate-prompt guard
LAST_PROMPT_SIG = {"text": None, "ts": 0.0}

# Rate-limit cooldown after repeated 429s
LLM_BLOCKED_UNTIL = 0.0

# =========================================
# Pre-render tiny colorbar → base64
# =========================================
fig, ax = plt.subplots(figsize=(4, 0.5))
norm = Normalize(vmin=0, vmax=1)
plt.colorbar(ScalarMappable(norm=norm, cmap="viridis"), cax=ax, orientation="horizontal").set_ticks([])
ax.set_xlabel("Low    High")
fig.tight_layout(pad=0)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100)
plt.close(fig)
buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(buf.read()).decode()

# =========================================
# Earth Engine auth
# =========================================
load_dotenv()
svc = json.loads(os.environ.get("GEE_SERVICE_ACCOUNT", "{}"))
creds = ee.ServiceAccountCredentials(svc.get("client_email"), key_data=json.dumps(svc))
ee.Initialize(creds)

# =========================================
# LLM setup: Together + HF serverless fallback
# =========================================
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
client = Together(api_key=TOGETHER_API_KEY) if TOGETHER_API_KEY else None

HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
hf_client = InferenceClient(model=HF_MODEL, token=HF_TOKEN) if HF_TOKEN else None

# Short tool-call prompt (keep it tiny)
SYSTEM_PROMPT = (
    "You are SpatChat SDM. If user asks to fetch predictors, respond EXACTLY:\n"
    '{"tool":"fetch","layers":[...],"landcover":[...]}\n'
    "If user asks to run the model, respond EXACTLY:\n"
    '{"tool":"run_model"}\n'
    "Otherwise answer briefly (<=2 sentences)."
)

FALLBACK_PROMPT = (
    "You are SpatChat SDM (concise). Be brief (<=2 sentences). Help with upload, fetching layers, and running the model."
)

def _messages_to_prompt(messages):
    """Minimal chat→instruct stitching for HF text-generation."""
    sys_txt = ""
    parts = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "").strip()
        if role == "system":
            sys_txt = content
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    prompt = (sys_txt + "\n\n" if sys_txt else "") + "\n".join(parts) + "\nAssistant:"
    return prompt

def chat_completion(messages, temperature=0.0, json_expected=False, max_new_tokens=256, max_retries=2):
    """
    Try Together first with exponential backoff + jitter. If rate-limited/unavailable, try HF Serverless.
    If both fail, raise RuntimeError('llm_rate_limited'|'llm_unavailable').
    """
    global LLM_BLOCKED_UNTIL

    now = time.time()
    # If we're in a cooldown, skip Together immediately
    use_together = bool(client) and now >= LLM_BLOCKED_UNTIL

    if use_together:
        delay = 0.8
        for attempt in range(max_retries + 1):
            try:
                out = client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                ).choices[0].message.content
                return out
            except TogetherRateLimitError:
                # Backoff with jitter
                if attempt < max_retries:
                    time.sleep(delay * random.uniform(0.9, 1.3))
                    delay *= 1.9
                    continue
                # Set cooldown to avoid storms
                LLM_BLOCKED_UNTIL = time.time() + 35
                break
            except Exception:
                # Unknown Together error → try HF
                break

    if hf_client:
        try:
            prompt = _messages_to_prompt(messages)
            out = hf_client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=1.05,
                do_sample=(temperature > 0),
                return_full_text=False,
                timeout=25,
            )
            return out
        except Exception:
            # fall through to hard failure
            pass

    # Both failed
    if time.time() < LLM_BLOCKED_UNTIL:
        raise RuntimeError("llm_rate_limited")
    raise RuntimeError("llm_unavailable")

# =========================================
# Utilities
# =========================================
def clear_all():
    for d in ("predictor_rasters", "outputs", "inputs"):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs("inputs", exist_ok=True)
    for fp in ("spatchat_results.zip",):
        if os.path.exists(fp):
            os.remove(fp)

def detect_coords(df, fuzz_threshold=80):
    cols = list(df.columns)
    low = [c.lower().strip() for c in cols]
    LAT = {'lat','latitude','y','y_coordinate','decilatitude','dec_latitude','dec lat',
           'decimallatitude','decimal latitude'}
    LON = {'lon','long','longitude','x','x_coordinate','decilongitude','dec_longitude',
           'dec longitude','decimallongitude','decimal longitude'}
    # Exact
    lat_idx = next((i for i, n in enumerate(low) if n in LAT), None)
    lon_idx = next((i for i, n in enumerate(low) if n in LON), None)
    if lat_idx is not None and lon_idx is not None:
        return cols[lat_idx], cols[lon_idx]
    # Fuzzy
    lat_fz = difflib.get_close_matches("latitude", low, n=1, cutoff=fuzz_threshold/100)
    lon_fz = difflib.get_close_matches("longitude", low, n=1, cutoff=fuzz_threshold/100)
    if lat_fz and lon_fz:
        return cols[low.index(lat_fz[0])], cols[low.index(lon_fz[0])]
    # Numeric heuristics
    numerics = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    lat_opts = [c for c in numerics if pd.to_numeric(df[c], errors="coerce").between(-90, 90).mean() > 0.98]
    lon_opts = [c for c in numerics if pd.to_numeric(df[c], errors="coerce").between(-180, 180).mean() > 0.98]
    if len(lat_opts) == 1 and len(lon_opts) == 1:
        return lat_opts[0], lon_opts[0]
    return None, None

def parse_epsg_code(s):
    m = re.match(r"^(\d{4,5})$", s.strip())
    return int(m.group(1)) if m else None

def parse_utm_crs(s):
    m = re.search(r"utm\s*zone\s*(\d+)\s*([NS])", s, re.I)
    if m:
        zone, hemi = int(m.group(1)), m.group(2).upper()
        return (32600 if hemi == 'N' else 32700) + zone
    return None

def resolve_crs(raw):
    # Try pure parsers first, skip LLM to avoid rate-limit here
    for fn in (parse_epsg_code, parse_utm_crs):
        code = fn(raw)
        if code:
            return code
    # Fallback via LLM only if clients exist
    if client or hf_client:
        try:
            resp = chat_completion(
                [
                    {"role": "system", "content": 'GIS expert. Return only JSON like {"epsg": ####} or {"epsg": null}.'},
                    {"role": "user", "content": f"CRS: '{raw}'"},
                ],
                temperature=0.0,
                json_expected=True,
                max_new_tokens=64,
            )
            data = json.loads(resp) if isinstance(resp, str) else {}
            epsg = data.get("epsg")
            if epsg:
                return int(epsg)
        except Exception:
            pass
    raise ValueError("Unable to parse CRS")

# =========================================
# Local intent parser (avoid LLM calls)
# =========================================
_FETCH_VERBS = r"(?:fetch|get|grab|download|add|want|need|use|select|choose|pull)"
_LAYER_PATTERN = r"(?:bio(?:1[0-9]?|[2-9])|elevation|slope|aspect|ndvi|landcover)"
def parse_intent_local(user_msg):
    """
    Return {"tool":"fetch","layers":[...],"landcover":[...]} or {"tool":"run_model"} or None.
    """
    text = user_msg.strip().lower()
    if re.fullmatch(r"\s*(?:run\s+)?model\s*|^\s*run\s*$", text, re.I):
        return {"tool": "run_model"}

    # match "I want bio1, ndvi and elevation"
    if re.search(_FETCH_VERBS, text, re.I):
        # find all layer tokens from VALID_LAYERS (robust to punctuation)
        tokens = re.findall(_LAYER_PATTERN, text)
        layers = []
        landcover = []
        for t in tokens:
            t = t.lower()
            if t == "landcover":
                landcover.append("all")  # semantic flag; server-side script expects env var only
            elif t in VALID_LAYERS:
                layers.append(t)
        # dedupe
        layers = sorted(set(layers))
        landcover = sorted(set(landcover))
        if layers or landcover:
            return {"tool": "fetch", "layers": layers, "landcover": landcover}
    return None

# =========================================
# Map & I/O helpers
# =========================================
def create_map():
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)

    ppath = "inputs/presence_points.csv"
    if os.path.exists(ppath):
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

    rasdir = "predictor_rasters/wgs84"
    if os.path.isdir(rasdir):
        for fn in sorted(os.listdir(rasdir)):
            if fn.endswith(".tif"):
                with rasterio.open(os.path.join(rasdir, fn)) as src:
                    arr = src.read(1)
                    bnd = src.bounds
                vmin, vmax = np.nanmin(arr), np.nanmax(arr)
                if not np.isnan(vmin) and vmin != vmax:
                    rgba = colormaps["viridis"]((arr - vmin) / (vmax - vmin + 1e-12))
                    folium.raster_layers.ImageOverlay(
                        rgba, bounds=[[bnd.bottom, bnd.left], [bnd.top, bnd.right]], opacity=1.0, name=f"🟨 {fn}"
                    ).add_to(m)

    sf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        with rasterio.open(sf) as src:
            arr = src.read(1)
            bnd = src.bounds
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        if not np.isnan(vmin) and vmin != vmax:
            rgba = colormaps["viridis"]((arr - vmin) / (vmax - vmin + 1e-12))
            folium.raster_layers.ImageOverlay(
                rgba, bounds=[[bnd.bottom, bnd.left], [bnd.top, bnd.right]], opacity=0.7, name="🎯 Suitability"
            ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    img_html = f'<img src="data:image/png;base64,{COLORBAR_BASE64}" style="position:absolute; bottom:20px; right:10px; width:200px; height:30px; z-index:1000;"/>'
    m.get_root().html.add_child(Element(img_html))
    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" style="width:100%; height:450px; border:none;"></iframe>'

def zip_results():
    archive = "spatchat_results.zip"
    if os.path.exists(archive):
        os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters", "outputs"):
            for root, _, files in os.walk(fld):
                for fn in files:
                    full = os.path.join(root, fn)
                    zf.write(full, arcname=os.path.relpath(full, "."))
    return archive

def run_fetch(sl, lc):
    # 1) Make request list
    layers = list(sl)
    if lc:
        layers.append("landcover")

    if not layers:
        return create_map(), "⚠️ Please select at least one predictor."

    # 2) Validate predictors
    bad_layers = [l for l in layers if l not in VALID_LAYERS]
    if bad_layers:
        suggestions = []
        for b in bad_layers:
            match = difflib.get_close_matches(b, VALID_LAYERS, n=1, cutoff=0.6)
            if match:
                suggestions.append(f"Did you mean '{match[0]}' instead of '{b}'?")
        if suggestions:
            return create_map(), "⚠️ " + " ".join(suggestions)
        return create_map(), f"⚠️ Unknown predictors: {', '.join(bad_layers)}."

    # 3) Validate landcover subclasses (if you add filtering later)
    # (Currently just accepts 'landcover' as a top-level flag)

    # 4) Execute fetch script
    print(f"🧪 [run_fetch] SL={sl!r}   LC={lc!r}", file=sys.stdout)
    os.environ["SELECTED_LAYERS"] = ",".join(sl)
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(lc)  # may be 'all' or blank

    cmd = [sys.executable, "-u", os.path.join("scripts", "fetch_predictors.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return create_map(), f"❌ Fetch failed:\n```\n{logs}\n```"
    return create_map(), f"✅ Predictors fetched.\n\n```bash\n{logs}\n```"

def run_model():
    proc = subprocess.run(["python", "scripts/run_logistic_sdm.py"], capture_output=True, text=True)
    if proc.returncode != 0:
        return create_map(), f"❌ Model run failed:\n```\n{proc.stderr}\n```", None, None
    perf_df = pd.read_csv("outputs/performance_metrics.csv")
    coef_df = pd.read_csv("outputs/coefficients.csv")
    zip_results()
    return create_map(), "✅ Model ran successfully! Download using the button below the map.", perf_df, coef_df

# =========================================
# Chat handler
# =========================================
def chat_step(file, user_msg, history, state):
    # Debounce: ignore identical prompt in rapid succession
    sig = (user_msg or "").strip()
    now = time.time()
    if sig and sig == LAST_PROMPT_SIG["text"] and (now - LAST_PROMPT_SIG["ts"] < 1.5):
        # no state change, just return
        return history, create_map(), state
    LAST_PROMPT_SIG["text"] = sig
    LAST_PROMPT_SIG["ts"] = now

    # If no CSV yet → short fallback reply (no LLM needed)
    if not os.path.exists("inputs/presence_points.csv"):
        reply = (
            "Hi! Upload your presence CSV to start. "
            "Then say things like: 'I want bio1, ndvi' or 'run model'."
        )
        history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}])
        return history, create_map(), state

    # Shortcut: explicit "run model"
    if re.fullmatch(r"\s*(?:run\s+)?model\s*$", user_msg or "", re.I):
        m_out, status, perf_df, coef_df = run_model()
        if perf_df is not None:
            perf = pd.read_csv("outputs/performance_metrics.csv")
            first, second = perf.iloc[:, :3], perf.iloc[:, 3:]
            perf_md = (
                "**Model Performance (1 of 2):**\n\n"
                + first.to_markdown(index=False)
                + "\n\n**Model Performance (2 of 2):**\n\n"
                + second.to_markdown(index=False)
            )
            coef = pd.read_csv("outputs/coefficients.csv").dropna(axis=1, how="all")
            status += "\n\n**Model Performance:**\n\n" + perf_md
            status += "\n\n**Predictor Coefficients:**\n\n" + coef.to_markdown(index=False)
        history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": status}])
        return history, m_out, state

    # Local parse first (avoid LLM calls for common requests)
    parsed = parse_intent_local(user_msg or "")
    if parsed and parsed.get("tool") == "fetch":
        m_out, status = run_fetch(parsed.get("layers", []), parsed.get("landcover", []))
        assistant_txt = f"{status}\n\nGreat! You can say **run model** when ready."
        history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_txt}])
        return history, m_out, state
    if parsed and parsed.get("tool") == "run_model":
        m_out, status, perf_df, coef_df = run_model()
        history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": status}])
        return history, m_out, state

    # Build compact LLM messages for tool-call mode
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_msg},
    ]

    # LLM call with Together → HF fallback
    try:
        resp = chat_completion(msgs, temperature=0.0, json_expected=True, max_new_tokens=160)
        try:
            call = json.loads(resp)
        except Exception:
            call = None
    except RuntimeError as e:
        # Graceful, actionable answer with no promises
        tag = str(e)
        hint = (
            "I can't reach the model right now. You can still do:\n"
            "• **I want bio1, ndvi** (fetch layers)\n"
            "• **run model**\n"
            "• Or upload a different CSV."
        )
        history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": hint}])
        return history, create_map(), state

    # Dispatch tool calls
    if call and call.get("tool") == "fetch":
        m_out, status = run_fetch(call.get("layers", []), call.get("landcover", []))
        assistant_txt = f"{status}\n\nGreat! You can say **run model** when ready."
    elif call and call.get("tool") == "run_model":
        m_out, status, perf_df, coef_df = run_model()
        assistant_txt = status
    else:
        # Natural language fallback (short)
        try:
            nl = chat_completion(
                [{"role": "system", "content": FALLBACK_PROMPT}, {"role": "user", "content": user_msg}],
                temperature=0.3,
                json_expected=False,
                max_new_tokens=120,
            )
        except Exception:
            nl = "I can fetch predictors (e.g., 'I want bio1, ndvi') or run the model ('run model')."
        assistant_txt = nl
        m_out = create_map()

    history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_txt}])
    return history, m_out, state

# =========================================
# Upload & CRS handlers
# =========================================
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
            history2.append(
                {
                    "role": "assistant",
                    "content": (
                        "✅ Found your coordinate columns.\n"
                        "Now say things like **'I want bio1, ndvi, elevation'** to fetch layers, "
                        "or **'run model'** to train & predict."
                    ),
                }
            )
            return history2, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            history2.append({"role": "assistant", "content": "I couldn't detect coordinate columns. Pick them and enter the CRS below."})
            cols = list(df.columns)
            return history2, create_map(), state, gr.update(choices=cols, visible=True), gr.update(choices=cols, visible=True), gr.update(visible=True), gr.update(visible=True)
    return history2, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def confirm_coords(lat_col, lon_col, crs_raw, history, state):
    df = pd.read_csv("inputs/presence_points.csv")
    try:
        src_epsg = resolve_crs(crs_raw) if crs_raw else 4326
    except Exception:
        history.append({"role": "assistant", "content": "Sorry, I couldn't recognize that CRS. Try another format (e.g., 32610, 'UTM zone 10N')."})
        return history, create_map(), state, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    src_crs = RioCRS.from_epsg(int(src_epsg))
    dst_crs = RioCRS.from_epsg(4326)
    lon_vals, lat_vals = rio_transform(src_crs, dst_crs, df[lon_col].tolist(), df[lat_col].tolist())
    df["latitude"], df["longitude"] = lat_vals, lon_vals
    df.to_csv("inputs/presence_points.csv", index=False)
    history.append(
        {
            "role": "assistant",
            "content": (
                "✅ Coordinates set!\n"
                "Fetch predictors with **'I want bio1, ndvi'** (add more like elevation/slope/aspect/landcover), "
                "then **'run model'**."
            ),
        }
    )
    return history, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

# =========================================
# UI
# =========================================
with gr.Blocks(title="SpatChat: SDM") as demo:
    gr.Image(value="logo_long1.png", show_label=False, show_download_button=False, show_share_button=False, type="filepath", elem_id="logo-img")
    gr.HTML(
        """
    <style>
    #logo-img img { height: 90px; margin: 10px 50px 10px 10px; border-radius: 6px; }
    </style>
    """
    )
    gr.Markdown("## 🗺️ SpatChat: Species Distribution Model {sdm}  🐢🐍🦅🦋🦉🦊🐞 ")
    gr.HTML(
        """
    <div style="margin-top: -10px; margin-bottom: 15px;">
      <input type="text" value="https://spatchat.org/browse/?room=sdm" id="shareLink" readonly style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
      <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">📋 Copy Share Link</button>
      <div style="margin-top: 10px; font-size: 14px;">
        <b>Share:</b>
        <a href="https://twitter.com/intent/tweet?text=Checkout+Spatchat!&url=https://spatchat.org/browse/?room=sdm" target="_blank">🐦 Twitter</a> |
        <a href="https://www.facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=sdm" target="_blank">📘 Facebook</a>
      </div>
    </div>
    """
    )
    gr.Markdown(
        """
        <div style="font-size: 14px;">
        © 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
        If you use Spatchat in research, please cite:<br>
        <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>Spatchat: Species Distribution Model.</i>
        </div>
        """
    )

    state = gr.State({"stage": "await_upload"})
    with gr.Row():
        with gr.Column(scale=1):
            map_out = gr.HTML(create_map(), label="🗺️ Map Preview")
            download_btn = gr.DownloadButton("📥 Download Results", zip_results)
        with gr.Column(scale=1):
            chat = gr.Chatbot(
                value=[
                    {
                        "role": "assistant",
                        "content": "👋 Hello! Upload your presence CSV to begin. Then say: **'I want bio1, ndvi'** or **'run model'**.",
                    }
                ],
                type="messages",
                label="💬 Chat",
                height=400,
            )
            user_in = gr.Textbox(label="Ask SpatChat", placeholder="e.g., I want bio1, ndvi", lines=1)
            file_input = gr.File(label="📄 Upload Presence CSV", type="filepath", file_types=[".csv"])
            lat_dropdown = gr.Dropdown(choices=[], label="Latitude column", visible=False)
            lon_dropdown = gr.Dropdown(choices=[], label="Longitude column", visible=False)
            crs_input = gr.Textbox(label="Input CRS (code, zone, or name)", placeholder="e.g. 32610, UTM zone 10N", visible=False)
            confirm_btn = gr.Button("Confirm Coordinates", visible=False)

    file_input.change(
        on_upload,
        inputs=[file_input, chat, state],
        outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn],
        concurrency_limit=1,
    )
    confirm_btn.click(
        confirm_coords,
        inputs=[lat_dropdown, lon_dropdown, crs_input, chat, state],
        outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn],
        concurrency_limit=1,
    )
    user_in.submit(chat_step, inputs=[file_input, user_in, chat, state], outputs=[chat, map_out, state], concurrency_limit=1)
    user_in.submit(lambda: "", None, user_in)

# Queue with low concurrency to avoid bursty duplicates; disable SSR to reduce duplicate backend mounts
try:
    demo.queue(max_size=32).launch(ssr_mode=False)
except TypeError:
    demo.launch(ssr_mode=False)