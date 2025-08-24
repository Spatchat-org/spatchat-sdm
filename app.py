import os
import io
import sys
import re
import json
import time
import base64
import shutil
import zipfile
import difflib
from typing import List, Tuple, Optional

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

# Optional LLMs
from together import Together
try:
    # HF Serverless client (optional fallback)
    from huggingface_hub import InferenceClient as HFClient
except Exception:
    HFClient = None

print("Starting SpatChat SDM (Together → HF Serverless fallback)")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
HF_TOKEN         = os.getenv("HF_TOKEN", "")

TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
HF_MODEL       = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")  # small, fast fallback

client_together = Together(api_key=TOGETHER_API_KEY) if TOGETHER_API_KEY else None
client_hf = HFClient(token=HF_TOKEN) if (HF_TOKEN and HFClient is not None) else None

# Which top-level predictors we support (all lower-case)
PREDICTOR_CHOICES = [f"bio{i}" for i in range(1, 20)] + ["elevation", "slope", "aspect", "ndvi", "landcover"]
VALID_LAYERS = {p.lower() for p in PREDICTOR_CHOICES}

# All available MODIS landcover classes (snake_case)
LANDCOVER_CLASSES = {
    c.lower() for c in (
        "water", "evergreen_needleleaf_forest", "evergreen_broadleaf_forest",
        "deciduous_needleleaf_forest", "deciduous_broadleaf_forest", "mixed_forest",
        "closed_shrublands", "open_shrublands", "woody_savannas", "savannas",
        "grasslands", "permanent_wetlands", "croplands", "urban_and_built_up",
        "cropland_natural_vegetation_mosaic", "snow_and_ice", "barren_or_sparsely_vegetated"
    )
}

# Pre-render a colorbar → base64 for map UI
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

# Authenticate Earth Engine via Service Account (fail soft)
try:
    svc = json.loads(os.environ.get("GEE_SERVICE_ACCOUNT", "{}"))
    creds = ee.ServiceAccountCredentials(svc.get("client_email"), key_data=json.dumps(svc))
    ee.Initialize(creds)
    print("✅ Earth Engine authenticated successfully!")
except Exception as e:
    print(f"⚠️ Earth Engine init skipped/failed: {e}", file=sys.stderr)

# Paths
INPUT_CSV = "inputs/presence_points.csv"
PRED_DIR_RAW = "predictor_rasters/raw"
PRED_DIR_WGS = "predictor_rasters/wgs84"
OUTPUTS_DIR  = "outputs"
ZIP_PATH     = "spatchat_results.zip"

# ──────────────────────────────────────────────────────────────────────────────
# LLM prompts (tool JSON only)
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are SpatChat (SDM), a species distribution modeling helper.
When the user asks to fetch environmental layers (verbs like fetch, download, get, grab, add, include, "I want ..."), respond with EXACT JSON:
{"tool":"fetch","layers":[<layer names>],"landcover":[<landcover classes>]}
When the user asks to run the model ("run model", "build model", "train", "fit", "predict", "build an SDM"), respond with EXACT:
{"tool":"run_model"}
If the user asks to explain the model results ("explain those stats", "what do those numbers mean?"), respond:
{"tool":"explain_stats"}
Otherwise, reply NATURALLY in ≤2 sentences (no JSON).
Valid top-level layers: bio1–bio19, elevation, slope, aspect, ndvi, landcover.
Valid landcover classes (snake_case): water, urban_and_built_up, croplands, grasslands, etc.
""".strip()

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def clear_all():
    for d in ("predictor_rasters", OUTPUTS_DIR, "inputs"):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs("inputs", exist_ok=True)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

# Never crash on markdown table formatting (no tabulate dependency)
def _safe_table(df, title=None, max_rows=50, max_cols=30) -> str:
    if df is None:
        return ""
    df2 = df.copy()
    if df2.shape[0] > max_rows:
        df2 = df2.head(max_rows)
    if df2.shape[1] > max_cols:
        df2 = df2.iloc[:, :max_cols]
    head = f"**{title}**\n\n" if title else ""
    try:
        return head + df2.to_markdown(index=False)
    except Exception:
        return head + "```\n" + df2.to_string(index=False) + "\n```"

def _split_perf_tables(perf_df) -> Tuple[str, str]:
    if perf_df is None or perf_df.empty:
        return "", ""
    try:
        first, second = perf_df.iloc[:, :3], perf_df.iloc[:, 3:]
    except Exception:
        first, second = perf_df, None
    t1 = _safe_table(first, "Performance (1/2)")
    t2 = _safe_table(second, "Performance (2/2)") if (second is not None and second.shape[1] > 0) else ""
    return t1, t2

def explain_those_stats() -> str:
    perf_fp = os.path.join(OUTPUTS_DIR, "performance_metrics.csv")
    coef_fp = os.path.join(OUTPUTS_DIR, "coefficients.csv")
    parts = []
    try:
        if os.path.exists(perf_fp):
            perf = pd.read_csv(perf_fp)
            t1, t2 = _split_perf_tables(perf)
            if t1: parts.append(t1)
            if t2: parts.append(t2)
    except Exception as e:
        parts.append(f"⚠️ Couldn't read performance metrics: {e}")
    try:
        if os.path.exists(coef_fp):
            coef = pd.read_csv(coef_fp).dropna(axis=1, how='all')
            if not coef.empty:
                parts.append(_safe_table(coef, "Predictor Coefficients"))
    except Exception as e:
        parts.append(f"⚠️ Couldn't read coefficients: {e}")
    return parts and "\n\n".join(parts) or "I don’t see any stats yet—try **run model** first."

def zip_results() -> str:
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in ("predictor_rasters", OUTPUTS_DIR):
            if not os.path.isdir(folder):
                continue
            for root, _, files in os.walk(folder):
                for fn in files:
                    full = os.path.join(root, fn)
                    zf.write(full, arcname=os.path.relpath(full, "."))
    return ZIP_PATH

def detect_coords(df: pd.DataFrame, fuzz_threshold=80):
    cols = list(df.columns)
    low  = [c.lower().strip() for c in cols]
    LAT_ALIASES = {'lat','latitude','y','y_coordinate','decilatitude','dec_latitude','dec lat','decimallatitude','decimal latitude'}
    LON_ALIASES = {'lon','long','longitude','x','x_coordinate','decilongitude','dec_longitude','dec longitude','decimallongitude','decimal longitude'}
    lat_idx = next((i for i,n in enumerate(low) if n in LAT_ALIASES), None)
    lon_idx = next((i for i,n in enumerate(low) if n in LON_ALIASES), None)
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

def parse_epsg_code(s):
    m = re.match(r"^(\d{4,5})$", s.strip())
    return int(m.group(1)) if m else None

def parse_utm_crs(s):
    m = re.search(r"utm\s*zone\s*(\d+)\s*([NS])", s, re.I)
    if m:
        zone, hemi = int(m.group(1)), m.group(2).upper()
        return (32600 if hemi=='N' else 32700) + zone
    return None

def resolve_crs(raw):
    code = parse_epsg_code(raw) or parse_utm_crs(raw)
    return code if code else 4326

def available_layers_help() -> str:
    return (
        "Now say things like **\"I want bio1, ndvi, elevation\"** to fetch layers, or **\"run model\"** to train & predict.\n\n"
        "📦 **Available layers (exact datasets)**\n"
        "• **Bioclim:** `bio1–bio19` — WorldClim v1 bioclim normals (**WORLDCLIM/V1/BIO**, ~1 km)\n"
        "• **Topography:** `elevation`, `slope`, `aspect` — **USGS/SRTMGL1_003** (30 m)\n"
        "• **Remote sensing:** `ndvi` — **MODIS/061/MOD13Q1** mean (2022-01-01→2024-01-01) (250 m)\n"
        "• **Landcover:** MODIS IGBP classes — **MODIS/061/MCD12Q1** `LC_Type1` (500 m)\n\n"
        "💬 Say it like this\n"
        "• \"I want bio1, ndvi, elevation\" (fetch layers)\n"
        "• \"Fetch bio5, bio12 and slope\"\n"
        "• \"Add landcover water and urban_and_built_up\"\n"
        "• \"Run model\" (train & predict)\n"
        "• \"What is bio5?\" / \"Where does ndvi come from?\"\n"
        "• \"Explain those stats\" (summarize latest performance & coefficients)\n"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Map
# ──────────────────────────────────────────────────────────────────────────────

def create_map() -> str:
    m = folium.Map(location=[0,0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)
    if os.path.exists(INPUT_CSV):
        try:
            df = pd.read_csv(INPUT_CSV)
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
            print(f"⚠️ map load csv failed: {e}", file=sys.stderr)

    # Predictor rasters
    if os.path.isdir(PRED_DIR_WGS):
        for fn in sorted(os.listdir(PRED_DIR_WGS)):
            if not fn.endswith(".tif"): 
                continue
            try:
                with rasterio.open(os.path.join(PRED_DIR_WGS, fn)) as src:
                    arr = src.read(1); bnd = src.bounds
                vmin, vmax = np.nanmin(arr), np.nanmax(arr)
                if not np.isnan(vmin) and vmin!=vmax:
                    rgba = colormaps["viridis"]((arr-vmin)/(vmax-vmin))
                    folium.raster_layers.ImageOverlay(
                        rgba, bounds=[[bnd.bottom,bnd.left],[bnd.top,bnd.right]],
                        opacity=1.0, name=f"🟨 {fn}"
                    ).add_to(m)
            except Exception as e:
                print(f"⚠️ map overlay failed for {fn}: {e}", file=sys.stderr)

    # Suitability (opaque)
    sf = os.path.join(OUTPUTS_DIR, "suitability_map_wgs84.tif")
    if os.path.exists(sf):
        try:
            with rasterio.open(sf) as src:
                arr = src.read(1); bnd = src.bounds
            vmin, vmax = np.nanmin(arr), np.nanmax(arr)
            rgba = colormaps["viridis"]((arr - vmin) / (vmax - vmin))
            folium.raster_layers.ImageOverlay(
                rgba, bounds=[[bnd.bottom,bnd.left],[bnd.top,bnd.right]],
                opacity=1.0, name="🎯 Suitability"
            ).add_to(m)
        except Exception as e:
            print(f"⚠️ suitability overlay failed: {e}", file=sys.stderr)

    folium.LayerControl(collapsed=False).add_to(m)
    img_html = f'<img src="data:image/png;base64,{COLORBAR_BASE64}" style="position:absolute; bottom:20px; right:10px; width:200px; height:30px; z-index:1000;"/>'
    m.get_root().html.add_child(Element(img_html))
    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" style="width:100%; height:450px; border:none;"></iframe>'

# ──────────────────────────────────────────────────────────────────────────────
# LLM wrapper (Together → HF fallback), JSON-only tool calls
# ──────────────────────────────────────────────────────────────────────────────

def llm_tool_call(history_msgs: List[dict], user_msg: str) -> Tuple[Optional[dict], Optional[str]]:
    """Return (tool_json, fallback_text). If no tool, fallback_text may be a normal reply."""
    messages = [{"role":"system","content":SYSTEM_PROMPT}] + history_msgs + [{"role":"user","content":user_msg}]

    # 1) Try Together (if available)
    if client_together:
        try:
            resp = client_together.chat.completions.create(
                model=TOGETHER_MODEL, messages=messages, temperature=0.0
            ).choices[0].message.content
            try:
                call = json.loads(resp)
                return call, None
            except Exception:
                # not a tool → natural reply
                return None, resp
        except Exception as e:
            # Rate limit or other → continue to HF
            err = str(e)
            print(f"Together failed: {err}", file=sys.stderr)

    # 2) HF Serverless fallback (if available)
    if client_hf:
        try:
            # simple chat format
            prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(
                [f"{m['role'].upper()}: {m['content']}" for m in history_msgs + [{"role":"user","content":user_msg}]]
            )
            resp = client_hf.text_generation(prompt, model=HF_MODEL, max_new_tokens=256, temperature=0.0)
            # Try to parse JSON first
            try:
                call = json.loads(resp.strip())
                return call, None
            except Exception:
                return None, resp.strip()
        except Exception as e:
            print(f"HF fallback failed: {e}", file=sys.stderr)

    # 3) No LLM → none
    return None, None

# ──────────────────────────────────────────────────────────────────────────────
# Lightweight local intent fallback (regex) when LLM is unavailable
# ──────────────────────────────────────────────────────────────────────────────

LAYER_PATTERN = re.compile(r"\b(bio1?\d|bio[1-9]|bio1[0-9]|elevation|slope|aspect|ndvi|land ?cover)\b", re.I)

def local_intent(user_msg: str) -> Optional[dict]:
    u = user_msg.lower()

    # Explain stats?
    if re.search(r"\b(explain|what.*mean|help.*understand).*(stat|number|result|metric|coef)", u):
        return {"tool":"explain_stats"}

    # Run model?
    if re.search(r"\b(run|train|fit|build|predict)\b.*\b(model|sdm)\b", u) or re.fullmatch(r"\s*run model\s*", u):
        return {"tool":"run_model"}

    # Fetch layers?
    if re.search(r"\b(fetch|get|grab|download|add|include|want|need|pull|use)\b", u):
        layers = set()
        landcodes = set()
        for m in LAYER_PATTERN.finditer(u.replace(" ", "")):
            token = m.group(0).replace(" ", "")
            token = token.lower()
            if token.startswith("landcover") or token == "landcover":
                layers.add("landcover")
            elif token in VALID_LAYERS:
                layers.add(token)
        # also detect landcover subclasses
        for lab in LANDCOVER_CLASSES:
            if re.search(rf"\b{re.escape(lab)}\b", u):
                landcodes.add(lab)
        if layers:
            return {"tool":"fetch", "layers": sorted(layers), "landcover": sorted(landcodes)}
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Fetch & model runners
# ──────────────────────────────────────────────────────────────────────────────

def run_fetch(sl: List[str], lc: List[str]) -> Tuple[str, str]:
    # Require at least one predictor
    layers = [l for l in sl if l]
    if not layers and not lc:
        return create_map(), "⚠️ Please specify at least one layer."

    # Validate predictors
    bad_layers = [l for l in layers if l not in VALID_LAYERS]
    if bad_layers:
        suggestions = []
        for b in bad_layers:
            match = difflib.get_close_matches(b, list(VALID_LAYERS), n=1, cutoff=0.6)
            if match:
                suggestions.append(f"Did you mean '{match[0]}' instead of '{b}'?")
        msg = "⚠️ " + (" ".join(suggestions) if suggestions else f"Unknown layer(s): {', '.join(bad_layers)}.")
        return create_map(), msg

    # Validate landcover subclasses
    bad_codes = [c for c in lc if c not in LANDCOVER_CLASSES]
    if bad_codes:
        suggestions = []
        for b in bad_codes:
            match = difflib.get_close_matches(b, list(LANDCOVER_CLASSES), n=1, cutoff=0.6)
            if match:
                suggestions.append(f"Did you mean landcover class '{match[0]}' instead of '{b}'?")
        msg = "⚠️ " + (" ".join(suggestions) if suggestions else f"Unknown landcover classes: {', '.join(bad_codes)}.")
        return create_map(), msg

    # Marshal env and invoke export script
    os.environ["SELECTED_LAYERS"] = ",".join(layers + (["landcover"] if lc else []))
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(lc)

    cmd = [sys.executable, "-u", os.path.join("scripts", "fetch_predictors.py")]
    proc = shutil.which(sys.executable)
    print(f"🧪 [run_fetch] layers={layers}   landcover={lc}   py={proc}", file=sys.stdout)

    result = subprocess.run(cmd, capture_output=True, text=True)
    logs = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    if result.returncode != 0:
        return create_map(), f"❌ Fetch failed:\n```\n{logs}\n```"
    else:
        return create_map(), f"✅ Predictors fetched.\n\n```bash\n{logs}\n```"

def run_model():
    proc = subprocess.run([sys.executable, os.path.join("scripts","run_logistic_sdm.py")], capture_output=True, text=True)
    if proc.returncode != 0:
        return create_map(), f"❌ Model run failed:\n```\n{proc.stderr}\n```", None, None
    # Try to load tables for immediate display
    perf_df = None
    coef_df = None
    try:
        perf_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "performance_metrics.csv"))
    except Exception:
        pass
    try:
        coef_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "coefficients.csv")).dropna(axis=1, how='all')
    except Exception:
        pass
    zip_results()
    return create_map(), "✅ Model trained. Download results below (ZIP).", perf_df, coef_df

# ──────────────────────────────────────────────────────────────────────────────
# Gradio callbacks
# ──────────────────────────────────────────────────────────────────────────────

def on_upload(f, history, state):
    history2 = list(history)
    # Reset only outputs & predictors (keep inputs folder)
    shutil.rmtree("predictor_rasters", ignore_errors=True)
    shutil.rmtree(OUTPUTS_DIR, ignore_errors=True)
    os.makedirs("inputs", exist_ok=True)
    if f and hasattr(f, "name"):
        shutil.copy(f.name, INPUT_CSV)
        df = pd.read_csv(INPUT_CSV)
        lat, lon = detect_coords(df)
        if lat and lon:
            df = df.rename(columns={lat: "latitude", lon: "longitude"})
            df.to_csv(INPUT_CSV, index=False)
            history2.append({"role":"assistant","content":"✅ I found your **latitude** and **longitude** columns."})
            state["stage"] = "ready"
            if not state.get("showed_help_ready"):
                history2.append({"role":"assistant","content":available_layers_help()})
                state["showed_help_ready"] = True
            return history2, create_map(), state
        else:
            # ask for columns & CRS
            state["stage"] = "need_coords"
            cols = list(df.columns)
            history2.append({"role":"assistant","content":"I couldn't detect coordinate columns. Please tell me which columns are latitude/longitude and your CRS (e.g., 'UTM zone 10N' or '4326')."})
            state["awaiting_crs"] = True
            state["columns"] = cols
            return history2, create_map(), state
    return history2, create_map(), state

def chat_step(user_msg, history, state):
    history2 = list(history)
    state = dict(state or {})
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return history2, create_map(), state

    # Quick path: coordinate confirmation if needed
    if state.get("stage") == "need_coords":
        try:
            df = pd.read_csv(INPUT_CSV)
            # crude parse: look for two known column names in user's text
            cols = state.get("columns", list(df.columns))
            u = user_msg.lower()
            chosen_lat = next((c for c in cols if c.lower() in u and re.search(r"\b(lat|latitude|y)\b", c.lower()+u)), None)
            chosen_lon = next((c for c in cols if c.lower() in u and re.search(r"\b(lon|long|longitude|x)\b", c.lower()+u)), None)
            # CRS parse
            crs_match = re.search(r"\b(\d{4,5}|utm\s*zone\s*\d+\s*[ns])\b", u, re.I)
            epsg = resolve_crs(crs_match.group(0)) if crs_match else 4326

            if chosen_lat and chosen_lon:
                src_epsg = epsg
                src_crs = RioCRS.from_epsg(src_epsg)
                dst_crs = RioCRS.from_epsg(4326)
                lon_vals, lat_vals = rio_transform(src_crs, dst_crs, df[chosen_lon].tolist(), df[chosen_lat].tolist())
                df['latitude'], df['longitude'] = lat_vals, lon_vals
                df.to_csv(INPUT_CSV, index=False)
                history2.append({"role":"assistant","content":"✅ Coordinates set! Converted to EPSG:4326."})
                state["stage"] = "ready"
                if not state.get("showed_help_ready"):
                    history2.append({"role":"assistant","content":available_layers_help()})
                    state["showed_help_ready"] = True
                return history2, create_map(), state
            else:
                history2.append({"role":"assistant","content":"I still need the latitude and longitude column names (and optionally the CRS)."})
                return history2, create_map(), state
        except Exception as e:
            history2.append({"role":"assistant","content":f"Sorry, couldn't set coordinates: {e}"})
            return history2, create_map(), state

    # If no CSV yet, keep it conversational (short)
    if not os.path.exists(INPUT_CSV):
        history2.append({"role":"user","content":user_msg})
        history2.append({"role":"assistant","content":"Please upload a presence CSV to begin."})
        return history2, create_map(), state

    # 1) Try LLM for tool call
    call, normal_reply = llm_tool_call(history2, user_msg)

    # 2) Local intent fallback when no tool/LLM
    if call is None:
        call = local_intent(user_msg)

    # 3) Dispatch
    if call and call.get("tool") == "fetch":
        layers = call.get("layers", [])
        landcover = call.get("landcover", [])
        m_out, status = run_fetch(layers, landcover)
        history2.extend([
            {"role":"user","content":user_msg},
            {"role":"assistant","content":status}
        ])
        state["stage"] = "layers_fetched"
        return history2, m_out, state

    if call and call.get("tool") == "run_model":
        m_out, status, perf_df, coef_df = run_model()
        # Build chat text with safe tables
        perf_md = ""
        coef_md = ""
        try:
            if perf_df is None and os.path.exists(os.path.join(OUTPUTS_DIR, "performance_metrics.csv")):
                perf_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "performance_metrics.csv"))
            if perf_df is not None and not perf_df.empty:
                t1, t2 = _split_perf_tables(perf_df)
                perf_md = ("\n\n" + t1) + (("\n\n" + t2) if t2 else "")
            if coef_df is None and os.path.exists(os.path.join(OUTPUTS_DIR, "coefficients.csv")):
                coef_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "coefficients.csv")).dropna(axis=1, how='all')
            if coef_df is not None and not coef_df.empty:
                coef_md = "\n\n" + _safe_table(coef_df, "Predictor Coefficients")
        except Exception as e:
            perf_md += f"\n\n⚠️ Couldn't format tables: {e}"

        assistant_txt = status + (perf_md or "") + (coef_md or "")
        history2.extend([
            {"role":"user","content":user_msg},
            {"role":"assistant","content":assistant_txt}
        ])
        state["stage"] = "model_ran"
        return history2, m_out, state

    if call and call.get("tool") == "explain_stats":
        summary = explain_those_stats()
        history2.extend([
            {"role":"user","content":user_msg},
            {"role":"assistant","content":summary}
        ])
        return history2, create_map(), state

    # 4) Not a tool → normal text (from LLM if present, else context help)
    history2.append({"role":"user","content":user_msg})
    if normal_reply:
        history2.append({"role":"assistant","content":normal_reply})
    else:
        # Step-aware micro help
        stage = state.get("stage", "ready")
        if stage == "ready":
            history2.append({"role":"assistant","content":"If you want to fetch layers, say e.g. **\"I want bio1, ndvi\"**. To train, say **\"run model\"**."})
        elif stage == "layers_fetched":
            history2.append({"role":"assistant","content":"Layers are ready. Say **\"run model\"** to train & predict, or fetch more layers."})
        elif stage == "model_ran":
            history2.append({"role":"assistant","content":"Model completed. You can **Download Results**, or say **\"Explain those stats\"** for a summary."})
        else:
            history2.append({"role":"assistant","content":"How can I help? Upload CSV, fetch layers, or run model."})
    return history2, create_map(), state

# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="SpatChat: SDM") as demo:
    gr.Image(
        value="logo_long1.png",
        show_label=False,
        show_download_button=False,
        show_share_button=False,
        type="filepath",
        elem_id="logo-img"
    )
    gr.HTML("""
    <style>
    #logo-img img { height: 90px; margin: 10px 50px 10px 10px; border-radius: 6px; }
    </style>
    """)
    gr.Markdown("## 🗺️ SpatChat: Species Distribution Model {sdm}  🐢🐍🦅🦋🦉🦊🐞 ")
    gr.Markdown("👋 Hello, I'm **SpatChat (SDM)**! Upload a presence CSV to begin.")

    state = gr.State({"stage":"await_upload", "showed_help_ready": False})
    with gr.Row():
        with gr.Column(scale=1):
            map_out = gr.HTML(create_map(), label="🗺️ Map Preview")
            download_btn = gr.DownloadButton("📥 Download Results", value=lambda: zip_results())
        with gr.Column(scale=1):
            chat = gr.Chatbot(type="messages", label="💬 Chat", height=420, value=[])
            user_in = gr.Textbox(label="Ask SpatChat", placeholder="Type commands (e.g., I want bio1, ndvi)…", lines=1)
            file_input = gr.File(label="📄 Upload Presence CSV", type="filepath", file_types=[".csv"])

    file_input.change(on_upload, inputs=[file_input, chat, state], outputs=[chat, map_out, state])
    user_in.submit(chat_step, inputs=[user_in, chat, state], outputs=[chat, map_out, state])
    user_in.submit(lambda: "", None, user_in)

# Avoid deprecated args; enable queue for reliability
demo.queue().launch(ssr_mode=False)
