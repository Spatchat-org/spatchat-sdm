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
import requests

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

# Optional Together client (we'll handle missing key gracefully)
try:
    from together import Together
    from together import error as togerr
except Exception:
    Together = None
    class togerr:  # lightweight shims
        class RateLimitError(Exception): ...
        class APIError(Exception): ...
        class AuthenticationError(Exception): ...
        class Timeout(Exception): ...

# =========================
# Config / Environment
# =========================
load_dotenv()
TOGETHER_API_KEY   = os.getenv("TOGETHER_API_KEY", "")
HF_SERVERLESS_URL  = os.getenv("HF_SERVERLESS_URL", "")  # e.g. https://api-inference.huggingface.co/models/<org>/<model>
HF_API_TOKEN       = os.getenv("HF_API_TOKEN", "")
MODEL_NAME         = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")

# EE auth (primary app only; fetch script also authenticates)
svc = json.loads(os.environ.get("GEE_SERVICE_ACCOUNT", "{}"))
if svc:
    try:
        creds = ee.ServiceAccountCredentials(svc.get("client_email"), key_data=json.dumps(svc))
        ee.Initialize(creds)
    except Exception as e:
        print(f"⚠️ Earth Engine init warning: {e}")
else:
    print("⚠️ GEE_SERVICE_ACCOUNT not set; scripts/fetch_predictors.py will still attempt auth.")

# Together client (optional)
client = None
if Together and TOGETHER_API_KEY:
    try:
        client = Together(api_key=TOGETHER_API_KEY)
    except Exception as e:
        print(f"⚠️ Together client init warning: {e}")
else:
    print("ℹ️ TOGETHER_API_KEY not set or together lib missing; will rely on HF/local routing.")

# =========================
# Predictors / Landcover
# =========================
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

# =========================
# Colorbar (base64)
# =========================
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

# =========================
# Utilities
# =========================
def clear_all():
    for d in ("predictor_rasters", "outputs", "inputs"):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs("inputs", exist_ok=True)
    csv_fp = "inputs/presence_points.csv"
    if os.path.exists(csv_fp):
        os.remove(csv_fp)
    os.environ.pop("SELECTED_LAYERS", None)
    os.environ.pop("SELECTED_LANDCOVER_CLASSES", None)
    if os.path.exists("spatchat_results.zip"):
        os.remove("spatchat_results.zip")

clear_all()

def detect_coords(df, fuzz_threshold=80):
    cols = list(df.columns)
    low  = [c.lower().strip() for c in cols]
    LAT_ALIASES = {
        'lat','latitude','y','y_coordinate','decilatitude','dec_latitude','dec lat',
        'decimallatitude','decimal latitude'
    }
    LON_ALIASES = {
        'lon','long','longitude','x','x_coordinate','decilongitude','dec_longitude',
        'dec longitude','decimallongitude','decimal longitude'
    }
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

def llm_parse_crs(raw):
    if not client:
        raise ValueError("LLM unavailable")
    system = {"role":"system","content":"You're a GIS expert. Given a CRS description, respond with only JSON {\"epsg\": ###} or {\"epsg\": null}."}
    user = {"role":"user","content":f"CRS: '{raw}'"}
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[system, user], temperature=0.0
    ).choices[0].message.content
    code = json.loads(resp).get("epsg")
    if not code:
        raise ValueError("LLM couldn't parse CRS")
    return code

def resolve_crs(raw):
    for fn in (parse_epsg_code, parse_utm_crs):
        code = fn(raw)
        if code:
            return code
    return llm_parse_crs(raw)

def zip_results():
    archive = "spatchat_results.zip"
    if os.path.exists(archive): os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters","outputs"):
            if not os.path.isdir(fld): 
                continue
            for root,_,files in os.walk(fld):
                for fn in files:
                    full = os.path.join(root,fn)
                    zf.write(full, arcname=os.path.relpath(full,"."))
    return archive

def fetched_layer_names():
    rasdir = "predictor_rasters/wgs84"
    if os.path.isdir(rasdir):
        return sorted(os.path.splitext(f)[0] for f in os.listdir(rasdir) if f.endswith(".tif"))
    return []

def have_outputs():
    return os.path.exists("outputs/performance_metrics.csv") or os.path.exists("outputs/coefficients.csv")

# =========================
# Map
# =========================
def create_map():
    m = folium.Map(location=[0,0], zoom_start=2, control_scale=True)
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
                    arr = src.read(1); bnd = src.bounds
                vmin, vmax = np.nanmin(arr), np.nanmax(arr)
                if not np.isnan(vmin) and vmin!=vmax:
                    rgba = colormaps["viridis"]((arr-vmin)/(vmax-vmin))
                    folium.raster_layers.ImageOverlay(rgba, bounds=[[bnd.bottom,bnd.left],[bnd.top,bnd.right]], opacity=1.0, name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})").add_to(m)
    sf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        with rasterio.open(sf) as src:
            arr = src.read(1); bnd = src.bounds
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        rgba = colormaps["viridis"]((arr-vmin)/(vmax-vmin))
        # NON-transparent by default, per request
        folium.raster_layers.ImageOverlay(rgba, bounds=[[bnd.bottom,bnd.left],[bnd.top,bnd.right]], opacity=1.0, name="🎯 Suitability").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    img_html = f'<img src="data:image/png;base64,{COLORBAR_BASE64}" style="position:absolute; bottom:20px; right:10px; width:200px; height:30px; z-index:1000;"/>'
    m.get_root().html.add_child(Element(img_html))
    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" style="width:100%; height:450px; border:none;"></iframe>'

# =========================
# LLM Prompts
# =========================
SYSTEM_PROMPT = """You are SpatChat (SDM). Respond with JSON ONLY for intents below.

Schema:
{"tool": "<fetch|run_model|explain_stats|list_layers|help>", "args": {...}}

Intents:
- fetch: {"layers": ["bio1","bio2","ndvi","slope","elevation","aspect","landcover"], "landcover": ["urban_and_built_up","permanent_wetlands", ...] }
  Accept natural phrasing like "bio 2", "bio-2", "urban", "wetlands", "get me ndvi", "can you help me download ndvi".
- run_model: {}
- explain_stats: {}  # user asks: explain those numbers/stats/results/metrics
- list_layers: {}    # user asks what’s available or about datasets
- help: {"reply": "<short natural language reply>"}  # all other questions

Examples:
User: can you help me download ndvi and bio 2?
Assistant: {"tool":"fetch","args":{"layers":["ndvi","bio2"],"landcover":[]}}

User: add landcover urban and wetlands
Assistant: {"tool":"fetch","args":{"layers":["landcover"],"landcover":["urban_and_built_up","permanent_wetlands"]}}

User: run model
Assistant: {"tool":"run_model","args":{}}

User: can you explain those numbers?
Assistant: {"tool":"explain_stats","args":{}}

User: what layers do you have?
Assistant: {"tool":"list_layers","args":{}}
""".strip()

FALLBACK_PROMPT = """You are SpatChat (SDM). Be concise (≤2 sentences) and helpful about the SDM workflow: upload CSV, fetch layers, run model, explain outputs.""".strip()

# =========================
# LLM Router (Together → HF → Local)
# =========================
def _normalize_bio_tokens(t: str) -> str:
    return re.sub(r"\bbio\s*[-_ ]*\s*(\d{1,2})\b",
                  lambda m: f"bio{int(m.group(1))}", t, flags=re.I)

_LC_SYNONYMS = {
    "urban": "urban_and_built_up",
    "wetland": "permanent_wetlands",
    "wetlands": "permanent_wetlands",
    "barren": "barren_or_sparsely_vegetated",
    "snow": "snow_and_ice",
    "grassland": "grasslands",
    "savanna": "savannas",
}

def _local_light_parse(user_text: str):
    """Minimal backstop. Handles fetch/run_model/explain_stats/list_layers heuristics."""
    t = user_text.lower().strip()
    if re.search(r"\brun (?:the )?model\b|\brun sdm\b|\btrain\b", t):
        return {"tool":"run_model","args":{}}
    if re.search(r"\b(explain|understand|interpret)\b.+\b(stats?|numbers?|metrics?|results?)\b", t):
        return {"tool":"explain_stats","args":{}}
    if re.search(r"\bwhat (layers|datasets)|available (layers|datasets)|what can i fetch\b", t):
        return {"tool":"list_layers","args":{}}
    # fetch detection
    if re.search(r"\b(get|fetch|download|add|i want|add layer|grab)\b", t) or re.search(r"\b(bio|ndvi|elevation|slope|aspect|land ?cover)\b", t):
        t = _normalize_bio_tokens(t)
        layers = []
        for k in VALID_LAYERS:
            if k=="landcover": continue
            if re.search(rf"\b{k}\b", t):
                layers.append(k)
        want_lc = bool(re.search(r"\bland\s*cover|landcover\b", t))
        land = []
        for lc in LANDCOVER_CLASSES:
            if re.search(rf"\b{re.escape(lc)}\b", t):
                land.append(lc)
        for k,v in _LC_SYNONYMS.items():
            if re.search(rf"\b{k}\b", t):
                land.append(v)
        if want_lc and "landcover" not in layers:
            layers.append("landcover")
        layers = list(dict.fromkeys(layers))
        land   = list(dict.fromkeys(land))
        if layers or land or want_lc:
            return {"tool":"fetch","args":{"layers":layers or (["landcover"] if want_lc else []), "landcover": land}}
    return None

def _extract_json(s: str):
    if not s or not isinstance(s, str): return None
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _together_route(messages, timeout=12):
    if not client:
        return {"_error":"together-not-configured"}
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            timeout=timeout,
        )
        return resp.choices[0].message.content
    except (togerr.RateLimitError, togerr.APIError, togerr.AuthenticationError, togerr.Timeout) as e:
        return {"_error": f"{type(e).__name__}: {e}"}
    except Exception as e:
        return {"_error": f"Unknown: {e}"}

def _hf_serverless_route(messages, timeout=12):
    if not HF_SERVERLESS_URL or not HF_API_TOKEN:
        return {"_error": "hf-not-configured"}
    try:
        # Simple serverless: feed last user message + rely on model prompt adherence
        payload = {"inputs": messages[-1]["content"], "parameters": {"max_new_tokens": 256, "temperature": 0.0}}
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        r = requests.post(HF_SERVERLESS_URL, json=payload, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return {"_error": f"HF status {r.status_code}: {r.text[:200]}"}
        data = r.json()
        # Some endpoints return a list of dicts (text-generation-inference)
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        if isinstance(data, str):
            return data
        # try generic
        return str(data)
    except Exception as e:
        return {"_error": str(e)}

def llm_route(user_text: str, history: list):
    sys_msg = {"role":"system","content": SYSTEM_PROMPT}
    messages = [sys_msg] + history[-6:] + [{"role":"user","content": user_text}]

    r = _together_route(messages)
    if isinstance(r, str):
        call = _extract_json(r)
        if call: return (call, None, "together")

    r2 = _hf_serverless_route(messages)
    if isinstance(r2, str):
        call = _extract_json(r2)
        if call: return (call, None, "hf")

    local = _local_light_parse(user_text)
    if local: return (local, None, "local")

    fb_msgs = [{"role":"system","content":FALLBACK_PROMPT},{"role":"user","content":user_text}]
    r3 = _together_route(fb_msgs)
    if isinstance(r3, str):
        return (None, r3, "together-fallback")
    return (None, "(LLM unavailable) Try: 'I want bio1, ndvi' or 'run model'.", "none")

# =========================
# Actions
# =========================
def run_fetch(sl, lc):
    layers = list(sl) if sl else []
    if lc: layers.append("landcover")

    if not layers:
        return create_map(), "⚠️ Please select at least one predictor."

    bad_layers = [l for l in layers if l not in VALID_LAYERS]
    if bad_layers:
        suggestions = []
        for b in bad_layers:
            match = difflib.get_close_matches(b, VALID_LAYERS, n=1, cutoff=0.6)
            if match:
                suggestions.append(f"Did you mean '{match[0]}' instead of '{b}'?")
        if suggestions:
            return create_map(), "⚠️ " + " ".join(suggestions)
        return create_map(), f"⚠️ Unknown layers: {', '.join(bad_layers)}"

    bad_codes = [c for c in lc if c not in LANDCOVER_CLASSES]
    if bad_codes:
        suggestions = []
        for b in bad_codes:
            match = difflib.get_close_matches(b, LANDCOVER_CLASSES, n=1, cutoff=0.6)
            if match:
                suggestions.append(f"Did you mean landcover '{match[0]}' instead of '{b}'?")
        if suggestions:
            return create_map(), "⚠️ " + " ".join(suggestions)
        return create_map(), f"⚠️ Unknown landcover classes: {', '.join(bad_codes)}"

    print(f"🧪 [run_fetch] SL={sl!r}   LC={lc!r}", file=sys.stdout)
    os.environ["SELECTED_LAYERS"] = ",".join([l for l in sl if l] if sl else [])
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(lc or [])

    cmd = [sys.executable, "-u", os.path.join("scripts", "fetch_predictors.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return create_map(), f"❌ Fetch failed:\n```\n{logs}\n```"
    else:
        return create_map(), f"✅ Predictors fetched.\n\n```bash\n{logs}\n```"

def run_model():
    proc = subprocess.run([sys.executable, os.path.join("scripts","run_logistic_sdm.py")], capture_output=True, text=True)
    if proc.returncode!=0:
        return create_map(), f"❌ Model run failed:\n```\n{proc.stderr}\n```", None, None
    perf_df = None
    coef_df = None
    try:
        perf_df = pd.read_csv("outputs/performance_metrics.csv")
    except Exception:
        pass
    try:
        coef_df = pd.read_csv("outputs/coefficients.csv").dropna(axis=1, how='all')
    except Exception:
        pass
    zip_results()
    return create_map(), "✅ Model trained. Download results below (ZIP).", perf_df, coef_df

def render_layers_help():
    return (
        "📦 **Available layers (exact datasets)**\n"
        "- **Bioclim**: bio1–bio19 — WorldClim v1 bioclim normals (WORLDCLIM/V1/BIO, ~1 km)\n"
        "- **Topography**: elevation, slope, aspect — USGS/SRTMGL1_003 (30 m)\n"
        "- **Remote sensing**: ndvi — MODIS/061/MOD13Q1 mean (2022-01-01→2024-01-01) (250 m)\n"
        "- **Landcover**: MODIS IGBP classes — MODIS/061/MCD12Q1 LC_Type1 (500 m)\n"
        "\nSay things like:\n"
        "• \"I want bio1, ndvi, elevation\" (fetch layers)\n"
        "• \"Fetch bio5, bio12 and slope\"\n"
        "• \"Add landcover water and urban_and_built_up\"\n"
        "• \"Run model\" (train & predict)\n"
        "• \"What is bio5?\" / \"Where does ndvi come from?\"\n"
        "• \"Explain those stats\" (summarize latest performance & coefficients)\n"
    )

def explain_those_stats():
    perf_fp = "outputs/performance_metrics.csv"
    coef_fp = "outputs/coefficients.csv"
    parts = []
    try:
        if os.path.exists(perf_fp):
            perf = pd.read_csv(perf_fp)
            first, second = perf.iloc[:, :3], perf.iloc[:, 3:]
            parts.append("**Performance (1/2)**\n\n" + first.to_markdown(index=False))
            if second.shape[1] > 0:
                parts.append("**Performance (2/2)**\n\n" + second.to_markdown(index=False))
    except Exception:
        pass
    try:
        if os.path.exists(coef_fp):
            coef = pd.read_csv(coef_fp).dropna(axis=1, how='all')
            if not coef.empty:
                parts.append("**Predictor Coefficients**\n\n" + coef.to_markdown(index=False))
    except Exception:
        pass
    return parts and "\n\n".join(parts) or "I don’t see any stats yet—try **run model** first."

# =========================
# Chat Step
# =========================
def chat_step(file, user_msg, history, state):
    if not isinstance(state, dict):
        state = {}

    # If no CSV yet: allow only general help/list_layers; otherwise prompt to upload
    if not os.path.exists("inputs/presence_points.csv"):
        tool, natural, backend = llm_route(user_msg, history)
        if tool and tool.get("tool") in ("list_layers","help"):
            assistant_txt = render_layers_help() if tool["tool"]=="list_layers" else (tool.get("args",{}).get("reply") or "Upload a presence CSV to begin.")
        else:
            assistant_txt = "Please upload your presence CSV to begin. Then say “I want bio1, ndvi, elevation”."
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":assistant_txt}])
        return history, create_map(), state

    # Reset
    if re.search(r"\b(start over|restart|clear everything|reset|clear all)\b", user_msg, re.I):
        clear_all()
        new_hist = [{"role":"assistant","content":"👋 All cleared! Please upload your presence CSV to begin."}]
        return new_hist, create_map(), {"stage":"await_upload"}

    # Route (LLM first)
    tool, natural, backend = llm_route(user_msg, history)

    # Consume pending (disambiguation) if any
    if state.get("pending") == "fetch_needed":
        # treat this message as layer spec; re-parse locally
        parsed = _local_light_parse(user_msg)
        if parsed and parsed.get("tool")=="fetch":
            tool = parsed
        state["pending"] = None

    assistant_txt = None
    m_out = create_map()

    if tool:
        tname = tool.get("tool")
        args = tool.get("args", {})

        if tname == "fetch":
            layers = args.get("layers", [])
            land   = args.get("landcover", [])
            if not layers and not land:
                state["pending"] = "fetch_needed"
                assistant_txt = "Which layers should I fetch? e.g., **bio1, ndvi, elevation** (you can also say landcover classes like **urban, wetlands**)."
            else:
                m_out, status = run_fetch(layers, land)
                assistant_txt = status + "\n\n*Tip:* now say **run model** to train & predict."

        elif tname == "run_model":
            # Helpful guard: if nothing fetched yet, nudge first
            if not fetched_layer_names():
                assistant_txt = "You haven’t fetched any predictors yet. Say **I want bio1, ndvi, elevation** (or the layers you want), then **run model**."
            else:
                m_out, status, perf_df, coef_df = run_model()
                perf_md = ""
                coef_md = ""
                try:
                    if perf_df is None and os.path.exists("outputs/performance_metrics.csv"):
                        perf_df = pd.read_csv("outputs/performance_metrics.csv")
                    if perf_df is not None:
                        first, second = perf_df.iloc[:, :3], perf_df.iloc[:, 3:]
                        perf_md = "\n\n**Performance (1/2)**\n\n" + first.to_markdown(index=False)
                        if second.shape[1] > 0:
                            perf_md += "\n\n**Performance (2/2)**\n\n" + second.to_markdown(index=False)
                    if coef_df is None and os.path.exists("outputs/coefficients.csv"):
                        coef_df = pd.read_csv("outputs/coefficients.csv").dropna(axis=1, how='all')
                    if coef_df is not None and not coef_df.empty:
                        coef_md = "\n\n**Predictor Coefficients**\n\n" + coef_df.to_markdown(index=False)
                except Exception:
                    pass
                assistant_txt = status + (perf_md or "") + (coef_md or "")

        elif tname == "explain_stats":
            assistant_txt = explain_those_stats()

        elif tname == "list_layers":
            assistant_txt = render_layers_help()

        elif tname == "help":
            # context-aware help
            fetched = fetched_layer_names()
            if not fetched:
                assistant_txt = "You can fetch predictors like **bio1, ndvi, elevation**. Then say **run model**."
            elif not have_outputs():
                assistant_txt = "Predictors are ready. Say **run model** to train & predict."
            else:
                assistant_txt = "Model finished. You can **download the ZIP**, or say **Explain those stats**."
        else:
            assistant_txt = "I didn’t recognize that action—try: “I want bio1, ndvi” or “run model”."
    else:
        assistant_txt = natural or "(LLM unavailable) Try: “I want bio1, ndvi” or “run model”."

    history.extend([
        {"role":"user","content":user_msg},
        {"role":"assistant","content":assistant_txt}
    ])
    return history, m_out, state

# =========================
# Upload & CRS Confirmation
# =========================
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
            msg = "✅ I found your **latitude** and **longitude** columns.\n\n" + render_layers_help()
            history2.append({"role":"assistant","content":msg})
            return history2, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            history2.append({"role":"assistant","content":"I couldn't detect coordinate columns. Please select them and enter CRS below."})
            cols = list(df.columns)
            return history2, create_map(), state, gr.update(choices=cols, visible=True), gr.update(choices=cols, visible=True), gr.update(visible=True), gr.update(visible=True)
    return history2, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def confirm_coords(lat_col, lon_col, crs_raw, history, state):
    df = pd.read_csv("inputs/presence_points.csv")
    try:
        src_epsg = resolve_crs(crs_raw) if crs_raw else 4326
    except Exception:
        history.append({"role":"assistant","content":"Sorry, I couldn't recognize that CRS. Could you try another format?"})
        return history, create_map(), state, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    src_crs = RioCRS.from_epsg(src_epsg)
    dst_crs = RioCRS.from_epsg(4326)
    lon_vals, lat_vals = rio_transform(src_crs, dst_crs, df[lon_col].tolist(), df[lat_col].tolist())
    df['latitude'], df['longitude'] = lat_vals, lon_vals
    df.to_csv("inputs/presence_points.csv", index=False)
    history.append({"role": "assistant","content": "✅ Coordinates set!\n\n" + render_layers_help()})
    return history, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

# =========================
# Gradio UI
# =========================
with gr.Blocks() as demo:
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
    #logo-img img {
        height: 90px;
        margin: 10px 50px 10px 10px;
        border-radius: 6px;
    }
    </style>
    """)
    gr.Markdown("## 🗺️ SpatChat: Species Distribution Model {sdm}  🐢🐍🦅🦋🦉🦊🐞")
    state = gr.State({"stage": "await_upload"})
    with gr.Row():
        with gr.Column(scale=1):
            map_out = gr.HTML(create_map(), label="🗺️ Map Preview")
            download_btn = gr.DownloadButton("📥 Download Results", zip_results)
        with gr.Column(scale=1):
            chat = gr.Chatbot(
                value=[{"role":"assistant","content":"👋 Hello, I'm SpatChat (SDM)! Upload a presence CSV to begin."}],
                type="messages",
                label="💬 Chat",
                height=400
            )
            user_in = gr.Textbox(label="Ask SpatChat", placeholder="e.g., I want bio1, ndvi, elevation")
            file_input = gr.File(label="📄 Upload Presence CSV", type="filepath", file_types=[".csv"])
            lat_dropdown = gr.Dropdown(choices=[], label="Latitude column", visible=False)
            lon_dropdown = gr.Dropdown(choices=[], label="Longitude column", visible=False)
            crs_input = gr.Textbox(label="Input CRS (code, zone, or name)", placeholder="e.g. 32610, UTM zone 10N, LCC…", visible=False)
            confirm_btn = gr.Button("Confirm Coordinates", visible=False)

    file_input.change(
        on_upload,
        inputs=[file_input, chat, state],
        outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn]
    )
    confirm_btn.click(
        confirm_coords,
        inputs=[lat_dropdown, lon_dropdown, crs_input, chat, state],
        outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn]
    )
    user_in.submit(chat_step, inputs=[file_input, user_in, chat, state], outputs=[chat, map_out, state])
    user_in.submit(lambda: "", None, user_in)

if __name__ == "__main__":
    print("Starting SpatChat SDM (LLM router: Together → HF Serverless → local)")
    # Keep it simple; avoid old queue args that break on some Gradio versions
    demo.queue().launch()
