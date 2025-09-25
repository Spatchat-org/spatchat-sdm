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
import threading

import gradio as gr
import geemap.foliumap as foliumap
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

# LLM providers
from huggingface_hub import InferenceClient
from together import Together
from together.error import RateLimitError, ServiceUnavailableError

# --- Which top-level predictors we support (all lower-case) ---
PREDICTOR_CHOICES = (
    [f"bio{i}" for i in range(1, 20)]
    + ["elevation", "slope", "aspect", "ndvi", "landcover"]
)
# force everything to lower-case so our .lower() tokens always match
VALID_LAYERS = {p.lower() for p in PREDICTOR_CHOICES}

# All available MODIS landcover classes
LANDCOVER_CLASSES = {
    c.lower() for c in (
        "water", "evergreen_needleleaf_forest", "evergreen_broadleaf_forest",
        "deciduous_needleleaf_forest", "deciduous_broadleaf_forest", "mixed_forest",
        "closed_shrublands", "open_shrublands", "woody_savannas", "savannas",
        "grasslands", "permanent_wetlands", "croplands", "urban_and_built_up",
        "cropland_natural_vegetation_mosaic", "snow_and_ice", "barren_or_sparsely_vegetated"
    )
}

# --- Small helper to list available layers ---
def available_layers_markdown():
    return (
        "You can fetch these predictors:\n"
        "• bio1–bio19\n"
        "• elevation\n"
        "• slope\n"
        "• aspect\n"
        "• NDVI\n"
        "• landcover (e.g. " + ", ".join(sorted(LANDCOVER_CLASSES)) + ")\n\n"
        "Example: **I want elevation, ndvi, bio1**"
    )

# --- Pre-render colorbar → base64 ---
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

# --- Authenticate Earth Engine ---
load_dotenv()
svc = json.loads(os.environ.get("GEE_SERVICE_ACCOUNT", "{}"))
creds = ee.ServiceAccountCredentials(svc.get("client_email"), key_data=json.dumps(svc))
ee.Initialize(creds)

# ──────────────────────────────────────────────────────────────
# Helpers for LLM response parsing (HF + Together)
# ──────────────────────────────────────────────────────────────

def _choice_content(choice):
    msg = getattr(choice, "message", None)
    if msg is None and isinstance(choice, dict):
        msg = choice.get("message")
    content = None
    if msg is not None:
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            content = "".join(parts)
    if content is None:
        content = ""
    return content

def _delta_text(delta):
    if isinstance(delta, dict):
        return delta.get("content", "")
    return getattr(delta, "content", "")

HF_MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOGETHER_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

class _SpacedCallLimiter:
    def __init__(self, min_interval_seconds: float):
        self.min_interval = float(min_interval_seconds)
        self._lock = threading.Lock()
        self._last = 0.0
    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()

class UnifiedLLM:
    def __init__(self):
        hf_model_or_url = (os.getenv("HF_ENDPOINT_URL") or HF_MODEL_DEFAULT).strip()
        hf_token = (os.getenv("HF_TOKEN") or "").strip()
        self.hf_client = InferenceClient(model=hf_model_or_url, token=hf_token, timeout=300)
        self.together = None
        self.together_model = (os.getenv("TOGETHER_MODEL") or TOGETHER_MODEL_DEFAULT).strip()
        tg_key = (os.getenv("TOGETHER_API_KEY") or "").strip()
        if tg_key:
            self.together = Together(api_key=tg_key)
            self._tg_limiter = _SpacedCallLimiter(min_interval_seconds=100.0)

    def _hf_chat(self, messages, max_tokens=512, temperature=0.3, stream=False):
        tries, delay = 3, 2.5
        last_err = None
        for _ in range(tries):
            try:
                if hasattr(self.hf_client, "chat_completion"):
                    resp = self.hf_client.chat_completion(
                        messages=messages, max_tokens=max_tokens, temperature=temperature, stream=stream
                    )
                    if stream:
                        text = "".join(_delta_text(ch.choices[0].delta) for ch in resp)
                    else:
                        text = _choice_content(resp.choices[0])
                    return text
                else:
                    prompt = self._messages_to_prompt(messages)
                    text = self.hf_client.text_generation(
                        prompt, max_new_tokens=512, temperature=temperature, stream=False, return_full_text=False
                    )
                    return text
            except Exception as e:
                last_err = e
                time.sleep(delay)
                delay *= 1.8
        raise last_err

    @staticmethod
    def _messages_to_prompt(messages):
        parts = []
        for m in messages:
            role = m.get("role", "user"); content = m.get("content", "")
            if role == "system": parts.append(f"<|system|>\n{content}\n")
            elif role == "user": parts.append(f"<|user|>\n{content}\n")
            else: parts.append(f"<|assistant|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def chat(self, messages, temperature=0.3, max_tokens=512, stream=False):
        try:
            return self._hf_chat(messages, max_tokens=max_tokens, temperature=temperature, stream=stream)
        except Exception as hf_err:
            print(f"[LLM] HF primary failed: {hf_err}", file=sys.stderr)
            if self.together is None:
                raise
            self._tg_limiter.wait()
            backoff = 12.0
            for attempt in range(4):
                try:
                    resp = self.together.chat.completions.create(
                        model=self.together_model, messages=messages,
                        temperature=temperature, max_tokens=max_tokens, stream=stream,
                    )
                    return _choice_content(resp.choices[0])
                except (RateLimitError, ServiceUnavailableError):
                    if attempt == 3:
                        raise
                    time.sleep(backoff + random.uniform(0, 3))
                    backoff *= 1.8

llm = UnifiedLLM()

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

def _clean_dir(path: str):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

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

# --- CRS parsing helpers (handle 'UTM 10T') ---
def parse_epsg_code(s: str):
    m = re.search(r'(?:^|\b)epsg\s*:\s*(\d{4,5})\b', s, re.I)
    if not m:
        m = re.match(r'^\s*(\d{4,5})\s*$', s.strip())
    return int(m.group(1)) if m else None

def parse_utm_crs(s: str):
    txt = s.strip()
    patterns = [
        r'\butm\b[^0-9]*?(\d{1,2})\s*([A-Za-z])?',
        r'\bzone\s*(\d{1,2})\s*([A-Za-z])?',
        r'\b(\d{1,2})\s*([C-HJ-NP-X])\b',
        r'\b(\d{1,2})\s*([NS])\b',
    ]
    m = None
    for p in patterns:
        m = re.search(p, txt, re.I)
        if m:
            break
    if not m:
        return None
    zone = int(m.group(1))
    letter = (m.group(2) or '').upper()
    if letter in ('N', 'S'):
        hemi = 'N' if letter == 'N' else 'S'
    elif letter:
        hemi = 'N' if letter >= 'N' else 'S'
    else:
        hemi = 'N'
    return (32600 if hemi == 'N' else 32700) + zone

def llm_parse_crs(raw):
    system = {"role":"system","content":"You're a GIS expert. Given a CRS description, respond with only JSON {\"epsg\": ###} or {\"epsg\": null}."}
    user = {"role":"user","content":f"CRS: '{raw}'"}
    resp = llm.chat([system, user], temperature=0.0, max_tokens=32, stream=False)
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

SYSTEM_PROMPT = """
You are SpatChat, a friendly species distribution modeling assistant.
When the user asks to fetch environmental layers (using verbs like fetch, download, get, grab, "I want", etc.), respond with exactly a JSON object:
{"tool":"fetch","layers":[<layer names>],"landcover":[<landcover classes>]}
When the user asks to run the model (e.g., "run model", "run species distribution model", "run SDM", etc.), respond with exactly:
{"tool":"run_model"}
If the user's request does not match either of these intents, reply naturally without JSON.
Examples:
User: I want bio2 and ndvi
Assistant: {"tool":"fetch","layers":["bio2","ndvi"],"landcover":[]}
User: Grab slope, elevation
Assistant: {"tool":"fetch","layers":["slope","elevation"],"landcover":[]}
User: Run model now
Assistant: {"tool":"run_model"}
User: How many points are uploaded?
Assistant: There are currently 193 presence points uploaded.
""".strip()

FALLBACK_PROMPT = """
You are SpatChat, a friendly assistant for species distribution modeling.
Keep your answers short—no more than two sentences—while still being helpful.
Guide the user to next steps: upload data, fetch layers, run model, etc.
""".strip()

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
                
    # background/absence points (if available)
    abs_fp = "outputs/absence_points_coordinates.csv"
    if os.path.exists(abs_fp):
        df_abs = pd.read_csv(abs_fp)
        if {'latitude','longitude'}.issubset(df_abs.columns):
            fg_abs = folium.FeatureGroup(name="🟥 Background / Absence Points")
            for lat, lon in df_abs[['latitude','longitude']].dropna().values.tolist():
                folium.CircleMarker(
                    location=[lat, lon], radius=3, color="red", fill=True, fill_opacity=0.6
                ).add_to(fg_abs)
            fg_abs.add_to(m)

    rasdir = "predictor_rasters/wgs84"
    if os.path.isdir(rasdir):
        for fn in sorted(os.listdir(rasdir)):
            if fn.endswith(".tif"):
                with rasterio.open(os.path.join(rasdir, fn)) as src:
                    arr = src.read(1); bnd = src.bounds
                vmin, vmax = np.nanmin(arr), np.nanmax(arr)
                if not np.isnan(vmin) and vmin!=vmax:
                    rgba = colormaps["viridis"]((arr-vmin)/(vmax-vmin))
                    folium.raster_layers.ImageOverlay(
                        rgba, bounds=[[bnd.bottom,bnd.left],[bnd.top,bnd.right]],
                        opacity=1.0, name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})"
                    ).add_to(m)
    sf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        with rasterio.open(sf) as src:
            arr = src.read(1); bnd = src.bounds
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        rgba = colormaps["viridis"]((arr - vmin) / (vmax - vmin))
        folium.raster_layers.ImageOverlay(
            rgba, bounds=[[bnd.bottom,bnd.left],[bnd.top,bnd.right]],
            opacity=1.0, name="🎯 Suitability"
        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    img_html = f'<img src="data:image/png;base64,{COLORBAR_BASE64}" style="position:absolute; bottom:20px; right:10px; width:200px; height:30px; z-index:1000;"/>'
    m.get_root().html.add_child(Element(img_html))
    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" style="width:100%; height:450px; border:none;"></iframe>'

def zip_results():
    archive = "spatchat_results.zip"
    if os.path.exists(archive): os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters","outputs"):
            for root,_,files in os.walk(fld):
                for fn in files:
                    full = os.path.join(root,fn)
                    zf.write(full, arcname=os.path.relpath(full,"."))
    return archive

def run_fetch(sl, lc):
    # --- keep previous layers; just ensure folders exist (NO clearing) ---
    os.makedirs("predictor_rasters", exist_ok=True)
    os.makedirs("predictor_rasters/wgs84", exist_ok=True)

    # normalize inputs to lower-case
    sl = [s.lower() for s in sl]
    lc = [c.lower() for c in lc]

    # 1) Build the list of requested predictors
    layers = list(sl)
    if lc and "landcover" not in layers:
        layers.append("landcover")

    # 2) Require at least one predictor
    if not layers:
        return create_map(), "⚠️ Please select at least one predictor."

    # 3) Validate top-level predictors
    bad_layers = [l for l in layers if l not in VALID_LAYERS]
    if bad_layers:
        suggestions = []
        for b in bad_layers:
            match = difflib.get_close_matches(b, VALID_LAYERS, n=1, cutoff=0.6)
            if match:
                suggestions.append(f"Did you mean '{match[0]}' instead of '{b}'?")
        if suggestions:
            return create_map(), "⚠️ " + " ".join(suggestions)
        prompt = (
            f"You requested these predictors: {', '.join(layers)}. "
            f"I don't recognize: {', '.join(bad_layers)}. "
            "Could you please clarify which predictors you want?"
        )
        clar = llm.chat(
            messages=[{"role": "system", "content": FALLBACK_PROMPT},
                      {"role": "user",   "content": prompt}],
            temperature=0.7, max_tokens=256, stream=False
        )
        return create_map(), clar

    # 4) Validate landcover subclasses
    bad_codes = [c for c in lc if c not in LANDCOVER_CLASSES]
    if bad_codes:
        suggestions = []
        for b in bad_codes:
            match = difflib.get_close_matches(b, LANDCOVER_CLASSES, n=1, cutoff=0.6)
            if match:
                suggestions.append(f"Did you mean landcover class '{match[0]}' instead of '{b}'?")
        if suggestions:
            return create_map(), "⚠️ " + " ".join(suggestions)
        prompt = (
            f"You requested landcover classes: {', '.join(lc)}. "
            f"I don't recognize: {', '.join(bad_codes)}. "
            "Could you please clarify which landcover classes you want?"
        )
        clar = llm.chat(
            messages=[{"role": "system", "content": FALLBACK_PROMPT},
                      {"role": "user",   "content": prompt}],
            temperature=0.7, max_tokens=256, stream=False
        )
        return create_map(), clar

    # 5) All inputs valid → proceed with fetch
    print(f"🧪 [run_fetch] SL={sl!r}   LC={lc!r}", file=sys.stdout)

    # ensure SELECTED_LAYERS includes landcover if subclasses present
    sl_env = list(sl)
    if lc and "landcover" not in sl_env:
        sl_env.append("landcover")

    os.environ["SELECTED_LAYERS"] = ",".join(sl_env)
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(lc)

    cmd = [sys.executable, "-u", os.path.join("scripts", "fetch_predictors.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return create_map(), f"❌ Fetch failed:\n```\n{logs}\n```"
    else:
        return create_map(), f"✅ Predictors fetched.\n\n```bash\n{logs}\n```"

# --- Deterministic “I want …” fetch intent (handles spaces in landcover names) ---
def try_parse_fetch_from_text(user_msg: str):
    if not re.search(r'\b(fetch|get|grab|download|want|need|add)\b', user_msg, re.I):
        return None, None
    text = user_msg.lower()
    tokens = re.findall(r'[a-zA-Z0-9_]+', text)
    layers, classes = [], []
    for t in tokens:
        if re.fullmatch(r'bio(?:[1-9]|1[0-9])', t):
            if t not in layers: layers.append(t)
            continue
        if t in VALID_LAYERS and t not in layers:
            layers.append(t); continue
        if t in LANDCOVER_CLASSES and t not in classes:
            classes.append(t); continue
    snake_text = re.sub(r'[^a-z0-9]+', '_', text)
    for cls in LANDCOVER_CLASSES:
        if cls in snake_text and cls not in classes:
            classes.append(cls)
    if classes and "landcover" not in layers:
        layers.append("landcover")
    if layers or classes:
        return layers, classes
    return None, None

def run_model():
    _clean_dir("outputs")
    proc = subprocess.run([sys.executable, "-u", os.path.join("scripts","run_logistic_sdm.py")],
                          capture_output=True, text=True)
    if proc.returncode!=0:
        return create_map(), f"❌ Model run failed:\n{proc.stderr}", None, None, None
    perf_df = pd.read_csv("outputs/performance_metrics_cv.csv")
    coef_df = pd.read_csv("outputs/coefficients.csv")
    zip_path = zip_results()
    return create_map(), "✅ Model ran successfully! Download the SDM using the button below the map!", perf_df, coef_df, zip_path

def chat_step(file, user_msg, history, state):
    # available layers Q&A
    if re.search(r'\b(what|which)\b.*\b(layers|predictors)\b.*\b(available|supported|have)\b', user_msg, re.I) \
       or re.search(r'\bavailable layers\b', user_msg, re.I):
        reply = available_layers_markdown()
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":reply}])
        return history, create_map(), state, gr.update()

    # deterministic fetch
    parsed_layers, parsed_classes = try_parse_fetch_from_text(user_msg)
    if parsed_layers is not None:
        m_out, status = run_fetch(parsed_layers, parsed_classes)
        assistant_txt = f"{status}\n\nGreat! Now you can run the model or fetch more layers."
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":assistant_txt}])
        return history, m_out, state, gr.update()

    # LLM tool route
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":user_msg}]
    resp = llm.chat(msgs, temperature=0.0, max_tokens=256, stream=False)
    try:
        call = json.loads(resp); tool = call.get("tool")
    except Exception:
        tool = None
    if tool == "fetch":
        m_out, status = run_fetch(call.get("layers", []), call.get("landcover", []))
        assistant_txt = f"{status}\n\nGreat! Now you can run the model or fetch more layers."
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":assistant_txt}])
        return history, m_out, state, gr.update()
    elif tool == "run_model":
        m_out, status, perf_df, coef_df, zip_path = run_model()
        if perf_df is None:
            assistant_txt = status; dl_update = gr.update()
        else:
            cv = pd.read_csv("outputs/performance_metrics_cv.csv")
            row = cv.iloc[0] if not cv.empty else pd.Series({})
            
            def getv(k):
                return row[k] if (k in row and pd.notna(row[k])) else np.nan
            def fmt(x):
                return "—" if pd.isna(x) else f"{float(x):.3f}"
            def fmt_pm(mu, sd):
                if pd.isna(mu): return "—"
                return f"{float(mu):.3f}" + (f" ± {float(sd):.3f}" if not pd.isna(sd) else "")
            
            folds = int(getv("n_folds")) if not pd.isna(getv("n_folds")) else 0
            auc_str  = fmt_pm(getv("AUC_mean"),   getv("AUC_sd"))
            tss_str  = fmt_pm(getv("TSS_mean"),   getv("TSS_sd"))
            kap_str  = fmt_pm(getv("Kappa_mean"), getv("Kappa_sd"))
            sens_str = fmt(getv("Sensitivity_mean"))
            spec_str = fmt(getv("Specificity_mean"))
            thr_str  = fmt(getv("Threshold_mean"))
            
            perf_md = (
                "| Metric | Value |\n"
                "|---|---:|\n"
                f"| AUC | {auc_str} |\n"
                f"| TSS | {tss_str} |\n"
                f"| Kappa | {kap_str} |\n"
                f"| Sens | {sens_str} |\n"
                f"| Spec | {spec_str} |\n"
                f"| Thr | {thr_str} |\n"
            )
            
            coef = pd.read_csv("outputs/coefficients.csv").dropna(axis=1, how='all')
            coef_md = coef.to_markdown(index=False)
            
            assistant_txt = (
                f"{status}\n\n"
                f"**Cross-validated Performance (spatial blocks; n_folds={folds})**\n\n{perf_md}\n\n"
                f"**Predictor Coefficients:**\n\n{coef_md}"
            )
            dl_update = gr.update(value=zip_path)
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":assistant_txt}])
        return history, m_out, state, dl_update

    # no CSV yet -> small talk
    if not os.path.exists("inputs/presence_points.csv"):
        fb = [{"role":"system","content":FALLBACK_PROMPT}, {"role":"user","content":user_msg}]
        reply = llm.chat(fb, temperature=0.7, max_tokens=256, stream=False)
        history.extend([{"role":"user","content":user_msg}, {"role":"assistant","content":reply}])
        return history, create_map(), state, gr.update()

    # run model shortcut
    if re.fullmatch(r"\s*(?:run\s+)?model\s*$", user_msg, re.I):
        m_out, status, perf_df, coef_df, zip_path = run_model()
        if perf_df is not None:
            cv = pd.read_csv("outputs/performance_metrics_cv.csv")
            row = cv.iloc[0] if not cv.empty else pd.Series({})
            
            def getv(k):
                return row[k] if (k in row and pd.notna(row[k])) else np.nan
            def fmt(x):
                return "—" if pd.isna(x) else f"{float(x):.3f}"
            def fmt_pm(mu, sd):
                if pd.isna(mu): return "—"
                return f"{float(mu):.3f}" + (f" ± {float(sd):.3f}" if not pd.isna(sd) else "")
            
            folds = int(getv("n_folds")) if not pd.isna(getv("n_folds")) else 0
            auc_str  = fmt_pm(getv("AUC_mean"),   getv("AUC_sd"))
            tss_str  = fmt_pm(getv("TSS_mean"),   getv("TSS_sd"))
            kap_str  = fmt_pm(getv("Kappa_mean"), getv("Kappa_sd"))
            sens_str = fmt(getv("Sensitivity_mean"))
            spec_str = fmt(getv("Specificity_mean"))
            thr_str  = fmt(getv("Threshold_mean"))
            
            perf_md = (
                "| Metric | Value |\n"
                "|---|---:|\n"
                f"| AUC | {auc_str} |\n"
                f"| TSS | {tss_str} |\n"
                f"| Kappa | {kap_str} |\n"
                f"| Sens | {sens_str} |\n"
                f"| Spec | {spec_str} |\n"
                f"| Thr | {thr_str} |\n"
            )
            
            coef = pd.read_csv("outputs/coefficients.csv").dropna(axis=1, how='all')
            
            status += "\n\n**Cross-validated Performance (spatial blocks; n_folds=" + str(folds) + ")**\n\n" + perf_md
            status += "\n\n**Predictor Coefficients:**\n\n" + coef.to_markdown(index=False)
        assistant_txt = status
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":assistant_txt}])
        return history, m_out, state, gr.update(value=zip_path)

    # reset
    if re.search(r"\b(start over|restart|clear everything|reset|clear all)\b", user_msg, re.I):
        clear_all()
        new_hist = [{"role":"assistant","content":"👋 All cleared! Please upload your presence-points CSV to begin."}]
        return new_hist, create_map(), state, gr.update(value=None)

    # fallback tool prompt again
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":user_msg}]
    resp = llm.chat(msgs, temperature=0.0, max_tokens=256, stream=False)
    try:
        call = json.loads(resp); tool = call.get("tool")
    except Exception:
        tool = None

    if tool == "fetch":
        m_out, status = run_fetch(call.get("layers", []), call.get("landcover", []))
        assistant_txt = f"{status}\n\nGreat! Now you can run the model or fetch more layers."
        dl_update = gr.update()
    elif tool == "run_model":
        m_out, status, perf_df, coef_df, zip_path = run_model()
        if perf_df is None:
            assistant_txt = status; dl_update = gr.update()
        else:
            perf = pd.read_csv("outputs/performance_metrics.csv")
            first, second = perf.iloc[:, :3], perf.iloc[:, 3:]
            perf_md = ("**Model Performance (1 of 2):**\n\n" + first.to_markdown(index=False)
                       + "\n\n**Model Performance (2 of 2):**\n\n" + second.to_markdown(index=False))
            coef = pd.read_csv("outputs/coefficients.csv").dropna(axis=1, how='all')
            coef_md = coef.to_markdown(index=False)
            assistant_txt = f"{status}\n\n**Model Performance:**\n\n{perf_md}\n\n**Predictor Coefficients:**\n\n{coef_md}"
            dl_update = gr.update(value=zip_path)
    elif tool == "download":
        m_out, _ = create_map(), zip_results()
        assistant_txt = "✅ ZIP is downloading…"
        dl_update = gr.update(value="spatchat_results.zip")
    else:
        try:
            n_pts = len(pd.read_csv("inputs/presence_points.csv"))
        except:
            n_pts = 0
        rasdir = "predictor_rasters/wgs84"
        if os.path.isdir(rasdir):
            fetched = sorted(os.path.splitext(f)[0] for f in os.listdir(rasdir) if f.endswith(".tif"))
        else:
            fetched = []
        perf_table = ""
            perf_fp = "outputs/performance_metrics_cv.csv"
            folds_txt = ""
            perf_md = "*none*"
            if os.path.exists(perf_fp):
                cv = pd.read_csv(perf_fp)
                if not cv.empty:
                    row = cv.iloc[0]
            
                    def getv(k):
                        return row[k] if (k in row and pd.notna(row[k])) else np.nan
                    def fmt(x):
                        return "—" if pd.isna(x) else f"{float(x):.3f}"
                    def fmt_pm(mu, sd):
                        if pd.isna(mu): return "—"
                        return f"{float(mu):.3f}" + (f" ± {float(sd):.3f}" if not pd.isna(sd) else "")
            
                    nfolds = getv("n_folds")
                    if not pd.isna(nfolds):
                        folds_txt = f" (n_folds={int(nfolds)})"
            
                    auc_str  = fmt_pm(getv("AUC_mean"),   getv("AUC_sd"))
                    tss_str  = fmt_pm(getv("TSS_mean"),   getv("TSS_sd"))
                    kap_str  = fmt_pm(getv("Kappa_mean"), getv("Kappa_sd"))
                    sens_str = fmt(getv("Sensitivity_mean"))
                    spec_str = fmt(getv("Specificity_mean"))
                    thr_str  = fmt(getv("Threshold_mean"))
            
                    perf_md = (
                        "| Metric | Value |\n"
                        "|---|---:|\n"
                        f"| AUC | {auc_str} |\n"
                        f"| TSS | {tss_str} |\n"
                        f"| Kappa | {kap_str} |\n"
                        f"| Sens | {sens_str} |\n"
                        f"| Spec | {spec_str} |\n"
                        f"| Thr | {thr_str} |\n"
                    )
            
            summary = (
                f"- Presence points: {n_pts}\n"
                f"- Layers fetched ({len(fetched)}): {', '.join(fetched) or 'none'}\n\n"
                "**Cross-validated Performance (spatial blocks" + folds_txt + ")**\n"
                f"{perf_md}\n\n"
                "**Predictor Coefficients**\n"
                f"{coef_table or '*none*'}"
            )
        explain_sys = {
            "role":"system",
            "content":"You are SpatChat, an expert in species distribution modeling. Use ALL of the context below to answer the user's question as clearly as possible."
        }
        msgs = [explain_sys, {"role":"system","content":"Data summary:\n" + summary}, {"role":"user","content":user_msg}]
        assistant_txt = llm.chat(msgs, temperature=0.7, max_tokens=384, stream=False)
        m_out = create_map(); dl_update = gr.update()

    history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":assistant_txt}])
    return history, m_out, state, dl_update

def on_upload(f, history, state):
    history2 = history.copy()
    if not isinstance(state, dict):
        state = {}
    state.setdefault("layers_help_shown", False)

    clear_all()
    if f and hasattr(f, "name"):
        shutil.copy(f.name, "inputs/presence_points.csv")
        df = pd.read_csv("inputs/presence_points.csv")
        lat, lon = detect_coords(df)
        if lat and lon:
            df = df.rename(columns={lat: "latitude", lon: "longitude"})
            df.to_csv("inputs/presence_points.csv", index=False)
            msg = "✅ Sweet! I found your `latitude` and `longitude` columns."
            if not state.get("layers_help_shown"):
                msg += "\n\n" + available_layers_markdown()
                state["layers_help_shown"] = True
            history2.append({"role":"assistant","content": msg})
            return (history2, create_map(), state, gr.update(),
                    gr.update(choices=[], visible=False),
                    gr.update(choices=[], visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False))
        else:
            history2.append({"role":"assistant","content":
                "I couldn't detect coordinate columns. Please select them and enter CRS below (e.g., `UTM 10T` or `32610`)."
            })
            cols = list(df.columns)
            return (history2, create_map(), state, gr.update(),
                    gr.update(choices=cols, visible=True, value=None),
                    gr.update(choices=cols, visible=True, value=None),
                    gr.update(visible=True, value="UTM 10T"),
                    gr.update(visible=True))
    return (history2, create_map(), state, gr.update(),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False))

def confirm_coords(lat_col, lon_col, crs_raw, history, state):
    if not isinstance(state, dict):
        state = {}
    state.setdefault("layers_help_shown", False)

    df = pd.read_csv("inputs/presence_points.csv")
    try:
        src_epsg = resolve_crs(crs_raw) if crs_raw else 4326
    except:
        history.append({"role":"assistant","content":"Sorry, I couldn't recognize that CRS. Try formats like `32610`, `EPSG:32610`, or `UTM 10T`."})
        return (history, create_map(), state, gr.update(),
                gr.update(visible=True), gr.update(visible=True),
                gr.update(visible=True), gr.update(visible=True))

    src_crs = RioCRS.from_epsg(src_epsg)
    dst_crs = RioCRS.from_epsg(4326)
    lon_vals, lat_vals = rio_transform(src_crs, dst_crs, df[lon_col].tolist(), df[lat_col].tolist())
    df['latitude'], df['longitude'] = lat_vals, lon_vals
    df.to_csv("inputs/presence_points.csv", index=False)

    success = f"✅ Coordinates transformed from EPSG:{src_epsg} to WGS84 (lat/lon)."
    if not state.get("layers_help_shown"):
        success += "\n\n" + available_layers_markdown()
        state["layers_help_shown"] = True
    history.append({"role":"assistant","content": success})

    return (history, create_map(), state, gr.update(),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False))

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
    gr.Markdown("## 🗺️ SpatChat: Species Distribution Model {sdm}  🐢🐍🦅🦋🦉🦊🐞 ")
    gr.HTML("""
    <div style="margin-top: -10px; margin-bottom: 15px;">
      <input type="text" value="https://spatchat.org/browse/?room=sdm" id="shareLink" readonly style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
      <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">
        📋 Copy Share Link
      </button>
      <div style="margin-top: 10px; font-size: 14px;">
        <b>Share:</b>
        <a href="https://twitter.com/intent/tweet?text=Checkout+Spatchat!&url=https://spatchat.org/browse/?room=sdm" target="_blank">🐦 Twitter</a> |
        <a href="https://www.facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=sdm" target="_blank">📘 Facebook</a>
      </div>
    </div>
    """)
    gr.Markdown("""
                <div style="font-size: 14px;">
                © 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
                If you use Spatchat in research, please cite:<br>
                <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>Spatchat: Species Distribution Model.</i>
                </div>
                """)
    state = gr.State({"stage": "await_upload", "layers_help_shown": False})
    with gr.Row():
        with gr.Column(scale=1):
            map_out = gr.HTML(create_map(), label="🗺️ Map Preview")
            download_btn = gr.DownloadButton("📥 Download Results")
        with gr.Column(scale=1):
            chat = gr.Chatbot(
                value=[{"role":"assistant","content":"👋 Hello, I'm SpatChat, your SDM assistant! I'm here to help you build your species distribution model. Please upload your presence CSV to begin."}],
                type="messages", label="💬 Chat", height=400
            )
            user_in = gr.Textbox(label="Ask SpatChat", placeholder="Type commands…")
            file_input = gr.File(label="📄 Upload Presence CSV", type="filepath", file_types=[".csv"])
            lat_dropdown = gr.Dropdown(choices=[], label="Latitude column", visible=False)
            lon_dropdown = gr.Dropdown(choices=[], label="Longitude column", visible=False)
            crs_input = gr.Textbox(label="Input CRS (code, zone, or name)", placeholder="e.g. 32610, UTM zone 10N, LCC…", visible=False)
            confirm_btn = gr.Button("Confirm Coordinates", visible=False)

    demo.queue(max_size=16)

    file_input.change(on_upload,
        inputs=[file_input, chat, state],
        outputs=[chat, map_out, state, download_btn,
                 lat_dropdown, lon_dropdown, crs_input, confirm_btn]
    )
    confirm_btn.click(confirm_coords,
        inputs=[lat_dropdown, lon_dropdown, crs_input, chat, state],
        outputs=[chat, map_out, state, download_btn,
                 lat_dropdown, lon_dropdown, crs_input, confirm_btn]
    )
    user_in.submit(chat_step, inputs=[file_input, user_in, chat, state], outputs=[chat, map_out, state, download_btn])
    user_in.submit(lambda: "", None, user_in)
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=True)
