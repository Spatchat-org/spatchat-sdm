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

# --- Small helper to list available layers (added) ---
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
    """
    Works for HF InferenceClient and Together pydantic objects.
    - choice.message.content may be str or list of content parts.
    """
    msg = getattr(choice, "message", None)
    if msg is None and isinstance(choice, dict):
        msg = choice.get("message")

    # get content (str or list)
    content = None
    if msg is not None:
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)

        # If content is a list (HF can return structured parts), join text pieces.
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            content = "".join(parts)

    if content is None:
        content = ""
    return content

def _delta_text(delta):
    # HF stream delta text accessor
    if isinstance(delta, dict):
        return delta.get("content", "")
    return getattr(delta, "content", "")

# ──────────────────────────────────────────────────────────────
# LLM: HF primary, Together fallback with pacing for 0.6 QPM
# ──────────────────────────────────────────────────────────────

HF_MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # good, lightweight default
TOGETHER_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

class _SpacedCallLimiter:
    """Ensure at least `min_interval_seconds` between calls (per process)."""
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
    """
    Primary: Hugging Face (Serverless or your Endpoint via HF_ENDPOINT_URL)
    Fallback: Together.ai (if TOGETHER_API_KEY is set)
    Returns: plain string (assistant content)
    """
    def __init__(self):
        # HF: model can be a model id OR an endpoint URL
        hf_model_or_url = (os.getenv("HF_ENDPOINT_URL") or HF_MODEL_DEFAULT).strip()
        hf_token = (os.getenv("HF_TOKEN") or "").strip()

        self.hf_client = InferenceClient(
            model=hf_model_or_url,
            token=hf_token,
            timeout=300,
        )

        # Together fallback (optional)
        self.together = None
        self.together_model = (os.getenv("TOGETHER_MODEL") or TOGETHER_MODEL_DEFAULT).strip()
        tg_key = (os.getenv("TOGETHER_API_KEY") or "").strip()
        if tg_key:
            self.together = Together(api_key=tg_key)
            # 0.6 QPM ≈ 100s between requests
            self._tg_limiter = _SpacedCallLimiter(min_interval_seconds=100.0)

    # minimal retry for transient HF 429/5xx
    def _hf_chat(self, messages, max_tokens=512, temperature=0.3, stream=False):
        tries, delay = 3, 2.5
        last_err = None
        for _ in range(tries):
            try:
                if hasattr(self.hf_client, "chat_completion"):
                    resp = self.hf_client.chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=stream,
                    )
                    if stream:
                        text = "".join(_delta_text(ch.choices[0].delta) for ch in resp)
                    else:
                        text = _choice_content(resp.choices[0])
                    return text
                else:
                    # fallback to text_generation if model lacks chat endpoint
                    prompt = self._messages_to_prompt(messages)
                    text = self.hf_client.text_generation(
                        prompt,
                        max_new_tokens=512,
                        temperature=temperature,
                        stream=False,
                        return_full_text=False,
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
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            else:
                parts.append(f"<|assistant|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def chat(self, messages, temperature=0.3, max_tokens=512, stream=False):
        """
        Try HF first; on error, fall back to Together (if configured).
        Together calls are spaced to ~100s and retried on 429/503.
        """
        try:
            return self._hf_chat(messages, max_tokens=max_tokens, temperature=temperature, stream=stream)
        except Exception as hf_err:
            print(f"[LLM] HF primary failed: {hf_err}", file=sys.stderr)
            if self.together is None:
                raise

            # Pace BEFORE first attempt to respect 0.6 QPM
            self._tg_limiter.wait()

            # Retry Together on 429/503 with exponential backoff + jitter
            backoff = 12.0
            for attempt in range(4):  # 1 try + up to 3 retries
                try:
                    resp = self.together.chat.completions.create(
                        model=self.together_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                    )
                    return _choice_content(resp.choices[0])
                except (RateLimitError, ServiceUnavailableError) as e:
                    if attempt == 3:
                        raise
                    time.sleep(backoff + random.uniform(0, 3))
                    backoff *= 1.8

# single shared instance
llm = UnifiedLLM()

# --- Utility to clear all data ---
def clear_all():
    for d in ("predictor_rasters", "outputs", "inputs"):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs("inputs", exist_ok=True)
    # remove any old input CSV, just in case
    csv_fp = "inputs/presence_points.csv"
    if os.path.exists(csv_fp):
        os.remove(csv_fp)

    # clear environment selection
    os.environ.pop("SELECTED_LAYERS", None)
    os.environ.pop("SELECTED_LANDCOVER_CLASSES", None)

    if os.path.exists("spatchat_results.zip"):
        os.remove("spatchat_results.zip")

clear_all()

# --- Utility to clear specific dirs ---
def _clean_dir(path: str):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

# --- Detection helpers ---
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
    # 1) Exact match
    lat_idx = next((i for i,n in enumerate(low) if n in LAT_ALIASES), None)
    lon_idx = next((i for i,n in enumerate(low) if n in LON_ALIASES), None)
    if lat_idx is not None and lon_idx is not None:
        return cols[lat_idx], cols[lon_idx]
    # 2) Fuzzy match
    lat_fz = difflib.get_close_matches("latitude", low, n=1, cutoff=fuzz_threshold/100)
    lon_fz = difflib.get_close_matches("longitude", low, n=1, cutoff=fuzz_threshold/100)
    if lat_fz and lon_fz:
        return cols[low.index(lat_fz[0])], cols[low.index(lon_fz[0])]
    # 3) Numeric heuristics
    numerics = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    lat_opts = [c for c in numerics if df[c].between(-90, 90).mean() > 0.98]
    lon_opts = [c for c in numerics if df[c].between(-180, 180).mean() > 0.98]
    if len(lat_opts) == 1 and len(lon_opts) == 1:
        return lat_opts[0], lon_opts[0]
    # 4) Nothing found
    return None, None


# --- CRS parsing helpers (updated to handle 'UTM 10T') ---
def parse_epsg_code(s: str):
    # Accept "32610", "EPSG:32610", "epsg 32610", etc.
    m = re.search(r'(?:^|\b)epsg\s*:\s*(\d{4,5})\b', s, re.I)
    if not m:
        m = re.match(r'^\s*(\d{4,5})\s*$', s.strip())
    return int(m.group(1)) if m else None

def parse_utm_crs(s: str):
    """
    Parse things like:
      "UTM 10T", "10T", "UTM zone 10N", "zone 10 N", etc.
    - If letter is N/S, use that directly.
    - If letter is a latitude band (C–X, no I/O), infer hemisphere:
        bands N–X => Northern; C–M => Southern.
    """
    txt = s.strip()

    patterns = [
        r'\butm\b[^0-9]*?(\d{1,2})\s*([A-Za-z])?',   # "UTM 10T", "UTM 10N"
        r'\bzone\s*(\d{1,2})\s*([A-Za-z])?',        # "zone 10T", "zone 10N"
        r'\b(\d{1,2})\s*([C-HJ-NP-X])\b',           # "10T" (band letters)
        r'\b(\d{1,2})\s*([NS])\b',                  # "10N"/"10S"
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
        # Latitude band letters: C–M south, N–X north (I and O are not used)
        hemi = 'N' if letter >= 'N' else 'S'
    else:
        hemi = 'N'  # default if unspecified

    return (32600 if hemi == 'N' else 32700) + zone

def llm_parse_crs(raw):
    system = {"role":"system","content":"You're a GIS expert. Given a CRS description, respond with only JSON {\"epsg\": ###} or {\"epsg\": null}."}
    user = {"role":"user","content":f"CRS: '{raw}'"}
    resp = llm.chat(
        [system, user],
        temperature=0.0,
        max_tokens=32,
        stream=False
    )
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

# --- LLM prompts ---
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

# --- Core app functions ---
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
        rgba = colormaps["viridis"]((arr - vmin) / (vmax - vmin))  # fixed denominator
        folium.raster_layers.ImageOverlay(rgba, bounds=[[bnd.bottom,bnd.left],[bnd.top,bnd.right]], opacity=0.7, name="🎯 Suitability").add_to(m)
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
    # ensure a clean predictor set for this fetch
    _clean_dir("predictor_rasters")
    _clean_dir("predictor_rasters/wgs84")

    # normalize inputs to lower-case
    sl = [s.lower() for s in sl]
    lc = [c.lower() for c in lc]

    # 1) Build the list of requested predictors
    layers = list(sl)
    if lc:
        layers.append("landcover")

    # 2) Require at least one predictor
    if not layers:
        return create_map(), "⚠️ Please select at least one predictor."

    # 3) Validate top-level predictors
    bad_layers = [l for l in layers if l not in VALID_LAYERS]
    if bad_layers:
        # 3a) Try difflib suggestions
        suggestions = []
        for b in bad_layers:
            match = difflib.get_close_matches(b, VALID_LAYERS, n=1, cutoff=0.6)
            if match:
                suggestions.append(f"Did you mean '{match[0]}' instead of '{b}'?")
        if suggestions:
            return create_map(), "⚠️ " + " ".join(suggestions)

        # 3b) LLM fallback for top-level names (HF primary)
        prompt = (
            f"You requested these predictors: {', '.join(layers)}. "
            f"I don't recognize: {', '.join(bad_layers)}. "
            "Could you please clarify which predictors you want?"
        )
        clar = llm.chat(
            messages=[
                {"role": "system", "content": FALLBACK_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.7,
            max_tokens=256,
            stream=False
        )
        return create_map(), clar

    # 4) Validate landcover subclasses
    bad_codes = [c for c in lc if c not in LANDCOVER_CLASSES]
    if bad_codes:
        # 4a) difflib suggestions for codes
        suggestions = []
        for b in bad_codes:
            match = difflib.get_close_matches(b, LANDCOVER_CLASSES, n=1, cutoff=0.6)
            if match:
                suggestions.append(
                    f"Did you mean landcover class '{match[0]}' instead of '{b}'?"
                )
        if suggestions:
            return create_map(), "⚠️ " + " ".join(suggestions)

        # 4b) LLM fallback for landcover codes (HF primary)
        prompt = (
            f"You requested landcover classes: {', '.join(lc)}. "
            f"I don't recognize: {', '.join(bad_codes)}. "
            "Could you please clarify which landcover classes you want?"
        )
        clar = llm.chat(
            messages=[
                {"role": "system", "content": FALLBACK_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.7,
            max_tokens=256,
            stream=False
        )
        return create_map(), clar

    # 5) All inputs valid → proceed with fetch
    print(f"🧪 [run_fetch] SL={sl!r}   LC={lc!r}", file=sys.stdout)

    os.environ["SELECTED_LAYERS"] = ",".join(sl)
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(lc)

    cmd = [
        sys.executable,
        "-u",
        os.path.join("scripts", "fetch_predictors.py")
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return create_map(), f"❌ Fetch failed:\n```\n{logs}\n```"
    else:
        return create_map(), f"✅ Predictors fetched.\n\n```bash\n{logs}\n```"

def run_model():
    # clear previous outputs so we only ship fresh results
    _clean_dir("outputs")

    proc = subprocess.run([sys.executable, "-u", os.path.join("scripts","run_logistic_sdm.py")], capture_output=True, text=True)
    if proc.returncode!=0:
        return create_map(), f"❌ Model run failed:\n{proc.stderr}", None, None, None
    perf_df = pd.read_csv("outputs/performance_metrics.csv")
    coef_df = pd.read_csv("outputs/coefficients.csv")

    # prebuild the zip and return its path
    zip_path = zip_results()

    return create_map(), "✅ Model ran successfully! Download the SDM using the button below the map!", perf_df, coef_df, zip_path

def chat_step(file, user_msg, history, state):
    # --- Quick intent: "what/which layers are available/supported?" (added) ---
    if re.search(r'\b(what|which)\b.*\b(layers|predictors)\b.*\b(available|supported|have)\b', user_msg, re.I) \
       or re.search(r'\bavailable layers\b', user_msg, re.I):
        reply = available_layers_markdown()
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":reply}])
        return history, create_map(), state, gr.update()

    # --- TOOL INTENT FIRST (so "I want bio3" triggers fetch even pre-CSV) ---
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":user_msg}]
    resp = llm.chat(msgs, temperature=0.0, max_tokens=256, stream=False)

    try:
        call = json.loads(resp)
        tool = call.get("tool")
    except Exception:
        tool = None

    if tool == "fetch":
        m_out, status = run_fetch(call.get("layers", []), call.get("landcover", []))
        assistant_txt = f"{status}\n\nGreat! Now you can run the model or fetch more layers."
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":assistant_txt}])
        return history, m_out, state, gr.update()  # no change to download button
    elif tool == "run_model":
        m_out, status, perf_df, coef_df, zip_path = run_model()
        if perf_df is None:
            assistant_txt = status
            dl_update = gr.update()
        else:
            perf = pd.read_csv("outputs/performance_metrics.csv")
            first, second = perf.iloc[:, :3], perf.iloc[:, 3:]
            perf_md = (
                "**Model Performance (1 of 2):**\n\n" + first.to_markdown(index=False)
                + "\n\n**Model Performance (2 of 2):**\n\n" + second.to_markdown(index=False)
            )
            coef = pd.read_csv("outputs/coefficients.csv").dropna(axis=1, how='all')
            coef_md = coef.to_markdown(index=False)
            assistant_txt = (
                f"{status}\n\n**Model Performance:**\n\n{perf_md}\n\n"
                f"**Predictor Coefficients:**\n\n{coef_md}"
            )
            dl_update = gr.update(value=zip_path)
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":assistant_txt}])
        return history, m_out, state, dl_update

    # --- If no tool was detected, proceed with the original logic ---
    # 0a) If no CSV yet, fallback to conversational LLM
    if not os.path.exists("inputs/presence_points.csv"):
        fb = [
            {"role":"system","content":FALLBACK_PROMPT},
            {"role":"user","content":user_msg}
        ]
        reply = llm.chat(
            fb,
            temperature=0.7,
            max_tokens=256,
            stream=False
        )
        history.extend([{"role":"user","content":user_msg}, {"role":"assistant","content":reply}])
        return history, create_map(), state, gr.update()

    # 0b) “run model” shortcut (also catch “model” or “run” alone)
    if re.fullmatch(r"\s*(?:run\s+)?model\s*$", user_msg, re.I):
        m_out, status, perf_df, coef_df, zip_path = run_model()
        if perf_df is not None:
            perf = pd.read_csv("outputs/performance_metrics.csv")
            first, second = perf.iloc[:, :3], perf.iloc[:, 3:]
            perf_md = (
                "**Model Performance (1 of 2):**\n\n"
                + first.to_markdown(index=False)
                + "\n\n**Model Performance (2 of 2):**\n\n"
                + second.to_markdown(index=False)
            )
            coef = pd.read_csv("outputs/coefficients.csv").dropna(axis=1, how='all')
            status += "\n\n**Model Performance:**\n\n" + perf_md
            status += "\n\n**Predictor Coefficients:**\n\n" + coef.to_markdown(index=False)
        assistant_txt = status
        history.extend([
            {"role":"user","content":user_msg},
            {"role":"assistant","content":assistant_txt}
        ])
        return history, m_out, state, gr.update(value=zip_path)

    # 1) Handle reset
    if re.search(r"\b(start over|restart|clear everything|reset|clear all)\b", user_msg, re.I):
        clear_all()
        new_hist = [{"role":"assistant","content":"👋 All cleared! Please upload your presence-points CSV to begin."}]
        return new_hist, create_map(), state, gr.update(value=None)

    # 2) Build the JSON-tool prompt (legacy order kept for non-tool Q&A)
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":user_msg}]
    resp = llm.chat(
        msgs,
        temperature=0.0,
        max_tokens=256,
        stream=False
    )

    try:
        call = json.loads(resp)
        tool = call.get("tool")
    except Exception:
        tool = None

    if tool == "fetch":
        m_out, status = run_fetch(call.get("layers", []), call.get("landcover", []))
        assistant_txt = f"{status}\n\nGreat! Now you can run the model or fetch more layers."
        dl_update = gr.update()
    elif tool == "run_model":
        m_out, status, perf_df, coef_df, zip_path = run_model()
        if perf_df is None:
            assistant_txt = status
            dl_update = gr.update()
        else:
            perf = pd.read_csv("outputs/performance_metrics.csv")
            first, second = perf.iloc[:, :3], perf.iloc[:, 3:]
            perf_md = (
                "**Model Performance (1 of 2):**\n\n" + first.to_markdown(index=False)
                + "\n\n**Model Performance (2 of 2):**\n\n" + second.to_markdown(index=False)
            )
            coef = pd.read_csv("outputs/coefficients.csv").dropna(axis=1, how='all')
            coef_md = coef.to_markdown(index=False)
            assistant_txt = (
                f"{status}\n\n**Model Performance:**\n\n{perf_md}\n\n"
                f"**Predictor Coefficients:**\n\n{coef_md}"
            )
            dl_update = gr.update(value=zip_path)
    elif tool == "download":
        m_out, _ = create_map(), zip_results()
        assistant_txt = "✅ ZIP is downloading…"
        dl_update = gr.update(value="spatchat_results.zip")
    else:
        # summary block
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
        perf_fp = "outputs/performance_metrics.csv"
        if os.path.exists(perf_fp):
            perf = pd.read_csv(perf_fp)
            perf_table = perf.to_markdown(index=False)
        coef_table = ""
        coef_fp = "outputs/coefficients.csv"
        if os.path.exists(coef_fp):
            coef = pd.read_csv(coef_fp).dropna(axis=1, how='all')
            coef_table = coef.to_markdown(index=False)
        summary = (
            f"- Presence points: {n_pts}\n"
            f"- Layers fetched ({len(fetched)}): {', '.join(fetched) or 'none'}\n\n"
            "**Performance Metrics**\n"
            f"{perf_table or '*none*'}\n\n"
            "**Predictor Coefficients**\n"
            f"{coef_table or '*none*'}"
        )
        explain_sys = {
            "role":"system",
            "content":(
                "You are SpatChat, an expert in species distribution modeling. "
                "Use ALL of the context below to answer the user's question as clearly as possible."
            )
        }
        msgs = [explain_sys, {"role":"system","content":"Data summary:\n" + summary}, {"role":"user","content":user_msg}]
        assistant_txt = llm.chat(
            msgs,
            temperature=0.7,
            max_tokens=384,
            stream=False
        )
        m_out = create_map()
        dl_update = gr.update()

    history.extend([
        {"role":"user","content":user_msg},
        {"role":"assistant","content":assistant_txt}
    ])
    return history, m_out, state, dl_update

# --- Upload callback (returns expanded outputs to toggle pickers) ---
def on_upload(f, history, state):
    history2 = history.copy()
    clear_all()
    if f and hasattr(f, "name"):
        # 1. copy original CSV
        shutil.copy(f.name, "inputs/presence_points.csv")
        # 2. load & detect whatever the user called their coords
        df = pd.read_csv("inputs/presence_points.csv")
        lat, lon = detect_coords(df)
        if lat and lon:
            # 3. rename to the exact column names fetch_predictors expects
            df = df.rename(columns={lat: "latitude", lon: "longitude"})
            df.to_csv("inputs/presence_points.csv", index=False)

            history2.append({"role":"assistant","content":(
                "✅ Sweet! I found your `latitude` and `longitude` columns.\n\n"
                + available_layers_markdown()
            )})
            # hide pickers when auto-detected
            return (history2, create_map(), state, gr.update(),
                    gr.update(choices=[], visible=False),
                    gr.update(choices=[], visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False))
        else:
            history2.append({"role":"assistant","content":
                "I couldn't detect coordinate columns. Please select them and enter CRS below (e.g., `UTM 10T` or `32610`).\n\n"
                + available_layers_markdown()
            })
            cols = list(df.columns)
            # show pickers when needed
            return (history2, create_map(), state, gr.update(),
                    gr.update(choices=cols, visible=True, value=None),
                    gr.update(choices=cols, visible=True, value=None),
                    gr.update(visible=True, value="UTM 10T"),
                    gr.update(visible=True))
    # default: keep hidden
    return (history2, create_map(), state, gr.update(),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False))

# --- CRS confirm callback (returns expanded outputs to toggle pickers) ---
def confirm_coords(lat_col, lon_col, crs_raw, history, state):
    df = pd.read_csv("inputs/presence_points.csv")
    try:
        src_epsg = resolve_crs(crs_raw) if crs_raw else 4326
    except:
        history.append({"role":"assistant","content":"Sorry, I couldn't recognize that CRS. Try formats like `32610`, `EPSG:32610`, or `UTM 10T`."})
        # keep inputs visible for another try
        return (history, create_map(), state, gr.update(),
                gr.update(visible=True), gr.update(visible=True),
                gr.update(visible=True), gr.update(visible=True))
    src_crs = RioCRS.from_epsg(src_epsg)
    dst_crs = RioCRS.from_epsg(4326)
    # Note: inputs are x=easting (lon_col) and y=northing (lat_col) in source CRS.
    lon_vals, lat_vals = rio_transform(src_crs, dst_crs, df[lon_col].tolist(), df[lat_col].tolist())
    df['latitude'], df['longitude'] = lat_vals, lon_vals
    df.to_csv("inputs/presence_points.csv", index=False)
    history.append({
        "role": "assistant",
        "content": (
            f"✅ Coordinates transformed from EPSG:{src_epsg} to WGS84 (lat/lon).\n\n"
            + available_layers_markdown()
        )
    })
    # hide after success
    return (history, create_map(), state, gr.update(),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False))

# --- Gradio UI ---
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
        margin: 10px 50px 10px 10px;  /* top, right, bottom, left */
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
                <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>Spatchat: Specides Distribution Model.</i>
                </div>
                """)
    state = gr.State({"stage": "await_upload"})
    with gr.Row():
        with gr.Column(scale=1):
            map_out = gr.HTML(create_map(), label="🗺️ Map Preview")
            # value is set dynamically after model finishes
            download_btn = gr.DownloadButton("📥 Download Results")
        with gr.Column(scale=1):
            chat = gr.Chatbot(value=[{"role":"assistant","content":"👋 Hello, I'm SpatChat, your SDM assistant! I'm here to help you build your species distribution model. Please upload your presence CSV to begin."}], type="messages", label="💬 Chat", height=400)
            user_in = gr.Textbox(label="Ask SpatChat", placeholder="Type commands…")
            file_input = gr.File(label="📄 Upload Presence CSV", type="filepath", file_types=[".csv"])
            lat_dropdown = gr.Dropdown(choices=[], label="Latitude column", visible=False)
            lon_dropdown = gr.Dropdown(choices=[], label="Longitude column", visible=False)
            crs_input = gr.Textbox(label="Input CRS (code, zone, or name)", placeholder="e.g. 32610, UTM zone 10N, LCC…", visible=False)
            confirm_btn = gr.Button("Confirm Coordinates", visible=False)

    # Older Gradio compatibility: only pass max_size
    demo.queue(max_size=16)

    # include picker visibility outputs
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
