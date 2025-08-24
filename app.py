import os
import io
import re
import sys
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

from matplotlib import pyplot as plt, colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from folium import Element
from dotenv import load_dotenv
from rasterio.crs import CRS as RioCRS
from rasterio.warp import transform as rio_transform

# Optional LLM clients
TOGETHER_OK = False
HF_OK = False
try:
    from together import Together
    TOGETHER_OK = True
except Exception:
    TOGETHER_OK = False

try:
    # Hugging Face Serverless fallback
    from huggingface_hub import InferenceClient
    HF_OK = True
except Exception:
    HF_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Config & constants
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()

SPATCHAT_MODEL_TOGETHER = os.getenv("SPATCHAT_LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
TOGETHER_KEY = os.getenv("TOGETHER_API_KEY", "")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", os.getenv("HF_TOKEN", ""))
HF_SERVERLESS_MODEL = os.getenv("HF_SERVERLESS_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

# Predictor menu
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

# Pre-render a tiny colorbar for the map (used once)
_fig, _ax = plt.subplots(figsize=(4, 0.5))
_norm = Normalize(vmin=0, vmax=1)
plt.colorbar(ScalarMappable(norm=_norm, cmap="viridis"), cax=_ax, orientation="horizontal").set_ticks([])
_ax.set_xlabel("Low    High")
_fig.tight_layout(pad=0)
_buf = io.BytesIO()
_fig.savefig(_buf, format="png", dpi=100)
plt.close(_fig)
_buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(_buf.read()).decode()

# LLM instruction for JSON tool-call
SYSTEM_PROMPT = """
You are SpatChat, a helpful SDM (species distribution modeling) assistant.
If the user intends to FETCH predictors, respond with EXACT JSON, no code fences:
{"tool":"fetch","layers":[...],"landcover":[...]}
- "layers" are any of: bio1..bio19, elevation, slope, aspect, ndvi, landcover
- "landcover" is a list of MODIS class names in snake_case (e.g., "water", "urban_and_built_up")

If the user intends to RUN the model, respond EXACTLY:
{"tool":"run_model"}

If the user asks to DOWNLOAD results (zip), respond EXACTLY:
{"tool":"download"}

If the user asks to EXPLAIN model stats, respond EXACTLY:
{"tool":"explain_stats"}

If the user asks what a dataset/layer is (e.g., "what is bio5", "where does ndvi come from"), respond EXACTLY:
{"tool":"dataset_info","query":"<the question or token>"}

Otherwise, reply with a short, natural-sounding one-sentence answer.
""".strip()

FALLBACK_PROMPT = """
You are a concise, friendly SDM assistant. Answer in ≤2 sentences and suggest the next step if obvious.
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def intro_text():
    return "👋 Hello, I'm **SpatChat (SDM)**! Upload a presence CSV to begin."

def layers_help_text() -> str:
    return (
        "Now say things like **\"I want bio1, ndvi, elevation\"** to fetch layers, or **\"run model\"** to train & predict.\n\n"
        "### 📦 Available layers (exact datasets)\n"
        "- **Bioclim:** `bio1–bio19` — WorldClim v1 bioclim normals *(WORLDCLIM/V1/BIO, ~1 km)*\n"
        "- **Topography:** `elevation`, `slope`, `aspect` — USGS/SRTMGL1_003 *(30 m)*\n"
        "- **Remote sensing:** `ndvi` — MODIS/061/MOD13Q1 **mean (2022-01-01→2024-01-01)** *(250 m)*\n"
        "- **Landcover:** MODIS IGBP classes — MODIS/061/MCD12Q1 `LC_Type1` *(500 m)*\n"
        "### 💬 Say it like this\n"
        "• \"I want bio1, ndvi, elevation\" *(fetch layers)*\n"
        "• \"Fetch bio5, bio12 and slope\"\n"
        "• \"Add landcover water and urban_and_built_up\"\n"
        "• \"Run model\" *(train & predict)*\n"
        "• \"What is bio5?\" / \"Where does ndvi come from?\"\n"
        "• \"Explain those stats\" *(summarize latest performance & coefficients)*"
    )

def detect_coords(df: pd.DataFrame, fuzz_threshold=80) -> Tuple[Optional[str], Optional[str]]:
    cols = list(df.columns)
    low = [c.lower().strip() for c in cols]
    LAT_ALIASES = {
        "lat",
        "latitude",
        "y",
        "y_coordinate",
        "decilatitude",
        "dec_latitude",
        "dec lat",
        "decimallatitude",
        "decimal latitude",
    }
    LON_ALIASES = {
        "lon",
        "long",
        "longitude",
        "x",
        "x_coordinate",
        "decilongitude",
        "dec_longitude",
        "dec longitude",
        "decimallongitude",
        "decimal longitude",
    }
    lat_idx = next((i for i, n in enumerate(low) if n in LAT_ALIASES), None)
    lon_idx = next((i for i, n in enumerate(low) if n in LON_ALIASES), None)
    if lat_idx is not None and lon_idx is not None:
        return cols[lat_idx], cols[lon_idx]
    # fuzzy
    lat_fz = difflib.get_close_matches("latitude", low, n=1, cutoff=fuzz_threshold / 100)
    lon_fz = difflib.get_close_matches("longitude", low, n=1, cutoff=fuzz_threshold / 100)
    if lat_fz and lon_fz:
        return cols[low.index(lat_fz[0])], cols[low.index(lon_fz[0])]
    # numeric heuristics
    numerics = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    lat_opts = [c for c in numerics if df[c].between(-90, 90).mean() > 0.98]
    lon_opts = [c for c in numerics if df[c].between(-180, 180).mean() > 0.98]
    if len(lat_opts) == 1 and len(lon_opts) == 1:
        return lat_opts[0], lon_opts[0]
    return None, None


def step_state() -> str:
    """Simple state machine based on what files exist."""
    if not os.path.exists("inputs/presence_points.csv"):
        return "await_upload"
    has_rasters = os.path.isdir("predictor_rasters/wgs84") and any(
        f.endswith(".tif") for f in os.listdir("predictor_rasters/wgs84")
    )
    if not has_rasters:
        return "await_fetch"
    # if there is a suitability output, consider "trained"
    if os.path.exists("outputs/suitability_map_wgs84.tif"):
        return "trained"
    return "ready_to_train"


def clear_all():
    for d in ("predictor_rasters", "outputs", "inputs"):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs("inputs", exist_ok=True)
    if os.path.exists("spatchat_results.zip"):
        os.remove("spatchat_results.zip")


def explain_stats_local() -> str:
    """Summarize outputs/performance_metrics.csv and outputs/coefficients.csv in plain text + compact tables."""
    perf_fp = "outputs/performance_metrics.csv"
    coef_fp = "outputs/coefficients.csv"
    parts = ["**Model summary**"]

    have_any = False
    if os.path.exists(perf_fp):
        try:
            perf = pd.read_csv(perf_fp)
            have_any = True
            parts.append(f"- Performance metrics rows: {len(perf)}; columns: {len(perf.columns)}")
            # Show up to 8 cols/12 rows for chat readability
            show = perf.iloc[:12, :8]
            parts.append("\n**Performance (preview)**\n\n" + show.to_markdown(index=False))
        except Exception as e:
            parts.append(f"- Could not read performance metrics: {e}")

    if os.path.exists(coef_fp):
        try:
            coef = pd.read_csv(coef_fp).dropna(axis=1, how="all")
            have_any = True
            parts.append(f"\n- Coefficients rows: {len(coef)}; columns: {len(coef.columns)}")
            showc = coef.iloc[:20, :6]
            parts.append("\n**Coefficients (preview)**\n\n" + showc.to_markdown(index=False))
        except Exception as e:
            parts.append(f"- Could not read coefficients: {e}")

    if not have_any:
        return "I don’t see any exported stats yet. Run the model first, then say “explain those stats.”"
    return "\n".join(parts)


def data_info_answer(q: str) -> str:
    ql = q.lower()
    # BioClim quick explainer
    m = re.search(r"\bbio(\d{1,2})\b", ql)
    if m:
        k = int(m.group(1))
        BIO = {
            1: "BIO1: Annual Mean Temperature",
            2: "BIO2: Mean Diurnal Range (Mean of monthly (max temp − min temp))",
            3: "BIO3: Isothermality (BIO2/BIO7) (*100)",
            4: "BIO4: Temperature Seasonality (standard deviation *100)",
            5: "BIO5: Max Temperature of Warmest Month",
            6: "BIO6: Min Temperature of Coldest Month",
            7: "BIO7: Temperature Annual Range (BIO5 − BIO6)",
            8: "BIO8: Mean Temperature of Wettest Quarter",
            9: "BIO9: Mean Temperature of Driest Quarter",
            10: "BIO10: Mean Temperature of Warmest Quarter",
            11: "BIO11: Mean Temperature of Coldest Quarter",
            12: "BIO12: Annual Precipitation",
            13: "BIO13: Precipitation of Wettest Month",
            14: "BIO14: Precipitation of Driest Month",
            15: "BIO15: Precipitation Seasonality (Coefficient of Variation)",
            16: "BIO16: Precipitation of Wettest Quarter",
            17: "BIO17: Precipitation of Driest Quarter",
            18: "BIO18: Precipitation of Warmest Quarter",
            19: "BIO19: Precipitation of Coldest Quarter",
        }
        desc = BIO.get(k, "")
        meta = "WorldClim v1 (EE: WORLDCLIM/V1/BIO, ~1 km)."
        return f"{desc}\n\nSource: {meta}"

    if "ndvi" in ql:
        return (
            "NDVI (Normalized Difference Vegetation Index) measures greenness from red/near-IR reflectance. "
            "We use MODIS v061 MOD13Q1, averaged 2022-01-01→2024-01-01 (250 m)."
        )
    if "elevation" in ql or "slope" in ql or "aspect" in ql:
        return "SRTM (USGS/SRTMGL1_003, 30 m) via Earth Engine. ‘slope’ and ‘aspect’ are derived from elevation."
    if "landcover" in ql or "igbp" in ql:
        return "MODIS MCD12Q1 v061 IGBP LC_Type1 (500 m). If you specify classes (e.g. ‘water’), we one-hot encode those."
    return "Which layer are you curious about? Try: “what is bio5?”, or “where does ndvi come from?”"


def create_map() -> str:
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)

    # presence points
    ppath = "inputs/presence_points.csv"
    if os.path.exists(ppath):
        df = pd.read_csv(ppath)
        lat_col, lon_col = detect_coords(df)
        if lat_col and lon_col:
            pts = df[[lat_col, lon_col]].dropna().values.tolist()
            if pts:
                fg = folium.FeatureGroup(name="🟦 Presence Points")
                for lat, lon in pts:
                    folium.CircleMarker(
                        location=[lat, lon], radius=5, color="blue", fill=True, fill_opacity=0.8
                    ).add_to(fg)
                fg.add_to(m)
                m.fit_bounds(pts)

    # any predictors that were fetched
    rasdir = "predictor_rasters/wgs84"
    if os.path.isdir(rasdir):
        for fn in sorted(os.listdir(rasdir)):
            if not fn.endswith(".tif"):
                continue
            try:
                with rasterio.open(os.path.join(rasdir, fn)) as src:
                    arr = src.read(1)
                    bnd = src.bounds
                vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
                if np.isnan(vmin) or vmin == vmax:
                    continue
                rgba = colormaps["viridis"]((arr - vmin) / (vmax - vmin))
                folium.raster_layers.ImageOverlay(
                    rgba,
                    bounds=[[bnd.bottom, bnd.left], [bnd.top, bnd.right]],
                    opacity=1.0,  # keep fully opaque
                    name=f"🟨 {os.path.splitext(fn)[0]} ({vmin:.2f}–{vmax:.2f})",
                ).add_to(m)
            except Exception:
                continue

    # suitability map if exists
    sf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        with rasterio.open(sf) as src:
            arr = src.read(1)
            bnd = src.bounds
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
        if not (np.isnan(vmin) or vmin == vmax):
            rgba = colormaps["viridis"]((arr - vmin) / (vmax - vmin))
            folium.raster_layers.ImageOverlay(
                rgba,
                bounds=[[bnd.bottom, bnd.left], [bnd.top, bnd.right]],
                opacity=1.0,  # non-transparent as requested
                name="🎯 Suitability",
            ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    img_html = f'<img src="data:image/png;base64,{COLORBAR_BASE64}" style="position:absolute; bottom:20px; right:10px; width:200px; height:30px; z-index:1000;"/>'
    m.get_root().html.add_child(Element(img_html))
    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" style="width:100%; height:450px; border:none;"></iframe>'


def zip_results() -> str:
    archive = "spatchat_results.zip"
    if os.path.exists(archive):
        os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters", "outputs"):
            if not os.path.isdir(fld):
                continue
            for root, _, files in os.walk(fld):
                for fn in files:
                    full = os.path.join(root, fn)
                    zf.write(full, arcname=os.path.relpath(full, "."))
    return archive


# ──────────────────────────────────────────────────────────────────────────────
# LLM routing (Together → regex → HF serverless)
# ──────────────────────────────────────────────────────────────────────────────

def regex_intent(user_msg: str) -> Optional[dict]:
    text = user_msg.lower()

    # explain stats
    if re.search(r"\b(explain (those|the)?\s*(stats|statistics|numbers)|help me understand (them|these|those|the numbers))\b", text):
        return {"tool": "explain_stats"}

    # run model
    if re.search(r"\b(run|train|build)\s+(sdm|model)\b", text) or text.strip() in {"run model", "build sdm", "train model"}:
        return {"tool": "run_model"}

    # download results
    if re.search(r"\bdownload\b.*\b(zip|results|files?)\b", text) or text.strip() in {"download", "download results", "download zip"}:
        return {"tool": "download"}

    # dataset info
    if re.search(r"\bwhat\s+is\s+bio\d+\b", text) or "where does ndvi come from" in text or "what is ndvi" in text or "what is landcover" in text:
        return {"tool": "dataset_info", "query": user_msg}

    # fetch / add
    if re.search(r"\b(fetch|download|get|grab|add|want|need|pull|import)\b", text):
        layers = set()
        landcovers = set()

        # find tokens that look like bioN
        for m in re.finditer(r"\bbio(\d{1,2})\b", text):
            try:
                k = int(m.group(1))
                if 1 <= k <= 19:
                    layers.add(f"bio{k}")
            except:
                pass
        # exact layer keywords
        for token in ("elevation", "slope", "aspect", "ndvi", "landcover"):
            if re.search(rf"\b{token}\b", text):
                layers.add(token)

        # landcover subclasses
        for lc in LANDCOVER_CLASSES:
            if re.search(rf"\b{re.escape(lc)}\b", text):
                landcovers.add(lc)
        if landcovers and "landcover" not in layers:
            layers.add("landcover")

        if layers or landcovers:
            return {"tool": "fetch", "layers": sorted(layers), "landcover": sorted(landcovers)}

    # help
    if re.search(r"\b(help|what can i do|what next|how to (start|begin)|guide me)\b", text):
        return {"tool": "help"}

    return None


def llm_toolcall(user_msg: str, history_msgs: List[dict]) -> Tuple[Optional[dict], Optional[str]]:
    """
    Try Together -> if fails or JSON parse fails, try regex -> if still None, try HF serverless.
    Return (tool_dict_or_None, plain_text_reply_or_None)
    """
    # 1) TOGETHER
    # Build minimal context: system + last turn only (keeps tokens low)
    if TOGETHER_OK and TOGETHER_KEY:
        try:
            client = Together(api_key=TOGETHER_KEY)
            resp = client.chat.completions.create(
                model=SPATCHAT_MODEL_TOGETHER,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
                temperature=0.0,
            ).choices[0].message.content
            try:
                parsed = json.loads(resp)
                if isinstance(parsed, dict) and "tool" in parsed:
                    return parsed, None
            except Exception:
                # If it's not JSON, treat it as a short answer
                if resp and isinstance(resp, str) and len(resp) < 800:
                    return None, resp
        except Exception as e:
            # swallow and fall through
            pass

    # 2) REGEX fallback
    r = regex_intent(user_msg)
    if r:
        return r, None

    # 3) HF Serverless fallback
    if HF_OK and HF_TOKEN:
        try:
            ic = InferenceClient(model=HF_SERVERLESS_MODEL, token=HF_TOKEN)
            # Keep it short + deterministic
            prompt = (
                SYSTEM_PROMPT
                + "\n\nUser: "
                + user_msg
                + "\nAssistant: (Return either EXACT JSON tool call or a single short sentence.)"
            )
            out = ic.text_generation(prompt=prompt, max_new_tokens=256, temperature=0.0, top_p=1.0, do_sample=False)
            txt = out.strip()
            # strip code fences if any
            if txt.startswith("```"):
                txt = re.sub(r"^```[a-zA-Z0-9]*\n", "", txt).strip()
                txt = txt.replace("```", "").strip()
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, dict) and "tool" in parsed:
                    return parsed, None
            except Exception:
                if txt and len(txt) < 800:
                    return None, txt
        except Exception:
            pass

    # 4) Last-ditch: very short heuristic answer
    return None, "If you want to fetch layers, try: “I want bio1, ndvi”. To train, say: “run model”. To summarize results, say: “explain those stats”."


# ──────────────────────────────────────────────────────────────────────────────
# App actions
# ──────────────────────────────────────────────────────────────────────────────

def run_fetch(selected_layers: List[str], landcover_list: List[str]) -> Tuple[str, str]:
    """
    Calls the separate Earth Engine exporter script.
    Uses environment variables SELECTED_LAYERS and SELECTED_LANDCOVER_CLASSES for scripts/fetch_predictors.py
    """
    layers = [l.lower().strip() for l in (selected_layers or []) if l]
    landcover = [c.lower().strip() for c in (landcover_list or []) if c]

    # Validate
    bad_layers = [l for l in layers if l not in VALID_LAYERS]
    if bad_layers:
        sug = []
        for b in bad_layers:
            m = difflib.get_close_matches(b, list(VALID_LAYERS), n=1, cutoff=0.6)
            if m:
                sug.append(f"Did you mean '{m[0]}' instead of '{b}'?")
        if sug:
            return create_map(), "⚠️ " + " ".join(sug)
        return create_map(), f"⚠️ Unknown layers: {', '.join(bad_layers)}"

    bad_lc = [c for c in landcover if c not in LANDCOVER_CLASSES]
    if bad_lc:
        sug = []
        for b in bad_lc:
            m = difflib.get_close_matches(b, list(LANDCOVER_CLASSES), n=1, cutoff=0.6)
            if m:
                sug.append(f"Did you mean landcover class '{m[0]}' instead of '{b}'?")
        if sug:
            return create_map(), "⚠️ " + " ".join(sug)
        return create_map(), f"⚠️ Unknown landcover classes: {', '.join(bad_lc)}"

    # Env for the script
    os.environ["SELECTED_LAYERS"] = ",".join(layers)
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(landcover)

    # Execute
    cmd = [sys.executable, "-u", os.path.join("scripts", "fetch_predictors.py")]
    proc = None
    try:
        proc = shutil.which(sys.executable)
        r = os.popen(" ".join(cmd)).read()  # simple, synchronous
        # Even if we didn't capture returncode, show logs
        out = "✅ Predictors fetched.\n\n```bash\n" + r + "\n```"
        return create_map(), out
    except Exception as e:
        return create_map(), f"❌ Fetch failed: {e}"


def run_model() -> Tuple[str, str]:
    try:
        p = os.popen(f"{sys.executable} scripts/run_logistic_sdm.py").read()
        # Refresh map and summarize
        m = create_map()
        summary = explain_stats_local()
        msg = "✅ Model trained. Download results below (ZIP).\n\n" + summary
        return m, msg
    except Exception as e:
        return create_map(), f"❌ Model run failed: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# Gradio callbacks
# ──────────────────────────────────────────────────────────────────────────────

def on_upload(f, history, state):
    history2 = list(history)
    # reset workspace
    clear_all()
    if f and hasattr(f, "name"):
        shutil.copy(f.name, "inputs/presence_points.csv")
        df = pd.read_csv("inputs/presence_points.csv")
        lat, lon = detect_coords(df)
        if lat and lon:
            df = df.rename(columns={lat: "latitude", lon: "longitude"})
            df.to_csv("inputs/presence_points.csv", index=False)
            history2.append({"role": "assistant", "content": "✅ I found your **latitude** and **longitude** columns.\n\n" + layers_help_text()})
            return history2, create_map(), state
        else:
            cols = list(df.columns)
            history2.append(
                {
                    "role": "assistant",
                    "content": "I couldn’t detect coordinate columns. Please choose them and specify the input CRS (or enter EPSG like 4326).",
                }
            )
            # If you want a manual coordinate confirmation flow, add extra widgets; omitted here per your “few buttons” preference.
            return history2, create_map(), state
    return history2, create_map(), state


def chat_step(file, user_msg, history, state):
    history2 = list(history)
    user = user_msg.strip()
    if not user:
        return history2, create_map(), state

    # Route with LLM (Together -> regex -> HF)
    tool, short_answer = llm_toolcall(user, history2)

    # If tool requires a CSV but we don't have one yet
    if tool and tool.get("tool") in {"fetch", "run_model"} and not os.path.exists("inputs/presence_points.csv"):
        history2.extend(
            [{"role": "user", "content": user}, {"role": "assistant", "content": "Please upload a presence CSV first."}]
        )
        return history2, create_map(), state

    # Dispatch tools
    assistant_txt = None
    if tool:
        t = tool.get("tool")
        if t == "fetch":
            layers = tool.get("layers", [])
            landcover = tool.get("landcover", [])
            m, status = run_fetch(layers, landcover)
            assistant_txt = status + "\n\n" + "You can now **run model** when ready."
            map_html = m
        elif t == "run_model":
            map_html, assistant_txt = run_model()
        elif t == "download":
            zip_results()
            assistant_txt = "✅ ZIP is ready — click **Download Results**."
            map_html = create_map()
        elif t == "explain_stats":
            assistant_txt = explain_stats_local()
            map_html = create_map()
        elif t == "dataset_info":
            assistant_txt = data_info_answer(tool.get("query", ""))
            map_html = create_map()
        elif t == "help":
            st = step_state()
            if st == "await_upload":
                assistant_txt = intro_text()
            elif st == "await_fetch":
                assistant_txt = "You’ve uploaded points. Next, fetch predictors. For example: “I want bio1, ndvi, elevation”."
            elif st == "ready_to_train":
                assistant_txt = "Predictors fetched. Say “run model” to train & predict, or fetch more layers."
            else:
                assistant_txt = "Model trained. You can say “explain those stats” or download results."
            map_html = create_map()
        else:
            assistant_txt = "I’m not sure what you need. Try: “I want bio1, ndvi” or “run model”."
            map_html = create_map()
    else:
        # No tool (short natural answer or generic hint)
        assistant_txt = short_answer or "Try: “I want bio1, ndvi” (fetch) or “run model”."
        map_html = create_map()

    history2.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant_txt}])
    return history2, map_html, state


# ──────────────────────────────────────────────────────────────────────────────
# Build UI
# ──────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="SpatChat SDM") as demo:
    # Logo (optional if you have the file)
    if os.path.exists("logo_long1.png"):
        gr.Image(
            value="logo_long1.png",
            show_label=False,
            show_download_button=False,
            show_share_button=False,
            type="filepath",
            elem_id="logo-img",
        )
        gr.HTML(
            """
        <style>
        #logo-img img { height: 90px; margin: 10px 50px 10px 10px; border-radius: 6px; }
        </style>
        """
        )

    # Intro in-chat
    chat = gr.Chatbot(
        type="messages",
        label="💬 Chat",
        height=420,
        value=[{"role": "assistant", "content": intro_text()}],
    )

    with gr.Row():
        with gr.Column(scale=1):
            map_out = gr.HTML(create_map(), label="🗺️ Map Preview")
            download_btn = gr.DownloadButton("📥 Download Results", zip_results)
        with gr.Column(scale=1):
            user_in = gr.Textbox(label="Ask SpatChat", placeholder="e.g., I want bio1, ndvi, elevation")
            file_input = gr.File(label="📄 Upload Presence CSV", type="filepath", file_types=[".csv"])

    file_input.change(on_upload, inputs=[file_input, chat, gr.State()], outputs=[chat, map_out, gr.State()])
    user_in.submit(chat_step, inputs=[file_input, user_in, chat, gr.State()], outputs=[chat, map_out, gr.State()])
    user_in.submit(lambda: "", None, user_in)

if __name__ == "__main__":
    print("Starting SpatChat SDM (LLM-first with regex + HF fallback)")
    # No special args; keep it light for Spaces
    demo.queue().launch(ssr_mode=False)
