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

import gradio as gr
import geemap.foliumap as foliumap  # noqa: F401 (kept for future map features)
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
from together import Together
from dotenv import load_dotenv
from rasterio.crs import CRS as RioCRS
from rasterio.warp import transform as rio_transform

print("Starting SpatChat SDM (with layer explainer + stats interpreter)")

# ------------------------- Layer catalog (no LLM needed) -------------------------
# Short, safe, vendor-neutral summaries. You can customize sources in scripts/fetch_predictors.py.
BIO_DEFS = {
    "bio1": "Annual mean temperature",
    "bio2": "Mean diurnal range (mean of monthly (max temp − min temp))",
    "bio3": "Isothermality (BIO2/BIO7 × 100)",
    "bio4": "Temperature seasonality (std. dev. × 100)",
    "bio5": "Max temperature of warmest month",
    "bio6": "Min temperature of coldest month",
    "bio7": "Temperature annual range (BIO5 − BIO6)",
    "bio8": "Mean temperature of wettest quarter",
    "bio9": "Mean temperature of driest quarter",
    "bio10": "Mean temperature of warmest quarter",
    "bio11": "Mean temperature of coldest quarter",
    "bio12": "Annual precipitation",
    "bio13": "Precipitation of wettest month",
    "bio14": "Precipitation of driest month",
    "bio15": "Precipitation seasonality (coefficient of variation)",
    "bio16": "Precipitation of wettest quarter",
    "bio17": "Precipitation of driest quarter",
    "bio18": "Precipitation of warmest quarter",
    "bio19": "Precipitation of coldest quarter",
}
LAYER_CATALOG = {
    "bioclim": {
        "name": "Bioclim variables (Bio1–Bio19)",
        "source": (
            "Climatology normals (temperature & precipitation–derived variables). "
            "Commonly distributed as WorldClim v2 or CHELSA; your Space fetches them via Earth Engine "
            "(see scripts/fetch_predictors.py for the exact dataset/version)."
        ),
        "resolution": "Typically ~1 km (depends on dataset/version)",
        "notes": "Use names bio1 … bio19. Examples: bio1=annual mean temp, bio12=annual precip.",
    },
    "elevation": {
        "name": "Elevation",
        "source": (
            "Global DEM via Earth Engine (e.g., SRTM or NASA DEM). "
            "Your exact DEM is defined in scripts/fetch_predictors.py."
        ),
        "resolution": "30–90 m (dataset dependent)",
        "notes": "Slope and aspect are derived from elevation.",
    },
    "slope": {
        "name": "Slope (derived)",
        "source": "Derived from the selected DEM in Earth Engine.",
        "resolution": "Same as elevation",
        "notes": "Units usually degrees.",
    },
    "aspect": {
        "name": "Aspect (derived)",
        "source": "Derived from the selected DEM in Earth Engine.",
        "resolution": "Same as elevation",
        "notes": "Degrees clockwise from North (0–360).",
    },
    "ndvi": {
        "name": "NDVI (vegetation greenness)",
        "source": (
            "Normalized Difference Vegetation Index derived from satellite reflectance "
            "(commonly MODIS or Sentinel/Landsat composites). "
            "See scripts/fetch_predictors.py for which collection is used."
        ),
        "resolution": "250 m – 10 m (sensor dependent)",
        "notes": "Range ~ −1 to 1; higher = greener vegetation.",
    },
    "landcover": {
        "name": "Land cover classes",
        "source": (
            "MODIS MCD12Q1 (IGBP scheme) is commonly used. "
            "Class list matches the IGBP categories."
        ),
        "resolution": "500 m (for MODIS MCD12Q1)",
        "notes": "Use snake_case class names, e.g., water, urban_and_built_up, croplands, etc.",
    },
}

# --- Which top-level predictors we support (all lower-case) ---
PREDICTOR_CHOICES = ([f"bio{i}" for i in range(1, 20)] + ["elevation", "slope", "aspect", "ndvi", "landcover"])
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

# --- LLM client (primary: Together) ---
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# ------------------------- Utility: layer help & info -------------------------
def layers_help_text() -> str:
    return (
        "### 📦 Available layers\n"
        "- **Bioclim**: **bio1–bio19** (e.g., bio1=Annual Mean Temp, bio12=Annual Precip)\n"
        "- **Topography**: **elevation**, **slope**, **aspect**\n"
        "- **Remote sensing**: **ndvi**\n"
        "- **Landcover** (MODIS IGBP): "
        "water, urban_and_built_up, croplands, grasslands, evergreen_broadleaf_forest, "
        "evergreen_needleleaf_forest, deciduous_broadleaf_forest, deciduous_needleleaf_forest, "
        "mixed_forest, closed_shrublands, open_shrublands, woody_savannas, savannas, "
        "permanent_wetlands, cropland_natural_vegetation_mosaic, snow_and_ice, barren_or_sparsely_vegetated\n"
        "\n"
        "### 💬 Say it like this\n"
        "• **\"I want bio1, ndvi, elevation\"** (fetch layers)\n"
        "• **\"Fetch bio5, bio12 and slope\"**\n"
        "• **\"Add landcover water and urban_and_built_up\"**\n"
        "• **\"Run model\"** (train & predict)\n"
        "• **\"What is bio5?\"** or **\"Where does landcover come from?\"**\n"
        "• **\"Explain those stats\"** (summarize latest performance & coefficients)\n"
    )

def describe_layer_token(tok: str) -> str:
    t = tok.strip().lower()
    if t in BIO_DEFS:
        return f"**{t.upper()}** — {BIO_DEFS[t]}\n• Source: {LAYER_CATALOG['bioclim']['source']}\n• Notes: {LAYER_CATALOG['bioclim']['notes']}"
    if t in ("elevation", "slope", "aspect"):
        info = LAYER_CATALOG[t]
        return f"**{info['name']}**\n• Source: {info['source']}\n• Resolution: {info['resolution']}\n• Notes: {info['notes']}"
    if t == "ndvi":
        info = LAYER_CATALOG["ndvi"]
        return f"**{info['name']}**\n• Source: {info['source']}\n• Resolution: {info['resolution']}\n• Notes: {info['notes']}"
    if t == "landcover":
        info = LAYER_CATALOG["landcover"]
        cats = ", ".join(sorted(list(LANDCOVER_CLASSES))[:10]) + ", …"
        return f"**{info['name']}**\n• Source: {info['source']}\n• Resolution: {info['resolution']}\n• Notes: {info['notes']}\n• Example classes: {cats}"
    # Try to interpret "bio 5" or "Bio-5"
    m = re.match(r"bio\s*[-_ ]?(\d{1,2})$", t)
    if m:
        key = f"bio{int(m.group(1))}"
        if key in BIO_DEFS:
            return describe_layer_token(key)
    return f"Sorry — I don’t recognize **{tok}** as a fetchable layer."

def explain_layers_from_text(user_msg: str) -> str:
    # Pull all tokens we might explain
    toks = re.findall(r"(bio\s*[-_ ]?\d{1,2}|ndvi|elevation|slope|aspect|landcover)", user_msg, flags=re.I)
    if not toks:
        # general layer explainer
        parts = [
            "### ℹ️ Data sources (summary)",
            f"**Bioclim** — {LAYER_CATALOG['bioclim']['source']}",
            f"**Elevation/Slope/Aspect** — {LAYER_CATALOG['elevation']['source']}",
            f"**NDVI** — {LAYER_CATALOG['ndvi']['source']}",
            f"**Landcover** — {LAYER_CATALOG['landcover']['source']}",
            "",
            layers_help_text()
        ]
        return "\n".join(parts)
    out = ["### ℹ️ Layer details"]
    seen = set()
    for tok in toks:
        key = tok.lower().strip()
        if key not in seen:
            out.append("- " + describe_layer_token(tok))
            seen.add(key)
    return "\n".join(out)

# ------------------------- Stats explanation (no LLM) -------------------------
def explain_latest_stats() -> str:
    """Summarize latest outputs/performance_metrics.csv and outputs/coefficients.csv in human language."""
    perf_fp = "outputs/performance_metrics.csv"
    coef_fp = "outputs/coefficients.csv"
    lines = []
    had_any = False

    if os.path.exists(perf_fp):
        try:
            perf = pd.read_csv(perf_fp)
            if not perf.empty:
                had_any = True
                row = perf.iloc[0]  # take first row (or only)
                # pick common metrics if present
                metrics = []
                def pick(name):
                    for c in perf.columns:
                        if c.lower() == name:
                            return c
                    return None
                for key, label in [
                    ("auc", "AUC"),
                    ("accuracy", "Accuracy"),
                    ("sensitivity", "Sensitivity"),
                    ("specificity", "Specificity"),
                    ("precision", "Precision"),
                    ("recall", "Recall"),
                    ("f1", "F1"),
                    ("tss", "TSS"),
                ]:
                    col = pick(key)
                    if col and pd.api.types.is_numeric_dtype(perf[col]):
                        val = row[col]
                        if pd.notna(val):
                            metrics.append(f"{label} ≈ {val:.3f}")
                header = "### 🔍 Model performance"
                if metrics:
                    lines.append(header)
                    lines.append("• " + " | ".join(metrics))
                else:
                    lines.append(header + "\n(Performance file present, but no standard metric columns detected.)")
        except Exception as e:
            lines.append(f"⚠️ Couldn’t parse performance_metrics.csv: {e}")

    if os.path.exists(coef_fp):
        try:
            coef = pd.read_csv(coef_fp).dropna(axis=1, how="all")
            # try to find coefficient columns
            term_col = next((c for c in coef.columns if c.lower() in ("term", "feature", "variable", "predictor")), None)
            val_col = next((c for c in coef.columns if c.lower() in ("coef", "coefficient", "estimate", "beta", "weight")), None)
            if term_col and val_col and not coef.empty:
                had_any = True
                # drop intercept-ish rows
                mask = ~coef[term_col].str.lower().str.contains(r"^intercept$|^const$", regex=True, na=False)
                slim = coef.loc[mask, [term_col, val_col]].copy()
                slim["abs"] = slim[val_col].abs()
                top = slim.sort_values("abs", ascending=False).head(5)
                pos = top[top[val_col] > 0].head(3)
                neg = top[top[val_col] < 0].head(3)
                lines.append("### 🧠 Important predictors")
                if not top.empty:
                    if not pos.empty:
                        lines.append("• Strongest positive effects: " + ", ".join(f"{r[term_col]} (+{r[val_col]:.3g})" for _, r in pos.iterrows()))
                    if not neg.empty:
                        lines.append("• Strongest negative effects: " + ", ".join(f"{r[term_col]} ({r[val_col]:.3g})" for _, r in neg.iterrows()))
                else:
                    lines.append("(Couldn’t rank predictors.)")
            else:
                # still show the table size
                if not os.path.exists(coef_fp):
                    pass
                else:
                    lines.append("### 🧠 Coefficients")
                    lines.append("(Coefficients file present, but couldn’t detect ‘term/coef’ columns.)")
        except Exception as e:
            lines.append(f"⚠️ Couldn’t parse coefficients.csv: {e}")

    if not had_any:
        return "I couldn’t find any model outputs yet. Try **run model**, then say **explain those stats**."
    # Friendly closer
    lines.append("\nTip: If any metric is unclear, ask e.g. “What does AUC mean?”")
    return "\n".join(lines)

# ------------------------- App state & file housekeeping -------------------------
def clear_all():
    for d in ("predictor_rasters", "outputs", "inputs"):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs("inputs", exist_ok=True)
    if os.path.exists("spatchat_results.zip"):
        os.remove("spatchat_results.zip")

clear_all()

# ------------------------- Detection helpers -------------------------
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

# ------------------------- CRS parsing helpers -------------------------
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
    system = {"role":"system","content":"You're a GIS expert. Given a CRS description, respond with only JSON {\"epsg\": ###} or {\"epsg\": null}."}
    user = {"role":"user","content":f"CRS: '{raw}'"}
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
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

# ------------------------- LLM prompts -------------------------
SYSTEM_PROMPT = """
You are SpatChat, a friendly species distribution modeling assistant.
When the user asks to fetch environmental layers (verbs like fetch, download, get, grab, "I want", "add"), respond with exactly a JSON object:
{"tool":"fetch","layers":[<layer names>],"landcover":[<landcover classes>]}
When the user asks to run the model (e.g., "run model", "run SDM", "train"), respond with exactly:
{"tool":"run_model"}
If the request doesn’t match these tools, reply naturally without JSON.
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
Keep answers short (<=2 sentences) and guide the user to next steps: upload data, fetch layers, run model, or ask about layers ("What is bio5?").
""".strip()

# ------------------------- Map / ZIP -------------------------
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

# ------------------------- Actions -------------------------
def run_fetch(sl, lc):
    layers = list(sl)
    if lc:
        layers.append("landcover")
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
        prompt = (
            f"You requested these predictors: {', '.join(layers)}. "
            f"I don't recognize: {', '.join(bad_layers)}. "
            "Could you please clarify which predictors you want?"
        )
        clar = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "system", "content": FALLBACK_PROMPT},{"role": "user", "content": prompt}],
            temperature=0.7
        ).choices[0].message.content
        return create_map(), clar

    # Validate landcover subclasses
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
        clar = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "system", "content": FALLBACK_PROMPT},{"role": "user", "content": prompt}],
            temperature=0.7
        ).choices[0].message.content
        return create_map(), clar

    print(f"🧪 [run_fetch] SL={sl!r}   LC={lc!r}", file=sys.stdout)
    os.environ["SELECTED_LAYERS"] = ",".join(sl)
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(lc)
    cmd = [sys.executable, "-u", os.path.join("scripts", "fetch_predictors.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return create_map(), f"❌ Fetch failed:\n```\n{logs}\n```"
    else:
        return create_map(), f"✅ Predictors fetched.\n\n```bash\n{logs}\n```"

def run_model():
    proc = subprocess.run(["python","scripts/run_logistic_sdm.py"], capture_output=True, text=True)
    if proc.returncode!=0:
        return create_map(), f"❌ Model run failed:\n{proc.stderr}", None, None
    perf_df = pd.read_csv("outputs/performance_metrics.csv")
    coef_df = pd.read_csv("outputs/coefficients.csv")
    zip_results()
    return create_map(), "✅ Model ran successfully! Download the SDM using the button below the map!", perf_df, coef_df

# ------------------------- Chat loop -------------------------
def chat_step(file, user_msg, history, state):
    # 0) If no CSV yet, fallback to conversational LLM
    if not os.path.exists("inputs/presence_points.csv"):
        fb = [{"role":"system","content":FALLBACK_PROMPT},{"role":"user","content":user_msg}]
        reply = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", messages=fb, temperature=0.7
        ).choices[0].message.content
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":reply}])
        return history, create_map(), state

    # 1) Reset
    if re.search(r"\b(start over|restart|clear everything|reset|clear all)\b", user_msg, re.I):
        clear_all()
        new_hist = [{"role":"assistant","content":"👋 All cleared! Please upload your presence-points CSV to begin."}]
        return new_hist, create_map(), state

    # 2) Quick “help / layers” menu
    if re.search(r"\b(help|layers|what can i (?:fetch|get|use)|available layers|what layers)\b", user_msg, re.I):
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":layers_help_text()}])
        return history, create_map(), state

    # 3) Layer explainer (no LLM): “what is bio5?”, “where does landcover come from?”
    if re.search(r"\b(what\s+is|what\s+does|where\s+does|which\s+dataset|source of)\b", user_msg, re.I) and re.search(
        r"(bio\s*[-_ ]?\d{1,2}|ndvi|elevation|slope|aspect|landcover|layers?)", user_msg, re.I
    ):
        ans = explain_layers_from_text(user_msg)
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":ans}])
        return history, create_map(), state

    # 4) Explain stats (no LLM): “explain those stats”, “interpret the results”
    if re.search(r"\b(explain|interpret|help me understand|summarize)\b", user_msg, re.I) and re.search(
        r"\b(stats?|results?|model|performance|coefficients?)\b", user_msg, re.I
    ):
        ans = explain_latest_stats()
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":ans}])
        return history, create_map(), state
    if re.fullmatch(r"\s*explain those stats\s*", user_msg, re.I):
        ans = explain_latest_stats()
        history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":ans}])
        return history, create_map(), state

    # 5) Build the JSON-tool prompt and dispatch
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":user_msg}]
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", messages=msgs, temperature=0.0
    ).choices[0].message.content

    try:
        call = json.loads(resp)
        tool = call.get("tool")
    except Exception:
        tool = None

    if tool == "fetch":
        m_out, status = run_fetch(call.get("layers", []), call.get("landcover", []))
        assistant_txt = f"{status}\n\nGreat! Now you can run the model or fetch more layers."
    elif tool == "run_model":
        m_out, status, perf_df, coef_df = run_model()
        if perf_df is None:
            assistant_txt = status
        else:
            perf = pd.read_csv("outputs/performance_metrics.csv")
            first, second = perf.iloc[:, :3], perf.iloc[:, 3:]
            perf_md = (
                "**Model Performance (1 of 2):**\n\n" + first.to_markdown(index=False)
                + "\n\n**Model Performance (2 of 2):**\n\n" + second.to_markdown(index=False)
            )
            coef = pd.read_csv("outputs/coefficients.csv").dropna(axis=1, how='all')
            coef_md = coef.to_markdown(index=False)
            assistant_txt = f"{status}\n\n**Model Performance:**\n\n{perf_md}\n\n**Predictor Coefficients:**\n\n{coef_md}"
    elif tool == "download":
        m_out, _ = create_map(), zip_results()
        assistant_txt = "✅ ZIP is downloading…"
    else:
        # Non-tool general conversation (fallback)
        try:
            n_pts = len(pd.read_csv("inputs/presence_points.csv"))
        except Exception:
            n_pts = 0
        rasdir = "predictor_rasters/wgs84"
        if os.path.isdir(rasdir):
            fetched = sorted(os.path.splitext(f)[0] for f in os.listdir(rasdir) if f.endswith(".tif"))
        else:
            fetched = []
        perf_table = ""
        if os.path.exists("outputs/performance_metrics.csv"):
            perf_table = pd.read_csv("outputs/performance_metrics.csv").to_markdown(index=False)
        coef_table = ""
        if os.path.exists("outputs/coefficients.csv"):
            coef_table = pd.read_csv("outputs/coefficients.csv").dropna(axis=1, how='all').to_markdown(index=False)
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
            "content":"You are SpatChat, an expert in species distribution modeling. Use ALL of the context below to answer clearly."
        }
        msgs = [explain_sys, {"role":"system","content":"Data summary:\n" + summary}, {"role":"user","content":user_msg}]
        assistant_txt = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", messages=msgs, temperature=0.7
        ).choices[0].message.content
        m_out = create_map()

    history.extend([{"role":"user","content":user_msg},{"role":"assistant","content":assistant_txt}])
    return history, m_out, state

# ------------------------- Upload & CRS confirm -------------------------
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
                "✅ Sweet! I found your `latitude` and `longitude` columns.\n\n"
                "Now say things like **\"I want bio1, ndvi, elevation\"** to fetch layers, "
                "or **\"run model\"** to train & predict.\n\n"
                + layers_help_text()
            )})
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
    history.append({
        "role": "assistant",
        "content": (
            "✅ Coordinates set! You're doing awesome!\n\n"
            "Now say things like **\"I want bio1, ndvi, elevation\"** to fetch layers, "
            "or **\"run model\"** to train & predict.\n\n"
            + layers_help_text()
        )
    })
    return history, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

# ------------------------- UI -------------------------
WELCOME_MD = (
    "👋 Hello, I'm **SpatChat (SDM)**! Upload a presence CSV to begin.\n\n"
    "Now say things like **\"I want bio1, ndvi, elevation\"** to fetch layers, or **\"run model\"** to train & predict.\n\n"
    + layers_help_text()
)

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
      <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">📋 Copy Share Link</button>
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
    state = gr.State({"stage": "await_upload"})
    with gr.Row():
        with gr.Column(scale=1):
            map_out = gr.HTML(create_map(), label="🗺️ Map Preview")
            download_btn = gr.DownloadButton("📥 Download Results", zip_results)
        with gr.Column(scale=1):
            chat = gr.Chatbot(value=[{"role":"assistant","content":WELCOME_MD}], type="messages", label="💬 Chat", height=400)
            user_in = gr.Textbox(label="Ask SpatChat", placeholder="Type commands…")
            file_input = gr.File(label="📄 Upload Presence CSV", type="filepath", file_types=[".csv"])
            lat_dropdown = gr.Dropdown(choices=[], label="Latitude column", visible=False)
            lon_dropdown = gr.Dropdown(choices=[], label="Longitude column", visible=False)
            crs_input = gr.Textbox(label="Input CRS (code, zone, or name)", placeholder="e.g. 32610, UTM zone 10N, LCC…", visible=False)
            confirm_btn = gr.Button("Confirm Coordinates", visible=False)
    file_input.change(on_upload, inputs=[file_input, chat, state], outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn])
    confirm_btn.click(confirm_coords, inputs=[lat_dropdown, lon_dropdown, crs_input, chat, state], outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn])
    user_in.submit(chat_step, inputs=[file_input, user_in, chat, state], outputs=[chat, map_out, state])
    user_in.submit(lambda: "", None, user_in)

if __name__ == "__main__":
    # On Gradio 5.x, queue() has no concurrency_count param; HF Spaces handles scaling
    try:
        demo.queue(max_size=32).launch(ssr_mode=False)
    except TypeError:
        demo.launch(ssr_mode=False)
