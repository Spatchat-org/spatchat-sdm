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
from together import Together
from dotenv import load_dotenv
from rasterio.crs import CRS as RioCRS
from rasterio.warp import transform as rio_transform

# --- Which top‑level predictors we support (all lower‑case) ---
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
svc = json.loads(os.environ.get("GEE_SERVICE_ACCOUNT", "{}"))
creds = ee.ServiceAccountCredentials(svc.get("client_email"), key_data=json.dumps(svc))
ee.Initialize(creds)

# --- LLM client ---
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

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


# --- CRS parsing helpers ---
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

# --- LLM prompts ---
SYSTEM_PROMPT = """
You are SpatChat, a friendly assistant and an expert in species distribution modeling.
When the user issues a **command**:
  - fetch   → {"tool":"fetch",   "layers":…, "landcover":…}
  - run_model → {"tool":"run_model"}
  - download  → {"tool":"download"}
you MUST reply in JSON exactly as above.
All *other* inputs are *not* commands.

For landcover synonyms, map user‑friendly words into the exact MODIS codes:
    • water, lake, river, ocean                → water
    • evergreen needleleaf forest, pine forest → evergreen_needleleaf_forest
    • evergreen broadleaf forest               → evergreen_broadleaf_forest
    • deciduous needleleaf forest              → deciduous_needleleaf_forest
    • deciduous broadleaf forest               → deciduous_broadleaf_forest
    • mixed forest, mixed woods                → mixed_forest
    • closed shrublands, dense shrubs          → closed_shrublands
    • open shrublands, sparse shrubs           → open_shrublands
    • woody savannas                           → woody_savannas
    • savannas, grassy savanna                 → savannas
    • grass, grassland                         → grasslands
    • permanent wetlands, marsh, swamp         → permanent_wetlands
    • cropland, agriculture                    → croplands
    • cropland‑natural mosaic                  → cropland_natural_vegetation_mosaic
    • urban, built‑up, artificial surfaces     → urban_and_built_up
    • snow, ice, glacier                       → snow_and_ice
    • barren, bare ground, rock                → barren_or_sparsely_vegetated

Available layers or predictors to fetch: bio1–bio19, elevation, slope, aspect, ndvi, landcover

 **Example**  
 User: fetch urban, bio1  
 Assistant:  
 ```json
 {"tool":"fetch","layers":["bio1"],"landcover":["urban_and_built_up"]}
 ```

SpatChat SDM Pipeline: Methods Summary

Data Ingestion & Coordinate Handling
- Users upload a CSV of presence points.
- Column names are auto‑detected (aliases, fuzzy match, numeric heuristics) and renamed to latitude/longitude.
- All coordinates are reprojected to EPSG:4326 if needed.
- If the user does not yet have occurrence points, recommend obtaining them from public repositories such as GBIF (https://www.gbif.org), iNaturalist (https://www.inaturalist.org), or OBIS (https://obis.org).

Predictor Selection & Fetching
- Available layers:
  • BIOCLIM variables bio1–bio19 (WorldClim V1 at 1 km)
  • Elevation, Slope, Aspect (SRTM at 30 m)
  • NDVI (MODIS MCD13Q1 mean 2022–2024 at 250 m)
  • Landcover (MODIS MCD12Q1 LC_Type1 at 500 m, one‑hot–encoded subclasses)
- Fetch workflow via Earth Engine:
  1. Clip to study‐area bbox (points’ min/max ± 0.25°).
  2. Export at native resolution in EPSG:4326.
  3. Reproject & resample to a uniform 30 m lat/lon grid with rasterio.warp.reproject.

Data Preparation & Sampling
- Presence samples: extract pixel values at each presence point.
- Background samples: randomly draw 5× as many background pixels from the valid grid.
- Stack predictors into an (n_pixels×n_layers) array and build X, y matrices.

Spatial Block Cross‑Validation
- Cluster presence coordinates into n_blocks=5 via KMeans.
- Use GroupKFold(n_splits=5, groups=blocks) to ensure each fold holds out one spatial block of presences.
- For each fold:
  • Sample background separately for training and test.
  • Fit LogisticRegression(max_iter=1000).
  • Compute AUC, threshold (max TSS), Sensitivity, Specificity, TSS, and Kappa.
- Report mean ± std of AUC, TSS, Kappa across folds.

Final Model Fit & Metrics
- Retrain LogisticRegression on all cleaned data (Xc, yc).
- Select threshold maximizing TSS.
- Compute final AUC, TSS, Kappa and save to outputs/performance_metrics.csv.
- Extract and save intercept + coefficients per predictor to outputs/coefficients.csv.

Suitability Map Generation
- Predict probabilities across the full grid.
- Write GeoTIFF outputs/suitability_map_wgs84.tif.

Interactive App & LLM Integration
- Gradio routes commands (fetch, run_model, download, query) via JSON.
- Custom Python handlers answer layer counts, point counts, map stats, and model stats.
- All other questions go to the LLM, primed with this methods summary (and optionally your scripts) so it can answer implementation‑level queries accurately.

Try to keep your answers short—no more than two sentences if possible—while still being helpful.
Guide the user to next steps: upload data, fetch layers, run model, etc.

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

def run_fetch(sl, lc):
    # 1) Build the list of requested predictors
    layers = list(sl)
    if lc:
        layers.append("landcover")

    # 2) Require at least one predictor
    if not layers:
        return create_map(), "⚠️ Please select at least one predictor."

    # 3) Validate top‑level predictors
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

        # 3b) LLM fallback for top‑level names
        prompt = (
            f"You requested these predictors: {', '.join(layers)}. "
            f"I don't recognize: {', '.join(bad_layers)}. "
            "Could you please clarify which predictors you want?"
        )
        clar = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "system", "content": FALLBACK_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.7
        ).choices[0].message.content
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

        # 4b) LLM fallback for landcover codes
        prompt = (
            f"You requested landcover classes: {', '.join(lc)}. "
            f"I don't recognize: {', '.join(bad_codes)}. "
            "Could you please clarify which landcover classes you want?"
        )
        clar = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "system", "content": FALLBACK_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.7
        ).choices[0].message.content
        return create_map(), clar

    # 5) All inputs valid → proceed with fetch
    # —————————————————————————————————————————————
    #  a) Echo what we're about to do
    print(f"🧪 [run_fetch] SL={sl!r}   LC={lc!r}", file=sys.stdout)

    #  b) Set the SAME python executable and unbuffered mode
    os.environ["SELECTED_LAYERS"] = ",".join(sl)

    # map snake_case landcover names -> numeric codes
    modis_landcover_map = {
        0:"water", 1:"evergreen_needleleaf_forest", 2:"evergreen_broadleaf_forest",
        3:"deciduous_needleleaf_forest",4:"deciduous_broadleaf_forest",5:"mixed_forest",
        6:"closed_shrublands",7:"open_shrublands",8:"woody_savannas",9:"savannas",
        10:"grasslands",11:"permanent_wetlands",12:"croplands",
        13:"urban_and_built_up",14:"cropland_natural_vegetation_mosaic",
        15:"snow_and_ice",16:"barren_or_sparsely_vegetated"
    }
    # reverse lookup
    name_to_code = {v:str(k) for k,v in modis_landcover_map.items()}
    codes = [ name_to_code[c] for c in lc if c in name_to_code ]
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(codes)
    cmd = [
        sys.executable,
        "-u",  # unbuffered: so stdout appears as it’s printed
        os.path.join("scripts", "fetch_predictors.py")
    ]

    #  c) Run and capture *both* stdout and stderr
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # capture everything
    logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return create_map(), f"❌ Fetch failed:\n```\n{logs}\n```"
    else:
        # show you the full export log on success
        return create_map(), f"✅ Predictors fetched.\n\n```bash\n{logs}\n```"

    # d) Success
    return create_map(), "✅ Predictors fetched."

def run_model():
    proc = subprocess.run(["python","scripts/run_logistic_sdm.py"], capture_output=True, text=True)
    if proc.returncode!=0:
        return create_map(), f"❌ Model run failed:\n{proc.stderr}", None, None
    perf_df = pd.read_csv("outputs/performance_metrics.csv")
    coef_df = pd.read_csv("outputs/coefficients.csv")
    zip_results()
    return create_map(), "✅ Model ran successfully! Results are ready below.", perf_df, coef_df
    
def chat_step(file, user_msg, history, state):
    # —————————————————————————
    # 0) Layer‑only shortcut (must run before LLM)
    # —————————————————————————
    # tokenise on commas and spaces
    tokens = [t.strip().lower() for t in re.split(r"[,\s]+", user_msg) if t.strip()]
    # filter out connector words
    core = [t for t in tokens if t not in {"and","with","plus","&"}]
    # if *all* core tokens are either valid top‑levels or landcover classes:
    if core and all((t in VALID_LAYERS) or (t in LANDCOVER_CLASSES) for t in core):
        top = [t for t in core if t in VALID_LAYERS and t != "landcover"]
        lc  = [t for t in core if t in LANDCOVER_CLASSES]
        if lc:
            top.append("landcover")
        m_out, status = run_fetch(top, lc)
        txt = f"{status}\n\n 🎉Nice work! Just say “run model” or grab more layers."
        history.extend([
            {"role":"user",     "content": user_msg},
            {"role":"assistant","content": txt}
        ])
        return history, m_out, state
        
    # 1) No CSV yet? delegate to fallback LLM  
    if not os.path.exists("inputs/presence_points.csv"):
        fb = [{"role":"system","content":FALLBACK_PROMPT}, {"role":"user","content":user_msg}]
        reply = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", messages=fb, temperature=0.7).choices[0].message.content
        history.extend([{"role":"user","content":user_msg}, {"role":"assistant","content":reply}])
        return history, create_map(), state
    
    # 2) Reset?
    if re.search(r"\b(start over|restart|clear everything|reset|clear all)\b", user_msg, re.I):
        clear_all()
        # just reset history + map + state
        history = [{"role":"assistant","content":"👋 All cleared! Please upload your presence-points CSV to begin."}]
        return history, create_map(), state
        
    # 3) Tool call via LLM → JSON
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":user_msg}]
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=msgs,
        temperature=0.0
    ).choices[0].message.content

    try:
        call = json.loads(response)
        tool = call["tool"]
    except:
        history.append({"role":"assistant","content":
            "Sorry, I couldn't understand that. Please say 'fetch …', 'run model', or 'download'."})
        return history, create_map(), state

    # 4) Invoke the tool
    if tool == "fetch":
        m_out, status = run_fetch(call.get("layers",[]), call.get("landcover",[]))
        txt = f"{status}\n\nGreat! Now say “run model” or fetch more layers."

    elif tool == "run_model":
        m_out, status, perf_df, coef_df = run_model()
        if perf_df is None:
            txt = status
        else:
            # same performance+coef rendering as before
            status += " You can download the suitability map and all rasters using the 📥 button below the map."
            perf = pd.read_csv("outputs/performance_metrics.csv")
            first, second = perf.iloc[:,:3], perf.iloc[:,3:]
            perf_md = (
               "**Model Performance (1 of 2):**\n\n"
               f"{first.to_markdown(index=False)}\n\n"
               "**Model Performance (2 of 2):**\n\n"
               f"{second.to_markdown(index=False)}"
            )
            coefs = pd.read_csv("outputs/coefficients.csv").dropna(axis=1, how="all")
            txt = (
                f"{status}\n\n"
                f"{perf_md}\n\n"
                f"**Predictor Coefficients:**\n\n{coefs.to_markdown(index=False)}"
            )

    else:  # download
        m_out, _ = create_map(), zip_results()
        txt = "✅ ZIP is downloading…"

    history.extend([
        {"role":"user","content":user_msg},
        {"role":"assistant","content":txt}
    ])
    return history, m_out, state


# --- Upload callback ---
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
                "✅ Sweet! I found your `latitude` and `longitude` columns.\n"
                "You can now pick from these predictors:\n"
                "• bio1–bio19\n"
                "• elevation\n"
                "• slope\n"
                "• aspect\n"
                "• NDVI\n"
                "• landcover (e.g. water, urban, cropland, etc.)\n\n"
                "Feel free to ask me what each Bio1–Bio19 variable represents or which landcover classes you can use.\n"
                "When you’re ready, just say **'I want elevation, ndvi, bio1'** to grab those layers."
            )})
            return history2, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            history2.append({"role":"assistant","content":"I couldn't detect coordinate columns. Please select them and enter CRS below."})
            cols = list(df.columns)
            return history2, create_map(), state, gr.update(choices=cols, visible=True), gr.update(choices=cols, visible=True), gr.update(visible=True), gr.update(visible=True)
    return history2, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

# --- CRS confirm callback ---
def confirm_coords(lat_col, lon_col, crs_raw, history, state):
    df = pd.read_csv("inputs/presence_points.csv")
    try:
        src_epsg = resolve_crs(crs_raw) if crs_raw else 4326
    except:
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
            "Now you can pick from these predictors:\n"
            "• bio1–bio19\n"
            "• elevation\n"
            "• slope\n"
            "• aspect\n"
            "• NDVI\n"
            "• landcover (e.g. water, urban, cropland, etc.)\n\n"
            "Feel free to ask me what each Bio1–Bio19 variable represents or which landcover classes you can use.\n"
            "When you’re ready, just say **'I want elevation, ndvi, bio1'** to grab those layers."
        )
    })
    return history, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

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
            download_btn = gr.DownloadButton("📥 Download Results", zip_results)
        with gr.Column(scale=1):
            chat = gr.Chatbot(value=[{"role":"assistant","content":"👋 Hello, I'm SpatChat, your SDM assistant! I'm here to help you build your species distribution model. Please upload your presence CSV to begin."}], type="messages", label="💬 Chat", height=400)
            user_in = gr.Textbox(label="Ask SpatChat", placeholder="Type commands…")
            file_input = gr.File(label="📄 Upload Presence CSV", type="filepath")
            lat_dropdown = gr.Dropdown(choices=[], label="Latitude column", visible=False)
            lon_dropdown = gr.Dropdown(choices=[], label="Longitude column", visible=False)
            crs_input = gr.Textbox(label="Input CRS (code, zone, or name)", placeholder="e.g. 32610, UTM zone 10N, LCC…", visible=False)
            confirm_btn = gr.Button("Confirm Coordinates", visible=False)
    file_input.change(on_upload, inputs=[file_input, chat, state], outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn])
    confirm_btn.click(confirm_coords, inputs=[lat_dropdown, lon_dropdown, crs_input, chat, state], outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn])
    user_in.submit(chat_step, inputs=[file_input, user_in, chat, state], outputs=[chat, map_out, state])
    user_in.submit(lambda: "", None, user_in)
    demo.launch()
