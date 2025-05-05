import os
import io
import json
import base64
import shutil
import subprocess
import zipfile
import re
import difflib

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
    """
    Try to auto‑detect latitude & longitude columns from a wide set of aliases.
    Falls back to fuzzy‑matching “latitude”/“longitude” and numeric heuristics.
    """
    cols = list(df.columns)
    low = [c.lower().strip() for c in cols]

    # 1) Exact alias lists
    LAT_ALIASES = {
        'lat', 'latitude', 'y', 'decilatitude', 'dec_latitude', 'dec lat',
        'decimallatitude', 'decimal latitude'
    }
    LON_ALIASES = {
        'lon', 'long', 'longitude', 'x', 'decilongitude', 'dec_longitude',
        'dec longitude', 'decimallongitude', 'decimal longitude'
    }

    # Try exact alias match first
    lat_idx = next((i for i,name in enumerate(low) if name in LAT_ALIASES), None)
    lon_idx = next((i for i,name in enumerate(low) if name in LON_ALIASES), None)
    if lat_idx is not None and lon_idx is not None:
        return cols[lat_idx], cols[lon_idx]

    # 2) Fuzzy match “latitude” / “longitude”
    lat_fz = difflib.get_close_matches("latitude", low, n=1, cutoff=fuzz_threshold/100)
    lon_fz = difflib.get_close_matches("longitude", low, n=1, cutoff=fuzz_threshold/100)
    if lat_fz and lon_fz:
        return cols[low.index(lat_fz[0])], cols[low.index(lon_fz[0])]

    # 3) Numeric heuristics (fall back)
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
You are SpatChat, a friendly assistant for species distribution modeling.

Available environmental predictors:
• bio1–bio19  
• elevation  
• slope  
• aspect  
• ndvi  
• landcover (choose from:
    water, evergreen_needleleaf_forest, 
    evergreen_broadleaf_forest, deciduous_needleleaf_forest,
    deciduous_broadleaf_forest, mixed_forest,
    closed_shrublands, open_shrublands,
    woody_savannas, savannas, grasslands,
    permanent_wetlands, croplands,
    urban_and_built_up, cropland_natural_vegetation_mosaic,
    snow_and_ice, barren_or_sparsely_vegetated
)

Whenever you need to perform an action, respond ONLY with a JSON object selecting one of these tools:

  • Fetch predictors:     {"tool":"fetch","layers":["bio1","ndvi",...]}  
  • Run the model:        {"tool":"run_model"}  
  • Download results ZIP: {"tool":"download"}  
  • Query data or metadata: {"tool":"query","query":"<your question>"}  

Examples of queries you can make:
  – “What is the maximum suitability value?”  
  – “How many presence points do I have?”  
  – “What do bio1–bio19 stand for?”  
  – “Which landcover classes are available?”  

After we run that function, we'll display its output and then prompt the user on next steps.
If the user asks for statistical results, show them from "outputs/performance_metrics.csv" and "outputs/coefficients.csv".
If the question is vague, ask for clarification.
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
    # 1. Read previous selection (empty string → [])
    prev = os.environ.get("SELECTED_LAYERS", "")
    prev_layers = set(prev.split(",")) if prev else set()

    # 2. Add the new ones
    new_layers = set(sl)
    if lc:
        new_layers.add("landcover")
    all_layers = prev_layers | new_layers

    # 3. Persist back to env
    os.environ["SELECTED_LAYERS"] = ",".join(sorted(all_layers))

    # 4. Call the fetch script
    proc = subprocess.run(
        ["python","scripts/fetch_predictors.py"],
        capture_output=True, text=True
    )
    return (
        create_map(),
        "✅ Predictors fetched." if proc.returncode == 0
        else f"❌ Fetch failed:\n{proc.stderr}"
    )


def run_model():
    proc = subprocess.run(["python","scripts/run_logistic_sdm.py"], capture_output=True, text=True)
    if proc.returncode!=0:
        return create_map(), f"❌ Model run failed:\n{proc.stderr}", None, None
    perf_df = pd.read_csv("outputs/performance_metrics.csv")
    coef_df = pd.read_csv("outputs/coefficients.csv")
    zip_results()
    return create_map(), "✅ Model ran successfully! Results are ready below.", perf_df, coef_df

def run_query(query_text: str):
    """
    Handle natural‑language queries about:
    - The suitability map (outputs/suitability_map_wgs84.tif)
    - The uploaded points (inputs/presence_points.csv)
    - The model stats CSVs
    """
    q = query_text.lower()
    # 1) Suitability‐map questions
    if "highest" in q or "maximum" in q:
        with rasterio.open("outputs/suitability_map_wgs84.tif") as src:
            arr = src.read(1, masked=True)
        val = float(arr.max())
        return f"The maximum suitability value is {val:.3f}."
    if "lowest" in q or "minimum" in q:
        with rasterio.open("outputs/suitability_map_wgs84.tif") as src:
            arr = src.read(1, masked=True)
        val = float(arr.min())
        return f"The minimum suitability value is {val:.3f}."
    # 2) Presence‐points questions
    if "how many points" in q or "number of points" in q:
        df = pd.read_csv("inputs/presence_points.csv")
        return f"There are {len(df)} presence points."
    # 3) Performance‐metrics questions
    if "auc" in q or "tss" in q or "kappa" in q:
        perf = pd.read_csv("outputs/performance_metrics.csv")
        lines = [f"{col}: {perf.at[0,col]:.3f}" for col in perf.columns]
        return "Model performance:\n" + "\n".join(lines)
    # fallback
    return "Sorry, I don't know how to answer that. Try asking about max/min suitability, count of points, or model metrics."

def chat_step(file, user_msg, history, state):
    if not os.path.exists("inputs/presence_points.csv"):
        fb = [{"role":"system","content":FALLBACK_PROMPT}, {"role":"user","content":user_msg}]
        reply = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", messages=fb, temperature=0.7).choices[0].message.content
        history.extend([{"role":"user","content":user_msg}, {"role":"assistant","content":reply}])
        return history, create_map(), state
    if re.search(r"\b(start over|restart|clear everything|reset|clear all)\b", user_msg, re.I):
        clear_all()
        return (
           [{"role":"assistant","content":"👋 All cleared! Please upload your presence-points CSV to begin."}],
           create_map(),
           state,
           gr.update(value=None)        # clear the upload box
        )
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":user_msg}]
    resp = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", messages=msgs, temperature=0.0).choices[0].message.content
    try:
        call = json.loads(resp)
        tool = call.get("tool")
        query_text = call.get("query", "")
    except:
        tool = None
    if tool == "fetch":
        m_out, status = run_fetch(call.get("layers", []), call.get("landcover", []))
        txt = f"{status}\n\nGreat! Now run the model or fetch more layers."
    elif tool == "run_model":
        m_out, status, perf_df, coef_df = run_model()

        # Only convert to markdown if the run succeeded:
        if perf_df is None:
            # The subprocess failed; just show the stderr message
            txt = status
        else:
            # Success: show performance and coefficients
            status += " You can download the suitability map and all rasters using the 📥 button below the map."
            perf_md = perf_df.to_markdown(index=False)
            coef_df = coef_df.dropna(axis=1, how='all')
            coef_md = coef_df.to_markdown(index=False)
            txt = (
                f"{status}\n\n"
                f"**Model Performance:**\n\n{perf_md}\n\n"
                f"**Predictor Coefficients:**\n\n{coef_md}"
            )

    elif tool == "download":
        m_out, _ = create_map(), zip_results()
        txt = "✅ ZIP is downloading…"
    
    elif tool == "query":
        m_out = create_map()
        # call our new query helper
        txt = run_query(call.get("query",""))
    else:
        fb = [{"role":"system","content":FALLBACK_PROMPT}, {"role":"user","content":user_msg}]
        txt = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", messages=fb, temperature=0.7).choices[0].message.content
        m_out = create_map()
    history.extend([{"role":"user","content":user_msg}, {"role":"assistant","content":txt}])
    return history, m_out, state, file

# --- Upload callback ---
def on_upload(f, history, state):
    history2 = history.copy()
    clear_all()
    if f and hasattr(f, "name"):
        shutil.copy(f.name, "inputs/presence_points.csv")
        df = pd.read_csv("inputs/presence_points.csv")
        lat, lon = detect_coords(df)
        if lat and lon:
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
    gr.Markdown("## Spatchat: Species Distribution Model {sdm}")
    gr.HTML("""
    <div style="margin-top: -10px; margin-bottom: 15px;">
      <input type="text" value="hhttps://spatchat.org/browse/?room=sdm" id="shareLink" readonly style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
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
    user_in.submit(chat_step, inputs=[file_input, user_in, chat, state], outputs=[chat, map_out, state, file_input])
    user_in.submit(lambda: "", None, user_in)
    demo.launch()
