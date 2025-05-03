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
    if os.path.exists("spatchat_results.zip"):
        os.remove("spatchat_results.zip")

clear_all()

# --- Layers & Tools ---
LAYERS = [f"bio{i}" for i in range(1, 20)] + ["elevation", "slope", "aspect", "ndvi", "landcover"]
TOOLS = {
    "fetch": lambda args: run_fetch(args.get("layers", []), args.get("landcover", [])),
    "run_model": lambda args: run_model()[:2],
    "download": lambda args: (create_map(), zip_results())
}

# --- Pre-render colorbar ---
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

# --- Landcover choices ---
landcover_options = {
    0: "water", 1: "evergreen needleleaf forest", 2: "evergreen broadleaf forest",
    3: "deciduous needleleaf forest", 4: "deciduous broadleaf forest", 5: "mixed forest",
    6: "closed shrublands", 7: "open shrublands", 8: "woody savannas", 9: "savannas",
    10: "grasslands", 11: "permanent wetlands", 12: "croplands",
    13: "urban and built up", 14: "cropland/natural vegetation mosaic",
    15: "snow and ice", 16: "barren or sparsely vegetated"
}
landcover_choices = [f"{k} – {v}" for k, v in landcover_options.items()]

# --- Detection helpers ---
def detect_coords(df, fuzz_threshold=80):
    cols = list(df.columns)
    low = [c.lower() for c in cols]
    lat_match = difflib.get_close_matches("latitude", low, n=1, cutoff=fuzz_threshold/100)
    lon_match = difflib.get_close_matches("longitude", low, n=1, cutoff=fuzz_threshold/100)
    if lat_match and lon_match:
        return cols[low.index(lat_match[0])], cols[low.index(lon_match[0])]
    num = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    lat_opts = [c for c in num if df[c].between(-90, 90).mean() > 0.98]
    lon_opts = [c for c in num if df[c].between(-180, 180).mean() > 0.98]
    if len(lat_opts) == 1 and len(lon_opts) == 1:
        return lat_opts[0], lon_opts[0]
    return None, None

# --- CRS parsing helpers ---
def parse_epsg_code(s):
    m = re.match(r"^(\d{4,5})$", s.strip())
    return int(m.group(1)) if m else None

def parse_utm_crs(s):
    m = re.search(r"utm\s*zone\s*(\d+)\s*([NS])", s, re.I)
    if m:
        zone, hemi = int(m.group(1)), m.group(2).upper()
        base = 32600 if hemi == 'N' else 32700
        return base + zone
    return None

def llm_parse_crs(raw):
    system = {
        "role": "system",
        "content": (
            "You're a GIS expert. Given a CRS description, respond with only JSON {\"epsg\": ###} "
            "or {\"epsg\": null}."
        )
    }
    user = {"role": "user", "content": f"CRS: '{raw}'"}
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[system, user],
        temperature=0.0
    ).choices[0].message.content
    data = json.loads(resp)
    code = data.get("epsg")
    if not code:
        raise ValueError("LLM couldn't parse CRS")
    return code

def resolve_crs(raw):
    for fn in (parse_epsg_code, parse_utm_crs):
        code = fn(raw)
        if code:
            return code
    return llm_parse_crs(raw)

# --- Core app functions ---
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
                    arr = src.read(1); bnd = src.bounds
                vmin, vmax = np.nanmin(arr), np.nanmax(arr)
                if not np.isnan(vmin) and vmin != vmax:
                    rgba = colormaps["viridis"]((arr - vmin) / (vmax - vmin))
                    folium.raster_layers.ImageOverlay(rgba, bounds=[[bnd.bottom, bnd.left], [bnd.top, bnd.right]], opacity=1.0, name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})").add_to(m)
    sf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        with rasterio.open(sf) as src:
            arr = src.read(1); bnd = src.bounds
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        rgba = colormaps["viridis"]((arr - vmin) / (vmax - vmin))
        folium.raster_layers.ImageOverlay(rgba, bounds=[[bnd.bottom, bnd.left], [bnd.top, bnd.right]], opacity=0.7, name="🎯 Suitability").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    img_html = f'<img src="data:image/png;base64,{COLORBAR_BASE64}" style="position:absolute; bottom:20px; right:10px; width:200px; height:30px; z-index:1000;" />'
    m.get_root().html.add_child(Element(img_html))
    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" style="width:100%; height:450px; border:none;"></iframe>'

def zip_results():
    archive = "spatchat_results.zip"
    if os.path.exists(archive): os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters", "outputs"):
            for root, _, files in os.walk(fld):
                for fn in files:
                    zf.write(os.path.join(root, fn), arcname=os.path.relpath(os.path.join(root, fn), "."))
    return archive

def run_fetch(sl, lc):
    if not sl and not lc:
        return create_map(), "⚠️ Select at least one predictor."
    layers = list(sl)
    if lc: layers.append("landcover")
    os.environ["SELECTED_LAYERS"] = ",".join(layers)
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(c.split(" – ")[0] for c in lc)
    proc = subprocess.run(["python", "scripts/fetch_predictors.py"], capture_output=True, text=True)
    return create_map(), ("✅ Predictors fetched." if proc.returncode == 0 else f"❌ Fetch failed:\n{proc.stderr}")

def run_model():
    proc = subprocess.run(["python", "scripts/run_logistic_sdm.py"], capture_output=True, text=True)
    if proc.returncode != 0:
        return create_map(), f"❌ Model run failed:\n{proc.stderr}", None, None
    stats_df = pd.read_csv("outputs/model_stats.csv")
    zip_results()
    return create_map(), "✅ Model ran successfully! Results are ready below.", stats_df, "outputs/model_stats.csv"

def chat_step(file, user_msg, history, state):
    if not os.path.exists("inputs/presence_points.csv"):
        fb = [{"role":"system","content":FALLBACK_PROMPT}, {"role":"user","content":user_msg}]
        reply = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", messages=fb, temperature=0.7).choices[0].message.content
        history.extend([{"role":"user","content":user_msg}, {"role":"assistant","content":reply}])
        return history, create_map(), state
    if re.search(r"\b(start over|restart|clear everything|reset)\b", user_msg, re.I):
        clear_all()
        return [{"role":"assistant","content":"👋 All cleared! Please upload your presence-points CSV."}], create_map(), state
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":user_msg}]
    resp = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", messages=msgs, temperature=0.0).choices[0].message.content
    try:
        call = json.loads(resp)
        tool = call.get("tool")
    except:
        tool = None
    if tool == "fetch":
        m, status = run_fetch(call.get("layers", []), call.get("landcover", []))
        txt = f"{status}\n\nGreat! Now run the model or fetch more layers."
    elif tool == "run_model":
        m, status, stats_df, _ = run_model()
        stats_md = stats_df.to_markdown(index=False)
        txt = f"{status}\n\n**Model Statistics:**\n{stats_md}"  
    elif tool == "download":
        m, _ = create_map(), zip_results()
        txt = "✅ ZIP is downloading…"
    else:
        fb = [{"role":"system","content":FALLBACK_PROMPT}, {"role":"user","content":user_msg}]
        txt = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", messages=fb, temperature=0.7).choices[0].message.content
        m = create_map()
    history.extend([{"role":"user","content":user_msg}, {"role":"assistant","content":txt}])
    return history, m, state

def on_upload(f, history):
    history2 = history.copy()
    clear_all()
    if f and hasattr(f, "name"):
        shutil.copy(f.name, "inputs/presence_points.csv")
        df = pd.read_csv("inputs/presence_points.csv")
        lat, lon = detect_coords(df)
        if lat and lon:
            history2.append({"role":"assistant","content":(
                """✅ Sweet! I found your `latitude` and `longitude` columns.
You can now pick from these predictors:
• bio1–bio19
• elevation
• slope
• aspect
• NDVI
• landcover (e.g. water, urban)

When you’re ready, just say something like **"fetch elevation, ndvi, bio1"** to grab those layers."""
            )})
            return history2, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
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
    except Exception:
        history.append({"role":"assistant","content":"Sorry, I couldn't recognize that CRS. Could you try another format?"})
        return history, create_map(), state, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    src_crs = RioCRS.from_epsg(src_epsg)
    dst_crs = RioCRS.from_epsg(4326)
    lon_vals, lat_vals = rio_transform(src_crs, dst_crs, df[lon_col].tolist(), df[lat_col].tolist())
    df['latitude'], df['longitude'] = lat_vals, lon_vals
    df.to_csv("inputs/presence_points.csv", index=False)
    history.append({"role":"assistant","content":"✅ Coordinates set! Now you can fetch layers."})
    return history, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

# --- Gradio UI ---
with gr.Blocks() as demo:
    state = gr.State({"stage": "await_upload"})
    with gr.Row():
        with gr.Column(scale=1):
            map_out = gr.HTML(create_map(), label="🗺️ Map Preview")
            download_btn = gr.DownloadButton("📥 Download Results", zip_results)
        with gr.Column(scale=1):
            chat = gr.Chatbot(value=[{"role": "assistant", "content": "👋 Hello, I'm SpatChat! Upload your presence CSV to begin."}], type="messages", label="💬 Chat", height=400)
            user_in = gr.Textbox(label="Ask SpatChat", placeholder="Type commands…")
            file_input = gr.File(label="📄 Upload Presence CSV", type="filepath")
            lat_dropdown = gr.Dropdown(choices=[], label="Latitude column", visible=False)
            lon_dropdown = gr.Dropdown(choices=[], label="Longitude column", visible=False)
            crs_input = gr.Textbox(label="Input CRS (code, zone, or name)", placeholder="e.g. 32610, UTM zone 10N, LCC…", visible=False)
            confirm_btn = gr.Button("Confirm Coordinates", visible=False)
    file_input.change(on_upload, inputs=[file_input, chat], outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn])
    confirm_btn.click(confirm_coords, inputs=[lat_dropdown, lon_dropdown, crs_input, chat, state], outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn])
    user_in.submit(chat_step, inputs=[file_input, user_in, chat, state], outputs=[chat, map_out, state])
    user_in.submit(lambda: "", None, user_in)
    demo.launch()
