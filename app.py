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
from pyproj import CRS, Transformer, database

# --- Authenticate Earth Engine ---
svc = json.loads(os.environ["GEE_SERVICE_ACCOUNT"])
creds = ee.ServiceAccountCredentials(
    svc["client_email"], key_data=json.dumps(svc)
)
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

# clear on startup
clear_all()

# --- Layers & Tools definitions ---
LAYERS = [f"bio{i}" for i in range(1, 20)] + ["elevation","slope","aspect","ndvi","landcover"]
TOOLS = {
    "fetch": lambda args: run_fetch(args.get("layers",[]), args.get("landcover",[])),
    "run_model": lambda args: run_model()[:2],
    "download": lambda args: (create_map(), zip_results())
}

# --- Pre-render colorbar → base64 ---
fig, ax = plt.subplots(figsize=(4,0.5))
norm = Normalize(vmin=0, vmax=1)
plt.colorbar(ScalarMappable(norm=norm,cmap="viridis"), cax=ax, orientation="horizontal").set_ticks([])
ax.set_xlabel("Low    High")
fig.tight_layout(pad=0)
buf = io.BytesIO(); fig.savefig(buf,format="png",dpi=100); plt.close(fig)
buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(buf.read()).decode()

# --- Landcover choices ---
landcover_options = {
    0:"water",1:"evergreen needleleaf forest",2:"evergreen broadleaf forest",
    3:"deciduous needleleaf forest",4:"deciduous broadleaf forest",5:"mixed forest",
    6:"closed shrublands",7:"open shrublands",8:"woody savannas",9:"savannas",
    10:"grasslands",11:"permanent wetlands",12:"croplands",
    13:"urban and built up",14:"cropland/natural vegetation mosaic",
    15:"snow and ice",16:"barren or sparsely vegetated"
}
landcover_choices = [f"{k} – {v}" for k,v in landcover_options.items()]

# --- Helpers for detection & CRS ---
def detect_coords(df, fuzz_threshold=80):
    cols = list(df.columns)
    low = [c.lower() for c in cols]
    # 1) fuzzy name match using difflib
    lat_matches = difflib.get_close_matches("latitude", low, n=1, cutoff=fuzz_threshold/100)
    lon_matches = difflib.get_close_matches("longitude", low, n=1, cutoff=fuzz_threshold/100)
    if lat_matches and lon_matches:
        return cols[low.index(lat_matches[0])], cols[low.index(lon_matches[0])]
    # 2) numeric range
    num = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    lat_opts = [c for c in num if df[c].between(-90,90).mean()>0.98]
    lon_opts = [c for c in num if df[c].between(-180,180).mean()>0.98]
    if len(lat_opts)==1 and len(lon_opts)==1:
        return lat_opts[0], lon_opts[0]
    return None, None

# parse EPSG patterns
def parse_epsg_code(s):
    m = re.match(r"^(\d{4,5})$", s.strip())
    return int(m.group(1)) if m else None

def parse_utm_crs(s):
    m = re.search(r"utm\s*zone\s*(\d+)\s*([NS])", s, re.I)
    if m:
        zone, hemi = int(m.group(1)), m.group(2).upper()
        base = 32600 if hemi=='N' else 32700
        return base+zone
    return None

# lookup named CRS
def lookup_named_crs(s):
    infos = database.query_crs_info("name", s, case_sensitive=False)
    return int(infos[0].code) if infos else None

# LLM fallback
def llm_parse_crs(raw):
    system = {"role":"system","content":"You're a GIS expert. Given a CRS description, respond with only JSON {\"epsg\":###} or {\"epsg\":null}."}
    user = {"role":"user","content":f"CRS: '{raw}'"}
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[system,user],
        temperature=0.0
    ).choices[0].message.content
    data = json.loads(resp)
    if not data.get("epsg"): raise ValueError("LLM couldn't parse CRS")
    return data["epsg"]

# resolve to CRS
def resolve_crs(raw):
    for fn in (parse_epsg_code, parse_utm_crs, lookup_named_crs):
        code = fn(raw)
        if code: return CRS.from_epsg(code)
    # fallback to LLM
    return CRS.from_epsg(llm_parse_crs(raw))

# --- Core app funcs (unchanged) ---
# ... (create_map, zip_results, run_fetch, run_model, chat_step stay the same)

# --- New confirm_coords callback ---
def confirm_coords(lat_col, lon_col, crs_raw, history, state):
    df = pd.read_csv("inputs/presence_points.csv")
    try:
        src_crs = resolve_crs(crs_raw) if crs_raw else CRS.from_epsg(4326)
    except Exception:
        history.append({"role":"assistant","content":"Sorry, I couldn't recognize that CRS. Could you try another format?"})
        return history, create_map(), state, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
    lon_vals, lat_vals = transformer.transform(df[lon_col].values, df[lat_col].values)
    df['latitude'], df['longitude'] = lat_vals, lon_vals
    df.to_csv("inputs/presence_points.csv", index=False)
    history.append({"role":"assistant","content":"✅ Coordinates set! Now you can fetch layers (e.g., 'fetch elevation, ndvi')."})
    return history, create_map(), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

# --- Gradio UI ---
with gr.Blocks() as demo:
    # ... logo, markdown, etc.
    state = gr.State({"stage":"await_upload"})
    # main row
    with gr.Row():
        with gr.Column(scale=1):
            map_out      = gr.HTML(create_map(), label="🗺️ Map Preview")
            download_btn = gr.DownloadButton("📥 Download Results", zip_results)
        with gr.Column(scale=1):
            chat         = gr.Chatbot(
                              value=[{"role":"assistant","content":"👋 Hello, I'm SpatChat! Upload your presence CSV to begin."}],
                              type="messages", label="💬 Chat", height=400)
            user_in      = gr.Textbox(label="Ask SpatChat", placeholder="Type commands…")
            file_input   = gr.File(label="📄 Upload Presence CSV", type="filepath")
            lat_dropdown = gr.Dropdown(choices=[], label="Latitude column", visible=False)
            lon_dropdown = gr.Dropdown(choices=[], label="Longitude column", visible=False)
            crs_input    = gr.Textbox(label="Input CRS (code, zone, or name)", placeholder="e.g. 32610, UTM zone 10N, LCC…", visible=False)
            confirm_btn  = gr.Button("Confirm Coordinates", visible=False)

    # upload callback
    file_input.change(
        on_upload,
        inputs=[file_input, chat],
        outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn]
    )
    # confirm coords
    confirm_btn.click(
        confirm_coords,
        inputs=[lat_dropdown, lon_dropdown, crs_input, chat, state],
        outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn]
    )
    # chat
    user_in.submit(chat_step, inputs=[file_input, user_in, chat, state], outputs=[chat, map_out, state])
    user_in.submit(lambda: "", None, user_in)
    demo.launch()
