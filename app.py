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
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(buf.read()).decode()

# --- Landcover choices ---
landcover_options = {0:"water",1:"evergreen needleleaf forest",2:"evergreen broadleaf forest",3:"deciduous needleleaf forest",4:"deciduous broadleaf forest",5:"mixed forest",6:"closed shrublands",7:"open shrublands",8:"woody savannas",9:"savannas",10:"grasslands",11:"permanent wetlands",12:"croplands",13:"urban and built up",14:"cropland/natural vegetation mosaic",15:"snow and ice",16:"barren or sparsely vegetated"}
landcover_choices = [f"{k} – {v}" for k, v in landcover_options.items()]

# --- Detection helpers ---
def detect_coords(df, fuzz_threshold=80):
    cols = list(df.columns)
    low = [c.lower() for c in cols]
    lat = difflib.get_close_matches("latitude", low, n=1, cutoff=fuzz_threshold/100)
    lon = difflib.get_close_matches("longitude", low, n=1, cutoff=fuzz_threshold/100)
    if lat and lon:
        return cols[low.index(lat[0])], cols[low.index(lon[0])]
    nums = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    lat_opts = [c for c in nums if df[c].between(-90, 90).mean() > 0.98]
    lon_opts = [c for c in nums if df[c].between(-180, 180).mean() > 0.98]
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
        return (32600 if hemi=='N' else 32700) + zone
    return None

def llm_parse_crs(raw):
    system = {"role":"system","content":"You're a GIS expert. Given a CRS description, respond with only JSON {\"epsg\": ###} or {\"epsg\": null}."}
    user = {"role":"user","content":f"CRS: '{raw}'"}
    resp = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", messages=[system, user], temperature=0.0).choices[0].message.content
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
You are SpatChat, a friendly assistant orchestrating SDM.
Your job is to explain to the user what options they have in each step,
guiding them through the whole process to build the SDM.
Whenever the user wants to perform an action, reply _only_ with a JSON object selecting one of your tools:
- To fetch layers:     {"tool":"fetch","layers":["bio1","ndvi",...]}
- To run the model:    {"tool":"run_model"}
- To download results: {"tool":"download"}
After we run that function, we'll display its output and then prompt the user on next steps.
If the user asks for stats, show them from stats_df.
If the question is vague, ask for clarification.
""".strip()

FALLBACK_PROMPT = """
You are SpatChat, a friendly assistant for species distribution modeling.
Keep your answers short—no more than two sentences—while still being helpful.
Guide the user to next steps: upload data, fetch layers, run model, etc.
""".strip()

# --- Core app functions ---
(definitions of create_map, zip_results, run_fetch, run_model, chat_step unchanged...)

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
                "• landcover (e.g. water, urban)\n\n"
                "When you’re ready, just say **'fetch elevation, ndvi, bio1'** to grab those layers."
            )})
            return history2, create_map(), state,
                   gr.update(visible=False), gr.update(visible=False),
                   gr.update(visible=False), gr.update(visible=False)
        else:
            history2.append({"role":"assistant","content":"I couldn't detect coordinate columns. Please select them and enter CRS below."})
            cols = list(df.columns)
            return history2, create_map(), state,
                   gr.update(choices=cols, visible=True), gr.update(choices=cols, visible=True),
                   gr.update(visible=True), gr.update(visible=True)
    return history2, create_map(), state,
           gr.update(visible=False), gr.update(visible=False),
           gr.update(visible=False), gr.update(visible=False)

# --- Gradio UI ---
with gr.Blocks() as demo:
    state = gr.State({"stage": "await_upload"})
    with gr.Row():
        with gr.Column(scale=1):
            map_out = gr.HTML(create_map(), label="🗺️ Map Preview")
            download_btn = gr.DownloadButton("📥 Download Results", zip_results)
        with gr.Column(scale=1):
            chat = gr.Chatbot(...)
            user_in = gr.Textbox(...)
            file_input = gr.File(...)
            lat_dropdown = gr.Dropdown(...)
            lon_dropdown = gr.Dropdown(...)
            crs_input = gr.Textbox(...)
            confirm_btn = gr.Button(...)
    file_input.change(on_upload, inputs=[file_input, chat, state], outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn])
    confirm_btn.click(confirm_coords, inputs=[lat_dropdown, lon_dropdown, crs_input, chat, state], outputs=[chat, map_out, state, lat_dropdown, lon_dropdown, crs_input, confirm_btn])
    user_in.submit(chat_step, inputs=[file_input, user_in, chat, state], outputs=[chat, map_out, state])
    user_in.submit(lambda: "", None, user_in)
    demo.launch()
