import os
import io
import json
import base64
import shutil
import subprocess
import zipfile
import re

import gradio as gr
import geemap.foliumap as foliumap
import folium
import html as html_lib
import pandas as pd
import numpy as np
import rasterio
import ee
import joblib

from matplotlib import pyplot as plt, colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from folium import Element
from together import Together
from dotenv import load_dotenv

# --- Authenticate Earth Engine ---
svc = json.loads(os.environ["GEE_SERVICE_ACCOUNT"])
creds = ee.ServiceAccountCredentials(
    svc["client_email"], key_data=json.dumps(svc)
)
ee.Initialize(creds)

# --- LLM client ---
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# --- Clean up last session ---
for d in ("predictor_rasters", "outputs", "inputs"):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs("inputs", exist_ok=True)

# --- tool registry ---
TOOLS = {
    "fetch": lambda args: run_fetch(args.get("layers",[]), args.get("landcover",[])),
    "run_model": lambda args: run_model()[:2],    # returns (map, status)
    "download": lambda args: (create_map(), zip_results())
}

LAYERS = [f"bio{i}" for i in range(1, 20)] + ["elevation", "slope", "aspect", "ndvi", "landcover"]

# --- Pre-render colorbar ---
fig, ax = plt.subplots(figsize=(4, 0.5))
norm = Normalize(vmin=0, vmax=1)
cbar = plt.colorbar(ScalarMappable(norm=norm, cmap="viridis"),
                    cax=ax, orientation="horizontal")
cbar.set_ticks([])
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

# --- System prompt for the LLM ---
SYSTEM_PROMPT = """
You are SpatChat, a friendly assistant orchestrating SDM.
Your job is to explain to the user what options they have in each step, 
guiding them through the whole process to build the SDM.
Whenever the user
wants to perform an action, reply _only_ with a JSON object selecting one of
your tools:
- To fetch layers:     {"tool":"fetch",     "layers":["bio1","ndvi",...]}
- To run the model:    {"tool":"run_model"}
- To download results: {"tool":"download"}
After we run that function in Python, we'll show its output back to the user,
and then continue the conversation, 
and provide a prompt to the user after each action to guide them on what they should do next.
If the use ask for statistical results (e.g., show stats), then show them the results from stats_df.
Be conversational and helpful, but keep the conversation brief.
If the question is vague, ask the user to clarify.
""".strip()

def create_map():
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)

    # Presence points (auto-detect lat/lon columns)
    ppath = "inputs/presence_points.csv"
    if os.path.exists(ppath):
        df = pd.read_csv(ppath)
        lat_col = next((c for c in df.columns if c.lower() in ("latitude", "decimallatitude", "y")), None)
        lon_col = next((c for c in df.columns if c.lower() in ("longitude", "decimallongitude", "x")), None)
        if lat_col and lon_col:
            pts = df[[lat_col, lon_col]].values.tolist()
            fg = folium.FeatureGroup(name="🟦 Presence Points")
            for lat, lon in pts:
                folium.CircleMarker([lat, lon], radius=4,
                                    color="blue", fill=True, fill_opacity=0.8
                                    ).add_to(fg)
            fg.add_to(m)
            if pts:
                m.fit_bounds(pts)

    # Predictor rasters (Viridis)
    rasdir = "predictor_rasters/wgs84"
    if os.path.isdir(rasdir):
        for fn in sorted(os.listdir(r
