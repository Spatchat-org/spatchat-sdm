import os
import json
import shutil
import subprocess

import gradio as gr
import geemap.foliumap as foliumap
import folium
import html as html_lib
import pandas as pd
import numpy as np
import rasterio
import ee
import joblib

# new for color‐bar
from matplotlib import colormaps
from matplotlib.colors import to_hex
import branca.colormap as bcm

# --- Authenticate Earth Engine using Service Account ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)
print("✅ Earth Engine authenticated using Service Account!")

# --- Clean up previous session cache ---
shutil.rmtree("predictor_rasters", ignore_errors=True)
shutil.rmtree("outputs", ignore_errors=True)
shutil.rmtree("inputs", ignore_errors=True)
os.makedirs("inputs", exist_ok=True)

# --- Landcover Labels (unchanged) ---
landcover_options = {
    0: "water",
    1: "evergreen needleleaf forest",
    # … etc …
    16: "barren or sparsely vegetated"
}
landcover_choices = [f"{k} – {v}" for k, v in landcover_options.items()]


def create_map():
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer('OpenStreetMap').add_to(m)

    # 1) Draw presence points, fit to bounds
    presence_path = "inputs/presence_points.csv"
    if os.path.exists(presence_path):
        df = pd.read_csv(presence_path)
        if {'latitude','longitude'}.issubset(df.columns):
            latlons = df[['latitude','longitude']].values.tolist()
            fg = folium.FeatureGroup(name="🟦 Presence Points")
            for lat, lon in latlons:
                folium.CircleMarker(
                    [lat, lon],
                    radius=4, color='blue', fill=True, fill_opacity=0.8
                ).add_to(fg)
            fg.add_to(m)
            if latlons:
                m.fit_bounds(latlons)

    # 2) Overlay each predictor raster (normalized 0–1 → viridis)
    rasters_dir = "predictor_rasters/wgs84"
    if os.path.isdir(rasters_dir):
        for fname in sorted(os.listdir(rasters_dir)):
            if not fname.endswith('.tif'):
                continue
            path = os.path.join(rasters_dir, fname)
            with rasterio.open(path) as src:
                img = src.read(1)
                bounds = src.bounds

            vmin, vmax = np.nanmin(img), np.nanmax(img)
            if vmin == vmax or np.isnan(vmin) or np.isnan(vmax):
                continue

            norm = (img - vmin) / (vmax - vmin)
            viridis = colormaps['viridis']
            rgba = viridis(norm)  # (h, w, 4)

            folium.raster_layers.ImageOverlay(
                image=rgba,
                bounds=[[bounds.bottom, bounds.left],
                        [bounds.top,    bounds.right]],
                opacity=1.0,
                name=f"🟨 {fname} ({vmin:.1f}–{vmax:.1f})"
            ).add_to(m)

    # 3) Suitability overlay (same idea, but with plasma)
    suit_path = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(suit_path):
        with rasterio.open(suit_path) as src:
            img = src.read(1)
            bounds = src.bounds

        vmin, vmax = np.nanmin(img), np.nanmax(img)
        norm = (img - vmin) / (vmax - vmin)
        plasma = colormaps['plasma']
        rgba = plasma(norm)

        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[bounds.bottom, bounds.left],
                    [bounds.top,    bounds.right]],
            opacity=0.7,
            name=f"🎯 Suitability ({vmin:.2f}–{vmax:.2f})"
        ).add_to(m)
        # ensure view zooms to suit area:
        m.fit_bounds([[bounds.bottom, bounds.left],
                      [bounds.top,    bounds.right]])

    # 4) Layer control (always)
    folium.LayerControl(collapsed=False, position='topright').add_to(m)

    # 5) ALWAYS add a short vertical legend in the top‐right:
    #    two‐color step legend: purple=low, yellow=high
    #    so users immediately know the ramp direction.
    #    We sample 3 key colors from viridis and make a stepped bar.
    sample = colormaps['viridis'](np.linspace(0, 1, 3))
    hexes  = [to_hex(c) for c in sample]
    legend = bcm.LinearColormap(
        hexes, vmin=0, vmax=1, caption="Normalized (low → high)"
    ).to_step(2)   # two steps: low vs high
    legend.add_to(m)

    # 6) Render and return
    html = html_lib.escape(m.get_root().render())
    return f"<iframe srcdoc=\"{html}\" style=\"width:100%; height:600px; border:none;\"></iframe>"


# --- Build & launch Gradio UI (rest of your code unaffected) ---
with gr.Blocks() as demo:
    gr.Markdown("# SpatChat SDM – Species Distribution Modeling App")
    # … your upload/fetch/run buttons …
    map_output    = gr.HTML(value=create_map(), label="🗺️ Map Preview")
    status_output = gr.Textbox(label="Status", interactive=False)
    # … callbacks, demo.launch() …

