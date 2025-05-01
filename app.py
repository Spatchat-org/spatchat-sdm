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

# --- Landcover Labels ---
landcover_options = {
    0: "water",
    1: "evergreen needleleaf forest",
    2: "evergreen broadleaf forest",
    3: "deciduous needleleaf forest",
    4: "deciduous broadleaf forest",
    5: "mixed forest",
    6: "closed shrublands",
    7: "open shrublands",
    8: "woody savannas",
    9: "savannas",
    10: "grasslands",
    11: "permanent wetlands",
    12: "croplands",
    13: "urban and built up",
    14: "cropland/natural vegetation mosaic",
    15: "snow and ice",
    16: "barren or sparsely vegetated"
}
landcover_choices = [f"{k} – {v}" for k, v in landcover_options.items()]

def create_map():
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer('OpenStreetMap').add_to(m)

    # --- Presence points ---
    pp = "inputs/presence_points.csv"
    if os.path.exists(pp):
        df = pd.read_csv(pp)
        if {'latitude','longitude'}.issubset(df.columns):
            pts = folium.FeatureGroup(name="🟦 Presence Points")
            for lat, lon in df[['latitude','longitude']].values:
                folium.CircleMarker(
                    [lat, lon],
                    radius=4, color='blue',
                    fill=True, fill_opacity=0.8
                ).add_to(pts)
            pts.add_to(m)
            m.fit_bounds(df[['latitude','longitude']].values.tolist())

    # --- Predictor rasters ---
    rasters_dir = "predictor_rasters/wgs84"
    if os.path.isdir(rasters_dir):
        for fname in sorted(os.listdir(rasters_dir)):
            if not fname.endswith('.tif'): continue
            path = os.path.join(rasters_dir, fname)
            with rasterio.open(path) as src:
                img = src.read(1)
                bounds = src.bounds
            vmin, vmax = np.nanmin(img), np.nanmax(img)
            if np.isclose(vmin, vmax): continue
            norm = (img - vmin) / (vmax - vmin)
            cmap = colormaps['viridis']
            rgba = cmap(norm)
            folium.raster_layers.ImageOverlay(
                image=rgba,
                bounds=[[bounds.bottom, bounds.left],
                        [bounds.top,    bounds.right]],
                opacity=1.0,
                name=f"{fname} ({vmin:.2f}–{vmax:.2f})"
            ).add_to(m)

    # --- Suitability map ---
    suit = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(suit):
        with rasterio.open(suit) as src:
            img = src.read(1)
            bounds = src.bounds
        vmin, vmax = np.nanmin(img), np.nanmax(img)
        norm = (img - vmin) / (vmax - vmin)
        cmap = colormaps['viridis']
        rgba = cmap(norm)
        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[bounds.bottom, bounds.left],
                    [bounds.top,    bounds.right]],
            opacity=0.7,
            name=f"🎯 Suitability ({vmin:.2f}–{vmax:.2f})"
        ).add_to(m)
        m.fit_bounds([[bounds.bottom, bounds.left],
                      [bounds.top,    bounds.right]])

    # --- Layer control top-right ---
    folium.LayerControl(collapsed=False, position='topright').add_to(m)

    # --- Unified low→high legend at bottom-right ---
    low_hex  = to_hex(colormaps['viridis'](0.0))
    high_hex = to_hex(colormaps['viridis'](1.0))
    ramp = bcm.LinearColormap(
        [low_hex, high_hex],
        vmin=0, vmax=1,
        caption="Normalized (low → high)"
    ).to_step(2)
    ramp.position = 'bottomright'
    ramp.width    = 150   # narrow
    ramp.height   = 150  # shorter
    ramp.add_to(m)

    html = html_lib.escape(m.get_root().render())
    return f"<iframe srcdoc=\"{html}\" style=\"width:100%; height:600px; border:none;\"></iframe>"

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# SpatChat SDM – Species Distribution Modeling App")

    with gr.Row():
        with gr.Column(scale=1):
            upload_input       = gr.File(label="📄 Upload Presence CSV", file_types=['.csv'])
            layer_selector     = gr.CheckboxGroup(
                choices=[*[f"bio{i}" for i in range(1,20)], "elevation","slope","aspect","ndvi","landcover"],
                label="🧬 Environmental Layers"
            )
            landcover_selector = gr.CheckboxGroup(
                choices=landcover_choices,
                label="🌿 MODIS Landcover Classes (One-hot)"
            )
            fetch_button       = gr.Button("🌐 Fetch Predictors")
            run_button         = gr.Button("🧠 Run Model")

        with gr.Column(scale=3):
            map_output    = gr.HTML(create_map(), label="🗺️ Map Preview")
            status_output = gr.Textbox(label="Status", interactive=False)

    def handle_upload(file):
        if not file or not hasattr(file, "name"):
            return create_map(), "⚠️ No file uploaded."
        shutil.rmtree("predictor_rasters", ignore_errors=True)
        shutil.rmtree("outputs", ignore_errors=True)
        shutil.rmtree("inputs", ignore_errors=True)
        os.makedirs("inputs", exist_ok=True)
        shutil.copy(file.name, "inputs/presence_points.csv")
        return create_map(), "✅ Presence points uploaded!"

    def run_fetch(selected_layers, selected_landcover):
        if not selected_layers:
            return create_map(), "⚠️ Select at least one predictor."
        os.environ['SELECTED_LAYERS']            = ','.join(selected_layers)
        codes = [c.split(" – ")[0] for c in selected_landcover]
        os.environ['SELECTED_LANDCOVER_CLASSES'] = ','.join(codes)
        res = subprocess.run(
            ["python", "scripts/fetch_predictors.py"],
            capture_output=True, text=True
        )
        print(res.stdout, res.stderr)
        return (create_map(), "✅ Predictors fetched.") if res.returncode==0 else (create_map(), "❌ Fetch failed.")

    def run_model():
        res = subprocess.run(
            ["python", "scripts/run_logistic_sdm.py"],
            capture_output=True, text=True
        )
        print(res.stdout, res.stderr)
        return (create_map(), "✅ Model completed.") if res.returncode==0 else (create_map(), "❌ Model run failed.")

    upload_input.change(fn=handle_upload, inputs=[upload_input], outputs=[map_output, status_output])
    fetch_button.click(fn=run_fetch, inputs=[layer_selector, landcover_selector], outputs=[map_output, status_output])
    run_button.click(fn=run_model, outputs=[map_output, status_output])

    demo.launch()
