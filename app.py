import os
import json
import shutil
import subprocess
import zipfile

import gradio as gr
import geemap.foliumap as foliumap
import folium
import html as html_lib
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import ee
import joblib

from matplotlib import colormaps
from matplotlib.colors import to_hex
import branca.colormap as bcm

# --- Earth Engine auth ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)

# --- Clean up last session ---
shutil.rmtree("predictor_rasters", ignore_errors=True)
shutil.rmtree("outputs", ignore_errors=True)
shutil.rmtree("inputs", ignore_errors=True)
os.makedirs("inputs", exist_ok=True)

# --- Landcover labels ---
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
    # Base map
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer('OpenStreetMap').add_to(m)

    # 1) Presence points
    presence_csv = "inputs/presence_points.csv"
    if os.path.exists(presence_csv):
        df = pd.read_csv(presence_csv)
        if {'latitude','longitude'}.issubset(df.columns):
            pts = df[['latitude','longitude']].values.tolist()
            fg = folium.FeatureGroup(name="🟦 Presence Points")
            for lat, lon in pts:
                folium.CircleMarker(
                    [lat, lon], radius=4,
                    color='blue', fill=True, fill_opacity=0.8
                ).add_to(fg)
            fg.add_to(m)
            if pts:
                m.fit_bounds(pts)

    # 2) Predictor rasters
    any_pred = False
    rasters_dir = "predictor_rasters/wgs84"
    if os.path.isdir(rasters_dir):
        for fn in sorted(os.listdir(rasters_dir)):
            if not fn.endswith(".tif"):
                continue
            any_pred = True
            path = os.path.join(rasters_dir, fn)
            with rasterio.open(path) as src:
                img = src.read(1)
                bounds = src.bounds

            vmin, vmax = np.nanmin(img), np.nanmax(img)
            if vmin == vmax or np.isnan(vmin) or np.isnan(vmax):
                continue

            norm = (img - vmin) / (vmax - vmin)
            cmap = colormaps['viridis']
            rgba = cmap(norm)

            folium.raster_layers.ImageOverlay(
                image=rgba,
                bounds=[[bounds.bottom, bounds.left],
                        [bounds.top,    bounds.right]],
                opacity=1.0,
                name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})"
            ).add_to(m)

    # 3) Viridis legend (low→high) if any predictor
    if any_pred:
        vir = colormaps['viridis']
        vir_colors = [to_hex(c) for c in vir(np.linspace(0,1,256))]
        vir_legend = bcm.LinearColormap(
            colors=vir_colors,
            vmin=0, vmax=1,
            caption="Normalized (low → high)"
        )
        vir_legend.add_to(m)

    # 4) Suitability map + Plasma legend
    suit_tif = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(suit_tif):
        with rasterio.open(suit_tif) as src:
            img = src.read(1)
            bounds = src.bounds

        vmin, vmax = np.nanmin(img), np.nanmax(img)
        norm = (img - vmin) / (vmax - vmin)
        cmap = colormaps['plasma']
        rgba = cmap(norm)

        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[bounds.bottom, bounds.left],
                    [bounds.top,    bounds.right]],
            opacity=0.7,
            name=f"🎯 Suitability ({vmin:.2f}–{vmax:.2f})"
        ).add_to(m)

        plasma_colors = [to_hex(c) for c in cmap(np.linspace(0,1,256))]
        suit_legend = bcm.LinearColormap(
            colors=plasma_colors,
            vmin=vmin, vmax=vmax,
            caption="Suitability"
        )
        suit_legend.add_to(m)

    # 5) Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # 6) Render as iframe
    html = html_lib.escape(m.get_root().render())
    return f"<iframe srcdoc=\"{html}\" style=\"width:100%; height:600px; border:none;\"></iframe>"

# --- ZIP helper ---
def zip_results():
    """Bundle predictor_rasters + outputs into results.zip."""
    archive = "results.zip"
    if os.path.exists(archive):
        os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk("predictor_rasters"):
            for fn in files:
                path = os.path.join(root, fn)
                zf.write(path, arcname=os.path.relpath(path, start="."))
        for root, _, files in os.walk("outputs"):
            for fn in files:
                path = os.path.join(root, fn)
                zf.write(path, arcname=os.path.relpath(path, start="."))
    return archive, "✅ Archive ready!"

# --- Build Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# SpatChat SDM – Species Distribution Modeling App")

    with gr.Row():
        with gr.Column(scale=1):
            upload_input       = gr.File(label="📄 Upload Presence CSV", file_types=['.csv'])
            layer_selector     = gr.CheckboxGroup(
                                     choices=[*[f"bio{i}" for i in range(1,20)],
                                              "elevation","slope","aspect","ndvi","landcover"],
                                     label="🧬 Environmental Layers"
                                 )
            landcover_selector = gr.CheckboxGroup(
                                     choices=landcover_choices,
                                     label="🌿 MODIS Landcover Classes (One-hot)"
                                 )
            fetch_button       = gr.Button("🌐 Fetch Predictors")
            run_button         = gr.Button("🧠 Run Model")
            download_button    = gr.Button("📥 Download Results")

        with gr.Column(scale=3):
            map_output       = gr.HTML(value=create_map(), label="🗺️ Map Preview")
            status_output    = gr.Textbox(label="Status", interactive=False)
            download_output  = gr.File(label="Download .zip")

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
        msg = "✅ Predictors fetched." if res.returncode == 0 else "❌ Fetch failed; see logs."
        return create_map(), msg

    def run_model():
        res = subprocess.run(
            ["python", "scripts/run_logistic_sdm.py"],
            capture_output=True, text=True
        )
        print(res.stdout, res.stderr)
        msg = "✅ Model completed." if res.returncode == 0 else "❌ Model run failed."
        return create_map(), msg

    # hook up callbacks
    upload_input.change(fn=handle_upload, inputs=[upload_input], outputs=[map_output, status_output])
    fetch_button.click(fn=run_fetch,    inputs=[layer_selector, landcover_selector], outputs=[map_output, status_output])
    run_button.click(  fn=run_model,    outputs=[map_output, status_output])
    download_button.click(fn=zip_results, inputs=[], outputs=[download_output, status_output])

    demo.launch()
