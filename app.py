import os
import json
import shutil
import subprocess

import gradio as gr
import folium
import geemap.foliumap as foliumap
import html as html_lib
import pandas as pd
import numpy as np
import rasterio
import ee
import joblib

from matplotlib import colormaps
from matplotlib.colors import to_hex
import branca.colormap as bcm


# --- Authenticate Earth Engine ---
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
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer('OpenStreetMap').add_to(m)

    # Presence points
    presence_csv = "inputs/presence_points.csv"
    if os.path.exists(presence_csv):
        df = pd.read_csv(presence_csv)
        if {'latitude', 'longitude'}.issubset(df.columns):
            pts = df[['latitude','longitude']].values.tolist()
            fg = folium.FeatureGroup(name="🟦 Presence Points")
            for lat, lon in pts:
                folium.CircleMarker([lat, lon],
                                    radius=4,
                                    color='blue',
                                    fill=True,
                                    fill_opacity=0.8
                                   ).add_to(fg)
            fg.add_to(m)
            if pts:
                m.fit_bounds(pts)

    # Predictor rasters
    any_predictor = False
    rasters_dir = "predictor_rasters/wgs84"
    if os.path.isdir(rasters_dir):
        for fname in sorted(os.listdir(rasters_dir)):
            if not fname.endswith(".tif"):
                continue
            any_predictor = True
            path = os.path.join(rasters_dir, fname)
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
                name=f"🟨 {fname} ({vmin:.2f}–{vmax:.2f})"
            ).add_to(m)

    # Always show Viridis legend when any predictor is present
    if any_predictor:
        vir = colormaps['viridis']
        vir_colors = [ to_hex(c) for c in vir(np.linspace(0,1,256)) ]
        vir_legend = bcm.LinearColormap(
            tick_labels = [],
            scale_width = 200,
            colors=vir_colors,
            vmin=0, vmax=1,
            caption="Normalized (low → high)"
        )
        vir_legend.add_to(m)
        # move to bottom-right
        m.get_root().html.add_child(folium.Element("""
            <style>
            .linear-colormap {
              position: absolute !important;
              bottom: 10px !important;
              right: 10px !important;
              top: auto !important;
              width: 140px !important;
              height:  12px !important;
              margin: 0; padding: 0;
            }
            .linear-colormap .caption {
              margin-top: 2px; font-size: 0.8em;
            }
          </style>
        """))

    # Suitability map + legend
    suit_path = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(suit_path):
        with rasterio.open(suit_path) as src:
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

        plasma_colors = [ to_hex(c) for c in cmap(np.linspace(0,1,256)) ]
        suit_legend = bcm.LinearColormap(
            colors=plasma_colors,
            vmin=vmin, vmax=vmax,
            caption="Suitability"
        )
        suit_legend.add_to(m)
        # same bottom-right override, so it doesn't jump
        m.get_root().html.add_child(folium.Element("""
            <style>
              .linear-colormap { top: auto !important; bottom: 10px !important; right: 10px !important; }
            </style>
        """))

        m.fit_bounds([[bounds.bottom, bounds.left],
                      [bounds.top,    bounds.right]])

    folium.LayerControl(collapsed=False).add_to(m)
    html = html_lib.escape(m.get_root().render())
    return f"<iframe srcdoc=\"{html}\" style=\"width:100%; height:600px; border:none;\"></iframe>"


# --- Gradio UI ---
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

        with gr.Column(scale=3):
            map_output    = gr.HTML(value=create_map(), label="🗺️ Map Preview")
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
        os.environ['SELECTED_LANDCOVER_CLASSES'] = ','.join(c.split(" – ")[0]
                                                           for c in selected_landcover)
        res = subprocess.run(
            ["python", "scripts/fetch_predictors.py"],
            capture_output=True, text=True
        )
        print(res.stdout, res.stderr)
        msg = "✅ Predictors fetched." if res.returncode==0 else "❌ Fetch failed; see logs."
        return create_map(), msg

    def run_model():
        res = subprocess.run(
            ["python", "scripts/run_logistic_sdm.py"],
            capture_output=True, text=True
        )
        print(res.stdout, res.stderr)
        msg = "✅ Model completed." if res.returncode==0 else "❌ Model run failed."
        return create_map(), msg

    upload_input.change(fn=handle_upload,
                        inputs=[upload_input],
                        outputs=[map_output, status_output])
    fetch_button.click(fn=run_fetch,
                       inputs=[layer_selector, landcover_selector],
                       outputs=[map_output, status_output])
    run_button.click(fn=run_model,
                     outputs=[map_output, status_output])

    demo.launch()
