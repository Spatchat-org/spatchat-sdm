# =====================
# app.py
# =====================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

import gradio as gr
import geemap.foliumap as foliumap
import folium
import html as html_lib
import pandas as pd
import numpy as np
import rasterio
import os
import json
import shutil
import uuid

from predictor_fetcher import fetch_predictors
from sdm_runner import run_logistic_sdm

import ee
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)
print("✅ Earth Engine authenticated!")

uploaded_csv = None

landcover_options = {
    0: "water", 1: "evergreen needleleaf forest", 2: "evergreen broadleaf forest",
    3: "deciduous needleleaf forest", 4: "deciduous broadleaf forest", 5: "mixed forest",
    6: "closed shrublands", 7: "open shrublands", 8: "woody savannas", 9: "savannas",
    10: "grasslands", 11: "permanent wetlands", 12: "croplands", 13: "urban and built up",
    14: "cropland/natural vegetation mosaic", 15: "snow and ice", 16: "barren or sparsely vegetated"
}
landcover_choices = [f"{k} – {v}" for k, v in landcover_options.items()]

def reproject_to_wgs84(src_path, dst_path):
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    with rasterio.open(src_path) as src:
        if src.crs and src.crs.to_epsg() == 4326:
            shutil.copy(src_path, dst_path)
            return
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({'crs': "EPSG:4326", 'transform': transform, 'width': width, 'height': height})
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(rasterio.band(src, i), rasterio.band(dst, i),
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs="EPSG:4326",
                          resampling=Resampling.nearest)

def create_map(presence_points=None):
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer('OpenStreetMap').add_to(m)
    if presence_points is not None:
        try:
            df = pd.read_csv(presence_points.name)
            latlons = []
            for _, row in df.iterrows():
                latlon = [row['latitude'], row['longitude']]
                latlons.append(latlon)
                folium.CircleMarker(location=latlon, radius=3, color='blue', fill=True, fill_opacity=0.7).add_to(m)
            if latlons:
                m.fit_bounds(latlons)
        except Exception as e:
            print(f"CSV error: {e}")
    if os.path.exists("predictor_rasters/wgs84"):
        for tif in os.listdir("predictor_rasters/wgs84"):
            if tif.endswith(".tif"):
                with rasterio.open(os.path.join("predictor_rasters/wgs84", tif)) as src:
                    img = src.read(1)
                    bounds = src.bounds
                    if np.nanmin(img) != np.nanmax(img):
                        img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))
                    folium.raster_layers.ImageOverlay(
                        image=img,
                        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                        opacity=0.4,
                        colormap=lambda x: (0, 1, 0, x),
                        name=tif.replace("_", " ")
                    ).add_to(m)
    suitability_path = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(suitability_path):
        with rasterio.open(suitability_path) as src:
            img = src.read(1)
            bounds = src.bounds
            if np.nanmin(img) != np.nanmax(img):
                img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))
            folium.raster_layers.ImageOverlay(
                image=img,
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                opacity=0.6,
                colormap=lambda x: (1, 0, 0, x),
                name="Suitability"
            ).add_to(m)
            m.fit_bounds([[bounds.bottom, bounds.left], [bounds.top, bounds.right]])
    folium.LayerControl(collapsed=False).add_to(m)
    html = html_lib.escape(m.get_root().render())
    return f'<iframe key="{uuid.uuid4().hex}" srcdoc="{html}" style="width:100%; height:600px; border:none;"></iframe>'

def handle_upload(file, selected_layers, selected_classes):
    global uploaded_csv
    uploaded_csv = file
    shutil.copy(file.name, "predictor_rasters/presence_points.csv")
    class_ids = [s.split(" –")[0].strip() for s in selected_classes]
    fetch_predictors("predictor_rasters/presence_points.csv", selected_layers, list(map(int, class_ids)))
    os.makedirs("predictor_rasters/wgs84", exist_ok=True)
    for tif in os.listdir("predictor_rasters"):
        if tif.endswith(".tif"):
            reproject_to_wgs84(os.path.join("predictor_rasters", tif), os.path.join("predictor_rasters/wgs84", tif))
    return create_map(uploaded_csv), "✅ Predictors fetched."

def run_model():
    if uploaded_csv is None:
        return "⚠️ Please upload a presence CSV first.", create_map()
    shutil.copy(uploaded_csv.name, "predictor_rasters/presence_points.csv")
    run_logistic_sdm("predictor_rasters/presence_points.csv")
    reproject_to_wgs84("outputs/suitability_map.tif", "outputs/suitability_map_wgs84.tif")
    return "✅ Model completed!", create_map(uploaded_csv)

with gr.Blocks() as app:
    gr.Markdown("## 🧬 SpatChat SDM: Species Distribution Modeling")

    map_output = gr.HTML(value=create_map(), label="🗺️ Map Preview")

    uploader = gr.File(label="📥 Upload Presence CSV")
    layer_selector = gr.CheckboxGroup(
        label="🌍 Environmental Predictors",
        choices=[f"bio{i}" for i in range(1, 20)] + ["elevation", "slope", "aspect", "ndvi", "landcover"]
    )
    landcover_selector = gr.CheckboxGroup(label="🌱 Landcover Classes", choices=landcover_choices, visible=False)
    fetch_status = gr.Markdown()

    def toggle_classes(selected):
        return gr.update(visible="landcover" in selected)

    layer_selector.change(toggle_classes, inputs=layer_selector, outputs=landcover_selector)
    uploader.change(handle_upload, inputs=[uploader, layer_selector, landcover_selector], outputs=[map_output, fetch_status])
    run_btn = gr.Button("🚀 Run SDM")
    run_status = gr.Markdown()
    run_btn.click(run_model, outputs=[run_status, map_output])

app.launch()