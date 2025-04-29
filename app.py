import gradio as gr
import geemap.foliumap as foliumap
import folium
import html as html_lib
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
import json
import ee
import joblib
import shutil

# --- Authenticate Earth Engine using Hugging Face Secret ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)
print("✅ Earth Engine authenticated using Service Account!")

# --- Global State ---
uploaded_csv = None

# --- Helper Functions ---

def reproject_to_wgs84(src_path, dst_path):
    with rasterio.open(src_path) as src:
        print(f"📏 Original CRS: {src.crs}")
        if src.crs.to_epsg() == 4326:
            shutil.copy(src_path, dst_path)
            print("✅ Already in WGS84; copied directly.")
            return

        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': "EPSG:4326",
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.nearest
                )
        print("🌐 Reprojection complete!")

def create_map(presence_points=None, include_rasters=True):
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer('OpenStreetMap').add_to(m)

    # Draw presence points
    if presence_points is not None:
        try:
            df = pd.read_csv(presence_points.name)
            if {'latitude', 'longitude'}.issubset(df.columns):
                points_layer = folium.FeatureGroup(name="🟦 Presence Points")
                for _, row in df.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=3,
                        color='blue',
                        fill=True,
                        fill_opacity=0.7
                    ).add_to(points_layer)
                points_layer.add_to(m)
        except Exception as e:
            print(f"⚠️ Error reading CSV: {e}")

    # Add predictor rasters
    if include_rasters and os.path.exists("predictor_rasters"):
        for tif in os.listdir("predictor_rasters"):
            if tif.endswith(".tif"):
                try:
                    path = os.path.join("predictor_rasters", tif)
                    with rasterio.open(path) as src:
                        print(f"🌍 Raster: {tif}, CRS: {src.crs}")
                        bounds = src.bounds
                        img = src.read(1)
                        raster_layer = folium.raster_layers.ImageOverlay(
                            image=img,
                            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                            opacity=0.4,
                            colormap=lambda x: (0, 1, 0, x),
                            name=f"🟨 {tif}"
                        )
                        raster_layer.add_to(m)
                except Exception as e:
                    print(f"⚠️ Error displaying raster {tif}: {e}")

    # Leaflet-native toggle panel
    folium.LayerControl(collapsed=False).add_to(m)

    raw_html = m.get_root().render()
    safe_html = html_lib.escape(raw_html)
    iframe = f"""<iframe srcdoc="{safe_html}" style="width:100%; height:600px; border:none;"></iframe>"""
    return iframe

def handle_upload(file):
    global uploaded_csv
    uploaded_csv = file
    os.makedirs("predictor_rasters", exist_ok=True)
    shutil.copy(file.name, "predictor_rasters/presence_points.csv")
    return create_map(uploaded_csv), "✅ Presence points uploaded!"

def fetch_predictors(selected):
    global uploaded_csv

    if not selected:
        return "⚠️ No predictors selected.", gr.update(choices=[]), create_map(uploaded_csv)

    selected_layers = ",".join(selected)
    os.environ['SELECTED_LAYERS'] = selected_layers
    os.system("python scripts/fetch_predictors.py")

    available_files = [f for f in os.listdir("predictor_rasters") if f.endswith(".tif")]
    map_html = create_map(uploaded_csv)
    return "✅ Predictors fetched.", gr.update(choices=available_files), map_html

def run_model():
    if uploaded_csv is None:
        return "⚠️ Please upload presence points first."

    os.system("python scripts/run_logistic_sdm.py")

    if os.path.exists("outputs/suitability_map.tif"):
        reproject_to_wgs84(
            "outputs/suitability_map.tif",
            "outputs/suitability_map_wgs84.tif"
        )
        return "✅ Model trained and reprojected!"
    else:
        return "❗ Model ran but no suitability map was generated."

def show_suitability_map():
    path = "outputs/suitability_map_wgs84.tif"
    if not os.path.exists(path):
        return "❗ No suitability map available yet."

    with rasterio.open(path) as src:
        print(f"🗺️ Suitability map CRS: {src.crs}")
        bounds = src.bounds
        img = src.read(1)
        m = folium.Map(control_scale=True)
        folium.TileLayer('OpenStreetMap').add_to(m)

        folium.raster_layers.ImageOverlay(
            image=img,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            opacity=0.6,
            colormap=lambda x: (1, 0, 0, x),
            name="🎯 Suitability Map"
        ).add_to(m)

        m.fit_bounds([[bounds.bottom, bounds.left], [bounds.top, bounds.right]])
        folium.LayerControl(collapsed=False).add_to(m)

    raw_html = m.get_root().render()
    safe_html = html_lib.escape(raw_html)
    iframe = f"""<iframe srcdoc="{safe_html}" style="width:100%; height:600px; border:none;"></iframe>"""
    return iframe

# --- Gradio App Layout ---

with gr.Blocks() as app:
    gr.Markdown("## 🧬 Spatchat-SDM: Global Species Distribution Modeling")

    map_output = gr.HTML(value=create_map(), label="🗺️ Live Preview")

    with gr.Row():
        uploader = gr.File(label="📥 Upload Presence Points (CSV)")
        upload_status = gr.Markdown()

    with gr.Row():
        layer_selector = gr.CheckboxGroup(
            label="🌎 Select Environmental Predictors",
            choices=[
                "elevation",
                "slope",
                "aspect",
                "ndvi",
                "precipitation",
                "mean_temperature",
                "min_temperature",
                "max_temperature",
                "landcover"
            ]
        )
        fetch_btn = gr.Button("📥 Fetch Predictors")
        fetch_status = gr.Markdown()

    with gr.Row():
        run_btn = gr.Button("🚀 Run SDM Model")
        run_status = gr.Markdown()

    with gr.Row():
        show_map_btn = gr.Button("🎯 Show Suitability Map")
        suitability_map_output = gr.HTML()

    # --- Actions ---

    uploader.change(fn=handle_upload, inputs=[uploader], outputs=[map_output, upload_status])
    fetch_btn.click(fn=fetch_predictors, inputs=[layer_selector], outputs=[fetch_status, layer_selector, map_output])
    run_btn.click(fn=run_model, outputs=[run_status])
    show_map_btn.click(fn=show_suitability_map, outputs=[suitability_map_output])

# --- Launch App ---

app.launch()
