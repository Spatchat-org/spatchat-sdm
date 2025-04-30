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

# --- Ensure input directory exists ---
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

# --- Helper Functions ---

def reproject_to_wgs84(src_path, dst_path):
    with rasterio.open(src_path) as src:
        print(f"📏 Original CRS for {src_path}: {src.crs}")
        if src.crs and src.crs.to_epsg() == 4326:
            shutil.copy(src_path, dst_path)
            print("✅ Already in WGS84; copied directly.")
            return

        if src.count == 0:
            print(f"⚠️ Skipping {src_path} — has no raster bands.")
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
        print(f"🌐 Reprojected: {dst_path}")


def create_map(presence_points=None):
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer('OpenStreetMap').add_to(m)

    if presence_points is not None:
        try:
            df = pd.read_csv(presence_points.name)
            if {'latitude', 'longitude'}.issubset(df.columns):
                latlons = []
                points_layer = folium.FeatureGroup(name="🟦 Presence Points")
                for _, row in df.iterrows():
                    latlon = [row['latitude'], row['longitude']]
                    latlons.append(latlon)
                    folium.CircleMarker(
                        location=latlon,
                        radius=3,
                        color='blue',
                        fill=True,
                        fill_opacity=0.7
                    ).add_to(points_layer)
                points_layer.add_to(m)
                if latlons:
                    m.fit_bounds(latlons)
        except Exception as e:
            print(f"⚠️ Error reading CSV: {e}")

    wgs84_dir = "predictor_rasters/wgs84"
    if os.path.exists(wgs84_dir):
        for tif in os.listdir(wgs84_dir):
            if tif.endswith(".tif"):
                try:
                    path = os.path.join(wgs84_dir, tif)
                    with rasterio.open(path) as src:
                        if src.count == 0:
                            continue
                        bounds = src.bounds
                        img = src.read(1)
                        if np.nanmin(img) != np.nanmax(img):
                            img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))
                        folium.raster_layers.ImageOverlay(
                            image=img,
                            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                            opacity=0.4,
                            colormap=lambda x: (0, 1, 0, x),
                            name=f"🟨 {tif}"
                        ).add_to(m)
                except Exception as e:
                    print(f"⚠️ Error displaying raster {tif}: {e}")

    suitability_path = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(suitability_path):
        try:
            with rasterio.open(suitability_path) as src:
                bounds = src.bounds
                img = src.read(1)
                if np.nanmin(img) != np.nanmax(img):
                    img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))
                folium.raster_layers.ImageOverlay(
                    image=img,
                    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                    opacity=0.6,
                    colormap=lambda x: (1, 0, 0, x),
                    name="🎯 Suitability Map"
                ).add_to(m)
                m.fit_bounds([[bounds.bottom, bounds.left], [bounds.top, bounds.right]])
        except Exception as e:
            print(f"⚠️ Could not load suitability map: {e}")

    folium.LayerControl(collapsed=False).add_to(m)
    raw_html = m.get_root().render()
    safe_html = html_lib.escape(raw_html)
    return f"""<iframe srcdoc=\"{safe_html}\" style=\"width:100%; height:600px; border:none;\"></iframe>"""


def handle_upload(file):
    global uploaded_csv
    uploaded_csv = file
    os.makedirs("inputs", exist_ok=True)
    shutil.copy(file.name, "inputs/presence_points.csv")
    return create_map(uploaded_csv), "✅ Presence points uploaded!"


def fetch_predictors(selected_layers, selected_classes):
    global uploaded_csv
    if not selected_layers:
        return "⚠️ No predictors selected.", gr.update(choices=[]), create_map(uploaded_csv)

    if uploaded_csv is not None:
        shutil.copy(uploaded_csv.name, "inputs/presence_points.csv")

    os.environ["SELECTED_LAYERS"] = ",".join(selected_layers)
    class_ids = [s.split("–")[0].strip() for s in selected_classes]
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(class_ids)

    os.system("python scripts/fetch_predictors.py")

    os.makedirs("predictor_rasters/wgs84", exist_ok=True)
    for tif in os.listdir("predictor_rasters"):
        if tif.endswith(".tif"):
            src_path = os.path.join("predictor_rasters", tif)
            dst_path = os.path.join("predictor_rasters/wgs84", tif)
            reproject_to_wgs84(src_path, dst_path)

    available_files = [f for f in os.listdir("predictor_rasters") if f.endswith(".tif")]
    return "✅ Predictors fetched.", gr.update(choices=available_files), create_map(uploaded_csv)


def run_model():
    if uploaded_csv is None:
        return "⚠️ Please upload presence points first.", create_map(uploaded_csv)

    shutil.copy(uploaded_csv.name, "inputs/presence_points.csv")
    os.system("python scripts/run_logistic_sdm.py")

    if os.path.exists("outputs/suitability_map.tif"):
        reproject_to_wgs84(
            "outputs/suitability_map.tif",
            "outputs/suitability_map_wgs84.tif"
        )
        return "✅ Model trained and reprojected!", create_map(uploaded_csv)
    else:
        return "❗ Model ran but no suitability map was generated.", create_map(uploaded_csv)

# --- Gradio UI ---

with gr.Blocks() as app:
    gr.Markdown("## 🧬 Spatchat-SDM: Global Species Distribution Modeling")

    map_output = gr.HTML(value=create_map(), label="🗺️ Live Preview")

    with gr.Row():
        uploader = gr.File(label="📥 Upload Presence Points (CSV)")
        upload_status = gr.Markdown()

    with gr.Row():
        layer_selector = gr.CheckboxGroup(
            label="🌎 Select Environmental Predictors",
            choices=[f"bio{i}" for i in range(1, 20)] + ["elevation", "slope", "aspect", "ndvi", "landcover"]
        )
        landcover_class_selector = gr.CheckboxGroup(
            label="🧮 Landcover Classes (only if 'landcover' is selected)",
            choices=landcover_choices,
            visible=False
        )

    with gr.Row():
        fetch_btn = gr.Button("📥 Fetch Predictors")
        fetch_status = gr.Markdown()

    with gr.Row():
        run_btn = gr.Button("🚀 Run SDM Model")
        run_status = gr.Markdown()

    with gr.Row():
        show_map_btn = gr.Button("🎯 Show Suitability Map")

    def toggle_landcover_class_selector(selected):
        return gr.update(visible="landcover" in selected)

    layer_selector.change(fn=toggle_landcover_class_selector, inputs=[layer_selector], outputs=[landcover_class_selector])
    uploader.change(fn=handle_upload, inputs=[uploader], outputs=[map_output, upload_status])
    fetch_btn.click(fn=fetch_predictors, inputs=[layer_selector, landcover_class_selector], outputs=[fetch_status, layer_selector, map_output])
    run_btn.click(fn=run_model, outputs=[run_status, map_output])
    show_map_btn.click(fn=create_map, outputs=[map_output])

app.launch()
