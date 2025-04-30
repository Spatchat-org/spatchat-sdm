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
import matplotlib.cm as mpl_cm

# --- Authenticate Earth Engine using Hugging Face Secret ---
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

# --- Helper Functions ---

def create_map():
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer('OpenStreetMap').add_to(m)

    # Presence points
    presence_path = "inputs/presence_points.csv"
    if os.path.exists(presence_path):
        df = pd.read_csv(presence_path)
        if {'latitude','longitude'}.issubset(df.columns):
            latlons = df[['latitude','longitude']].values.tolist()
            layer = folium.FeatureGroup(name="🟦 Presence Points")
            for lat, lon in latlons:
                folium.CircleMarker([lat, lon], radius=4, color='blue', fill=True, fill_opacity=0.8).add_to(layer)
            layer.add_to(m)
            if latlons:
                m.fit_bounds(latlons)

    # Predictor rasters
    rasters_dir = "predictor_rasters/wgs84"
    if os.path.isdir(rasters_dir):
        for fname in sorted(os.listdir(rasters_dir)):
            if fname.endswith('.tif'):
                path = os.path.join(rasters_dir, fname)
                with rasterio.open(path) as src:
                    img = src.read(1)
                    bounds = src.bounds
                # normalize
                vmin, vmax = np.nanmin(img), np.nanmax(img)
                if vmin == vmax:
                    continue
                norm = (img - vmin) / (vmax - vmin)
                # apply a matplotlib colormap
                cmap = mpl_cm.get_cmap('viridis')
                rgba = cmap(norm)  # shape (h, w, 4)
                folium.raster_layers.ImageOverlay(
                    image=rgba,
                    bounds=[[bounds.bottom, bounds.left],[bounds.top, bounds.right]],
                    opacity=1.0,
                    name=f"🟨 {fname}"
                ).add_to(m)

    # Suitability map
    suit_path = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(suit_path):
        with rasterio.open(suit_path) as src:
            img = src.read(1)
            bounds = src.bounds
        vmin, vmax = np.nanmin(img), np.nanmax(img)
        norm = (img - vmin) / (vmax - vmin)
        cmap = mpl_cm.get_cmap('plasma')
        rgba = cmap(norm)
        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[bounds.bottom, bounds.left],[bounds.top, bounds.right]],
            opacity=0.7,
            name="🎯 Suitability Map"
        ).add_to(m)
        m.fit_bounds([[bounds.bottom, bounds.left],[bounds.top, bounds.right]])

    folium.LayerControl(collapsed=False).add_to(m)
    html = html_lib.escape(m.get_root().render())
    return f"<iframe srcdoc=\"{html}\" style=\"width:100%; height:600px; border:none;\"></iframe>"

# --- Launch app ---
with gr.Blocks() as demo:
    gr.Markdown("""# SpatChat SDM – Species Distribution Modeling App""")
    with gr.Row():
        with gr.Column(scale=1):
            upload_input = gr.File(label="📄 Upload Presence CSV", file_types=['.csv'])
            layer_selector = gr.CheckboxGroup(
                choices=[
                    "bio1", "bio2", "bio3", "bio4", "bio5", "bio6", "bio7", "bio8", "bio9",
                    "bio10", "bio11", "bio12", "bio13", "bio14", "bio15", "bio16", "bio17",
                    "bio18", "bio19", "elevation", "slope", "aspect", "ndvi", "landcover"
                ],
                label="🧬 Environmental Layers"
            )
            landcover_selector = gr.CheckboxGroup(
                choices=landcover_choices,
                label="🌿 MODIS Landcover Classes (One-hot Encoded)"
            )
            fetch_button = gr.Button("🌐 Fetch Predictors")
            run_button = gr.Button("🧠 Run Model")
        with gr.Column(scale=3):
            map_output.render()
            status_output = gr.Textbox(label="Status", interactive=False)

    def handle_upload(file):
        if file is None or not hasattr(file, "name"):
            return create_map(), "⚠️ No file uploaded."

        print(f"📄 Received new file: {file.name}")

        shutil.rmtree("predictor_rasters", ignore_errors=True)
        shutil.rmtree("outputs", ignore_errors=True)
        shutil.rmtree("inputs", ignore_errors=True)
        os.makedirs("inputs", exist_ok=True)

        shutil.copy(file.name, "inputs/presence_points.csv")
        return create_map(), "✅ Presence points uploaded!"

    def run_fetch(selected_layers, selected_landcover):
        if not selected_layers:
            return status_output.value, "⚠️ Please select at least one layer."

        os.environ['SELECTED_LAYERS'] = ','.join(selected_layers)
        selected_codes = [c.split(" – ")[0] for c in selected_landcover]
        os.environ['SELECTED_LANDCOVER_CLASSES'] = ','.join(selected_codes)

        result = subprocess.run(["python", "scripts/fetch_predictors.py"], capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        if result.returncode == 0:
            return create_map(), "✅ Predictors fetched successfully."
        else:
            return status_output.value, "❌ Fetching failed. Check logs."

    def run_model():
        result = subprocess.run(["python", "scripts/run_logistic_sdm.py"], capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        if result.returncode == 0:
            return create_map(), "✅ Model ran successfully."
        else:
            return status_output.value, "❌ Model run failed. Check logs."

    upload_input.change(fn=handle_upload, inputs=upload_input, outputs=[map_output, status_output])
    fetch_button.click(fn=run_fetch, inputs=[layer_selector, landcover_selector], outputs=[map_output, status_output])
    run_button.click(fn=run_model, outputs=[map_output, status_output])

    demo.launch()
