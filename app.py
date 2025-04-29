import gradio as gr
import geemap.foliumap as foliumap
import folium
import html as html_lib
import pandas as pd
import numpy as np
import rasterio
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

def create_map(presence_points=None):
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer('OpenStreetMap').add_to(m)

    if presence_points is not None:
        try:
            df = pd.read_csv(presence_points.name)
            if {'latitude', 'longitude'}.issubset(df.columns):
                for idx, row in df.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=3,
                        color='blue',
                        fill=True,
                        fill_opacity=0.7
                    ).add_to(m)
        except Exception as e:
            print(f"⚠️ Error reading CSV: {e}")

    raw_html = m.get_root().render()
    safe_html = html_lib.escape(raw_html)
    iframe = f"""<iframe srcdoc="{safe_html}" style="width:100%; height:600px; border:none;"></iframe>"""
    return iframe

def handle_upload(file):
    global uploaded_csv
    uploaded_csv = file

    # ✅ Ensure directory exists
    os.makedirs("predictor_rasters", exist_ok=True)

    shutil.copy(file.name, "predictor_rasters/presence_points.csv")  # ✅ Hugging Face compatible
    return create_map(uploaded_csv), "✅ Presence points uploaded!"

def fetch_predictors(selected):
    if not selected:
        return "⚠️ No predictors selected.", gr.update(choices=[])

    selected_layers = ",".join(selected)
    os.environ['SELECTED_LAYERS'] = selected_layers
    os.system("python scripts/fetch_predictors.py")

    available_files = [f for f in os.listdir("predictor_rasters") if f.endswith(".tif")]
    return "✅ Predictors fetched.", gr.update(choices=available_files)

def run_model():
    if uploaded_csv is None:
        return "⚠️ Please upload presence points first."

    os.system("python scripts/run_logistic_sdm.py")

    if os.path.exists("outputs/suitability_map.tif"):
        return "✅ Model trained and suitability map generated!"
    else:
        return "❗ Model ran but no suitability map was generated."

def show_suitability_map():
    if not os.path.exists("outputs/suitability_map.tif"):
        return "❗ No suitability map available yet."

    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer('OpenStreetMap').add_to(m)

    with rasterio.open("outputs/suitability_map.tif") as src:
        bounds = src.bounds
        img = src.read(1)
        img_min = np.nanmin(img)
        img_max = np.nanmax(img)

        folium.raster_layers.ImageOverlay(
            image=img,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            opacity=0.6,
            colormap=lambda x: (1, 0, 0, x)  # simple red scale
        ).add_to(m)

    raw_html = m.get_root().render()
    safe_html = html_lib.escape(raw_html)
    iframe = f"""<iframe srcdoc="{safe_html}" style="width:100%; height:600px; border:none;"></iframe>"""
    return iframe

# --- Gradio App Layout ---

with gr.Blocks() as app:
    gr.Markdown("## 🧬 Spatchat-SDM: Global Species Distribution Modeling")

    with gr.Row():
        uploader = gr.File(label="📥 Upload Presence Points (CSV)")
        upload_btn = gr.Button("⬆️ Upload")
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
        show_map_btn = gr.Button("🗺️ Show Suitability Map")
        map_output = gr.HTML()

    # --- Actions ---

    upload_btn.click(fn=handle_upload, inputs=[uploader], outputs=[map_output, upload_status])
    fetch_btn.click(fn=fetch_predictors, inputs=[layer_selector], outputs=[fetch_status, layer_selector])
    run_btn.click(fn=run_model, outputs=[run_status])
    show_map_btn.click(fn=show_suitability_map, outputs=[map_output])

# --- Launch App ---

app.launch()
