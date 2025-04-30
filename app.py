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

# (reproject_to_wgs84, create_map, handle_upload remain unchanged)

# --- Gradio App Launch ---
with gr.Blocks() as app:
    gr.Markdown("## 🧬 Spatchat-SDM")

    map_output = gr.HTML(value=create_map(), label="🗺️ Preview")

    with gr.Row():
        uploader = gr.File(label="📤 Upload CSV with Presence Points")
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

    def fetch_predictors(selected_layers, selected_classes):
        os.environ["SELECTED_LAYERS"] = ",".join(selected_layers)
        os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join([c.split("–")[0].strip() for c in selected_classes])
        os.system("python scripts/fetch_predictors.py")
        return "✅ Predictors fetched.", layer_selector.choices, create_map()

    def run_model():
        os.system("python scripts/run_logistic_sdm.py")
        return "✅ Model run complete.", create_map()

    layer_selector.change(fn=toggle_landcover_class_selector, inputs=[layer_selector], outputs=[landcover_class_selector])
    uploader.change(fn=handle_upload, inputs=[uploader], outputs=[map_output, upload_status])
    fetch_btn.click(fn=fetch_predictors, inputs=[layer_selector, landcover_class_selector], outputs=[fetch_status, layer_selector, map_output])
    run_btn.click(fn=run_model, outputs=[run_status, map_output])
    show_map_btn.click(fn=create_map, outputs=[map_output])

app.launch()
