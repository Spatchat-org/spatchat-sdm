import gradio as gr
import folium
import rasterio
import numpy as np
import pandas as pd
import subprocess
import os
import matplotlib.pyplot as plt

# --- Landcover Classes ---
landcover_class_dict = {
    0: "Water",
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-Up",
    14: "Cropland/Natural Vegetation Mosaic",
    15: "Snow and Ice",
    16: "Barren or Sparsely Vegetated"
}

# --- Environmental Layers Available ---
env_layer_options = [
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

uploaded_csv_path = "predictor_rasters/presence_points.csv"

# --- Handle Upload ---
def handle_upload(csv_file):
    if csv_file is None:
        return "⚠️ No file uploaded.", None
    df = pd.read_csv(csv_file.name)
    os.makedirs("predictor_rasters", exist_ok=True)
    df.to_csv(uploaded_csv_path, index=False)
    return "✅ Presence points uploaded!", preview_presence_map(df)

# --- Preview Presence Points ---
def preview_presence_map(df):
    if 'longitude' not in df.columns or 'latitude' not in df.columns:
        return "⚠️ CSV must have longitude, latitude columns."
    m = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=5)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color='blue',
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
    return m._repr_html_()

# --- Show Suitability Map ---
def load_suitability_map():
    raster_path = "outputs/suitability_map.tif"
    if not os.path.exists(raster_path):
        return "⚠️ Suitability map not generated yet."
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        array = src.read(1)
    array_min, array_max = np.nanmin(array), np.nanmax(array)
    norm_array = (array - array_min) / (array_max - array_min)
    plt.imsave("outputs/suitability_map_temp.png", np.clip(norm_array, 0, 1), cmap="YlGn", vmin=0, vmax=1)
    center_lat = (bounds.top + bounds.bottom) / 2
    center_lon = (bounds.left + bounds.right) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    folium.raster_layers.ImageOverlay(
        name="Suitability",
        image="outputs/suitability_map_temp.png",
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.6
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m._repr_html_()

# --- Full Workflow ---
def full_workflow(selected_layers, selected_lc_classes):
    if not selected_layers:
        return "⚠️ Please select environmental layers.", None

    os.makedirs("scripts", exist_ok=True)
    with open("scripts/user_layer_selection.txt", "w") as f:
        for layer in selected_layers:
            f.write(f"{layer}\n")
    with open("scripts/user_landcover_selection.txt", "w") as f:
        for lc in selected_lc_classes:
            f.write(f"{lc.split(':')[0]}\n")  # Only class ID

    subprocess.run(["python", "scripts/run_full_pipeline.py"])

    return "✅ Model completed and map generated!", load_suitability_map()

# --- Gradio App ---
with gr.Blocks() as app:
    gr.Markdown("# 🧬 Spatchat-SDM - Upload, Select, Model, Map")

    with gr.Row():
        uploader = gr.File(label="📤 Upload Presence CSV (longitude, latitude)")
        upload_status = gr.Markdown("⬇️ Waiting for upload...")
        presence_map = gr.HTML()

    uploader.change(
        fn=handle_upload,
        inputs=[uploader],
        outputs=[upload_status, presence_map]
    )

    gr.Markdown("## 🌍 Select Environmental Layers")
    env_layer_selector = gr.CheckboxGroup(choices=env_layer_options, label="Environmental Layers")

    gr.Markdown("## 🌳 Select Landcover Classes (if using Landcover)")
    landcover_selector = gr.CheckboxGroup(
        choices=[f"{k}: {v}" for k,v in landcover_class_dict.items()],
        label="Landcover Classes"
    )

    run_button = gr.Button("🚀 Fetch Predictors + Run Model")
    model_status = gr.Markdown("🔄 Waiting to run...")
    suitability_map_output = gr.HTML()

    run_button.click(
        fn=full_workflow,
        inputs=[env_layer_selector, landcover_selector],
        outputs=[model_status, suitability_map_output]
    )

app.launch(share=True)
