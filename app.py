import gradio as gr
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

# --- Landcover Code Name Map (for use in fetch_predictors.py) ---
modis_landcover_code_name = {
    0: "water",
    1: "evergreen_needleleaf_forest",
    2: "evergreen_broadleaf_forest",
    3: "deciduous_needleleaf_forest",
    4: "deciduous_broadleaf_forest",
    5: "mixed_forest",
    6: "closed_shrublands",
    7: "open_shrublands",
    8: "woody_savannas",
    9: "savannas",
    10: "grasslands",
    11: "permanent_wetlands",
    12: "croplands",
    13: "urban_and_built_up",
    14: "cropland_natural_vegetation_mosaic",
    15: "snow_and_ice",
    16: "barren_or_sparsely_vegetated"
}

with open("modis_landcover_code_name.json", "w") as f:
    json.dump(modis_landcover_code_name, f)

# --- Generate map preview ---
def create_map():
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer('OpenStreetMap').add_to(m)

    presence_path = "inputs/presence_points.csv"
    if os.path.exists(presence_path):
        try:
            df = pd.read_csv(presence_path)
            if {'latitude', 'longitude'}.issubset(df.columns):
                latlons = []
                layer = folium.FeatureGroup(name="🟦 Presence Points")
                for _, row in df.iterrows():
                    latlon = [row['latitude'], row['longitude']]
                    latlons.append(latlon)
                    folium.CircleMarker(
                        location=latlon,
                        radius=3,
                        color='blue',
                        fill=True,
                        fill_opacity=0.7
                    ).add_to(layer)
                layer.add_to(m)
                if latlons:
                    m.fit_bounds(latlons)
        except Exception as e:
            print(f"⚠️ Error reading CSV: {e}")

    folium.LayerControl(collapsed=False).add_to(m)
    raw_html = m.get_root().render()
    safe_html = html_lib.escape(raw_html)
    return f"""<iframe srcdoc=\"{safe_html}\" style=\"width:100%; height:600px; border:none;\"></iframe>"""

map_output = gr.HTML(value=create_map(), label="🗺️ Preview")

# --- Launch app ---
with gr.Blocks() as demo:
    gr.Markdown("""# SpatChat SDM – Species Distribution Modeling App""")
    with gr.Row():
        with gr.Column(scale=1):
            upload_input = gr.File(label="📤 Upload Presence CSV", file_types=['.csv'])
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

        print(f"📤 Received new file: {file.name}")

        shutil.rmtree("predictor_rasters", ignore_errors=True)
        shutil.rmtree("outputs", ignore_errors=True)
        shutil.rmtree("inputs", ignore_errors=True)
        os.makedirs("inputs", exist_ok=True)

        shutil.copy(file.name, "inputs/presence_points.csv")
        return create_map(), "✅ Presence points uploaded!"

    upload_input.change(fn=handle_upload, inputs=upload_input, outputs=[map_output, status_output])

    demo.launch()
