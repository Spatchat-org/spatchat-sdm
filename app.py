import os
import io
import json
import base64
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
import ee
import joblib

from matplotlib import pyplot as plt, colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from folium import Element

# --- Authenticate Earth Engine ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)

# --- Clean up last session ---
for d in ("predictor_rasters","outputs","inputs"):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs("inputs", exist_ok=True)

# --- Pre-render a small Viridis color bar as PNG → base64 ---
fig, ax = plt.subplots(figsize=(4, 0.5))
norm = Normalize(vmin=0, vmax=1)
cbar = plt.colorbar(
    ScalarMappable(norm=norm, cmap='viridis'),
    cax=ax, orientation='horizontal'
)
cbar.set_ticks([])
ax.set_xlabel("Low                                  High")
fig.tight_layout(pad=0)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100)
plt.close(fig)
buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(buf.read()).decode()

# --- Landcover labels (for one-hot) ---
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

    # Suitability map (now using Viridis)
    suit_path = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(suit_path):
        with rasterio.open(suit_path) as src:
            img = src.read(1)
            bnd = src.bounds

        vmin, vmax = np.nanmin(img), np.nanmax(img)
        norm = (img - vmin)/(vmax-vmin)
        cmap = colormaps['viridis']      # ← changed from 'plasma'
        rgba = cmap(norm)

        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[bnd.bottom,bnd.left],[bnd.top,bnd.right]],
            opacity=0.7,
            name=f"🎯 Suitability ({vmin:.2f}–{vmax:.2f})"
        ).add_to(m)

    # 5) Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # 6) Overlay our static colorbar PNG bottom-right
    img_html = (
        f'<img src="data:image/png;base64,{COLORBAR_BASE64}" '
        'style="position:absolute; bottom:20px; right:10px; '
        'width:200px; height:30px; z-index:1000;" />'
    )
    m.get_root().html.add_child(Element(img_html))

    # 7) Render and return iframe
    html = html_lib.escape(m.get_root().render())
    return f'<iframe srcdoc="{html}" style="width:100%; height:600px; border:none;"></iframe>'

def zip_results():
    """Bundle predictor_rasters + outputs into results.zip and return its path."""
    archive = "results.zip"
    if os.path.exists(archive):
        os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in ("predictor_rasters", "outputs"):
            for root, _, files in os.walk(folder):
                for fn in files:
                    path = os.path.join(root, fn)
                    zf.write(path, arcname=os.path.relpath(path, start="."))
    return archive

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# SpatChat SDM – Species Distribution Modeling App")

    with gr.Row():
        with gr.Column(scale=1):
            upload_input       = gr.File(label="📄 Upload Presence CSV", file_types=['.csv'])
            layer_selector     = gr.CheckboxGroup(
                                     choices=[*[f"bio{i}" for i in range(1,20)],
                                              "elevation","slope","aspect","ndvi"],
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
            map_output    = gr.HTML(value=create_map(), label="🗺️ Map Preview")
            status_output = gr.Textbox(label="Status", interactive=False)
            download_output = gr.File(label="Download .zip")

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
        # require at least one env layer OR one landcover class
        if not selected_layers and not selected_landcover:
            return create_map(), "⚠️ Select at least one predictor."
        # auto-add 'landcover' if any classes picked
        layers = list(selected_layers)
        if selected_landcover:
            layers.append("landcover")
        os.environ['SELECTED_LAYERS']            = ','.join(layers)
        os.environ['SELECTED_LANDCOVER_CLASSES'] = ','.join(
            c.split(" – ")[0] for c in selected_landcover
        )
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

    upload_input.change(
        fn=handle_upload,
        inputs=[upload_input],
        outputs=[map_output, status_output]
    )
    fetch_button.click(
        fn=run_fetch,
        inputs=[layer_selector, landcover_selector],
        outputs=[map_output, status_output]
    )
    run_button.click(
        fn=run_model,
        outputs=[map_output, status_output]
    )
    download_button.click(
        fn=zip_results,
        inputs=[],
        outputs=[download_output]
    )

    demo.launch()
