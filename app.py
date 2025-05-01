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

# --- Clean previous session ---
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
cbar.set_ticks([])             # hide numbers
ax.set_xlabel("Low                                  High")
fig.tight_layout(pad=0)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100)
plt.close(fig)
buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(buf.read()).decode()

# --- Landcover labels (one-hot) ---
landcover_options = {
    0: "water",1:"evergreen needleleaf forest",2:"evergreen broadleaf forest",
    3:"deciduous needleleaf forest",4:"deciduous broadleaf forest",5:"mixed forest",
    6:"closed shrublands",7:"open shrublands",8:"woody savannas",9:"savannas",
    10:"grasslands",11:"permanent wetlands",12:"croplands",
    13:"urban and built up",14:"cropland/natural vegetation mosaic",
    15:"snow and ice",16:"barren or sparsely vegetated"
}
landcover_choices = [f"{k} – {v}" for k,v in landcover_options.items()]

def create_map():
    m = folium.Map(location=[0,0], zoom_start=2, control_scale=True)
    folium.TileLayer('OpenStreetMap').add_to(m)

    # Presence points
    csvp = "inputs/presence_points.csv"
    if os.path.exists(csvp):
        df = pd.read_csv(csvp)
        if {'latitude','longitude'}.issubset(df.columns):
            pts = df[['latitude','longitude']].values.tolist()
            fg = folium.FeatureGroup(name="🟦 Presence Points")
            for lat,lon in pts:
                folium.CircleMarker([lat,lon],
                    radius=4, color='blue', fill=True, fill_opacity=0.8
                ).add_to(fg)
            fg.add_to(m)
            if pts: m.fit_bounds(pts)

    # Predictor rasters (Viridis)
    ras_dir = "predictor_rasters/wgs84"
    if os.path.isdir(ras_dir):
        for fn in sorted(os.listdir(ras_dir)):
            if not fn.endswith(".tif"): continue
            path = os.path.join(ras_dir,fn)
            with rasterio.open(path) as src:
                img=src.read(1); b=src.bounds
            vmin,vmax = np.nanmin(img), np.nanmax(img)
            if np.isnan(vmin) or vmin==vmax: continue
            norm=(img-vmin)/(vmax-vmin)
            rgba=colormaps['viridis'](norm)
            folium.raster_layers.ImageOverlay(
                image=rgba,
                bounds=[[b.bottom,b.left],[b.top,b.right]],
                opacity=1.0,
                name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})"
            ).add_to(m)

    # Suitability map (also Viridis)
    suitf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(suitf):
        with rasterio.open(suitf) as src:
            img=src.read(1); b=src.bounds
        vmin,vmax = np.nanmin(img), np.nanmax(img)
        norm=(img-vmin)/(vmax-vmin)
        rgba=colormaps['viridis'](norm)
        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[b.bottom,b.left],[b.top,b.right]],
            opacity=0.7,
            name=f"🎯 Suitability ({vmin:.2f}–{vmax:.2f})"
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # embed the static colorbar at bottom-right
    img_html = (
      f'<img src="data:image/png;base64,{COLORBAR_BASE64}" '
      'style="position:absolute; bottom:15px; right:10px; '
      'width:200px; height:30px; z-index:1000;" />'
    )
    m.get_root().html.add_child(Element(img_html))

    return gr.HTML.update(value=html_lib.escape(m.get_root().render()))

def zip_results():
    """Zip predictor_rasters + outputs into results.zip and return its path."""
    out = "results.zip"
    if os.path.exists(out): os.remove(out)
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as zf:
        for fold in ("predictor_rasters","outputs"):
            for root,_,files in os.walk(fold):
                for fn in files:
                    p=os.path.join(root,fn)
                    zf.write(p,arcname=os.path.relpath(p,"."))
    return out  # return path for DownloadButton

with gr.Blocks() as demo:
    gr.Markdown("# SpatChat SDM – Species Distribution Modeling App")

    with gr.Row():
        with gr.Column(scale=1):
            upload_input      = gr.File(label="📄 Upload Presence CSV", file_types=['.csv'])
            layer_selector    = gr.CheckboxGroup(
                                   choices=[*[f"bio{i}" for i in range(1,20)],
                                            "elevation","slope","aspect","ndvi"],
                                   label="🧬 Environmental Layers"
                                )
            landcover_selector= gr.CheckboxGroup(
                                   choices=landcover_choices,
                                   label="🌿 MODIS Landcover Classes (One-hot)"
                                )
            fetch_button      = gr.Button("🌐 Fetch Predictors")
            run_button        = gr.Button("🧠 Run Model")
            download_button   = gr.DownloadButton("📥 Download Results")

        with gr.Column(scale=3):
            map_output    = gr.HTML(value=create_map(), label="🗺️ Map Preview")
            status_output = gr.Textbox(label="Status", interactive=False)

    def handle_upload(f):
        if not f or not hasattr(f,"name"):
            return create_map(), "⚠️ No file uploaded."
        for d in ("predictor_rasters","outputs","inputs"):
            shutil.rmtree(d,ignore_errors=True)
        os.makedirs("inputs",exist_ok=True)
        shutil.copy(f.name,"inputs/presence_points.csv")
        return create_map(), "✅ Presence points uploaded!"

    def run_fetch(sl, lc):
        if not sl and not lc:
            return create_map(), "⚠️ Select at least one predictor."
        layers = list(sl)
        if lc: layers.append("landcover")
        os.environ['SELECTED_LAYERS']            = ",".join(layers)
        os.environ['SELECTED_LANDCOVER_CLASSES'] = ",".join(c.split(" – ")[0] for c in lc)
        res = subprocess.run(["python","scripts/fetch_predictors.py"],
                             capture_output=True, text=True)
        return create_map(), ("✅ Predictors fetched." if res.returncode==0 else "❌ Fetch failed.")

    def run_model():
        res = subprocess.run(["python","scripts/run_logistic_sdm.py"],
                             capture_output=True, text=True)
        return create_map(), ("✅ Model completed." if res.returncode==0 else "❌ Model failed.")

    upload_input.change(handle_upload, inputs=[upload_input], outputs=[map_output,status_output])
    fetch_button.click(run_fetch, inputs=[layer_selector,landcover_selector],
                       outputs=[map_output,status_output])
    run_button.click(run_model, outputs=[map_output,status_output])
    download_button.click(zip_results, outputs=[download_button])

    demo.launch()
