import os
import json
import shutil
import subprocess

import gradio as gr
import folium
import geemap.foliumap as foliumap
import html as html_lib
import pandas as pd
import numpy as np
import rasterio
import ee
import joblib

from matplotlib import colormaps
from matplotlib.colors import to_hex
import branca.colormap as bcm


from matplotlib import colormaps
from matplotlib.colors import to_hex

# --- Authenticate Earth Engine ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)

# --- Clean up from last run ---
shutil.rmtree("predictor_rasters", ignore_errors=True)
shutil.rmtree("outputs", ignore_errors=True)
shutil.rmtree("inputs", ignore_errors=True)
os.makedirs("inputs", exist_ok=True)

# --- Landcover choices (one-hot) ---
landcover_options = {
    0: "water", 1: "evergreen needleleaf forest", 2: "evergreen broadleaf forest",
    3: "deciduous needleleaf forest", 4: "deciduous broadleaf forest",
    5: "mixed forest", 6: "closed shrublands", 7: "open shrublands",
    8: "woody savannas", 9: "savannas", 10: "grasslands",
    11: "permanent wetlands", 12: "croplands", 13: "urban and built up",
    14: "cropland/natural vegetation mosaic", 15: "snow and ice",
    16: "barren or sparsely vegetated"
}
landcover_choices = [f"{k} – {v}" for k, v in landcover_options.items()]

def create_map():
    # Base map
    m = folium.Map(location=[0,0], zoom_start=2, control_scale=True)
    folium.TileLayer('OpenStreetMap').add_to(m)

    # 1) Presence points
    pts_csv = "inputs/presence_points.csv"
    if os.path.exists(pts_csv):
        df = pd.read_csv(pts_csv)
        if {'latitude','longitude'}.issubset(df.columns):
            coords = df[['latitude','longitude']].values.tolist()
            fg = folium.FeatureGroup(name="🟦 Presence Points")
            for lat,lon in coords:
                folium.CircleMarker([lat,lon],
                                    radius=4, color='blue',
                                    fill=True, fill_opacity=0.8
                                   ).add_to(fg)
            fg.add_to(m)
            m.fit_bounds(coords)

    # 2) Predictor rasters (viridis)
    predictor_dir = "predictor_rasters/wgs84"
    show_vir = False
    if os.path.isdir(predictor_dir):
        for fn in sorted(os.listdir(predictor_dir)):
            if not fn.endswith(".tif"):
                continue
            path = os.path.join(predictor_dir, fn)
            with rasterio.open(path) as src:
                arr = src.read(1)
                bnd = src.bounds
            vmin, vmax = np.nanmin(arr), np.nanmax(arr)
            if np.isnan(vmin) or vmin==vmax:
                continue
            show_vir = True
            norm = (arr - vmin)/(vmax-vmin)
            cmap = colormaps['viridis']
            rgba = cmap(norm)   # (h,w,4)
            folium.raster_layers.ImageOverlay(
                image=rgba,
                bounds=[[bnd.bottom,bnd.left],[bnd.top,bnd.right]],
                opacity=1.0,
                name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})"
            ).add_to(m)

    # 3) Viridis legend if any predictor was added
    if show_vir:
        vir = colormaps['viridis']
        colors = [to_hex(c) for c in vir(np.linspace(0,1,256))]
        legend = bcm.LinearColormap(
            tick_labels=[],
            colors=colors,
            vmin=0, vmax=1,
            caption="Normalized (low → high)"
        )
        legend.add_to(m)

    # 4) Suitability map (plasma) + legend
    suit = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(suit):
        with rasterio.open(suit) as src:
            arr = src.read(1)
            bnd = src.bounds
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        norm = (arr - vmin)/(vmax-vmin)
        cmap = colormaps['plasma']
        rgba = cmap(norm)
        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[bnd.bottom,bnd.left],[bnd.top,bnd.right]],
            opacity=0.7,
            name=f"🎯 Suitability ({vmin:.2f}–{vmax:.2f})"
        ).add_to(m)
        colors = [to_hex(c) for c in cmap(np.linspace(0,1,256))]
        suit_legend = bcm.LinearColormap(
            colors=colors,
            vmin=vmin, vmax=vmax,
            caption="Suitability"
        )
        suit_legend.add_to(m)
        m.fit_bounds([[bnd.bottom,bnd.left],[bnd.top,bnd.right]])

    folium.LayerControl(collapsed=False).add_to(m)

    # ** Render to HTML and wrap in iframe **
    html = html_lib.escape(m.get_root().render())
    return f"<iframe srcdoc=\"{html}\" style=\"width:100%;height:600px;border:none;\"></iframe>"

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# SpatChat SDM – Species Distribution Modeling App")

    with gr.Row():
        with gr.Column(scale=1):
            upload           = gr.File(label="📄 Upload Presence CSV", file_types=['.csv'])
            layer_selector   = gr.CheckboxGroup(
                                   choices=[*[f"bio{i}" for i in range(1,20)],
                                            "elevation","slope","aspect","ndvi","landcover"],
                                   label="🧬 Environmental Layers"
                               )
            lc_selector      = gr.CheckboxGroup(
                                   choices=landcover_choices,
                                   label="🌿 Landcover Classes (One-hot)"
                               )
            fetch_btn        = gr.Button("🌐 Fetch Predictors")
            run_btn          = gr.Button("🧠 Run Model")
        with gr.Column(scale=3):
            map_out   = gr.HTML(value=create_map(), label="🗺️ Map Preview")
            status    = gr.Textbox(label="Status", interactive=False)

    def on_upload(f):
        if not f or not hasattr(f,"name"):
            return create_map(), "⚠️ No file!"
        shutil.rmtree("predictor_rasters", ignore_errors=True)
        shutil.rmtree("outputs", ignore_errors=True)
        shutil.rmtree("inputs", ignore_errors=True)
        os.makedirs("inputs", exist_ok=True)
        shutil.copy(f.name, "inputs/presence_points.csv")
        return create_map(), "✅ Points uploaded!"

    def on_fetch(layers, lc):
        if not layers:
            return create_map(), "⚠️ Pick at least one layer."
        os.environ['SELECTED_LAYERS'] = ','.join(layers)
        codes = [c.split(" – ")[0] for c in lc]
        os.environ['SELECTED_LANDCOVER_CLASSES'] = ','.join(codes)
        res = subprocess.run(["python","scripts/fetch_predictors.py"],
                              capture_output=True, text=True)
        print(res.stdout, res.stderr)
        msg = "✅ Fetched predictors." if res.returncode==0 else "❌ Fetch failed."
        return create_map(), msg

    def on_run():
        res = subprocess.run(["python","scripts/run_logistic_sdm.py"],
                              capture_output=True, text=True)
        print(res.stdout, res.stderr)
        msg = "✅ Model done." if res.returncode==0 else "❌ Run failed."
        return create_map(), msg

    upload.change(on_upload, inputs=[upload], outputs=[map_out, status])
    fetch_btn.click(on_fetch,
                    inputs=[layer_selector, lc_selector],
                    outputs=[map_out, status])
    run_btn.click(on_run, outputs=[map_out, status])

    demo.launch()
