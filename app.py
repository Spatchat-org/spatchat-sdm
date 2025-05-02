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

# for future LLM integration
from dotenv import load_dotenv
from together import Together

# --- Authenticate Earth Engine ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)

# --- (future) LLM client setup ---
load_dotenv()
llm = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# --- Clean up last session ---
for d in ("predictor_rasters","outputs","inputs"):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs("inputs", exist_ok=True)

# --- Pre-render Viridis color bar as base64 PNG ---
fig, ax = plt.subplots(figsize=(4,0.5))
norm = Normalize(vmin=0, vmax=1)
plt.colorbar(ScalarMappable(norm=norm,cmap="viridis"),
             cax=ax, orientation="horizontal").set_ticks([])
ax.set_xlabel("Low                                  High")
fig.tight_layout(pad=0)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100)
plt.close(fig)
buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(buf.read()).decode()

# --- Landcover labels (for one-hot) ---
landcover_options = {
    0:"water",1:"evergreen needleleaf forest",2:"evergreen broadleaf forest",
    3:"deciduous needleleaf forest",4:"deciduous broadleaf forest",5:"mixed forest",
    6:"closed shrublands",7:"open shrublands",8:"woody savannas",9:"savannas",
    10:"grasslands",11:"permanent wetlands",12:"croplands",
    13:"urban and built up",14:"cropland/natural vegetation mosaic",
    15:"snow and ice",16:"barren or sparsely vegetated"
}
landcover_choices = [f"{k} – {v}" for k,v in landcover_options.items()]

# --- Map builder (identical to your existing) ---
def create_map():
    m = folium.Map(location=[0,0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)
    # presence
    pcsv = "inputs/presence_points.csv"
    if os.path.exists(pcsv):
        df = pd.read_csv(pcsv)
        if {'latitude','longitude'}.issubset(df.columns):
            pts = df[['latitude','longitude']].values.tolist()
            fg = folium.FeatureGroup(name="🟦 Presence Points")
            for lat,lon in pts:
                folium.CircleMarker([lat,lon], radius=4,
                                    color="blue", fill=True, fill_opacity=0.8)\
                      .add_to(fg)
            fg.add_to(m)
            if pts: m.fit_bounds(pts)
    # predictors
    rasdir = "predictor_rasters/wgs84"
    if os.path.isdir(rasdir):
        for fn in sorted(os.listdir(rasdir)):
            if not fn.endswith(".tif"): continue
            path = os.path.join(rasdir,fn)
            with rasterio.open(path) as src:
                img = src.read(1); b = src.bounds
            vmin,vmax = np.nanmin(img), np.nanmax(img)
            if np.isnan(vmin) or vmin==vmax: continue
            rgba = colormaps['viridis']((img-vmin)/(vmax-vmin))
            folium.raster_layers.ImageOverlay(
                image=rgba,
                bounds=[[b.bottom,b.left],[b.top,b.right]],
                opacity=1.0,
                name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})"
            ).add_to(m)
    # suitability
    suitf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(suitf):
        with rasterio.open(suitf) as src:
            img = src.read(1); b = src.bounds
        vmin,vmax = np.nanmin(img), np.nanmax(img)
        rgba = colormaps['viridis']((img-vmin)/(vmax-vmin))
        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[b.bottom,b.left],[b.top,b.right]],
            opacity=0.7,
            name=f"🎯 Suitability ({vmin:.2f}–{vmax:.2f})"
        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    # static colorbar
    img_html = (
        f'<img src="data:image/png;base64,{COLORBAR_BASE64}" '
        'style="position:absolute; bottom:20px; right:10px; '
        'width:200px; height:30px; z-index:1000;" />'
    )
    m.get_root().html.add_child(Element(img_html))
    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" ' \
           'style="width:100%; height:600px; border:none;"></iframe>'

# --- ZIP helper, fetch, run_model unchanged ---
def zip_results():
    archive = "spatchat_results.zip"
    if os.path.exists(archive): os.remove(archive)
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters","outputs"):
            for r,_,fs in os.walk(fld):
                for f in fs:
                    full = os.path.join(r,f)
                    zf.write(full, arcname=os.path.relpath(full,"."))
    return archive

def run_fetch(sl, lc):
    if not sl and not lc:
        return create_map(), "⚠️ Select at least one predictor."
    layers = list(sl)
    if lc: layers.append("landcover")
    os.environ['SELECTED_LAYERS'] = ",".join(layers)
    os.environ['SELECTED_LANDCOVER_CLASSES'] = ",".join(c.split(" – ")[0] for c in lc)
    res = subprocess.run(["python","scripts/fetch_predictors.py"],
                         capture_output=True, text=True)
    ok = (res.returncode==0)
    return create_map(), ("✅ Predictors fetched." if ok else f"❌ Fetch failed:\n{res.stderr}")

def run_model():
    res = subprocess.run(["python","scripts/run_logistic_sdm.py"],
                         capture_output=True, text=True)
    if res.returncode!=0:
        return create_map(), f"❌ Model run failed:\n{res.stderr}", None, None
    stats_df = pd.read_csv("outputs/model_stats.csv")
    suitf = "outputs/suitability_map_wgs84.tif"
    if not os.path.exists(suitf):
        return create_map(), "⚠️ Model finished but no suitability map!", stats_df, None
    return (
        create_map(),
        "✅ Model ran successfully!",
        stats_df,
        "outputs/model_stats.csv"
    )

# --- stub LLM chat handler (just echoes) ---
def analyze_chat(user_msg, history):
    history = history[:]  # copy
    # for now just echo
    response = f"🤖 You said: “{user_msg}”"
    history.append((user_msg, response))
    return history

# --- Build Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# SpatChat SDM – Species Distribution Modeling App")

    with gr.Row():
        with gr.Column(scale=1):
            upload_input       = gr.File(label="📄 Upload Presence CSV", file_types=['.csv'])
            layer_selector     = gr.CheckboxGroup(
                                     choices=[f"bio{i}" for i in range(1,20)]
                                             + ["elevation","slope","aspect","ndvi"],
                                     label="🧬 Environmental Layers"
                                 )
            landcover_selector = gr.CheckboxGroup(
                                     choices=landcover_choices,
                                     label="🌿 MODIS Landcover Classes (One‑hot)"
                                 )
            fetch_button       = gr.Button("🌐 Fetch Predictors")
            run_button         = gr.Button("🧠 Run Model")
            download_button    = gr.DownloadButton("📥 Download Results", zip_results)

            stats_table        = gr.Dataframe(
                                     headers=["predictor","coefficient"],
                                     label="📊 Model Statistics",
                                     interactive=False
                                 )
            stats_download     = gr.DownloadButton("📥 Download Stats")

        with gr.Column(scale=3):
            map_output    = gr.HTML(value=create_map(), label="🗺️ Map Preview")
            status_output = gr.Textbox(label="Status", interactive=False)

    # wire up your existing callbacks
    upload_input.change(
        fn=lambda f: (
            create_map(), "⚠️ No file uploaded."
        ) if not (f and hasattr(f,"name")) else (
            shutil.rmtree("predictor_rasters", ignore_errors=True),
            shutil.rmtree("outputs", ignore_errors=True),
            shutil.rmtree("inputs", ignore_errors=True),
            os.makedirs("inputs", exist_ok=True),
            shutil.copy(f.name, "inputs/presence_points.csv"),
            create_map(), "✅ Presence points uploaded!"
        )[-2:],
        inputs=[upload_input],
        outputs=[map_output, status_output]
    )
    fetch_button.click(run_fetch,
                       inputs=[layer_selector, landcover_selector],
                       outputs=[map_output, status_output])
    run_button.click(run_model,
                     outputs=[map_output, status_output, stats_table, stats_download])
    download_button.click(zip_results, outputs=[download_button])

    # === NEW: LLM chat pane underneath ===
    gr.Markdown("## 💬 SpatChat Assistant")
    chat = gr.Chatbot(label="SpatChat Dialog", type="messages")
    user_in = gr.Textbox(placeholder="Ask me anything…")
    user_btn = gr.Button("Send")
    user_btn.click(analyze_chat, inputs=[user_in, chat], outputs=chat)
    user_in.submit(analyze_chat, inputs=[user_in, chat], outputs=chat)

    demo.launch()
