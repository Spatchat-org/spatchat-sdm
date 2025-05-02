import os
import io
import re
import json
import base64
import shutil
import subprocess
import zipfile

import gradio as gr
import folium
import html as html_lib
import pandas as pd
import numpy as np
import rasterio
import ee

from matplotlib import pyplot as plt, colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from folium import Element

from dotenv import load_dotenv
from together import Together

# --- INITIALIZE ---
load_dotenv()

# Earth Engine
svc   = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
creds = ee.ServiceAccountCredentials(svc['client_email'], key_data=json.dumps(svc))
ee.Initialize(creds)

# Together LLM
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# Clean workspace
for d in ("predictor_rasters","outputs","inputs"):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs("inputs", exist_ok=True)

LAYERS = [f"bio{i}" for i in range(1,20)] + ["elevation","slope","aspect","ndvi","landcover"]

# --- PRE‑RENDER COLORBAR ---
fig, ax = plt.subplots(figsize=(4,0.5))
norm = Normalize(0,1)
plt.colorbar(ScalarMappable(norm=norm,cmap="viridis"),
             cax=ax, orientation="horizontal")
ax.set_xticks([]); ax.set_xlabel("Low    High")
fig.tight_layout(pad=0)
buf = io.BytesIO()
fig.savefig(buf,format="png",dpi=100)
plt.close(fig)
buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(buf.read()).decode()

def create_map():
    m = folium.Map(location=[0,0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)

    # Presence points
    ppath = "inputs/presence_points.csv"
    if os.path.exists(ppath):
        df = pd.read_csv(ppath)
        # smart detect lat/lon columns
        lat_candidates = [c for c in df.columns if c.lower() in
                          ("latitude","lat","y","decimallatitude")]
        lon_candidates = [c for c in df.columns if c.lower() in
                          ("longitude","lon","x","decimallongitude")]
        if lat_candidates and lon_candidates:
            lat_col = lat_candidates[0]
            lon_col = lon_candidates[0]
            pts = df[[lat_col, lon_col]].values.tolist()
            fg = folium.FeatureGroup(name="🟦 Presence Points")
            for lat, lon in pts:
                folium.CircleMarker([lat, lon],
                                    radius=4,
                                    color="blue",
                                    fill=True,
                                    fill_opacity=0.8
                ).add_to(fg)
            fg.add_to(m)
            if pts: m.fit_bounds(pts)

    # Predictor rasters
    rasdir = "predictor_rasters/wgs84"
    if os.path.isdir(rasdir):
        for fn in sorted(os.listdir(rasdir)):
            if not fn.endswith(".tif"): continue
            with rasterio.open(os.path.join(rasdir,fn)) as src:
                img = src.read(1); b = src.bounds
            vmin, vmax = np.nanmin(img), np.nanmax(img)
            if np.isnan(vmin) or vmin==vmax: continue
            rgba = colormaps["viridis"]((img - vmin)/(vmax-vmin))
            folium.raster_layers.ImageOverlay(
                image=rgba,
                bounds=[[b.bottom, b.left],[b.top, b.right]],
                opacity=1.0,
                name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})"
            ).add_to(m)

    # Suitability
    sf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        with rasterio.open(sf) as src:
            img = src.read(1); b = src.bounds
        vmin, vmax = np.nanmin(img), np.nanmax(img)
        rgba = colormaps["viridis"]((img - vmin)/(vmax-vmin))
        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[b.bottom, b.left],[b.top, b.right]],
            opacity=0.7,
            name="🎯 Suitability"
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # embed static colorbar
    img_html = (
        f'<img src="data:image/png;base64,{COLORBAR_BASE64}" '
        'style="position:absolute; bottom:20px; right:10px; '
        'width:200px; height:30px; z-index:1000;" />'
    )
    m.get_root().html.add_child(Element(img_html))

    rendered = m.get_root().render()
    return f'<iframe srcdoc="{html_lib.escape(rendered)}" '\
           'style="width:100%; height:600px; border:none;"></iframe>'

# --- ZIP ---
def zip_results():
    z = "spatchat_results.zip"
    if os.path.exists(z): os.remove(z)
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters","outputs"):
            for r,_,fs in os.walk(fld):
                for f in fs:
                    full = os.path.join(r,f)
                    zf.write(full,arcname=os.path.relpath(full,"."))
    return z

SYSTEM_PROMPT = """
You are SpatChat, a friendly SDM assistant.
• “fetch …” runs fetch_predictors.py with the named layers.
• “run model” runs run_logistic_sdm.py & shows stats.
• “download” bundles everything into a ZIP.
Guide the user step by step.
""".strip()

# --- UPLOAD handler ---
def preview_upload(file):
    if not file or not hasattr(file,"name"):
        return (
            [{"role":"assistant","content":"👋 Please upload a presence CSV to begin."}],
            create_map(),
            gr.update(visible=False),
            {"stage":"await_upload"}
        )
    shutil.copy(file.name, "inputs/presence_points.csv")
    intro = (
        "✅ Got your points! Which predictors shall I fetch?\n"
        "Available: **bio1–bio19** "
        "(https://www.worldclim.org/data/bioclim.html), elevation, slope, aspect, ndvi, "
        "and landcover (e.g., water, urban, forest, cropland).\n"
        "For example: “fetch elevation, ndvi, bio1”"
    )
    return (
        [{"role":"assistant","content":intro}],
        create_map(),
        gr.update(visible=False),
        {"stage":"await_layers"}
    )

# --- CHAT handler ---
def analyze_sdm(file, user_msg, history, state):
    stage = state.get("stage","await_layers")
    hist  = history[:]  # list of dicts
    cmd   = user_msg.strip()
    cmd_lower = cmd.lower()
    info  = ""
    download_link = None

    # record user turn
    hist.append({"role":"user","content":cmd})

    # FETCH
    if stage=="await_layers" and re.search(r"\b(fetch|get|use)\b", cmd_lower):
        # parse which layers
        picked = [l for l in LAYERS if re.search(rf"\b{re.escape(l)}\b", cmd_lower)]
        if not picked:
            info = "❗ I didn't catch any valid layer names. Try “fetch bio1, ndvi” etc."
            # stay in await_layers
        else:
            os.environ['SELECTED_LAYERS'] = ",".join(picked)
            proc = subprocess.run(
                ["python","scripts/fetch_predictors.py"],
                capture_output=True, text=True
            )
            info = f"```bash\n{proc.stdout}{proc.stderr}```"
            stage = "await_run"

    # RUN MODEL
    elif stage=="await_run" and re.search(r"\b(run|train|create)\b.*model", cmd_lower):
        proc = subprocess.run(
            ["python","scripts/run_logistic_sdm.py"],
            capture_output=True, text=True
        )
        logs = proc.stdout + proc.stderr
        stats_md = ""
        stats_path = "outputs/model_stats.csv"
        if os.path.exists(stats_path):
            df = pd.read_csv(stats_path)
            stats_md = "\n**Model stats**:\n\n" + df.to_markdown(index=False)
        info = f"```bash\n{logs}```{stats_md}"
        stage = "await_download"

    # DOWNLOAD
    elif stage=="await_download" and re.search(r"\b(download|yes|y)\b", cmd_lower):
        download_link = zip_results()
        info = "📦 Here is your ZIP of predictors + outputs!"
        stage = "done"

    else:
        prompts = {
            "await_layers":   'Please say “fetch …” to fetch predictors.',
            "await_run":      'Please say “run model” to train the model.',
            "await_download": 'Please say “download” to get your ZIP.',
            "done":           'Session complete. Upload a new CSV to start over.'
        }
        info = prompts.get(stage, prompts["done"])
        if stage=="done":
            stage="await_upload"

    # record assistant turn
    hist.append({"role":"assistant","content":info})

    # LLM polish (optional—comment out if you prefer raw info)
    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    for msg in hist:
        messages.append(msg)
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        temperature=0.3
    ).choices[0].message.content
    # replace last assistant entry with polished
    hist[-1] = {"role":"assistant","content":resp}

    # prepare download file visibility/value
    dl_update = gr.update(visible=bool(download_link), value=download_link or "")
    return hist, create_map(), dl_update, {"stage":stage}


# --- GRADIO UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 SpatChat SDM – Chat‑Driven Modeling")

    with gr.Row():
        with gr.Column(scale=1):
            file_input    = gr.File(label="📄 Upload Presence CSV", type="filepath")

        with gr.Column(scale=3):
            map_out       = gr.HTML(create_map, label="🗺️ Map Preview")
            chat          = gr.Chatbot(
                                label="SpatChat Dialog",
                                type="messages",
                                value=[{"role":"assistant","content":"👋 Hello! Upload a CSV to begin."}]
                            )
            user_in       = gr.Textbox(placeholder="Type here…", label="")
            send_btn      = gr.Button("Send")
            download_file = gr.File(label="Download Results", visible=False)
            state         = gr.State({"stage":"await_upload"})

    # events
    file_input.change(
        preview_upload,
        inputs=[file_input],
        outputs=[chat, map_out, download_file, state]
    )

    send_btn.click(
        analyze_sdm,
        inputs=[file_input, user_in, chat, state],
        outputs=[chat, map_out, download_file, state]
    )
    send_btn.click(lambda: "", None, user_in)

    user_in.submit(
        analyze_sdm,
        inputs=[file_input, user_in, chat, state],
        outputs=[chat, map_out, download_file, state]
    )
    user_in.submit(lambda: "", None, user_in)

    demo.launch()
