import os
import io
import json
import shutil
import subprocess
import zipfile
import re

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

# ─── 0) SETUP ──────────────────────────────────────────────────────
load_dotenv()

# Earth Engine auth
svc   = json.loads(os.environ["GEE_SERVICE_ACCOUNT"])
creds = ee.ServiceAccountCredentials(svc["client_email"], key_data=json.dumps(svc))
ee.Initialize(creds)

# Together LLM
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# clean workspace
for d in ("predictor_rasters","outputs","inputs"):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs("inputs", exist_ok=True)

# available layers
LAYERS = [f"bio{i}" for i in range(1,20)] + ["elevation","slope","aspect","ndvi","landcover"]

# ─── 1) PRE‑RENDER COLORBAR ────────────────────────────────────────
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

# ─── 2) MAP BUILDER ────────────────────────────────────────────────
def create_map():
    m = folium.Map(location=[0,0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)

    # presence
    csv = "inputs/presence_points.csv"
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        latc = [c for c in df.columns if c.lower() in ("latitude","lat","y","decimallatitude")]
        lonc = [c for c in df.columns if c.lower() in ("longitude","lon","x","decimallongitude")]
        if latc and lonc:
            pts = df[[latc[0],lonc[0]]].values.tolist()
            fg = folium.FeatureGroup(name="🟦 Presence")
            for la,lo in pts:
                folium.CircleMarker([la,lo], radius=4, color="blue", fill=True, fill_opacity=0.8).add_to(fg)
            m.add_child(fg)
            if pts: m.fit_bounds(pts)

    # predictors
    for fn in sorted(os.listdir("predictor_rasters/wgs84") if os.path.isdir("predictor_rasters/wgs84") else []):
        if not fn.endswith(".tif"): continue
        with rasterio.open(f"predictor_rasters/wgs84/{fn}") as src:
            img = src.read(1); b=src.bounds
        vmin,vmax = np.nanmin(img),np.nanmax(img)
        if np.isnan(vmin) or vmin==vmax: continue
        rgba = colormaps["viridis"]((img-vmin)/(vmax-vmin))
        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[b.bottom,b.left],[b.top,b.right]],
            opacity=1.0,name=f"🟨 {fn}"
        ).add_to(m)

    # suitability
    sf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        with rasterio.open(sf) as src:
            img=src.read(1); b=src.bounds
        vmin,vmax=np.nanmin(img),np.nanmax(img)
        rgba = colormaps["viridis"]((img-vmin)/(vmax-vmin))
        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[b.bottom,b.left],[b.top,b.right]],
            opacity=0.7,name="🎯 Suitability"
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # embed colorbar
    m.get_root().html.add_child(Element(
        f'<img src="data:image/png;base64,{COLORBAR_BASE64}" '
        'style="position:absolute; bottom:20px; right:10px; width:200px; height:30px; z-index:1000;"/>'
    ))

    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" '\
           'style="width:100%; height:600px; border:none;"></iframe>'

# ─── 3) ZIP RESULTS ────────────────────────────────────────────────
def zip_results():
    z="spatchat_results.zip"
    if os.path.exists(z): os.remove(z)
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters","outputs"):
            for r,_,fs in os.walk(fld):
                for f in fs:
                    full = os.path.join(r,f)
                    zf.write(full,arcname=os.path.relpath(full,"."))
    return z

# ─── 4) UPLOAD HANDLER ─────────────────────────────────────────────
def preview_upload(file):
    if not file or not hasattr(file,"name"):
        return (
            [{"role":"assistant","content":"👋 Please upload a presence CSV to begin."}],
            create_map(), False, {"stage":"await_upload"}
        )
    # copy and reset
    for d in ("predictor_rasters","outputs","inputs"):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs("inputs", exist_ok=True)
    shutil.copy(file.name, "inputs/presence_points.csv")

    intro = (
        "✅ Points received!  Which predictors shall I fetch?\n\n"
        f"Available: **bio1–bio19** (https://www.worldclim.org/data/bioclim.html), "
        "elevation, slope, aspect, ndvi, landcover.\n\n"
        "E.g.: “fetch elevation, ndvi, bio1”"
    )
    return ([{"role":"assistant","content":intro}],
            create_map(), False, {"stage":"await_layers"})

# ─── 5) CHAT HANDLER ──────────────────────────────────────────────
SYSTEM = """
You are SpatChat, an assistant that wraps:
• fetch → runs scripts/fetch_predictors.py  
• run model → runs scripts/run_logistic_sdm.py and shows stats  
• download → bundles and provides a ZIP link  

Guide the user step by step. If unclear, ask for clarification.
""".strip()

def analyze_sdm(file, user_msg, history, state):
    stage = state.get("stage","await_layers")
    hist  = history[:] 
    cmd   = user_msg.strip()
    out   = ""
    download_link = None

    hist.append({"role":"user","content":cmd})

    # FETCH
    if stage=="await_layers" and re.search(r"\b(fetch|get|use)\b", cmd, re.I):
        picks = [L for L in LAYERS if re.search(rf"\b{L}\b", cmd, re.I)]
        if not picks:
            out = "❗ No valid layers found. Try e.g. “fetch bio1, ndvi”"
        else:
            os.environ["SELECTED_LAYERS"] = ",".join(picks)
            proc = subprocess.run(
                ["python","scripts/fetch_predictors.py"],
                capture_output=True, text=True
            )
            out = f"```bash\n{proc.stdout}{proc.stderr}\n```"
            stage = "await_run"

    # RUN MODEL
    elif stage=="await_run" and re.search(r"\b(run|train|create)\b.*model", cmd, re.I):
        proc = subprocess.run(
            ["python","scripts/run_logistic_sdm.py"],
            capture_output=True, text=True
        )
        logs = proc.stdout + proc.stderr
        if os.path.exists("outputs/model_stats.csv"):
            df = pd.read_csv("outputs/model_stats.csv")
            md = df.to_markdown(index=False)
            out = f"```bash\n{logs}```\n**Model stats**:\n\n{md}"
        else:
            out = f"```bash\n{logs}```"
        stage="await_download"

    # DOWNLOAD
    elif stage=="await_download" and re.search(r"\b(download|yes|y)\b", cmd, re.I):
        download_link = zip_results()
        out = "📦 Here’s your ZIP. Click below to download!"
        stage="done"

    # FALLBACK
    else:
        prompts = {
            "await_layers":   "Please say “fetch …” to fetch predictors.",
            "await_run":      "Please say “run model” to train the SDM.",
            "await_download": "Please say “download” to bundle everything."
        }
        out = prompts.get(stage, "Session over. Upload a new CSV to start again.")
        if stage=="done":
            stage="await_upload"

    hist.append({"role":"assistant","content":out})

    # final LLM polish
    msgs = [{"role":"system","content":SYSTEM}] + hist
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=msgs,
        temperature=0.3
    ).choices[0].message.content
    hist[-1] = {"role":"assistant","content":resp}

    return (
        hist,
        create_map(),
        download_link,
        {"stage":stage}
    )

# ─── 6) GRADIO UI ─────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 SpatChat SDM – Chat‑Driven Interface")

    with gr.Row():
        with gr.Column(scale=1):
            file_input    = gr.File(label="📄 Upload Presence CSV", type="filepath")

        with gr.Column(scale=3):
            map_out       = gr.HTML(create_map, label="🗺️ Map Preview")
            chatbox       = gr.Chatbot(type="messages", label="SpatChat Dialog",
                                      value=[{"role":"assistant","content":"👋 Hello! Upload a CSV to begin."}])
            user_in       = gr.Textbox(placeholder="e.g. “fetch elevation, ndvi”", label="")
            send_btn      = gr.Button("Send")
            download_file = gr.File(label="Download ZIP", visible=False)
            state         = gr.State({"stage":"await_upload"})

    # events
    file_input.change(
        preview_upload,
        inputs=[file_input],
        outputs=[chatbox, map_out, download_file, state]
    )

    for trigger in (send_btn, user_in.submit):
        trigger.click(
            analyze_sdm,
            inputs=[file_input, user_in, chatbox, state],
            outputs=[chatbox, map_out, download_file, state]
        )
        trigger.click(lambda: "", None, user_in)

    demo.launch()
