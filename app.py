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
from together import Together
from dotenv import load_dotenv

# --- Authenticate Earth Engine ---
svc = json.loads(os.environ["GEE_SERVICE_ACCOUNT"])
creds = ee.ServiceAccountCredentials(
    svc["client_email"], key_data=json.dumps(svc)
)
ee.Initialize(creds)

# --- LLM client ---
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# --- Clean up last session ---
for d in ("predictor_rasters","outputs","inputs"):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs("inputs", exist_ok=True)

LAYERS = [f"bio{i}" for i in range(1,20)] + ["elevation","slope","aspect","ndvi","landcover"]

# --- Pre-render colorbar ---
fig, ax = plt.subplots(figsize=(4,0.5))
norm = Normalize(vmin=0, vmax=1)
cbar = plt.colorbar(ScalarMappable(norm=norm, cmap="viridis"),
                    cax=ax, orientation="horizontal")
cbar.set_ticks([])
ax.set_xlabel("Low    High")
fig.tight_layout(pad=0)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100)
plt.close(fig)
buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(buf.read()).decode()

# --- Landcover choices ---
landcover_options = {
    0: "water", 1: "evergreen needleleaf forest", 2: "evergreen broadleaf forest",
    3: "deciduous needleleaf forest", 4: "deciduous broadleaf forest", 5: "mixed forest",
    6: "closed shrublands", 7: "open shrublands", 8: "woody savannas", 9: "savannas",
    10: "grasslands", 11: "permanent wetlands", 12: "croplands",
    13: "urban and built up", 14: "cropland/natural vegetation mosaic",
    15: "snow and ice", 16: "barren or sparsely vegetated"
}
landcover_choices = [f"{k} – {v}" for k,v in landcover_options.items()]

# --- System prompt for LLM ---
SYSTEM_PROMPT = """
You are SpatChat, a friendly assistant that orchestrates species distribution modeling:
1) After the user has uploaded presence points, if the user says "fetch …”, trigger our run_fetch() and report the output. Then, prompt the user to run model.
2) After the user has fetched the layers, if the user says “run model”, trigger run_model() and display its stats.
3) After the user has run the model, ask if the user wants to download the results. if the user says “download” or “yes”, trigger zip_results() and offer the ZIP.
Be conversational and guide the user at each step.
""".strip()

def create_map():
    m = folium.Map(location=[0,0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)
    # presence points
    ppath="inputs/presence_points.csv"
    if os.path.exists(ppath):
        df=pd.read_csv(ppath)
        # allow varied column names
        lat_col = next((c for c in df.columns if c.lower() in ("latitude","decimallatitude","y")),None)
        lon_col = next((c for c in df.columns if c.lower() in ("longitude","decimallongitude","x")),None)
        if lat_col and lon_col:
            pts=df[[lat_col,lon_col]].values.tolist()
            fg=folium.FeatureGroup(name="🟦 Presence Points")
            for lat,lon in pts:
                folium.CircleMarker([lat,lon],radius=4,
                    color="blue",fill=True,fill_opacity=0.8
                ).add_to(fg)
            fg.add_to(m)
            if pts: m.fit_bounds(pts)
    # predictor rasters
    rasdir="predictor_rasters/wgs84"
    if os.path.isdir(rasdir):
        for fn in sorted(os.listdir(rasdir)):
            if not fn.endswith(".tif"): continue
            with rasterio.open(os.path.join(rasdir,fn)) as src:
                img, b = src.read(1), src.bounds
            vmin, vmax = np.nanmin(img), np.nanmax(img)
            if np.isnan(vmin) or vmin==vmax: continue
            rgba = colormaps["viridis"]((img-vmin)/(vmax-vmin))
            folium.raster_layers.ImageOverlay(
                rgba, [[b.bottom,b.left],[b.top,b.right]],
                opacity=1.0, name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})"
            ).add_to(m)
    # suitability
    sf="outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        with rasterio.open(sf) as src:
            img, b = src.read(1), src.bounds
        rgba = colormaps["viridis"]((img-np.nanmin(img))/(np.nanmax(img)-np.nanmin(img)))
        folium.raster_layers.ImageOverlay(
            rgba, [[b.bottom,b.left],[b.top,b.right]],
            opacity=0.7, name="🎯 Suitability"
        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    # colorbar
    html = (
      f'<img src="data:image/png;base64,{COLORBAR_BASE64}" '
      'style="position:absolute; bottom:20px; right:10px; '
      'width:200px; height:30px; z-index:1000;" />'
    )
    m.get_root().html.add_child(Element(html))
    rendered = m.get_root().render()
    return f'<iframe srcdoc="{html_lib.escape(rendered)}" style="width:100%; height:600px; border:none;"></iframe>'

def run_fetch(sl, lc):
    if not sl and not lc:
        return create_map(), "⚠️ Select at least one predictor."
    layers = list(sl)
    if lc: layers.append("landcover")
    os.environ['SELECTED_LAYERS']            = ",".join(layers)
    os.environ['SELECTED_LANDCOVER_CLASSES'] = ",".join(c.split(" – ")[0] for c in lc)
    res = subprocess.run(["python","scripts/fetch_predictors.py"],
                         capture_output=True, text=True)
    ok = (res.returncode==0)
    msg = "✅ Predictors fetched." if ok else f"❌ Fetch failed:\n{res.stderr}"
    return create_map(), msg

def run_model():
    res = subprocess.run(["python","scripts/run_logistic_sdm.py"],
                         capture_output=True, text=True)
    if res.returncode!=0:
        return create_map(), f"❌ Model run failed:\n{res.stderr}", None, None
    stats_df = pd.read_csv("outputs/model_stats.csv")
    return create_map(), "✅ Model ran successfully!", stats_df, "outputs/model_stats.csv"

def zip_results():
    z="spatchat_results.zip"
    if os.path.exists(z): os.remove(z)
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters","outputs"):
            for r,_,fs in os.walk(fld):
                for f in fs:
                    full=os.path.join(r,f)
                    zf.write(full, arcname=os.path.relpath(full,"."))
    return z

# unified chat handler
def chat_step(f, msg, history, state):
    stage = state.get("stage","await_upload")
    cmd   = msg.strip().lower()

    # decide which op to run
    if stage=="await_fetch" and cmd.startswith(("fetch","get","use")):
        m_out, status = run_fetch([],[])  # we ignore sl, lc: LLM will not pass them
        op_out = status
        next_stage="await_run"

    elif stage=="await_run" and "run model" in cmd:
        m_out, status, stats_df, _ = run_model()
        op_out = status + ("\n\n" + stats_df.to_markdown(index=False) if stats_df is not None else "")
        next_stage="await_download"

    elif stage=="await_download" and cmd.startswith(("download","yes","y")):
        zipf = zip_results()
        op_out = f"✅ Here is your ZIP: {zipf}"
        next_stage="done"

    else:
        # fallback prompt
        if stage=="await_upload":
            op_out = "Please upload your presence-points CSV to begin."
            next_stage="await_upload"
        elif stage=="await_fetch":
            op_out = "Say “fetch …” to download your chosen layers."
            next_stage="await_fetch"
        elif stage=="await_run":
            op_out = "Say “run model” to train the SDM."
            next_stage="await_run"
        else:
            op_out = "Session complete. Upload a new CSV to restart."
            next_stage="await_upload"

    # hand off to LLM to craft a friendly reply
    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    for u,a in history:
        messages.append({"role":"user","content":u})
        messages.append({"role":"assistant","content":a})
    messages.append({"role":"user","content":msg})
    messages.append({"role":"system","content":op_out})

    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages, temperature=0.3
    ).choices[0].message.content

    history.append((msg, resp))
    return history, m_out, zipf if next_stage=="done" else None, {"stage":next_stage}

# Integrate upload → greeting continues
def on_upload(f, history):
    if not f or not hasattr(f,"name"):
        return history, create_map(), None, {"stage":"await_upload"}
    shutil.copy(f.name,"inputs/presence_points.csv")
    history += [("", "✅ Uploaded! You can now say “fetch elevation, ndvi, bio1”, etc.")]
    return history, create_map(), None, {"stage":"await_fetch"}

# --- GRADIO UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 SpatChat SDM – Chat‑Driven Species Distribution Modeling")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📄 Upload Presence CSV", type="filepath")
        with gr.Column(scale=3):
            map_out = gr.HTML(value=create_map(), label="🗺️ Map Preview")
            chat = gr.Chatbot(
                label = "SpatChat Dialog",
                type = "messages",
                value = [{"role":"assistant","content":"👋 Hello! Welcome to SpatChat. Please upload your presence-points CSV to begin."}]
            )

            user_in = gr.Textbox(placeholder="Type commands…", label="")
            send_btn = gr.Button("Send")
            download_blk = gr.File(label="Download Results", visible=False)
            state = gr.State()

            file_input.change(on_upload,
                      inputs=[file_input, chat],
                      outputs=[chat, map_out, download_blk, gr.State()])

    for trigger in (send_btn, user_in.submit()):
        trigger.click(chat_step,
                  inputs=[file_input, user_in, chat, state],
                  outputs=[chat, map_out, download_blk, state])
        trigger.click(lambda: "", None, user_in)

    demo.launch()
