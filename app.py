import os
import io
import json
import base64
import shutil
import subprocess
import zipfile
import re

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
for d in ("predictor_rasters", "outputs", "inputs"):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs("inputs", exist_ok=True)

LAYERS = [f"bio{i}" for i in range(1, 20)] + ["elevation", "slope", "aspect", "ndvi", "landcover"]

# --- Pre-render colorbar ---
fig, ax = plt.subplots(figsize=(4, 0.5))
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
landcover_choices = [f"{k} – {v}" for k, v in landcover_options.items()]

# --- System prompt for the LLM ---
SYSTEM_PROMPT = """
You are SpatChat, a friendly assistant that orchestrates species distribution modeling:
1) After the user uploads points, if they say "fetch …", run run_fetch() and report back.
2) After layers are fetched, if they say "run model", run run_model() and display its stats.
3) After the model, if they say "download" or "yes", run zip_results() and give them the ZIP link.
Guide them through each step conversationally.
""".strip()


def create_map():
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)

    # Presence points (auto-detect lat/lon columns)
    ppath = "inputs/presence_points.csv"
    if os.path.exists(ppath):
        df = pd.read_csv(ppath)
        lat_col = next((c for c in df.columns if c.lower() in ("latitude", "decimallatitude", "y")), None)
        lon_col = next((c for c in df.columns if c.lower() in ("longitude", "decimallongitude", "x")), None)
        if lat_col and lon_col:
            pts = df[[lat_col, lon_col]].values.tolist()
            fg = folium.FeatureGroup(name="🟦 Presence Points")
            for lat, lon in pts:
                folium.CircleMarker([lat, lon], radius=4,
                                    color="blue", fill=True, fill_opacity=0.8
                                    ).add_to(fg)
            fg.add_to(m)
            if pts:
                m.fit_bounds(pts)

    # Predictor rasters (Viridis)
    rasdir = "predictor_rasters/wgs84"
    if os.path.isdir(rasdir):
        for fn in sorted(os.listdir(rasdir)):
            if not fn.endswith(".tif"):
                continue
            path = os.path.join(rasdir, fn)
            with rasterio.open(path) as src:
                img = src.read(1)
                b = src.bounds
            vmin, vmax = np.nanmin(img), np.nanmax(img)
            if np.isnan(vmin) or vmin == vmax:
                continue
            rgba = colormaps["viridis"]((img - vmin) / (vmax - vmin))
            folium.raster_layers.ImageOverlay(
                rgba, [[b.bottom, b.left], [b.top, b.right]],
                opacity=1.0, name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})"
            ).add_to(m)

    # Suitability map (Viridis)
    sf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        with rasterio.open(sf) as src:
            img = src.read(1)
            b = src.bounds
        rgba = colormaps["viridis"]((img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img)))
        folium.raster_layers.ImageOverlay(
            rgba, [[b.bottom, b.left], [b.top, b.right]],
            opacity=0.7, name="🎯 Suitability"
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Static colorbar at bottom-right
    img_html = (
        f'<img src="data:image/png;base64,{COLORBAR_BASE64}" '
        'style="position:absolute; bottom:20px; right:10px; '
        'width:200px; height:30px; z-index:1000;" />'
    )
    m.get_root().html.add_child(Element(img_html))

    rendered = m.get_root().render()
    return f'<iframe srcdoc="{html_lib.escape(rendered)}" style="width:100%; height:600px; border:none;"></iframe>'


def run_fetch(sl, lc):
    if not sl and not lc:
        return create_map(), "⚠️ Select at least one predictor."
    layers = list(sl)
    if lc:
        layers.append("landcover")
    os.environ["SELECTED_LAYERS"] = ",".join(layers)
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(c.split(" – ")[0] for c in lc)
    proc = subprocess.run(["python", "scripts/fetch_predictors.py"], capture_output=True, text=True)
    ok = proc.returncode == 0
    msg = "✅ Predictors fetched." if ok else f"❌ Fetch failed:\n{proc.stderr}"
    return create_map(), msg


def run_model():
    proc = subprocess.run(["python", "scripts/run_logistic_sdm.py"], capture_output=True, text=True)
    if proc.returncode != 0:
        return create_map(), f"❌ Model run failed:\n{proc.stderr}", None, None
    stats_df = pd.read_csv("outputs/model_stats.csv")
    return create_map(), "✅ Model ran successfully!", stats_df, "outputs/model_stats.csv"


def zip_results():
    zipf = "spatchat_results.zip"
    if os.path.exists(zipf):
        os.remove(zipf)
    with zipfile.ZipFile(zipf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters", "outputs"):
            for root, _, files in os.walk(fld):
                for f in files:
                    full = os.path.join(root, f)
                    zf.write(full, arcname=os.path.relpath(full, "."))
    return zipf


def chat_step(f, msg, history, state):
    # ensure zipf always defined
    zipf = None
    stage = state.get("stage", "await_upload")
    cmd = msg.strip().lower()

    # 1) FETCH
    if stage == "await_fetch" and cmd.startswith(("fetch", "get", "use")):
        m_out, status = run_fetch([], [])
        op_out = status
        next_stage = "await_run"

    # 2) RUN MODEL
    elif stage == "await_run" and "run model" in cmd:
        m_out, status, stats_df, _ = run_model()
        suffix = ("\n\n" + stats_df.to_markdown(index=False)) if stats_df is not None else ""
        op_out = status + suffix
        next_stage = "await_download"

    # 3) DOWNLOAD
    elif stage == "await_download" and cmd.startswith(("download", "yes", "y")):
        zipf = zip_results()
        op_out = f"✅ Here is your ZIP: {zipf}"
        next_stage = "done"

    # FALLBACK
    else:
        if stage == "await_upload":
            op_out = "Please upload your presence-points CSV to begin."
            next_stage = "await_upload"
        elif stage == "await_fetch":
            op_out = 'Say “fetch …” to download your chosen layers.'
            next_stage = "await_fetch"
        elif stage == "await_run":
            op_out = 'Say “run model” to train the SDM.'
            next_stage = "await_run"
        else:
            op_out = "Session complete. Upload a new CSV to restart."
            next_stage = "await_upload"
        m_out = create_map()

    # Build LLM prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": msg})
    messages.append({"role": "system", "content": op_out})

    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        temperature=0.3
    ).choices[0].message.content

    # after you’ve computed `resp`…
    new_history = history.copy()
    new_history.append({ "role":"assistant", "content": resp })
    return new_history, m_out, download_path_if_any, {"stage": next_stage}

def on_upload(f, history):
    # history is a list of {role,content} dicts
    if not f or not hasattr(f, "name"):
        return history, create_map(), None, {"stage":"await_upload"}

    # copy in the CSV
    shutil.copy(f.name, "inputs/presence_points.csv")

    # make a brand‑new history list and append an assistant message
    new_history = history.copy()
    new_history.append({
        "role": "assistant",
        "content": "✅ Uploaded! You can now say “fetch elevation, ndvi, bio1”, etc."
    })
    return new_history, create_map(), None, {"stage":"await_fetch"}

# --- Build & launch UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 SpatChat SDM – Chat‑Driven Species Distribution Modeling")

    # Shared state
    state = gr.State({"stage": "await_upload"})

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📄 Upload Presence CSV", type="filepath")
        with gr.Column(scale=3):
            map_out     = gr.HTML(value=create_map(), label="🗺️ Map Preview")
            chat        = gr.Chatbot(
                             label="SpatChat Dialog",
                             type="messages",
                             value=[{"role":"assistant","content":"👋 Hello! Welcome to SpatChat. Please upload your presence‑points CSV to begin."}]
                         )
            user_in     = gr.Textbox(placeholder="Type commands…", label="")
            send_btn    = gr.Button("Send")
            download_blk= gr.File(label="Download Results", visible=False)

    # Wire up upload → on_upload
    file_input.change(
        on_upload,
        inputs=[file_input, chat],
        outputs=[chat, map_out, download_blk, state]
    )

    # When the user clicks “Send” …
    send_btn.click(
        chat_step,
        inputs=[file_input, user_in, chat, state],
        outputs=[chat, map_out, download_blk, state]
    )
    # … and clear the textbox
    send_btn.click(lambda: "", None, user_in)
    
    # When the user presses Enter in the textbox …
    user_in.submit(
        chat_step,
        inputs=[file_input, user_in, chat, state],
        outputs=[chat, map_out, download_blk, state]
    )
    # … and clear the textbox
    user_in.submit(lambda: "", None, user_in)

    demo.launch()
