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

# --- tool registry ---
TOOLS = {
    "fetch": lambda args: run_fetch(args.get("layers", []), args.get("landcover", [])),
    "run_model": lambda args: run_model()[:2],       # map, status
    "download": lambda args: (create_map(), zip_results())
}

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

# --- System prompt for tool‑picking LLM ---
SYSTEM_PROMPT = """
You are SpatChat, a friendly assistant orchestrating SDM.
Your job is to explain to the user what options they have in each step, 
guiding them through the whole process to build the SDM.
Whenever the user wants to perform an action, reply _only_ with a JSON object selecting one of your tools:
- To fetch layers:     {"tool":"fetch","layers":["bio1","ndvi",...]}
- To run the model:    {"tool":"run_model"}
- To download results: {"tool":"download"}
After we run that function, we'll display its output and then prompt the user on next steps.
If the user asks for stats, show them from stats_df.
If the question is vague, ask for clarification.
""".strip()

# --- Normal‑chat fallback prompt ---
FALLBACK_PROMPT = """
You are SpatChat, a friendly assistant for species distribution modeling.
Answer the user's question conversationally.
""".strip()

def create_map():
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)

    # Presence points
    ppath = "inputs/presence_points.csv"
    if os.path.exists(ppath):
        df = pd.read_csv(ppath)
        lat = next((c for c in df.columns if c.lower() in ("latitude","decimallatitude","y")), None)
        lon = next((c for c in df.columns if c.lower() in ("longitude","decimallongitude","x")), None)
        if lat and lon:
            pts = df[[lat, lon]].values.tolist()
            fg = folium.FeatureGroup(name="🟦 Presence Points")
            for la, lo in pts:
                folium.CircleMarker([la, lo], radius=4, color="blue", fill=True, fill_opacity=0.8).add_to(fg)
            fg.add_to(m)
            if pts: m.fit_bounds(pts)

    # Predictor rasters
    rasdir = "predictor_rasters/wgs84"
    if os.path.isdir(rasdir):
        for fn in sorted(os.listdir(rasdir)):
            if not fn.endswith(".tif"): continue
            path = os.path.join(rasdir, fn)
            with rasterio.open(path) as src:
                img = src.read(1); b = src.bounds
            vmin, vmax = np.nanmin(img), np.nanmax(img)
            if np.isnan(vmin) or vmin==vmax: continue
            rgba = colormaps["viridis"]((img - vmin)/(vmax-vmin))
            folium.raster_layers.ImageOverlay(
                rgba, [[b.bottom,b.left],[b.top,b.right]],
                opacity=1.0, name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})"
            ).add_to(m)

    # Suitability map
    sf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        with rasterio.open(sf) as src:
            img = src.read(1); b = src.bounds
        rgba = colormaps["viridis"]((img-np.nanmin(img))/(np.nanmax(img)-np.nanmin(img)))
        folium.raster_layers.ImageOverlay(
            rgba, [[b.bottom,b.left],[b.top,b.right]],
            opacity=0.7, name="🎯 Suitability"
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Static colorbar
    img_html = (
        f'<img src="data:image/png;base64,{COLORBAR_BASE64}" '
        'style="position:absolute; bottom:20px; right:10px; '
        'width:200px; height:30px; z-index:1000;" />'
    )
    m.get_root().html.add_child(Element(img_html))

    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" style="width:100%; height:600px; border:none;"></iframe>'

def run_fetch(sl, lc):
    if not sl and not lc:
        return create_map(), "⚠️ Select at least one predictor."
    layers = list(sl)
    if lc: layers.append("landcover")
    os.environ["SELECTED_LAYERS"] = ",".join(layers)
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(c.split(" – ")[0] for c in lc)
    proc = subprocess.run(["python","scripts/fetch_predictors.py"], capture_output=True, text=True)
    ok = proc.returncode==0
    return create_map(), ("✅ Predictors fetched." if ok else f"❌ Fetch failed:\n{proc.stderr}")

def run_model():
    proc = subprocess.run(["python","scripts/run_logistic_sdm.py"], capture_output=True, text=True)
    if proc.returncode!=0:
        return create_map(), f"❌ Model run failed:\n{proc.stderr}", None, None
    stats_df = pd.read_csv("outputs/model_stats.csv")
    return create_map(), "✅ Model ran successfully!", stats_df, "outputs/model_stats.csv"

def zip_results():
    zipf = "spatchat_results.zip"
    if os.path.exists(zipf): os.remove(zipf)
    with zipfile.ZipFile(zipf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters","outputs"):
            for root,_,files in os.walk(fld):
                for f in files:
                    full = os.path.join(root,f)
                    zf.write(full, arcname=os.path.relpath(full,"."))
    return zipf

def chat_step(file, user_msg, history, state):
    # 1) Tool‐picking prompt
    tool_msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":user_msg}]
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=tool_msgs, temperature=0.0
    ).choices[0].message.content

    # 2) Try JSON parse
    try:
        call = json.loads(resp)
        tool = call["tool"]
        m_out, status = TOOLS[tool](call)
        # 2a) fetch
        if tool=="fetch":
            assistant_txt = f"{status}\n\nGreat! You can “run model” when ready."
            download_path = None
        # 2b) run_model
        elif tool=="run_model":
            # auto‐zip immediately
            zip_results()
            assistant_txt = "✅ Model ran successfully! Results are ready for download using the Download Button!"
            download_path = None
        # 2c) download via chat JSON (rare now)
        else:  # download
            m_out, _ = (create_map(), zip_results())
            assistant_txt = (
                "✅ Download bundle:<br>"
                f"<a id='dl' href='{zip_results()}' download style='display:none;'></a>"
                "<script>document.getElementById('dl').click();</script>"
            )
            download_path = None

    except Exception:
        # 3) Fallback to normal LLM
        fb_msgs = [{"role":"system","content":FALLBACK_PROMPT},
                   {"role":"user","content":user_msg}]
        assistant_txt = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=fb_msgs, temperature=0.7
        ).choices[0].message.content
        m_out = history[-1].get("map", create_map()) if history else create_map()
        download_path = None

    # 4) Record
    history.append({"role":"user","content":user_msg})
    history.append({"role":"assistant","content":assistant_txt})

    return history, m_out, download_path, state

def on_upload(f, history):
    new_history = history.copy()
    if not f or not hasattr(f,"name"):
        return new_history, create_map(), None, {"stage":"await_upload"}
    shutil.copy(f.name, "inputs/presence_points.csv")

    extras = LAYERS[19:-1]
    last = LAYERS[-1]
    layers_str = f"bio1–bio19, {', '.join(extras)}, and {last}"
    new_history.append({"role":"assistant","content":
        "✅ Uploaded! Available layers are:\n\n"
        f"{layers_str}\n\n"
        "Now say “fetch elevation, ndvi, bio1” to grab those layers."
    })
    return new_history, create_map(), None, {"stage":"await_fetch"}

def on_download_click(chat_history, state):
    new_history = chat_history.copy()
    zipf = "spatchat_results.zip"
    if os.path.exists(zipf):
        zip_results()
        assistant_txt = (
            "✅ Download starting...<br>"
            f"<a id='dl' href='{zipf}' download style='display:none;'></a>"
            "<script>document.getElementById('dl').click();</script>"
        )
    else:
        assistant_txt = "⚠️ Results not ready yet. Please run the model first."
    new_history.append({"role":"assistant","content":assistant_txt})
    return new_history, gr.update(visible=False), state

# --- Build UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 SpatChat SDM – Chat‑Driven SDM")

    state = gr.State({"stage":"await_upload"})

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📄 Upload Presence CSV", type="filepath")
        with gr.Column(scale=3):
            map_out     = gr.HTML(value=create_map(), label="🗺️ Map Preview")
            chat        = gr.Chatbot(
                label="SpatChat Dialog", type="messages",
                value=[{"role":"assistant","content":
                        "👋 Hello! Welcome to SpatChat. Please upload your presence‑points CSV to begin."}]
            )
            user_in     = gr.Textbox(placeholder="Type commands…", label="")
            send_btn    = gr.Button("Send")
            download_btn= gr.Button("Download Results")
            download_blk= gr.File(label="Download Results", visible=False)

    file_input.change(on_upload,
        inputs=[file_input, chat],
        outputs=[chat, map_out, download_blk, state]
    )

    send_btn.click(chat_step,
        inputs=[file_input, user_in, chat, state],
        outputs=[chat, map_out, download_blk, state]
    )
    send_btn.click(lambda: "", None, user_in)
    user_in.submit(chat_step,
        inputs=[file_input, user_in, chat, state],
        outputs=[chat, map_out, download_blk, state]
    )
    user_in.submit(lambda: "", None, user_in)

    download_btn.click(on_download_click,
        inputs=[chat, state],
        outputs=[chat, download_blk, state]
    )

    demo.launch()
