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

# --- Utility to clear all data ---
def clear_all():
    for d in ("predictor_rasters", "outputs", "inputs"):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs("inputs", exist_ok=True)
    if os.path.exists("spatchat_results.zip"):
        os.remove("spatchat_results.zip")

# clear on startup
clear_all()

# --- Layers & Tools definitions ---
LAYERS = [f"bio{i}" for i in range(1, 20)] + ["elevation","slope","aspect","ndvi","landcover"]
TOOLS = {
    "fetch": lambda args: run_fetch(args.get("layers",[]), args.get("landcover",[])),
    "run_model": lambda args: run_model()[:2],
    "download": lambda args: (create_map(), zip_results())
}

# --- Pre-render colorbar → base64 ---
fig, ax = plt.subplots(figsize=(4,0.5))
norm = Normalize(vmin=0, vmax=1)
plt.colorbar(ScalarMappable(norm=norm,cmap="viridis"),
             cax=ax, orientation="horizontal").set_ticks([])
ax.set_xlabel("Low    High")
fig.tight_layout(pad=0)
buf = io.BytesIO(); fig.savefig(buf,format="png",dpi=100); plt.close(fig)
buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(buf.read()).decode()

# --- Landcover choices ---
landcover_options = {
    0:"water",1:"evergreen needleleaf forest",2:"evergreen broadleaf forest",
    3:"deciduous needleleaf forest",4:"deciduous broadleaf forest",5:"mixed forest",
    6:"closed shrublands",7:"open shrublands",8:"woody savannas",9:"savannas",
    10:"grasslands",11:"permanent wetlands",12:"croplands",
    13:"urban and built up",14:"cropland/natural vegetation mosaic",
    15:"snow and ice",16:"barren or sparsely vegetated"
}
landcover_choices = [f"{k} – {v}" for k,v in landcover_options.items()]

# --- LLM prompts ---
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

FALLBACK_PROMPT = """
You are SpatChat, a friendly assistant for species distribution modeling.
Answer the user's question conversationally.
""".strip()

def create_map():
    # Base map
    m = folium.Map(location=[0,0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)

    # 1) Presence points
    ppath = "inputs/presence_points.csv"
    if os.path.exists(ppath):
        df = pd.read_csv(ppath)
        lat_col = next((c for c in df.columns if c.lower() in ("latitude","decimallatitude","y")), None)
        lon_col = next((c for c in df.columns if c.lower() in ("longitude","decimallongitude","x")), None)
        if lat_col and lon_col:
            pts = df[[lat_col, lon_col]].dropna().values.tolist()
            if pts:
                fg = folium.FeatureGroup(name="🟦 Presence Points")
                for lat, lon in pts:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=5,
                        color="blue",
                        fill=True,
                        fill_opacity=0.8
                    ).add_to(fg)
                fg.add_to(m)
                m.fit_bounds(pts)

    # 2) Predictor rasters
    rasdir = "predictor_rasters/wgs84"
    if os.path.isdir(rasdir):
        for fn in sorted(os.listdir(rasdir)):
            if not fn.endswith(".tif"): continue
            path = os.path.join(rasdir, fn)
            with rasterio.open(path) as src:
                arr = src.read(1); bnd = src.bounds
            vmin,vmax = np.nanmin(arr), np.nanmax(arr)
            if not np.isnan(vmin) and vmin!=vmax:
                rgba = colormaps["viridis"]((arr-vmin)/(vmax-vmin))
                folium.raster_layers.ImageOverlay(
                    rgba,
                    bounds=[[bnd.bottom,bnd.left],[bnd.top,bnd.right]],
                    opacity=1.0,
                    name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})"
                ).add_to(m)

    # 3) Suitability map
    sf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        with rasterio.open(sf) as src:
            arr = src.read(1); bnd = src.bounds
        vmin,vmax = np.nanmin(arr), np.nanmax(arr)
        rgba = colormaps["viridis"]((arr-vmin)/(vmax-vmin))
        folium.raster_layers.ImageOverlay(
            rgba,
            bounds=[[bnd.bottom,bnd.left],[bnd.top,bnd.right]],
            opacity=0.7,
            name="🎯 Suitability"
        ).add_to(m)

    # 4) Controls and colorbar
    folium.LayerControl(collapsed=False).add_to(m)
    img_html = (
        f'<img src="data:image/png;base64,{COLORBAR_BASE64}" '
        'style="position:absolute; bottom:20px; right:10px; '
        'width:200px; height:30px; z-index:1000;" />'
    )
    m.get_root().html.add_child(Element(img_html))

    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" style="width:100%; height:600px; border:none;"></iframe>'

def zip_results():
    archive = "spatchat_results.zip"
    if os.path.exists(archive): os.remove(archive)
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters","outputs"):
            for root,_,files in os.walk(fld):
                for fn in files:
                    full = os.path.join(root,fn)
                    zf.write(full, arcname=os.path.relpath(full,"."))
    return archive

def run_fetch(sl, lc):
    if not sl and not lc:
        return create_map(), "⚠️ Select at least one predictor."
    layers = list(sl)
    if lc: layers.append("landcover")
    os.environ["SELECTED_LAYERS"] = ",".join(layers)
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(c.split(" – ")[0] for c in lc)
    proc = subprocess.run(["python","scripts/fetch_predictors.py"], capture_output=True, text=True)
    return create_map(), ("✅ Predictors fetched." if proc.returncode==0 else f"❌ Fetch failed:\n{proc.stderr}")

def run_model():
    proc = subprocess.run(["python","scripts/run_logistic_sdm.py"], capture_output=True, text=True)
    if proc.returncode!=0:
        return create_map(), f"❌ Model run failed:\n{proc.stderr}", None, None
    stats_df = pd.read_csv("outputs/model_stats.csv")
    zip_results()
    return create_map(), "✅ Model ran successfully! Results are ready below.", stats_df, "outputs/model_stats.csv"

def chat_step(file, user_msg, history, state):
    # reset logic
    if re.search(r"\b(start over|restart|clear everything)\b", user_msg, re.I):
        clear_all()
        new_hist = [{"role":"assistant",
                     "content":"👋 All cleared! Please upload your presence‑points CSV to begin."}]
        return new_hist, create_map(), state

    # tool-picking
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":user_msg}]
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=msgs, temperature=0.0
    ).choices[0].message.content

    try:
        call = json.loads(response)
        tool = call["tool"]
    except:
        assistant_txt = ("Sorry, I couldn't understand that. Please say 'fetch ...', 'run model', or use the download button.")
        history.append({"role":"assistant","content":assistant_txt})
        return history, create_map(), state

    if tool=="fetch":
        m_out, status = run_fetch(call.get("layers",[]), call.get("landcover",[]))
        assistant_txt = f"{status}\n\nWhen ready, say “run model.”"

    elif tool=="run_model":
        m_out, status, stats_df, stats_csv = run_model()
        stats_md = stats_df.to_markdown(index=False)
        assistant_txt = (
            f"{status}\n\n**Model Statistics:**\n\n{stats_md}\n\n"
            "Download your ZIP using the button on the left."
        )

    elif tool=="download":
        m_out, _ = (create_map(), zip_results())
        assistant_txt = "✅ ZIP is downloading…"

    else:
        m_out = create_map()
        assistant_txt = "Sorry, I don’t know that command."

    history.append({"role":"user","content":user_msg})
    history.append({"role":"assistant","content":assistant_txt})
    return history, m_out, state

def on_upload(f, history):
    new_history = history.copy()
    clear_all()
    if f and hasattr(f,"name"):
        shutil.copy(f.name,"inputs/presence_points.csv")
        extras = LAYERS[19:-1]
        last   = LAYERS[-1]
        layers_str = f"bio1–bio19, {', '.join(extras)}, and {last} (e.g., water, urban…)"
        new_history.append({
            "role":"assistant",
            "content":(
                "✅ Uploaded! Available layers are:\n\n"
                f"{layers_str}\n\n"
                "Now say “fetch elevation, ndvi, bio1” to grab those layers."
            )
        })
    return new_history, create_map(), state

with gr.Blocks() as demo:
    gr.Markdown("## 🌱 SpatChat SDM – Chat‑Driven Layout")

    state = gr.State({"stage":"await_upload"})

    with gr.Row():
        with gr.Column(scale=2):
            map_out      = gr.HTML(create_map(), label="🗺️ Map Preview")
            download_btn = gr.DownloadButton("📥 Download Results", zip_results)
        with gr.Column(scale=1):
            chat         = gr.Chatbot(
                              value=[{"role":"assistant",
                                      "content":"👋 Hello! Upload your presence‑points CSV to begin."}],
                              type="messages",
                              label="💬 Chat"
                          )
            gr.Markdown("**Ask SpatChat**")
            user_in      = gr.Textbox(placeholder="Type commands…", label="")
            file_input   = gr.File(label="📄 Upload Presence CSV", type="filepath")

    file_input.change(on_upload, inputs=[file_input, chat], outputs=[chat, map_out, state])

    user_in.submit(chat_step, inputs=[file_input, user_in, chat, state],
                   outputs=[chat, map_out, state])
    user_in.submit(lambda: "", None, user_in)

    demo.launch()
