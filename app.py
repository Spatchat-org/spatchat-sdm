import os
import io
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
from together import Together
from dotenv import load_dotenv

# --- Authenticate Earth Engine ---
svc = json.loads(os.environ["GEE_SERVICE_ACCOUNT"])
creds = ee.ServiceAccountCredentials(svc["client_email"], key_data=json.dumps(svc))
ee.Initialize(creds)

# --- LLM client ---
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# --- Clean up last session ---
for d in ("predictor_rasters","outputs","inputs"):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs("inputs", exist_ok=True)

# --- Available layers for prompts ---
LAYERS = [f"bio{i}" for i in range(1,20)] + ["elevation","slope","aspect","ndvi","landcover"]

# --- Pre-render Viridis colorbar as Base64 ---
fig, ax = plt.subplots(figsize=(4,0.5))
norm = Normalize(vmin=0, vmax=1)
cbar = plt.colorbar(ScalarMappable(norm=norm,cmap="viridis"),
                    cax=ax, orientation="horizontal")
cbar.set_ticks([])
ax.set_xlabel("Low    High")
fig.tight_layout(pad=0)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100)
plt.close(fig)
buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(buf.read()).decode()

# --- System prompt for tool selection ---
SYSTEM_PROMPT = """
You are SpatChat, an assistant that orchestrates SDM.
When the user asks to “fetch …”, “run model” or “download” you reply _only_ with JSON:

• Fetch:     {"tool":"fetch","layers":[…],"landcover":[…]}
• Run model: {"tool":"run_model"}
• Download:  {"tool":"download"}

We'll invoke that tool in Python, show the result, then you continue the convo.
""".strip()

def create_map():
    m = folium.Map(location=[0,0],zoom_start=2,control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)

    # presence pts
    p = "inputs/presence_points.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        lat = next((c for c in df if c.lower() in ("latitude","decimallatitude","y")),None)
        lon = next((c for c in df if c.lower() in ("longitude","decimallongitude","x")),None)
        if lat and lon:
            pts = df[[lat,lon]].values.tolist()
            fg = folium.FeatureGroup("🟦 Presence")
            for la,lo in pts:
                folium.CircleMarker([la,lo],radius=4,
                                    color="blue",fill=True,fill_opacity=0.8).add_to(fg)
            fg.add_to(m)
            if pts: m.fit_bounds(pts)

    # predictor rasters
    rd = "predictor_rasters/wgs84"
    if os.path.isdir(rd):
        for fn in sorted(os.listdir(rd)):
            if not fn.endswith(".tif"): continue
            with rasterio.open(os.path.join(rd,fn)) as src:
                img,b = src.read(1), src.bounds
            vmin,vmax = np.nanmin(img),np.nanmax(img)
            if np.isnan(vmin) or vmin==vmax: continue
            rgba = colormaps["viridis"]((img-vmin)/(vmax-vmin))
            folium.raster_layers.ImageOverlay(
                rgba, [[b.bottom,b.left],[b.top,b.right]],
                opacity=1.0,name=f"🟨 {fn}"
            ).add_to(m)

    # suitability
    sf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        with rasterio.open(sf) as src:
            img,b = src.read(1), src.bounds
        rgba = colormaps["viridis"]((img-np.nanmin(img))/(np.nanmax(img)-np.nanmin(img)))
        folium.raster_layers.ImageOverlay(
            rgba, [[b.bottom,b.left],[b.top,b.right]],
            opacity=0.7,name="🎯 Suitability"
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(Element(
        f'<img src="data:image/png;base64,{COLORBAR_BASE64}" '
        'style="position:absolute; bottom:20px; right:10px; '
        'width:200px; height:30px; z-index:1000;" />'
    ))
    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" '\
           'style="width:100%; height:600px; border:none;"></iframe>'

def run_fetch(layers, landcover):
    if not layers and not landcover:
        return create_map(), "⚠️ Pick at least one predictor."
    os.environ["SELECTED_LAYERS"] = ",".join(layers)
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(map(str,landcover))
    proc = subprocess.run(["python","scripts/fetch_predictors.py"],
                          capture_output=True,text=True)
    ok = proc.returncode==0
    return create_map(), ("✅ Predictors fetched." if ok else f"❌ Fetch failed:\n{proc.stderr}")

def run_model():
    proc = subprocess.run(["python","scripts/run_logistic_sdm.py"],
                          capture_output=True,text=True)
    if proc.returncode!=0:
        return create_map(), f"❌ Model run failed:\n{proc.stderr}"
    df = pd.read_csv("outputs/model_stats.csv")
    return create_map(), "✅ Model ran successfully!\n\n" + df.to_markdown(index=False)

def zip_results():
    zp = "spatchat_results.zip"
    if os.path.exists(zp): os.remove(zp)
    with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters","outputs"):
            for r,_,fs in os.walk(fld):
                for f in fs:
                    full=os.path.join(r,f)
                    zf.write(full, arcname=os.path.relpath(full,"."))
    return zp

TOOLS = {
    "fetch": run_fetch,
    "run_model": lambda _: run_model(),
    "download": lambda _: zip_results()
}

def chat_step(file, user_msg, history, state):
    # ask LLM which tool
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":user_msg}]
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=msgs, temperature=0.0
    ).choices[0].message.content

    # try parse JSON
    try:
        call = json.loads(resp)
        tool = call["tool"]
        if tool=="fetch":
            m_out, txt = TOOLS[tool]({"layers":call.get("layers",[]),
                                      "landcover":call.get("landcover",[])})
            dl = None
        elif tool=="run_model":
            m_out, txt = run_model()
            dl = None
        else:  # download
            dl = zip_results()
            m_out = create_map()
            txt = f"✅ Your ZIP is ready."
    except:
        txt = "❗️ Sorry, I didn’t get that—say “fetch …”, “run_model” or “download”."
        m_out, dl = create_map(), None

    history.append({"role":"user","content":user_msg})
    history.append({"role":"assistant","content":txt})
    return history, m_out, dl, state

def on_upload(f, history, state):
    if not f: 
        return history, create_map(), None, state
    shutil.copy(f.name,"inputs/presence_points.csv")
    history.append({"role":"assistant",
        "content":"✅ Uploaded! Available layers:\n\n"
                  + ", ".join(LAYERS)
                  + "\n\nSay “fetch elevation, ndvi, bio1” to start."})
    return history, create_map(), None, state

# --- UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 SpatChat SDM – Chat‑Driven SDM")
    state       = gr.State()
    file_input  = gr.File(label="Upload CSV",type="filepath")
    map_out     = gr.HTML(value=create_map(), label="Map")
    chatbot     = gr.Chatbot(label="SpatChat Dialog", type="messages",
                             value=[{"role":"assistant","content":"👋 Hi! Please upload your presence CSV."}])
    user_in     = gr.Textbox(placeholder="Type a command…")
    send_btn    = gr.Button("Send")
    download_blk= gr.File(label="Download ZIP", visible=False)

    # hook upload
    file_input.change(on_upload,
        inputs=[file_input, chatbot, state],
        outputs=[chatbot, map_out, download_blk, state]
    )
    # hook send & Enter
    send_btn.click(chat_step,
        inputs=[file_input, user_in, chatbot, state],
        outputs=[chatbot, map_out, download_blk, state]
    )
    user_in.submit(chat_step,
        inputs=[file_input, user_in, chatbot, state],
        outputs=[chatbot, map_out, download_blk, state]
    )
    # clear box
    send_btn.click(lambda: "", None, user_in)
    user_in.submit(lambda: "", None, user_in)

    demo.launch()
