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

# --- 0) Authenticate Earth Engine ---
svc = json.loads(os.environ["GEE_SERVICE_ACCOUNT"])
creds = ee.ServiceAccountCredentials(
    svc["client_email"], key_data=json.dumps(svc)
)
ee.Initialize(creds)

# --- 1) LLM client ---
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# --- 2) Clean workspace each launch ---
for d in ("predictor_rasters","outputs","inputs"):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs("inputs", exist_ok=True)

# --- 3) Define available layers and landcover classes ---
LAYERS = [f"bio{i}" for i in range(1,20)] + ["elevation","slope","aspect","ndvi","landcover"]
landcover_map = {
    0: "water", 1: "evergreen_needleleaf_forest", 2: "evergreen_broadleaf_forest",
    3: "deciduous_needleleaf_forest", 4: "deciduous_broadleaf_forest", 5: "mixed_forest",
    6: "closed_shrublands", 7: "open_shrublands", 8: "woody_savannas", 9: "savannas",
    10: "grasslands", 11: "permanent_wetlands", 12: "croplands",
    13: "urban_and_built_up", 14: "cropland_natural_vegetation_mosaic",
    15: "snow_and_ice", 16: "barren_or_sparsely_vegetated"
}
landcover_choices = [f"{k} – {v.replace('_',' ')}" for k,v in landcover_map.items()]

# --- 4) Pre-render a tiny Viridis colorbar as base64 ---
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

# --- 5) System prompt for JSON tool selection ---
SYSTEM_PROMPT = """
You are SpatChat, a friendly assistant orchestrating species distribution modeling.
When the user wants to fetch layers, run the model, or download results, reply _only_ with a JSON object picking one tool:

• Fetch layers:     {"tool":"fetch","layers":["bio1","ndvi",...],"landcover":[...]}
• Run the model:    {"tool":"run_model"}
• Download results: {"tool":"download"}

After we run that tool in Python, you will continue the conversation explaining what happened and what to do next.
If the user asks for stats, include them in your follow‑up. Be concise and helpful.
""".strip()

# --- 6) Map‐making helper (unchanged) ---
def create_map():
    m = folium.Map(location=[0,0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)

    # presence‐points (auto‐detect lat/lon)
    p = "inputs/presence_points.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        lat = next((c for c in df if c.lower() in ("latitude","decimallatitude","y")), None)
        lon = next((c for c in df if c.lower() in ("longitude","decimallongitude","x")), None)
        if lat and lon:
            pts = df[[lat,lon]].values.tolist()
            fg = folium.FeatureGroup(name="🟦 Presence Points")
            for la,lo in pts:
                folium.CircleMarker([la,lo],radius=4,
                    color="blue", fill=True, fill_opacity=0.8
                ).add_to(fg)
            fg.add_to(m)
            if pts: m.fit_bounds(pts)

    # predictor rasters
    rd = "predictor_rasters/wgs84"
    if os.path.isdir(rd):
        for fn in sorted(os.listdir(rd)):
            if not fn.endswith(".tif"): continue
            with rasterio.open(os.path.join(rd,fn)) as src:
                img, b = src.read(1), src.bounds
            vmin,vmax = np.nanmin(img),np.nanmax(img)
            if np.isnan(vmin) or vmin==vmax: continue
            rgba = colormaps["viridis"]((img-vmin)/(vmax-vmin))
            folium.raster_layers.ImageOverlay(
                rgba, [[b.bottom,b.left],[b.top,b.right]],
                opacity=1.0,name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})"
            ).add_to(m)

    # suitability overlay
    sf = "outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        with rasterio.open(sf) as src:
            img, b = src.read(1), src.bounds
        rgba = colormaps["viridis"]((img-np.nanmin(img))/(np.nanmax(img)-np.nanmin(img)))
        folium.raster_layers.ImageOverlay(
            rgba, [[b.bottom,b.left],[b.top,b.right]],
            opacity=0.7,name="🎯 Suitability"
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    # embed colorbar
    html = (
        f'<img src="data:image/png;base64,{COLORBAR_BASE64}" '
        'style="position:absolute; bottom:20px; right:10px; '
        'width:200px; height:30px; z-index:1000;" />'
    )
    m.get_root().html.add_child(Element(html))
    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" '\
           'style="width:100%;height:600px;border:none;"></iframe>'

# --- 7) Tool implementations ---
def run_fetch(layers, landcover):
    if not layers and not landcover:
        return create_map(), "⚠️ You must pick at least one predictor."
    os.environ["SELECTED_LAYERS"] = ",".join(layers + (["landcover"] if landcover else []))
    os.environ["SELECTED_LANDCOVER_CLASSES"] = ",".join(map(str,landcover))
    proc = subprocess.run(["python","scripts/fetch_predictors.py"],
                          capture_output=True, text=True)
    ok = (proc.returncode==0)
    return create_map(), ("✅ Predictors fetched." if ok else f"❌ Fetch failed:\n{proc.stderr}")

def run_model():
    proc = subprocess.run(["python","scripts/run_logistic_sdm.py"],
                          capture_output=True, text=True)
    if proc.returncode!=0:
        return create_map(), f"❌ Model run failed:\n{proc.stderr}"
    # load and format stats
    df = pd.read_csv("outputs/model_stats.csv")
    md = df.to_markdown(index=False)
    status = "✅ Model ran successfully!\n\n**Model stats**:\n\n" + md
    return create_map(), status

def zip_results():
    zf_path = "spatchat_results.zip"
    if os.path.exists(zf_path): os.remove(zf_path)
    with zipfile.ZipFile(zf_path,"w",zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters","outputs"):
            for r,_,fs in os.walk(fld):
                for f in fs:
                    full = os.path.join(r,f)
                    zf.write(full,arcname=os.path.relpath(full,"."))
    return create_map(), zf_path

TOOLS = {
    "fetch":     lambda call: run_fetch(call.get("layers",[]), call.get("landcover",[])),
    "run_model": lambda call: run_model(),
    "download":  lambda call: zip_results()
}

# --- 8) Chat handler ---
def chat_step(file, user_msg, history, state):
    # build LLM prompt
    msg_list = [{"role":"system","content":SYSTEM_PROMPT}]
    msg_list += history
    msg_list.append({"role":"user","content":user_msg})

    # get model’s selection JSON
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=msg_list,
        temperature=0.0
    ).choices[0].message.content

    # try to parse the tool call
    try:
        call = json.loads(resp)
        tool = call["tool"]
        m_out, tool_res = TOOLS[tool](call)
    except Exception:
        # if parsing fails
        history.append({"role":"user","content":user_msg})
        history.append({"role":"assistant",
                        "content":"❓ Sorry, I didn’t understand – please say “fetch …”, “run_model”, or “download”.​"})
        return history, create_map(), None, state

    # craft assistant reply
    if tool == "download":
        # tool_res is ZIP path
        link = tool_res
        assistant_txt = (
            "✅ All done! Your ZIP is ready:\n\n"
            f"[▶️ Click here to download results]({link})"
        )
    else:
        assistant_txt = tool_res

    # record both user + assistant
    history.append({"role":"user","content":user_msg})
    history.append({"role":"assistant","content":assistant_txt})

    # only the map and chat slot; no separate file component
    return history, m_out, None, state

# --- 9) Upload handler to kick off fetch stage ---
def on_upload(f, history, state):
    if not f or not hasattr(f,"name"):
        return history, create_map(), None, state
    shutil.copy(f.name, "inputs/presence_points.csv")
    # greet and list layers
    history.append({"role":"assistant",
                    "content":
                    "✅ Points uploaded! Available layers are:\n\n"
                    + ", ".join(LAYERS)
                    + "\n\nNow say “fetch elevation, ndvi, bio1” to grab them."})
    return history, create_map(), None, state

# --- 10) Launch the Gradio app ---
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 SpatChat SDM – Chat‑Driven SDM")

    state      = gr.State()  # we’re not doing staged fallbacks any more
    file_input = gr.File(label="Upload presence CSV", type="filepath")
    map_out    = gr.HTML(create_map, label="Map Preview")
    chatbot    = gr.Chatbot(label="SpatChat Dialog", type="messages",
                            value=[{"role":"assistant",
                                    "content":"👋 Hi! Please upload your presence‑points CSV to begin."}])
    user_in    = gr.Textbox(placeholder="Type a command…")
    send_btn   = gr.Button("Send")

    file_input.change(on_upload,
                      inputs=[file_input, chatbot, state],
                      outputs=[chatbot, map_out, gr.State(), state])

    # both click and Enter
    send_btn.click(chat_step,
                   inputs=[file_input, user_in, chatbot, state],
                   outputs=[chatbot, map_out, gr.State(), state])
    user_in.submit(chat_step,
                   inputs=[file_input, user_in, chatbot, state],
                   outputs=[chatbot, map_out, gr.State(), state])
    # clear input after send
    send_btn.click(lambda: "", None, user_in)
    user_in.submit(lambda: "", None, user_in)

    demo.launch()
