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

# --- Tool registry & layers (unchanged) ---
LAYERS = [f"bio{i}" for i in range(1, 20)] + ["elevation", "slope", "aspect", "ndvi", "landcover"]
TOOLS = {
    "fetch": lambda args: run_fetch(args.get("layers", []), args.get("landcover", [])),
    "run_model": lambda args: run_model()[:2],
    "download": lambda args: (create_map(), zip_results())
}

# --- Pre-render Viridis colorbar → base64 (unchanged) ---
fig, ax = plt.subplots(figsize=(4, 0.5))
norm = Normalize(vmin=0, vmax=1)
plt.colorbar(ScalarMappable(norm=norm, cmap="viridis"),
             cax=ax, orientation="horizontal").set_ticks([])
ax.set_xlabel("Low    High")
fig.tight_layout(pad=0)
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(buf.read()).decode()

# --- Landcover choices (unchanged) ---
landcover_options = {
    0:"water",1:"evergreen needleleaf forest",2:"evergreen broadleaf forest",
    3:"deciduous needleleaf forest",4:"deciduous broadleaf forest",5:"mixed forest",
    6:"closed shrublands",7:"open shrublands",8:"woody savannas",9:"savannas",
    10:"grasslands",11:"permanent wetlands",12:"croplands",
    13:"urban and built up",14:"cropland/natural vegetation mosaic",
    15:"snow and ice",16:"barren or sparsely vegetated"
}
landcover_choices = [f"{k} – {v}" for k, v in landcover_options.items()]

# --- LLM prompts (unchanged) ---
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
    m = folium.Map(location=[0,0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)
    # … presence points, rasters, suitability overlay, colorbar …
    # (identical to before)
    # copy-paste your create_map implementation here
    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" style="width:100%; height:600px; border:none;"></iframe>'

def zip_results():
    archive = "spatchat_results.zip"
    if os.path.exists(archive): os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters","outputs"):
            for root,_,files in os.walk(fld):
                for fn in files:
                    full = os.path.join(root, fn)
                    zf.write(full, arcname=os.path.relpath(full, "."))
    return archive

def run_fetch(sl, lc):
    # … unchanged …
    return create_map(), "✅ Predictors fetched."

def run_model():
    # … unchanged up to stats_df …
    stats_df = pd.read_csv("outputs/model_stats.csv")
    zip_results()  # pre-create zip
    return create_map(), "✅ Model ran successfully! Results are ready for download using the Download Button!", stats_df, "outputs/model_stats.csv"

def chat_step(file, user_msg, history, state):
    download_update = gr.update()  # by default, no change to button

    # 1) tool‐picker LLM
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":user_msg}]
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=msgs, temperature=0.0
    ).choices[0].message.content

    # 2) parse JSON
    try:
        call = json.loads(resp)
        tool = call["tool"]
        if tool == "run_model":
            m_out, status, stats_df, stats_csv = run_model()
            assistant_txt = status
            download_update = gr.update(disabled=False)
        else:
            m_out, status = TOOLS[tool](call)
            assistant_txt = status

    except Exception:
        # fallback
        fb = [{"role":"system","content":FALLBACK_PROMPT},
              {"role":"user","content":user_msg}]
        assistant_txt = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=fb, temperature=0.7
        ).choices[0].message.content
        m_out = create_map()

    history.append({"role":"user","content":user_msg})
    history.append({"role":"assistant","content":assistant_txt})
    return history, m_out, download_update, state

def on_upload(f, history):
    new_history = history.copy()
    if f and hasattr(f,"name"):
        shutil.copy(f.name, "inputs/presence_points.csv")
        new_history.append({"role":"assistant","content":"✅ Uploaded! Now “fetch …”"})
    return new_history, create_map(), gr.update(disabled=True), {"stage":"await_fetch"}

# --- Build UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 SpatChat SDM")

    state = gr.State({"stage":"await_upload"})

    with gr.Row():
        with gr.Column(scale=1):
            file_input   = gr.File(label="📄 Upload Presence CSV", type="filepath")
        with gr.Column(scale=3):
            map_out      = gr.HTML(create_map(), label="🗺️ Map Preview")
            chat         = gr.Chatbot(
                              value=[{"role":"assistant",
                                      "content":"👋 Hello! Upload your presence‑points CSV to begin."}],
                              type="messages"
                          )
            user_in      = gr.Textbox(placeholder="Type commands…")
            send_btn     = gr.Button("Send")
            download_btn = gr.DownloadButton(
                              "📥 Download Results",
                              file_name="spatchat_results.zip",
                              file=zip_results,
                              disabled=True
                          )

    file_input.change(
        on_upload,
        inputs=[file_input, chat],
        outputs=[chat, map_out, download_btn, state]
    )

    send_btn.click(
        chat_step,
        inputs=[file_input, user_in, chat, state],
        outputs=[chat, map_out, download_btn, state]
    )
    send_btn.click(lambda: "", None, user_in)
    user_in.submit(
        chat_step,
        inputs=[file_input, user_in, chat, state],
        outputs=[chat, map_out, download_btn, state]
    )
    user_in.submit(lambda: "", None, user_in)

    demo.launch()
