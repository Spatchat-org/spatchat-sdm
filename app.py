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
    "fetch": lambda args: run_fetch(args.get("layers",[]), args.get("landcover",[])),
    "run_model": lambda args: run_model()[:2],    # returns (map, status)
    "download": lambda args: (None, "download_trigger")
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

# --- System prompt for the LLM ---
SYSTEM_PROMPT = """
You are SpatChat, a friendly assistant orchestrating SDM.
Your job is to explain to the user what options they have in each step, 
guiding them through the whole process to build the SDM.
Whenever the user
wants to perform an action, reply _only_ with a JSON object selecting one of
your tools:
- To fetch layers:     {"tool":"fetch",     "layers":["bio1","ndvi",...]}
- To run the model:    {"tool":"run_model"}
- To download results: {"tool":"download"}
After we run that function in Python, we'll show its output back to the user,
and then continue the conversation, 
and provide a prompt to the user after each action to guide them on what they should do next.
If the use ask for statistical results (e.g., show stats), then show them the results from stats_df.
Be conversational and helpful, but keep the conversation brief.
If the question is vague, ask the user to clarify.
""".strip()

# --- Auxiliary download click handler ---
def on_download_click(history, state):
    zip_path = "spatchat_results.zip"
    if os.path.exists(zip_path):
        # Trigger actual ZIP creation
        zip_results()
        msg = "✅ Results are ready! Use the download button below."
        # Append assistant message and return path for file block
        return history + [{"role":"assistant","content":msg}], zip_path
    else:
        msg = "⚠️ Results not ready yet. Please run the model first."
        return history + [{"role":"assistant","content":msg}], None

# --- Build & launch UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 SpatChat SDM – Chat‑Driven Species Distribution Modeling")

    state = gr.State({"built": False})
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
            download_btn= gr.Button("Download Results")
            download_blk= gr.File(label="ZIP Bundle")

    # Upload and chat
    file_input.change(
        on_upload,
        inputs=[file_input, chat],
        outputs=[chat, map_out, download_blk, state]
    )
    send_btn.click(
        chat_step,
        inputs=[file_input, user_in, chat, state],
        outputs=[chat, map_out, download_blk, state]
    )
    send_btn.click(lambda: "", None, user_in)
    user_in.submit(
        chat_step,
        inputs=[file_input, user_in, chat, state],
        outputs=[chat, map_out, download_blk, state]
    )
    user_in.submit(lambda: "", None, user_in)

    # Download button logic
    download_btn.click(
        on_download_click,
        inputs=[chat, state],
        outputs=[chat, download_blk]
    )

    demo.launch()
