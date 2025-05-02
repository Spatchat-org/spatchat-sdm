#!/usr/bin/env python3
import os, io, json, shutil, subprocess, zipfile, re
import pandas as pd, numpy as np, gradio as gr, folium, rasterio, ee, joblib
from dotenv import load_dotenv
from together import Together
from matplotlib import pyplot as plt, colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from folium import Element
import html as html_lib

# --- 0) SETUP: Earth Engine + LLM client ---
# Earth Engine via Service Account
svc = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
creds = ee.ServiceAccountCredentials(svc['client_email'], key_data=json.dumps(svc))
ee.Initialize(creds)

# Together LLM
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# clean workspace
for d in ("predictor_rasters","outputs","inputs"):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs("inputs", exist_ok=True)

LAYERS = [f"bio{i}" for i in range(1,20)] + ["elevation","slope","aspect","ndvi","landcover"]

# --- 1) PRE‑RENDER tiny Viridis colorbar (base64) ---
fig, ax = plt.subplots(figsize=(4,0.5))
norm = Normalize(0,1)
plt.colorbar(ScalarMappable(norm=norm,cmap="viridis"),
             cax=ax, orientation="horizontal")
ax.set_xticks([]); ax.set_xlabel("Low    High")
fig.tight_layout(pad=0)
buf = io.BytesIO(); fig.savefig(buf,format="png",dpi=100); plt.close(fig)
buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(buf.read()).decode()

# --- 2) Map builder fn ---
def create_map():
    m = folium.Map(location=[0,0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)

    # presence
    ppath="inputs/presence_points.csv"
    if os.path.exists(ppath):
        df=pd.read_csv(ppath)
        if {'latitude','longitude'}.issubset(df.columns):
            pts=df[['latitude','longitude']].values.tolist()
            fg=folium.FeatureGroup(name="🟦 Presence Points")
            for lat,lon in pts:
                folium.CircleMarker([lat,lon],radius=4,
                                    color="blue",fill=True,fill_opacity=0.8)\
                       .add_to(fg)
            fg.add_to(m)
            if pts: m.fit_bounds(pts)

    # predictors overlay
    rasdir="predictor_rasters/wgs84"
    if os.path.isdir(rasdir):
        for fn in sorted(os.listdir(rasdir)):
            if not fn.endswith(".tif"): continue
            with rasterio.open(os.path.join(rasdir,fn)) as src:
                img=src.read(1); b=src.bounds
            vmin,vmax=np.nanmin(img),np.nanmax(img)
            if np.isnan(vmin) or vmin==vmax: continue
            rgba=colormaps["viridis"]((img-vmin)/(vmax-vmin))
            folium.raster_layers.ImageOverlay(
                rgba, [[b.bottom,b.left],[b.top,b.right]],
                opacity=1.0,name=f"🟨 {fn} ({vmin:.2f}–{vmax:.2f})"
            ).add_to(m)

    # suitability overlay
    sf="outputs/suitability_map_wgs84.tif"
    if os.path.exists(sf):
        with rasterio.open(sf) as src:
            img=src.read(1); b=src.bounds
        rgba=colormaps["viridis"]((img-np.nanmin(img))/(np.nanmax(img)-np.nanmin(img)))
        folium.raster_layers.ImageOverlay(
            rgba, [[b.bottom,b.left],[b.top,b.right]],
            opacity=0.7,name="🎯 Suitability"
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # embed static colorbar
    html = (
        f'<img src="data:image/png;base64,{COLORBAR_BASE64}" '
        'style="position:absolute; bottom:20px; right:10px; '
        'width:200px; height:30px; z-index:1000;" />'
    )
    m.get_root().html.add_child(Element(html))
    rendered = m.get_root().render()
    return f'<iframe srcdoc="{html_lib.escape(rendered)}" '\
           'style="width:100%; height:600px; border:none;"></iframe>'

# --- 3) ZIP results fn ---
def zip_results():
    z="spatchat_results.zip"
    if os.path.exists(z): os.remove(z)
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as zf:
        for fld in ("predictor_rasters","outputs"):
            for r,_,fs in os.walk(fld):
                for f in fs:
                    full=os.path.join(r,f)
                    zf.write(full,arcname=os.path.relpath(full,"."))
    return z

# --- 4) UPLOAD handler ---
def preview_upload(file):
    if not file or not hasattr(file,"name"):
        return [("", "👋 Please upload a presence CSV to begin.")], create_map(), None, {"stage":"await_upload"}
    # copy it in
    shutil.copy(file.name,"inputs/presence_points.csv")
    intro = (
        "✅ Got your points!  Which predictors shall I fetch?\n"
        f"Available layers: {', '.join(LAYERS)}\n"
        "e.g. “fetch elevation, ndvi, bio1”"
    )
    return [("","👋")+("",intro)], create_map(), None, {"stage":"await_layers"}

# --- 5) CHAT handler driving fetch→run→download via LLM + actual calls ---
SYSTEM_PROMPT = """
You are SpatChat, a friendly assistant that orchestrates species distribution modeling:
1) When asked to fetch layers, run fetch_predictors.py and report success or errors.
2) When asked to run/train the model, run run_logistic_sdm.py, read outputs/model_stats.csv, show it.
3) When asked to download, zip all results and provide the download link.
Be conversational and guide the user. If unclear, ask for clarification.
""".strip()

def analyze_sdm(file, user_msg, history, state):
    stage = state.get("stage","await_layers")
    hist = history[:]  # copy
    cmd = user_msg.strip()
    op_info = ""

    # 1) FETCH
    if stage=="await_layers" and re.search(r"\b(fetch|get|use)\b",cmd,re.I):
        # run actual fetch
        proc = subprocess.run(["python","scripts/fetch_predictors.py"],
                              capture_output=True, text=True)
        out = proc.stdout+proc.stderr
        op_info = f"```bash\n{out}\n```"
        stage="await_run"

    # 2) RUN MODEL
    elif stage=="await_run" and re.search(r"\b(run|train|create)\b.*\bmodel\b",cmd,re.I):
        proc = subprocess.run(["python","scripts/run_logistic_sdm.py"],
                              capture_output=True, text=True)
        out = proc.stdout+proc.stderr
        # read stats
        if os.path.exists("outputs/model_stats.csv"):
            stats = pd.read_csv("outputs/model_stats.csv")
            md = stats.to_markdown(index=False)
            op_info = f"```bash\n{out}\n```\n**Model stats**:\n\n{md}"
        else:
            op_info = f"```bash\n{out}\n```"
        stage="await_download"

    # 3) DOWNLOAD
    elif stage=="await_download" and re.match(r"^(yes|y|sure|download)\b",cmd,re.I):
        z = zip_results()
        op_info = f"Here is your ZIP! 📦\n\n"
        # in this case we'll return z in the download slot below
        hist.append((user_msg,op_info))
        return hist, create_map(), z, {"stage":"done"}

    # 4) FALLBACKS
    else:
        if stage=="await_layers":
            op_info = 'Please say “fetch …” to fetch predictors.'
        elif stage=="await_run":
            op_info = 'Please say “run model” to train the model.'
        elif stage=="await_download":
            op_info = 'Say “yes” to download results, or “no” to finish.'
        else:
            op_info = "Session complete. Upload a new CSV to start over."
            stage="await_upload"

    # build LLM prompt
    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    for u,a in hist:
        messages.append({"role":"user","content":u})
        messages.append({"role":"assistant","content":a})
    # inject the operation info
    messages.append({"role":"system","content":op_info})
    messages.append({"role":"user","content":user_msg})

    # ask the LLM to compose the final assistant response
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        temperature=0.3
    ).choices[0].message.content

    hist.append((user_msg, resp))
    return hist, create_map(), None, {"stage":stage}


# --- 6) BUILD & LAUNCH GRADIO UI ---

with gr.Blocks() as demo:
    gr.Markdown("## 🌱 SpatChat SDM – Chat‑Driven Species Distribution Modeling")

    with gr.Row():
        with gr.Column(scale=1):
            file_input  = gr.File(label="📄 Upload Presence CSV", type="filepath")
        with gr.Column(scale=3):
            map_out     = gr.HTML(value=create_map(), label="🗺️ Map Preview")
            chat        = gr.Chatbot(label="SpatChat Dialog",
                                     value=[("", "👋 Hello! Please upload your presence CSV to begin.")])
            user_in     = gr.Textbox(placeholder="Type commands…", label="")
            send_btn    = gr.Button("Send")
            download_f  = gr.File(label="Download Results", visible=False)

    # CSV upload
    file_input.change(
        preview_upload, 
        inputs=[file_input],
        outputs=[chat, map_out, download_f, gr.State()]
    )
    # Chat
    for trigger in (send_btn, user_in.submit):
        trigger.click(
            analyze_sdm,
            inputs=[file_input, user_in, chat, gr.State()],
            outputs=[chat, map_out, download_f, gr.State()]
        )
        # clear input
        trigger.click(lambda: "", None, user_in)

    demo.launch()
