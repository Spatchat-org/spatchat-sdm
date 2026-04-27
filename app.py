# app.py

import os
import io
import json
import base64
import shutil
import subprocess
import zipfile
import re
import difflib
import sys
import time
import random
import threading
import inspect
import uuid
from contextlib import contextmanager
from typing import Optional

import gradio as gr
import folium
import html as html_lib
import pandas as pd
import numpy as np
import rasterio
import ee


def _install_asyncio_fd_warning_filter() -> None:
    """
    Suppress noisy Python 3.13 asyncio loop-destructor warnings:
    ValueError: Invalid file descriptor: -1
    This is an unraisable exception during GC, not a runtime app failure.
    """
    original_hook = getattr(sys, "unraisablehook", None)
    if original_hook is None:
        return

    def _hook(unraisable):
        try:
            exc = getattr(unraisable, "exc_value", None)
            obj = getattr(unraisable, "object", None)
            if (
                isinstance(exc, ValueError)
                and "Invalid file descriptor: -1" in str(exc)
                and "BaseEventLoop.__del__" in repr(obj)
            ):
                return
        except Exception:
            pass
        original_hook(unraisable)

    sys.unraisablehook = _hook


_install_asyncio_fd_warning_filter()

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from folium import Element
from dotenv import load_dotenv
from rasterio.crs import CRS as RioCRS
from rasterio.warp import calculate_default_transform, reproject, transform as rio_transform, Resampling
from llm_utils import UnifiedLLM as SharedUnifiedLLM
from crs_utils import parse_crs_input
from coords_utils import detect_lonlat_columns
from map_layers import render_sdm_map
from analysis_tracker import record_step, reset as reset_analysis_steps, set_current_session
from session_snapshot import write_session_snapshot
from workflow.feature_workspace import (
    FEATURE_WORKSPACE_CSV,
    refresh_workspace_from_rasters,
    render_feature_popup_html,
    run_multicollinearity_check,
    run_morans_i_check,
    run_pre_model_diagnostics,
    set_predictor_active_state,
    get_active_predictors,
)

# LLM providers
from huggingface_hub import InferenceClient
from together import Together
from together.error import RateLimitError, TogetherException
try:
    from gradio.context import LocalContext
except Exception:
    LocalContext = None
try:
    from together.error import ServiceUnavailableError
except ImportError:
    ServiceUnavailableError = TogetherException

# --- Which top-level predictors we support (all lower-case) ---
PREDICTOR_CHOICES = (
    [f"bio{i}" for i in range(1, 20)]
    + ["elevation", "slope", "aspect", "ndvi", "landcover"]
)
# force everything to lower-case so our .lower() tokens always match
VALID_LAYERS = {p.lower() for p in PREDICTOR_CHOICES}

# All available MODIS landcover classes
LANDCOVER_CLASSES = {
    c.lower() for c in (
        "water", "evergreen_needleleaf_forest", "evergreen_broadleaf_forest",
        "deciduous_needleleaf_forest", "deciduous_broadleaf_forest", "mixed_forest",
        "closed_shrublands", "open_shrublands", "woody_savannas", "savannas",
        "grasslands", "permanent_wetlands", "croplands", "urban_and_built_up",
        "cropland_natural_vegetation_mosaic", "snow_and_ice", "barren_or_sparsely_vegetated"
    )
}

# --- Small helper to list available layers ---
def available_layers_markdown():
    return (
        "You can fetch these predictors:\n"
        "• bio1–bio19\n"
        "• elevation\n"
        "• slope\n"
        "• aspect\n"
        "• NDVI\n"
        "• landcover (e.g. " + ", ".join(sorted(LANDCOVER_CLASSES)) + ")\n\n"
        "Example: **I want elevation, ndvi, bio1**\n\n"
        "For very large study areas, upload your own GeoTIFF predictor rasters instead of using server fetch."
    )

# --- Pre-render colorbar → base64 ---
fig, ax = plt.subplots(figsize=(4, 0.5))
norm = Normalize(vmin=0, vmax=1)
plt.colorbar(ScalarMappable(norm=norm, cmap="viridis"), cax=ax, orientation="horizontal").set_ticks([])
ax.set_xlabel("Low    High")
fig.tight_layout(pad=0)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100)
plt.close(fig)
buf.seek(0)
COLORBAR_BASE64 = base64.b64encode(buf.read()).decode()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
SESSIONS_ROOT = os.path.join(APP_ROOT, "sessions")
REPRO_TEMPLATE_DIR = os.path.join(APP_ROOT, "reproducible_scripts")
os.makedirs(SESSIONS_ROOT, exist_ok=True)
_GRADIO_SESSION_MAP: dict[str, str] = {}

def _current_gradio_session_hash() -> Optional[str]:
    if LocalContext is None:
        return None
    try:
        request = LocalContext.request.get()
    except Exception:
        request = None
    if request is None:
        return None
    session_hash = getattr(request, "session_hash", None)
    if not session_hash:
        return None
    return str(session_hash).strip() or None

def _cleanup_session_workspace(session_id: Optional[str]) -> None:
    sid = str(session_id or "").strip()
    if not sid:
        return
    reset_analysis_steps(sid)
    session_root = os.path.join(SESSIONS_ROOT, sid)
    if os.path.exists(session_root):
        shutil.rmtree(session_root, ignore_errors=True)
    for key, value in list(_GRADIO_SESSION_MAP.items()):
        if value == sid:
            _GRADIO_SESSION_MAP.pop(key, None)

def _register_gradio_session(session_id: Optional[str]) -> None:
    sid = str(session_id or "").strip()
    if not sid:
        return
    request_hash = _current_gradio_session_hash()
    if not request_hash:
        return
    previous_sid = _GRADIO_SESSION_MAP.get(request_hash)
    _GRADIO_SESSION_MAP[request_hash] = sid
    if previous_sid and previous_sid != sid:
        _cleanup_session_workspace(previous_sid)

def _cleanup_current_browser_session() -> None:
    request_hash = _current_gradio_session_hash()
    if not request_hash:
        return
    sid = _GRADIO_SESSION_MAP.pop(request_hash, None)
    if sid:
        _cleanup_session_workspace(sid)

def _prune_stale_sessions(max_age_seconds: int = 60 * 60 * 12) -> None:
    cutoff = time.time() - max_age_seconds
    try:
        for entry in os.scandir(SESSIONS_ROOT):
            if not entry.is_dir():
                continue
            try:
                if entry.stat().st_mtime < cutoff:
                    _cleanup_session_workspace(entry.name)
            except OSError:
                continue
    except OSError:
        pass

def _start_session_pruner(interval_seconds: int = 10 * 60, max_age_seconds: int = 60 * 60 * 24) -> None:
    interval_seconds = max(30, int(interval_seconds))
    max_age_seconds = max(60, int(max_age_seconds))
    def _loop() -> None:
        while True:
            try:
                _prune_stale_sessions(max_age_seconds=max_age_seconds)
            except Exception:
                pass
            time.sleep(interval_seconds)
    threading.Thread(target=_loop, name="spatchat-sdm-session-pruner", daemon=True).start()

def _ensure_session_workspace(state: Optional[dict]) -> dict:
    if not isinstance(state, dict):
        state = {}
    session_id = str(state.get("session_id") or uuid.uuid4().hex)
    session_root = os.path.join(SESSIONS_ROOT, session_id)
    for path in (
        session_root,
        os.path.join(session_root, "inputs"),
        os.path.join(session_root, "outputs"),
        os.path.join(session_root, "predictor_rasters"),
        os.path.join(session_root, "predictor_rasters", "wgs84"),
    ):
        os.makedirs(path, exist_ok=True)
    state["session_id"] = session_id
    state["session_root"] = session_root
    state.setdefault("session_info", {"session_id": session_id})
    set_current_session(session_id)
    _register_gradio_session(session_id)
    return state

def _fresh_state() -> dict:
    return {
        "layers_help_shown": False,
        "session_id": None,
        "session_root": None,
        "awaiting_method_choice": False,
        "selected_method": None,
        "awaiting_prune_confirmation": False,
        "pending_prune_predictors": [],
        "pending_model_request": {},
        "model_prune_preference": None,
        "last_multicol_diag": {},
    }

@contextmanager
def _session_cwd(state: Optional[dict]):
    if isinstance(state, dict) and state.get("session_root"):
        root = state["session_root"]
    else:
        root = APP_ROOT
    prev = os.getcwd()
    os.makedirs(root, exist_ok=True)
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(prev)

# --- Authenticate Earth Engine ---
# Load .env from the app directory so credentials are found even if the
# process is launched from a different working directory.
load_dotenv(dotenv_path=os.path.join(APP_ROOT, ".env"))
EE_INITIALIZED = False
EE_AUTH_ERROR = None
def _parse_service_account_json(raw_text):
    text = (raw_text or "").strip()
    if not text:
        return None
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    candidates = [text]
    if "\\n" in text:
        candidates.append(text.replace("\\n", "\n"))
    try:
        decoded = base64.b64decode(text).decode("utf-8", errors="strict").strip()
        candidates.append(decoded)
    except Exception:
        pass

    for candidate in candidates:
        if not candidate or not candidate.lstrip().startswith("{"):
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None

def _load_service_account_from_env():
    # Priority: explicit JSON env var, then legacy GEE_SERVICE_ACCOUNT.
    raw_values = [
        os.environ.get("GEE_SERVICE_ACCOUNT_JSON", ""),
        os.environ.get("GEE_SERVICE_ACCOUNT", ""),
    ]
    for raw in raw_values:
        raw = (raw or "").strip()
        if not raw:
            continue
        # Allow passing a path in env var instead of inline JSON.
        maybe_path = raw.strip('"').strip("'")
        if os.path.isfile(maybe_path):
            with open(maybe_path, "r", encoding="utf-8") as f:
                return json.load(f)
        svc = _parse_service_account_json(raw)
        if svc:
            return svc

    for path_var in ("GEE_SERVICE_ACCOUNT_FILE", "GOOGLE_APPLICATION_CREDENTIALS"):
        path_val = (os.environ.get(path_var, "") or "").strip().strip('"').strip("'")
        if not path_val:
            continue
        if os.path.isfile(path_val):
            with open(path_val, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

def _initialize_earth_engine():
    project = (os.environ.get("GEE_PROJECT", "") or "").strip() or None
    errors = []

    svc = _load_service_account_from_env()
    if svc is not None:
        try:
            client_email = (svc.get("client_email") or "").strip()
            private_key = (svc.get("private_key") or "").strip()
            if not client_email or not private_key:
                raise RuntimeError("Service account JSON is missing `client_email` or `private_key`.")
            creds = ee.ServiceAccountCredentials(client_email, key_data=json.dumps(svc))
            ee.Initialize(credentials=creds, project=project)
            return
        except Exception as exc:
            errors.append(f"service-account auth failed: {exc}")

    # Final fallback: use locally persisted Earth Engine credentials.
    try:
        ee.Initialize(project=project)
        return
    except Exception as exc:
        errors.append(f"persistent credentials failed: {exc}")

    raise RuntimeError(" | ".join(errors) if errors else "No Earth Engine credentials found.")

try:
    _initialize_earth_engine()
    EE_INITIALIZED = True
except Exception as exc:
    EE_AUTH_ERROR = str(exc)

# ──────────────────────────────────────────────────────────────
# Helpers for LLM response parsing (HF + Together)
# ──────────────────────────────────────────────────────────────

def _choice_content(choice):
    msg = getattr(choice, "message", None)
    if msg is None and isinstance(choice, dict):
        msg = choice.get("message")
    content = None
    if msg is not None:
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            content = "".join(parts)
    if content is None:
        content = ""
    return content

def _delta_text(delta):
    if isinstance(delta, dict):
        return delta.get("content", "")
    return getattr(delta, "content", "")

HF_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct"
TOGETHER_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

class _SpacedCallLimiter:
    def __init__(self, min_interval_seconds: float):
        self.min_interval = float(min_interval_seconds)
        self._lock = threading.Lock()
        self._last = 0.0
    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()

class UnifiedLLM:
    def __init__(self):
        hf_model_or_url = (
            os.getenv("HF_ENDPOINT_URL")
            or os.getenv("HF_MODEL")
            or HF_MODEL_DEFAULT
        ).strip()
        hf_token = (os.getenv("HF_TOKEN") or "").strip()
        self.hf_client = InferenceClient(model=hf_model_or_url, token=hf_token, timeout=300)
        self.together = None
        self.together_model = (os.getenv("TOGETHER_MODEL") or TOGETHER_MODEL_DEFAULT).strip()
        tg_key = (os.getenv("TOGETHER_API_KEY") or "").strip()
        if tg_key:
            self.together = Together(api_key=tg_key)
            self._tg_limiter = _SpacedCallLimiter(min_interval_seconds=100.0)

    def _hf_chat(self, messages, max_tokens=512, temperature=0.3, stream=False):
        tries, delay = 3, 2.5
        last_err = None
        for _ in range(tries):
            try:
                if hasattr(self.hf_client, "chat_completion"):
                    resp = self.hf_client.chat_completion(
                        messages=messages, max_tokens=max_tokens, temperature=temperature, stream=stream
                    )
                    if stream:
                        text = "".join(_delta_text(ch.choices[0].delta) for ch in resp)
                    else:
                        text = _choice_content(resp.choices[0])
                    return text
                else:
                    prompt = self._messages_to_prompt(messages)
                    text = self.hf_client.text_generation(
                        prompt, max_new_tokens=512, temperature=temperature, stream=False, return_full_text=False
                    )
                    return text
            except Exception as e:
                last_err = e
                time.sleep(delay)
                delay *= 1.8
        raise last_err

    @staticmethod
    def _messages_to_prompt(messages):
        parts = []
        for m in messages:
            role = m.get("role", "user"); content = m.get("content", "")
            if role == "system": parts.append(f"<|system|>\n{content}\n")
            elif role == "user": parts.append(f"<|user|>\n{content}\n")
            else: parts.append(f"<|assistant|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def chat(self, messages, temperature=0.3, max_tokens=512, stream=False):
        try:
            return self._hf_chat(messages, max_tokens=max_tokens, temperature=temperature, stream=stream)
        except Exception as hf_err:
            print(f"[LLM] HF primary failed: {hf_err}", file=sys.stderr)
            if self.together is None:
                raise
            self._tg_limiter.wait()
            backoff = 12.0
            for attempt in range(4):
                try:
                    resp = self.together.chat.completions.create(
                        model=self.together_model, messages=messages,
                        temperature=temperature, max_tokens=max_tokens, stream=stream,
                    )
                    return _choice_content(resp.choices[0])
                except (RateLimitError, ServiceUnavailableError):
                    if attempt == 3:
                        raise
                    time.sleep(backoff + random.uniform(0, 3))
                    backoff *= 1.8

llm = SharedUnifiedLLM()

def _safe_llm_chat(messages, *, temperature=0.3, max_tokens=512, stream=False, default="", log_errors=True):
    try:
        return llm.chat(messages, temperature=temperature, max_tokens=max_tokens, stream=stream)
    except Exception as exc:
        if log_errors:
            print(f"[llm] fallback unavailable: {exc}", file=sys.stderr)
        return default

def clear_all(state=None):
    with _session_cwd(state):
        for d in ("predictor_rasters", "outputs", "inputs"):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs("inputs", exist_ok=True)
        csv_fp = "inputs/presence_points.csv"
        if os.path.exists(csv_fp):
            os.remove(csv_fp)
        os.environ.pop("SELECTED_LAYERS", None)
        os.environ.pop("SELECTED_LANDCOVER_CLASSES", None)
        if os.path.exists("spatchat_results.zip"):
            os.remove("spatchat_results.zip")
    if isinstance(state, dict):
        reset_analysis_steps(state.get("session_id"))
        state["awaiting_method_choice"] = False
        state["selected_method"] = None
        state["awaiting_prune_confirmation"] = False
        state["pending_prune_predictors"] = []
        state["pending_model_request"] = {}
        state["model_prune_preference"] = None
        state["last_multicol_diag"] = {}

def _clean_dir(path: str, state=None):
    with _session_cwd(state):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

def _run_subprocess(cmd, cwd=None, env=None):
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        env=env,
    )

def _render_feature_viewer_html(state=None):
    try:
        if not isinstance(state, dict) or not state.get("session_root"):
            return "<div class='spatchat-values-empty'>No extracted values yet. Upload points and fetch layers to populate this viewer.</div>"
        with _session_cwd(state):
            return render_feature_popup_html(auto_open=False)
    except Exception as exc:
        return f"<div class='spatchat-values-meta'>Values viewer unavailable: {html_lib.escape(str(exc))}</div>"

def _extract_requested_predictor_names(user_msg: str):
    text = str(user_msg or "").strip().lower()
    if not text:
        return []
    names = set(re.findall(r"\b[a-z][a-z0-9_]*\b", text))
    ignore = {
        "remove", "drop", "exclude", "add", "restore", "readd", "re_add", "again",
        "variable", "variables", "predictor", "predictors", "please", "the", "and",
    }
    return sorted([n for n in names if n not in ignore and len(n) >= 3])

def _parse_multicol_thresholds(user_msg: str):
    txt = str(user_msg or "")
    corr = None
    vif = None
    m_corr = re.search(r"(?:corr(?:elation)?|rho)\s*(?:threshold)?\s*[:=]?\s*(0?\.\d+|1(?:\.0+)?)", txt, re.I)
    if m_corr:
        try:
            corr = float(m_corr.group(1))
        except Exception:
            corr = None
    m_vif = re.search(r"\bvif\s*(?:threshold)?\s*[:=]?\s*(\d+(?:\.\d+)?)", txt, re.I)
    if m_vif:
        try:
            vif = float(m_vif.group(1))
        except Exception:
            vif = None
    return corr, vif

def _is_yes(text: str) -> bool:
    return bool(re.search(r"\b(yes|y|ok|sure|go ahead|proceed|prune|apply)\b", str(text or "").lower()))

def _is_no(text: str) -> bool:
    return bool(re.search(r"\b(no|n|skip|dont|don't|keep)\b", str(text or "").lower()))

def _wants_pruning(text: str) -> bool:
    return bool(
        re.search(
            r"\b(prune|pruning|with\s+pruning|with\s+correction|correction|correct(?:ion)?\s+(?:for|of)?\s*multicol|remove\s+collinear)\b",
            str(text or "").lower(),
        )
    )

def detect_coords(df, fuzz_threshold=80):
    try:
        detected = detect_lonlat_columns(df)
        if detected:
            return detected[1], detected[0]
    except Exception:
        pass
    cols = list(df.columns)
    low  = [c.lower().strip() for c in cols]
    LAT_ALIASES = {
        'lat','latitude','y','y_coordinate','decilatitude','dec_latitude','dec lat',
        'decimallatitude','decimal latitude'
    }
    LON_ALIASES = {
        'lon','long','longitude','x','x_coordinate','decilongitude','dec_longitude',
        'dec longitude','decimallongitude','decimal longitude'
    }
    lat_idx = next((i for i,n in enumerate(low) if n in LAT_ALIASES), None)
    lon_idx = next((i for i,n in enumerate(low) if n in LON_ALIASES), None)
    if lat_idx is not None and lon_idx is not None:
        return cols[lat_idx], cols[lon_idx]
    lat_fz = difflib.get_close_matches("latitude", low, n=1, cutoff=fuzz_threshold/100)
    lon_fz = difflib.get_close_matches("longitude", low, n=1, cutoff=fuzz_threshold/100)
    if lat_fz and lon_fz:
        return cols[low.index(lat_fz[0])], cols[low.index(lon_fz[0])]
    numerics = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    lat_opts = [c for c in numerics if df[c].between(-90, 90).mean() > 0.98]
    lon_opts = [c for c in numerics if df[c].between(-180, 180).mean() > 0.98]
    if len(lat_opts) == 1 and len(lon_opts) == 1:
        return lat_opts[0], lon_opts[0]
    return None, None

DATA_TYPE_CHOICES = [
    "Auto-detect",
    "Presence-only",
    "Presence-absence",
    "Movement: animal ID + x/y/time",
    "Movement: x/y/time",
]

INPUT_METADATA_PATH = "inputs/data_metadata.json"
UPLOADED_DATA_PATH = "inputs/uploaded_data.csv"
PRESENCE_POINTS_PATH = "inputs/presence_points.csv"
UPLOADED_ABSENCE_PATH = "inputs/absence_points_uploaded.csv"
MOVEMENT_POINTS_PATH = "inputs/movement_points.csv"
USER_RASTER_RAW_DIR = "predictor_rasters/user_uploaded"
USER_RASTER_WGS84_DIR = "predictor_rasters/wgs84"
FETCH_GRID_RES_METERS = 30
FETCH_MAX_AUTO_PIXELS = 100_000_000

MODEL_METHODS = {
    "logistic_regression": {
        "label": "Logistic Regression SDM",
        "aliases": ("logistic", "logistic regression", "glm", "default"),
    },
    "issa": {
        "label": "Integrated Step Selection Analysis (iSSA)",
        "aliases": ("issa", "iSSA", "integrated step selection", "step selection"),
    },
}

def _norm_col_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

def _coords_look_wgs84(df, lat_col, lon_col):
    if not lat_col or not lon_col:
        return False
    try:
        lats, lons = _coerce_wgs84_arrays(df, lat_col, lon_col)
    except Exception:
        return False
    denom = max(1, len(df))
    return len(lats) > 0 and (len(lats) / denom) >= 0.90


def _estimate_fetch_grid_from_points(points_csv: str) -> dict:
    if not os.path.exists(points_csv):
        return {"ok": False, "reason": "presence_points.csv not found."}
    df = pd.read_csv(points_csv)
    if not {"latitude", "longitude"}.issubset(df.columns):
        return {"ok": False, "reason": "presence_points.csv missing latitude/longitude."}
    valid = pd.DataFrame({
        "latitude": pd.to_numeric(df["latitude"], errors="coerce"),
        "longitude": pd.to_numeric(df["longitude"], errors="coerce"),
    }).dropna()
    if valid.empty:
        return {"ok": False, "reason": "No valid WGS84 points found."}

    buffer = 0.25
    min_lat, max_lat = float(valid.latitude.min()), float(valid.latitude.max())
    min_lon, max_lon = float(valid.longitude.min()), float(valid.longitude.max())
    x_size = int((max_lon + buffer - (min_lon - buffer)) * (111320 / FETCH_GRID_RES_METERS))
    y_size = int((max_lat + buffer - (min_lat - buffer)) * (110540 / FETCH_GRID_RES_METERS))
    x_size = max(32, x_size)
    y_size = max(32, y_size)
    return {
        "ok": True,
        "x_size": int(x_size),
        "y_size": int(y_size),
        "pixels": int(x_size) * int(y_size),
        "min_lat": min_lat - buffer,
        "max_lat": max_lat + buffer,
        "min_lon": min_lon - buffer,
        "max_lon": max_lon + buffer,
    }


def _large_fetch_message(grid: dict, layers: list[str]) -> str:
    pixels = int(grid.get("pixels") or 0)
    x_size = int(grid.get("x_size") or 0)
    y_size = int(grid.get("y_size") or 0)
    gb = pixels * 4 / (1024 ** 3)
    return (
        "This study area is too large for Spatchat's server-side predictor fetch.\n\n"
        f"Requested predictors: {', '.join(layers)}\n\n"
        f"Estimated 30 m alignment grid: {x_size:,} x {y_size:,} cells "
        f"({pixels:,} pixels; about {gb:.1f} GB per float32 layer before overhead).\n\n"
        "Please download or prepare the predictor rasters yourself, then upload GeoTIFF rasters "
        "in the **Upload predictor rasters** box. Spatchat will reproject them to WGS84 if needed, "
        "extract values at the points, draw them on the map, and continue with diagnostics/modeling."
    )

def _requested_data_type(choice):
    text = str(choice or "Auto-detect").strip().lower()
    if "presence-absence" in text or "presence_absence" in text:
        return "presence_absence"
    if "animal" in text:
        return "movement_with_id"
    if "movement" in text:
        return "movement_no_id"
    if "presence-only" in text or "presence_only" in text:
        return "presence_only"
    return "auto"

def _detect_timestamp_col(df):
    aliases = {
        "timestamp", "time", "datetime", "date_time", "date", "fix_time",
        "event_date", "event_time", "event_date_utc", "observed_at",
        "location_timestamp", "timestamp_utc",
    }
    normalized = {_norm_col_name(c): c for c in df.columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for key, col in normalized.items():
        if "timestamp" in key or key in {"datetime", "date_time"}:
            return col
    return None

def _detect_animal_id_col(df):
    aliases = {
        "animal_id", "animalid", "individual_id", "individual_local_identifier",
        "individual_local_id", "individual", "tag_id", "tag_local_identifier",
        "deployment_id", "track_id", "collar_id", "device_id",
    }
    normalized = {_norm_col_name(c): c for c in df.columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for key, col in normalized.items():
        if ("animal" in key and "id" in key) or ("individual" in key and "id" in key):
            return col
    if "id" in normalized:
        return normalized["id"]
    return None

def _coerce_presence_labels(series):
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        uniq = set(vals.dropna().astype(float).unique().tolist())
        if uniq and uniq.issubset({0.0, 1.0}):
            return vals.map(lambda x: np.nan if pd.isna(x) else int(float(x) > 0))
        return None

    true_vals = {"1", "true", "t", "yes", "y", "present", "presence", "detected", "used"}
    false_vals = {"0", "false", "f", "no", "n", "absent", "absence", "not_detected", "available", "background"}
    cleaned = series.astype(str).str.strip().str.lower().str.replace(r"[\s-]+", "_", regex=True)
    labels = []
    seen_valid = False
    for val in cleaned:
        if val in true_vals:
            labels.append(1)
            seen_valid = True
        elif val in false_vals:
            labels.append(0)
            seen_valid = True
        elif val in {"", "nan", "none", "null"}:
            labels.append(np.nan)
        else:
            return None
    return pd.Series(labels, index=series.index, dtype="float") if seen_valid else None

def _detect_response_col(df, exclude_cols=None):
    exclude = {_norm_col_name(c) for c in (exclude_cols or []) if c}
    aliases = {
        "presence", "present", "presence_absence", "pres_abs", "pa",
        "occurrence", "occurrence_status", "detected", "detection",
        "label", "response", "class", "used",
    }
    candidates = []
    for col in df.columns:
        key = _norm_col_name(col)
        if key in exclude:
            continue
        if key in aliases or ("presence" in key and "absence" in key):
            candidates.insert(0, col)
        else:
            candidates.append(col)

    for col in candidates:
        labels = _coerce_presence_labels(df[col])
        if labels is None:
            continue
        valid = labels.dropna()
        if len(valid) and set(valid.astype(int).unique()).issubset({0, 1}):
            return col, labels
    return None, None

def _load_input_metadata():
    try:
        if os.path.exists(INPUT_METADATA_PATH):
            with open(INPUT_METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _write_input_metadata(meta):
    os.makedirs("inputs", exist_ok=True)
    with open(INPUT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def _normalize_upload(df, data_type_choice="Auto-detect", lat_col=None, lon_col=None, src_epsg=4326):
    os.makedirs("inputs", exist_ok=True)
    original_rows = int(len(df))
    requested = _requested_data_type(data_type_choice)
    detected_lat, detected_lon = detect_coords(df)
    lat_col = lat_col or detected_lat
    lon_col = lon_col or detected_lon
    if not lat_col or not lon_col:
        raise ValueError("I couldn't detect coordinate columns. Please choose the Y/latitude and X/longitude columns and enter the CRS.")

    coord_df = df.copy()
    y_vals = pd.to_numeric(coord_df[lat_col], errors="coerce")
    x_vals = pd.to_numeric(coord_df[lon_col], errors="coerce")
    valid_xy = y_vals.notna() & x_vals.notna()
    coord_df = coord_df.loc[valid_xy].copy()
    y_vals = y_vals.loc[valid_xy]
    x_vals = x_vals.loc[valid_xy]

    epsg = int(src_epsg or 4326)
    if epsg == 4326:
        lat_vals = y_vals.astype(float).to_numpy()
        lon_vals = x_vals.astype(float).to_numpy()
    else:
        src_crs = RioCRS.from_epsg(epsg)
        dst_crs = RioCRS.from_epsg(4326)
        lon_vals, lat_vals = rio_transform(src_crs, dst_crs, x_vals.tolist(), y_vals.tolist())
        lat_vals = np.asarray(lat_vals, dtype=float)
        lon_vals = np.asarray(lon_vals, dtype=float)

    coord_df["latitude"] = lat_vals
    coord_df["longitude"] = lon_vals
    valid_wgs = (
        np.isfinite(coord_df["latitude"]) & np.isfinite(coord_df["longitude"]) &
        coord_df["latitude"].between(-90, 90) &
        coord_df["longitude"].between(-180, 180)
    )
    coord_df = coord_df.loc[valid_wgs].copy()
    if coord_df.empty:
        raise ValueError("No rows had valid WGS84 coordinates after parsing or CRS transformation.")

    timestamp_col = _detect_timestamp_col(coord_df)
    animal_id_col = _detect_animal_id_col(coord_df)
    response_col, labels = _detect_response_col(coord_df, exclude_cols=[lat_col, lon_col, timestamp_col, animal_id_col])

    if requested == "auto":
        if response_col is not None and labels is not None and set(labels.dropna().astype(int).unique()) == {0, 1}:
            data_type = "presence_absence"
        elif timestamp_col:
            data_type = "movement_with_id" if animal_id_col else "movement_no_id"
        else:
            data_type = "presence_only"
    else:
        data_type = requested

    if data_type == "presence_absence":
        if response_col is None or labels is None:
            raise ValueError("Presence-absence data needs a binary column such as `presence`, `pa`, `label`, or `response` with 1/0 or present/absent values.")
        coord_df["_presence_label"] = labels.loc[coord_df.index].astype("float")
        coord_df = coord_df.loc[coord_df["_presence_label"].isin([0.0, 1.0])].copy()
        pres_df = coord_df.loc[coord_df["_presence_label"] == 1.0].copy()
        abs_df = coord_df.loc[coord_df["_presence_label"] == 0.0].copy()
        if pres_df.empty or abs_df.empty:
            raise ValueError("Presence-absence data must include at least one presence row and one absence row.")
        pres_out = pres_df.drop(columns=["_presence_label"], errors="ignore").copy()
        abs_out = abs_df.drop(columns=["_presence_label"], errors="ignore").copy()
        pres_out["label"] = 1
        abs_out["label"] = 0
        pres_out.to_csv(PRESENCE_POINTS_PATH, index=False)
        abs_out.to_csv(UPLOADED_ABSENCE_PATH, index=False)
        n_presence, n_absence = len(pres_out), len(abs_out)
    else:
        pres_out = coord_df.copy()
        pres_out["label"] = 1
        pres_out.to_csv(PRESENCE_POINTS_PATH, index=False)
        n_presence, n_absence = len(pres_out), 0
        if os.path.exists(UPLOADED_ABSENCE_PATH):
            os.remove(UPLOADED_ABSENCE_PATH)

    if data_type in {"movement_with_id", "movement_no_id"}:
        if timestamp_col is None:
            raise ValueError("Movement data needs a timestamp column.")
        move_df = coord_df.copy()
        if animal_id_col:
            move_df["animal_id"] = move_df[animal_id_col].astype(str)
        else:
            move_df["animal_id"] = "track_1"
        move_df["timestamp"] = move_df[timestamp_col]
        cols = ["animal_id", "timestamp", "latitude", "longitude"]
        extra_cols = [c for c in move_df.columns if c not in cols]
        move_df[cols + extra_cols].to_csv(MOVEMENT_POINTS_PATH, index=False)
    elif os.path.exists(MOVEMENT_POINTS_PATH):
        os.remove(MOVEMENT_POINTS_PATH)

    meta = {
        "data_type": data_type,
        "requested_data_type": str(data_type_choice or "Auto-detect"),
        "coordinate_columns": {"y_or_latitude": lat_col, "x_or_longitude": lon_col, "epsg": epsg},
        "response_column": response_col,
        "timestamp_column": timestamp_col,
        "animal_id_column": animal_id_col,
        "n_rows_uploaded": original_rows,
        "n_rows_with_valid_coordinates": int(len(coord_df)),
        "n_presence": int(n_presence),
        "n_absence": int(n_absence),
    }
    _write_input_metadata(meta)
    return meta

def _upload_message(meta, state):
    dtype_labels = {
        "presence_only": "presence-only",
        "presence_absence": "presence-absence",
        "movement_with_id": "movement with animal IDs",
        "movement_no_id": "movement without animal IDs",
    }
    dtype = dtype_labels.get(meta.get("data_type"), meta.get("data_type", "uploaded"))
    msg = (
        f"Loaded {dtype} data with {meta.get('n_presence', 0)} presence/location rows"
        f" and {meta.get('n_absence', 0)} absence rows."
    )
    if str(meta.get("data_type", "")).startswith("movement"):
        msg += " Movement fixes were saved for iSSA. Recommended method: `issa`."
    if meta.get("data_type") == "presence_absence":
        msg += " Logistic regression will use the uploaded absences instead of generated background points."
    if not state.get("layers_help_shown"):
        msg += "\n\n" + available_layers_markdown()
        state["layers_help_shown"] = True
    return msg

def _normalize_method_name(raw):
    text = str(raw or "").strip().lower()
    if not text:
        return None
    for key, spec in MODEL_METHODS.items():
        if text == key or text in {a.lower() for a in spec.get("aliases", ())}:
            return key
    return None

def _detect_method_from_text(user_msg):
    msg = str(user_msg or "").strip()
    if not msg:
        return None
    lower_msg = msg.lower()
    if re.search(r"\bissa\b", lower_msg) or "integrated step selection" in lower_msg:
        return "issa"
    if "logistic regression" in lower_msg or "logistic_regression" in lower_msg or re.search(r"\bglm\b", lower_msg):
        return "logistic_regression"
    return None

def _is_model_request(user_msg):
    txt = str(user_msg or "").strip().lower()
    if not txt:
        return False
    if re.fullmatch(r"\s*(?:run|fit|train|build|create|start)?\s*(?:sdm|model|analysis)\s*", txt):
        return True
    return bool(re.search(r"\b(run|fit|train|build|create|start)\b.*\b(model|sdm|analysis)\b", txt))


def _is_method_explanation_question(user_msg: str, method_hint: str | None = None) -> bool:
    txt = str(user_msg or "").strip().lower()
    if not txt:
        return False
    question_mark = "?" in txt
    expl_phrase = bool(re.search(r"\b(what is|what's|how does|how do|explain|tell me about|definition of|mean)\b", txt))
    mentions_method = bool(method_hint) or bool(re.search(r"\b(logistic regression|logistic_regression|glm|issa|integrated step selection)\b", txt))
    return mentions_method and (question_mark or expl_phrase)


def _is_method_run_command(user_msg: str, method_hint: str | None = None) -> bool:
    txt = str(user_msg or "").strip().lower()
    if not txt:
        return False
    if _is_method_explanation_question(txt, method_hint):
        return False
    mentions_method = bool(method_hint) or bool(re.search(r"\b(logistic regression|logistic_regression|glm|issa|integrated step selection)\b", txt))
    if not mentions_method:
        return False
    return bool(re.search(r"\b(run|fit|train|build|create|start|use|give me|do)\b", txt))


def _has_movement_data(state) -> bool:
    with _session_cwd(state):
        meta = _load_input_metadata()
        data_type = str(meta.get("data_type", "") or "").strip().lower()
        return os.path.exists(MOVEMENT_POINTS_PATH) or data_type.startswith("movement")

def _method_guidance_message(state):
    has_movement = _has_movement_data(state)
    recommended = "issa" if has_movement else "logistic_regression"
    reason = (
        "movement data detected"
        if has_movement
        else "non-movement SDM data detected"
    )
    rec_label = MODEL_METHODS[recommended]["label"]
    if not has_movement:
        return (
            "Which method should I run?\n\n"
            f"- `logistic_regression`: {MODEL_METHODS['logistic_regression']['label']}\n\n"
            f"Recommended now: **{rec_label}** ({reason}).\n"
            "iSSA is hidden because no movement data with timestamps is loaded.\n"
            "Reply with `logistic_regression`."
        )
    return (
        "Which method should I run?\n\n"
        f"- `logistic_regression`: {MODEL_METHODS['logistic_regression']['label']}\n"
        f"- `issa`: {MODEL_METHODS['issa']['label']}\n\n"
        f"Recommended now: **{rec_label}** ({reason}).\n"
        "Reply with `logistic_regression` or `issa`."
    )


# --- CRS parsing helpers (handle 'UTM 10T') ---
def parse_epsg_code(s: str):
    m = re.search(r'(?:^|\b)epsg\s*:\s*(\d{4,5})\b', s, re.I)
    if not m:
        m = re.match(r'^\s*(\d{4,5})\s*$', s.strip())
    return int(m.group(1)) if m else None

def parse_utm_crs(s: str):
    txt = s.strip()
    patterns = [
        r'\butm\b[^0-9]*?(\d{1,2})\s*([A-Za-z])?',
        r'\bzone\s*(\d{1,2})\s*([A-Za-z])?',
        r'\b(\d{1,2})\s*([C-HJ-NP-X])\b',
        r'\b(\d{1,2})\s*([NS])\b',
    ]
    m = None
    for p in patterns:
        m = re.search(p, txt, re.I)
        if m:
            break
    if not m:
        return None
    zone = int(m.group(1))
    letter = (m.group(2) or '').upper()
    if letter in ('N', 'S'):
        hemi = 'N' if letter == 'N' else 'S'
    elif letter:
        hemi = 'N' if letter >= 'N' else 'S'
    else:
        hemi = 'N'
    return (32600 if hemi == 'N' else 32700) + zone

def llm_parse_crs(raw):
    system = {"role":"system","content":"You're a GIS expert. Given a CRS description, respond with only JSON {\"epsg\": ###} or {\"epsg\": null}."}
    user = {"role":"user","content":f"CRS: '{raw}'"}
    resp = llm.chat([system, user], temperature=0.0, max_tokens=32, stream=False)
    code = json.loads(resp).get("epsg")
    if not code:
        raise ValueError("LLM couldn't parse CRS")
    return code

def resolve_crs(raw):
    try:
        return parse_crs_input(raw)
    except Exception:
        pass
    for fn in (parse_epsg_code, parse_utm_crs):
        code = fn(raw)
        if code:
            return code
    return llm_parse_crs(raw)

SYSTEM_PROMPT = """
You are Spatchat, a friendly species distribution modeling assistant.
When the user asks to fetch environmental layers (using verbs like fetch, download, get, grab, "I want", etc.), respond with exactly a JSON object:
{"tool":"fetch","layers":[<layer names>],"landcover":[<landcover classes>]}
When the user asks to run the model (e.g., "run model", "run species distribution model", "run SDM", etc.), respond with exactly:
{"tool":"run_model","method":null}
If the user explicitly names a method, set `method` to one of:
- "logistic_regression"
- "issa"
If the user's request does not match either of these intents, reply naturally without JSON.
Examples:
User: I want bio2 and ndvi
Assistant: {"tool":"fetch","layers":["bio2","ndvi"],"landcover":[]}
User: Grab slope, elevation
Assistant: {"tool":"fetch","layers":["slope","elevation"],"landcover":[]}
User: Run model now
Assistant: {"tool":"run_model","method":null}
User: Run iSSA
Assistant: {"tool":"run_model","method":"issa"}
User: How many points are uploaded?
Assistant: There are currently 193 presence points uploaded.
""".strip()

FALLBACK_PROMPT = """
You are Spatchat, a friendly assistant for species distribution modeling.
Keep your answers short—no more than two sentences—while still being helpful.
Guide the user to next steps: upload data, fetch layers, run model, etc.
""".strip()

# ──────────────────────────────────────────────────────────────
# CV formatting + warning helpers
# ──────────────────────────────────────────────────────────────
GENERAL_CHAT_PROMPT = """
You are Spatchat, a Landscape Ecologist and species distribution modeling assistant.
Answer naturally and directly using the user's wording and context.
For method/explanation questions, provide clear ecological/statistical explanations without returning tool JSON.
When helpful, mention practical SDM next steps.
""".strip()

ROUTER_PROMPT = """
Classify the user's message into one action for Spatchat.
Return ONLY JSON with this schema:
{
  "action": "fetch_layers|run_model|remove_predictors|restore_predictors|check_multicollinearity|check_morans_i|check_both_diagnostics|reset_session|prune_yes|prune_no|general_chat",
  "method": null|"logistic_regression"|"issa",
  "layers": [string],
  "landcover": [string],
  "predictors": [string],
  "corr_threshold": null|number,
  "vif_threshold": null|number,
  "prune_preference": null|"apply"
}
Rules:
- Use `general_chat` for explanations, random questions, and non-execution requests.
- Use `run_model` only when user asks to execute/build/train/run a model.
- Use `fetch_layers` when requesting predictors/layers/environmental rasters.
- Use diagnostics actions for multicollinearity/Moran requests.
- Use prune_yes/prune_no only as a response to a pending yes/no prune confirmation.
- Use reset_session for clear/reset/start over.
Do not include any keys beyond this schema.
""".strip()

def _heuristic_route_intent(user_msg: str) -> dict:
    msg = str(user_msg or "")
    lower_msg = msg.lower()
    parsed_layers, parsed_classes = try_parse_fetch_from_text(msg)
    if parsed_layers is not None:
        return {
            "action": "fetch_layers",
            "method": None,
            "layers": parsed_layers,
            "landcover": parsed_classes,
            "predictors": [],
            "corr_threshold": None,
            "vif_threshold": None,
            "prune_preference": "apply" if _wants_pruning(msg) else None,
        }
    if re.search(r"\b(start over|restart|clear everything|reset|clear all)\b", lower_msg):
        return {"action": "reset_session"}
    if _is_yes(msg):
        return {"action": "prune_yes"}
    if _is_no(msg):
        return {"action": "prune_no"}
    asks_multicol = bool(re.search(r"\bmulticollinearity\b|\bvif\b|\bcollinearity\b|\bcorrelation\b", lower_msg))
    asks_moran = bool(re.search(r"\bmoran\b|\bspatial independence\b|\bspatial autocorrelation\b", lower_msg))
    if asks_multicol and asks_moran:
        c_thr, v_thr = _parse_multicol_thresholds(msg)
        return {"action": "check_both_diagnostics", "corr_threshold": c_thr, "vif_threshold": v_thr}
    if asks_multicol:
        c_thr, v_thr = _parse_multicol_thresholds(msg)
        return {"action": "check_multicollinearity", "corr_threshold": c_thr, "vif_threshold": v_thr}
    if asks_moran:
        return {"action": "check_morans_i"}
    if re.search(r"\b(remove|drop|exclude)\b", lower_msg):
        return {"action": "remove_predictors", "predictors": _extract_requested_predictor_names(msg)}
    if re.search(r"\b(add|restore|re-add|readd)\b", lower_msg):
        return {"action": "restore_predictors", "predictors": _extract_requested_predictor_names(msg)}
    method = _detect_method_from_text(msg) or _normalize_method_name(msg)
    if _is_model_request(msg) or _is_method_run_command(msg, method):
        return {
            "action": "run_model",
            "method": method,
            "prune_preference": "apply" if _wants_pruning(msg) else None,
        }
    return {"action": "general_chat"}

def _llm_route_intent(user_msg: str, state) -> dict:
    has_movement = _has_movement_data(state)
    router_messages = [
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "system", "content": f"Movement data available: {'yes' if has_movement else 'no'}"},
        {"role": "user", "content": str(user_msg or "")},
    ]
    raw = _safe_llm_chat(
        router_messages,
        temperature=0.0,
        max_tokens=260,
        stream=False,
        default="",
        log_errors=False,
    )
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("router output was not an object")
    except Exception:
        return _heuristic_route_intent(user_msg)

    action = str(parsed.get("action", "general_chat")).strip()
    allowed_actions = {
        "fetch_layers",
        "run_model",
        "remove_predictors",
        "restore_predictors",
        "check_multicollinearity",
        "check_morans_i",
        "check_both_diagnostics",
        "reset_session",
        "prune_yes",
        "prune_no",
        "general_chat",
    }
    if action not in allowed_actions:
        action = "general_chat"

    method = _normalize_method_name(parsed.get("method"))

    def _as_str_list(value):
        if not isinstance(value, list):
            return []
        out = []
        for item in value:
            text = str(item or "").strip().lower()
            if text:
                out.append(text)
        return out

    layers = []
    for name in _as_str_list(parsed.get("layers")):
        if re.fullmatch(r"bio(?:[1-9]|1[0-9])", name) or name in VALID_LAYERS:
            layers.append(name)
    layers = sorted(set(layers))

    landcover = [c for c in _as_str_list(parsed.get("landcover")) if c in LANDCOVER_CLASSES]
    landcover = sorted(set(landcover))

    predictors = sorted(set(_as_str_list(parsed.get("predictors"))))
    if not predictors and action in {"remove_predictors", "restore_predictors"}:
        predictors = _extract_requested_predictor_names(user_msg)

    corr_thr = parsed.get("corr_threshold")
    vif_thr = parsed.get("vif_threshold")
    try:
        corr_thr = float(corr_thr) if corr_thr is not None else None
    except Exception:
        corr_thr = None
    try:
        vif_thr = float(vif_thr) if vif_thr is not None else None
    except Exception:
        vif_thr = None

    prune_pref_raw = str(parsed.get("prune_preference") or "").strip().lower()
    prune_preference = "apply" if prune_pref_raw == "apply" else None

    if action == "fetch_layers" and not layers:
        fallback_layers, fallback_classes = try_parse_fetch_from_text(user_msg)
        if fallback_layers is not None:
            layers = fallback_layers
            landcover = fallback_classes

    return {
        "action": action,
        "method": method,
        "layers": layers,
        "landcover": landcover,
        "predictors": predictors,
        "corr_threshold": corr_thr,
        "vif_threshold": vif_thr,
        "prune_preference": prune_preference,
    }

def _general_llm_reply(user_msg: str, history, state) -> str:
    with _session_cwd(state):
        has_presence = os.path.exists(PRESENCE_POINTS_PATH)
        meta = _load_input_metadata()
        data_type = str(meta.get("data_type", "not_loaded") or "not_loaded")
        rasdir = USER_RASTER_WGS84_DIR
        fetched = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(rasdir)
            if f.endswith(".tif")
        ) if os.path.isdir(rasdir) else []
    context_note = (
        f"Session context: data_type={data_type}; has_presence_points={'yes' if has_presence else 'no'}; "
        f"active_predictor_count={len(fetched)}; active_predictors={', '.join(fetched) if fetched else 'none'}."
    )
    msgs = (
        [{"role": "system", "content": GENERAL_CHAT_PROMPT},
         {"role": "system", "content": context_note}]
        + list(history or [])[-10:]
        + [{"role": "user", "content": str(user_msg or "")}]
    )
    return _safe_llm_chat(
        msgs,
        temperature=0.6,
        max_tokens=420,
        stream=False,
        default="I can help with SDM tasks and related ecology/statistics questions. Ask anything and I'll adapt to your goal.",
    )

def _getv(row, k):
    return row[k] if (k in row and pd.notna(row[k])) else np.nan

def _fmt(x):
    return "—" if pd.isna(x) else f"{float(x):.3f}"

def _fmt_pm(mu, sd):
    if pd.isna(mu):
        return "—"
    return f"{float(mu):.3f}" + (f" ± {float(sd):.3f}" if not pd.isna(sd) else "")

def build_cv_section(cv_row, long_names=False):
    """
    Returns (perf_md_table:str, folds:int, warning_md:str)
    Detects cv_method and emits a clear warning if using non-spatial K-fold fallback.
    """
    folds = int(_getv(cv_row, "n_folds")) if not pd.isna(_getv(cv_row, "n_folds")) else 0
    cv_method = _getv(cv_row, "cv_method")
    cv_method = str(cv_method) if not pd.isna(cv_method) else "spatial_blocks"

    # Optional params that might exist
    k = _getv(cv_row, "kfold_k")
    r = _getv(cv_row, "kfold_repeats")

    # Metric strings
    auc_str  = _fmt_pm(_getv(cv_row, "AUC_mean"),   _getv(cv_row, "AUC_sd"))
    tss_str  = _fmt_pm(_getv(cv_row, "TSS_mean"),   _getv(cv_row, "TSS_sd"))
    kap_str  = _fmt_pm(_getv(cv_row, "Kappa_mean"), _getv(cv_row, "Kappa_sd"))
    if long_names:
        sens_str = _fmt_pm(_getv(cv_row, "Sensitivity_mean"), _getv(cv_row, "Sensitivity_sd"))
        spec_str = _fmt_pm(_getv(cv_row, "Specificity_mean"), _getv(cv_row, "Specificity_sd"))
        thr_str  = _fmt_pm(_getv(cv_row, "Threshold_mean"),   _getv(cv_row, "Threshold_sd"))
    else:
        sens_str = _fmt(_getv(cv_row, "Sensitivity_mean"))
        spec_str = _fmt(_getv(cv_row, "Specificity_mean"))
        thr_str  = _fmt(_getv(cv_row, "Threshold_mean"))

    headers = ("Sensitivity" if long_names else "Sens",
               "Specificity" if long_names else "Spec",
               "Threshold" if long_names else "Thr")

    perf_md = (
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"| AUC | {auc_str} |\n"
        f"| TSS | {tss_str} |\n"
        f"| Kappa | {kap_str} |\n"
        f"| {headers[0]} | {sens_str} |\n"
        f"| {headers[1]} | {spec_str} |\n"
        f"| {headers[2]} | {thr_str} |\n"
    )

    # Warning logic
    warn = ""
    if folds == 0:
        warn = (
            "> ⚠️ **No cross-validated metrics**. Not enough spatial spread for blocked CV and no fallback succeeded. "
            "Metrics below will show as em dashes. Consider adding more presences or widening the extent."
        )
    elif cv_method.lower() in {"kfold", "repeated_kfold", "nonspatial_kfold"}:
        details = []
        if not pd.isna(k): details.append(f"k={int(k)}")
        if not pd.isna(r): details.append(f"repeats={int(r)}")
        extra = f" ({', '.join(details)})" if details else ""
        warn = (
            f"> ⚠️ **Non-spatial K-fold fallback{extra}** — spatial blocked CV was not feasible with the current points/extent. "
            "These estimates can be optimistic if nearby points leak information across folds."
        )
    elif cv_method.lower() == "presence_absence_repeated_kfold":
        warn = "> Info: uploaded absence rows were used with repeated stratified K-fold cross-validation."
    elif cv_method.lower() not in {"spatial_blocks", "kmeans_blocks"}:
        warn = f"> ℹ️ CV method: **{cv_method}**."

    return perf_md, folds, warn

# ──────────────────────────────────────────────────────────────
# Mapping helpers (robust numeric coercion and WGS84 checks)
# ──────────────────────────────────────────────────────────────

def _coerce_wgs84_arrays(df, lat_col, lon_col):
    """
    Coerce the given lat/lon columns to numeric float arrays and filter to plausible WGS84 ranges.
    Returns (lats: np.ndarray, lons: np.ndarray) with NaNs/invalids removed.
    """
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    arr = np.column_stack([lat.values, lon.values]).astype(float)
    mask = np.isfinite(arr).all(axis=1)
    mask &= (arr[:, 0] >= -90) & (arr[:, 0] <= 90) & (arr[:, 1] >= -180) & (arr[:, 1] <= 180)
    return arr[mask, 0], arr[mask, 1]

def create_map(state=None):
    session_root = state.get("session_root") if isinstance(state, dict) else None
    return render_sdm_map(
        session_root=session_root,
        detect_coords_fn=detect_coords,
        colorbar_base64=COLORBAR_BASE64,
        uploaded_absence_path=UPLOADED_ABSENCE_PATH,
    )


def _write_repro_scripts(state=None) -> list[str]:
    with _session_cwd(state):
        script_dir = "reproducible_scripts"
        shutil.rmtree(script_dir, ignore_errors=True)
        os.makedirs(script_dir, exist_ok=True)
        created = []

        input_files = [
            ("inputs/presence_points.csv", "presence_points.csv"),
            ("inputs/movement_points.csv", "movement_points.csv"),
            ("inputs/absence_points_uploaded.csv", "absence_points_uploaded.csv"),
            ("inputs/feature_workspace.csv", "feature_workspace.csv"),
            ("inputs/data_metadata.json", "data_metadata.json"),
            ("inputs/feature_workspace_meta.json", "feature_workspace_meta.json"),
        ]
        copied_inputs = []
        for src, name in input_files:
            if os.path.exists(src):
                dst = os.path.join(script_dir, name)
                shutil.copy2(src, dst)
                copied_inputs.append(name)
                created.append(dst)

        copied_outputs = []
        output_files = [
            "performance_metrics_cv.csv",
            "performance_metrics_fitted.csv",
            "performance_metrics.csv",
            "coefficients.csv",
            "issa_summary.csv",
            "issa_coefficients.csv",
            "issa_projection_metrics.csv",
            "issa_projection_coefficients.csv",
            "dropped_predictors.csv",
            "multicollinearity_check.csv",
            "high_correlation_pairs.csv",
            "standardization_stats.csv",
            "sdm_point_samples_raw.csv",
            "sdm_point_samples_standardized.csv",
            "suitability_map_wgs84.tif",
            "multicollinearity_workspace_summary.csv",
            "multicollinearity_workspace_correlation_matrix.csv",
            "multicollinearity_workspace_pairs.csv",
            "multicollinearity_workspace_prune_candidates.csv",
            "pruned_variables_report.csv",
            "predictor_effect_plots_manifest.csv",
            "spatial_independence_morans_i.csv",
        ]
        outputs_dir = os.path.join(script_dir, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        for name in output_files:
            src = os.path.join("outputs", name)
            if os.path.exists(src):
                dst = os.path.join(outputs_dir, name)
                shutil.copy2(src, dst)
                copied_outputs.append(os.path.join("outputs", name))
                created.append(dst)

        effect_plot_dir = os.path.join("outputs", "predictor_effect_plots")
        if os.path.isdir(effect_plot_dir):
            dst_dir = os.path.join(outputs_dir, "predictor_effect_plots")
            os.makedirs(dst_dir, exist_ok=True)
            for fn in sorted(os.listdir(effect_plot_dir)):
                if not fn.lower().endswith(".png"):
                    continue
                src = os.path.join(effect_plot_dir, fn)
                dst = os.path.join(dst_dir, fn)
                shutil.copy2(src, dst)
                copied_outputs.append(os.path.join("outputs", "predictor_effect_plots", fn))
                created.append(dst)

        raster_manifest = []
        rasdir = os.path.join("predictor_rasters", "wgs84")
        if os.path.isdir(rasdir):
            for fn in sorted(os.listdir(rasdir)):
                if fn.lower().endswith(".tif"):
                    raster_manifest.append(os.path.join("..", "predictor_rasters", "wgs84", fn))

        config = {
            "room": "spatchat-sdm",
            "inputs": copied_inputs,
            "outputs": copied_outputs,
            "predictor_rasters": raster_manifest,
            "active_predictors": [],
            "selected_method": (state or {}).get("selected_method") if isinstance(state, dict) else None,
        }
        try:
            config["active_predictors"] = get_active_predictors()
        except Exception:
            config["active_predictors"] = []
        config_path = os.path.join(script_dir, "repro_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        created.append(config_path)

        for filename in (
            "README.txt",
            "preprocessing_reproduce.txt",
            "logistic_regression_reproduce.txt",
            "issa_reproduce.txt",
            "postprocessing_reproduce.txt",
            "postprocessing_effect_plots.py",
        ):
            if filename == "postprocessing_effect_plots.py":
                tpl = os.path.join(REPRO_TEMPLATE_DIR, filename)
                fallback = os.path.join(APP_ROOT, "workflow", "effect_plots.py")
                src = tpl if os.path.exists(tpl) else fallback
            else:
                src = os.path.join(REPRO_TEMPLATE_DIR, filename)
            if os.path.exists(src):
                dst = os.path.join(script_dir, filename)
                shutil.copyfile(src, dst)
                created.append(dst)
        return created


def zip_results(state=None):
    with _session_cwd(state):
        try:
            _write_repro_scripts(state)
        except Exception as exc:
            print(f"Warning: failed to write reproducible scripts: {exc}", file=sys.stderr)
        archive = "spatchat_results.zip"
        if os.path.exists(archive):
            os.remove(archive)
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for fld in ("predictor_rasters", "outputs", "inputs", "reproducible_scripts"):
                if not os.path.exists(fld):
                    continue
                for root, _, files in os.walk(fld):
                    for fn in files:
                        full = os.path.join(root, fn)
                        zf.write(full, arcname=os.path.relpath(full, "."))
        return os.path.abspath(archive)


def _sanitize_raster_name(path: str, used: set[str]) -> str:
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_").lower() or "predictor"
    base = name
    i = 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def _copy_raster_to_wgs84(src_path: str, out_name: str) -> str:
    os.makedirs(USER_RASTER_RAW_DIR, exist_ok=True)
    os.makedirs(USER_RASTER_WGS84_DIR, exist_ok=True)
    raw_path = os.path.join(USER_RASTER_RAW_DIR, f"{out_name}{os.path.splitext(src_path)[1].lower()}")
    shutil.copy(src_path, raw_path)
    out_path = os.path.join(USER_RASTER_WGS84_DIR, f"{out_name}.tif")

    with rasterio.open(raw_path) as src:
        if src.crs is None:
            raise ValueError("missing CRS; please upload a georeferenced GeoTIFF")
        src_crs = src.crs
        dst_crs = RioCRS.from_epsg(4326)
        if src_crs == dst_crs:
            profile = src.profile.copy()
            profile.update(driver="GTiff", crs=dst_crs, count=1)
            data = src.read(1)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data, 1)
            return out_path

        transform, width, height = calculate_default_transform(
            src_crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
        )
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            count=1,
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
    return out_path


def on_raster_upload(files, history, state):
    history2 = _history_to_messages(history)
    state = _ensure_session_workspace(state)
    if not files:
        history2.append({"role": "assistant", "content": "No raster files were uploaded."})
        return _history_for_chatbot(history2), create_map(state), state, gr.update(), gr.update(value=_render_feature_viewer_html(state))

    if not isinstance(files, list):
        files = [files]

    added = []
    failed = []
    with _session_cwd(state):
        existing = {
            os.path.splitext(fn)[0].lower()
            for fn in os.listdir(USER_RASTER_WGS84_DIR)
            if fn.lower().endswith(".tif")
        } if os.path.isdir(USER_RASTER_WGS84_DIR) else set()
        for f in files:
            src_path = f.name if hasattr(f, "name") else str(f)
            if not src_path or not os.path.exists(src_path):
                failed.append(f"{os.path.basename(str(src_path)) or 'unknown'}: file not found")
                continue
            if not src_path.lower().endswith((".tif", ".tiff")):
                failed.append(f"{os.path.basename(src_path)}: not a GeoTIFF")
                continue
            try:
                out_name = _sanitize_raster_name(src_path, existing)
                _copy_raster_to_wgs84(src_path, out_name)
                added.append(out_name)
            except Exception as exc:
                failed.append(f"{os.path.basename(src_path)}: {exc}")

        ws_msg = ""
        try:
            ws = refresh_workspace_from_rasters()
            if ws.get("ok"):
                new_predictors = ws.get("new_predictors", [])
                ws_msg = (
                    f"\n\nExtracted raster values for {len(new_predictors)} new predictor(s): "
                    f"{', '.join(new_predictors)}."
                    if new_predictors
                    else "\n\nExtracted values table is up to date."
                )
            else:
                ws_msg = f"\n\nValue extraction did not run: {ws.get('message', 'unknown error')}"
        except Exception as exc:
            ws_msg = f"\n\nRasters were copied, but value extraction failed: {exc}"

    msg_parts = []
    if added:
        msg_parts.append(f"Uploaded predictor raster(s): {', '.join(added)}.")
    if failed:
        msg_parts.append("Skipped file(s):\n" + "\n".join(f"- {x}" for x in failed))
    msg_parts.append(ws_msg.strip())
    history2.append({"role": "assistant", "content": "\n\n".join(p for p in msg_parts if p)})
    return _history_for_chatbot(history2), create_map(state), state, gr.update(), gr.update(value=_render_feature_viewer_html(state))


def run_fetch(sl, lc, state=None):
    global EE_INITIALIZED, EE_AUTH_ERROR
    def _public_ee_error_text(raw_error: str) -> str:
        txt = str(raw_error or "").strip()
        low = txt.lower()
        if ("earthengine authenticate" in low) or ("invalid_grant" in low) or ("authorization" in low):
            return "Server Earth Engine credentials are missing or invalid."
        return txt or "Server Earth Engine credentials are not available."
    with _session_cwd(state) as session_root:
        os.makedirs("predictor_rasters", exist_ok=True)
        os.makedirs("predictor_rasters/wgs84", exist_ok=True)

        sl = [s.lower() for s in sl]
        lc = [c.lower() for c in lc]

        layers = list(sl)
        if lc and "landcover" not in layers:
            layers.append("landcover")

        if not layers:
            return create_map(state), "Please select at least one predictor."

        grid = _estimate_fetch_grid_from_points(PRESENCE_POINTS_PATH)
        if grid.get("ok") and int(grid.get("pixels") or 0) > FETCH_MAX_AUTO_PIXELS:
            return create_map(state), _large_fetch_message(grid, layers)

        if not EE_INITIALIZED:
            try:
                _initialize_earth_engine()
                EE_INITIALIZED = True
                EE_AUTH_ERROR = None
            except Exception as exc:
                EE_INITIALIZED = False
                EE_AUTH_ERROR = str(exc)
                return (
                    create_map(state),
                    "Earth Engine is not configured for this process.\n\n"
                    f"{_public_ee_error_text(EE_AUTH_ERROR)}\n\n"
                    "Admin action required: configure `GEE_SERVICE_ACCOUNT_JSON` or legacy `GEE_SERVICE_ACCOUNT` (full JSON), or `GEE_SERVICE_ACCOUNT_FILE`/`GOOGLE_APPLICATION_CREDENTIALS` (path) on the server, then retry."
                )

        bad_layers = [l for l in layers if l not in VALID_LAYERS]
        if bad_layers:
            suggestions = []
            for b in bad_layers:
                match = difflib.get_close_matches(b, VALID_LAYERS, n=1, cutoff=0.6)
                if match:
                    suggestions.append(f"Did you mean '{match[0]}' instead of '{b}'?")
            if suggestions:
                return create_map(state), " ".join(suggestions)
            prompt = (
                f"You requested these predictors: {', '.join(layers)}. "
                f"I don't recognize: {', '.join(bad_layers)}. "
                "Could you please clarify which predictors you want?"
            )
            clar = _safe_llm_chat(
                messages=[{"role": "system", "content": FALLBACK_PROMPT},
                          {"role": "user",   "content": prompt}],
                temperature=0.7, max_tokens=256, stream=False, default=prompt,
            )
            return create_map(state), clar

        bad_codes = [c for c in lc if c not in LANDCOVER_CLASSES]
        if bad_codes:
            suggestions = []
            for b in bad_codes:
                match = difflib.get_close_matches(b, LANDCOVER_CLASSES, n=1, cutoff=0.6)
                if match:
                    suggestions.append(f"Did you mean landcover class '{match[0]}' instead of '{b}'?")
            if suggestions:
                return create_map(state), " ".join(suggestions)
            prompt = (
                f"You requested landcover classes: {', '.join(lc)}. "
                f"I don't recognize: {', '.join(bad_codes)}. "
                "Could you please clarify which landcover classes you want?"
            )
            clar = _safe_llm_chat(
                messages=[{"role": "system", "content": FALLBACK_PROMPT},
                          {"role": "user",   "content": prompt}],
                temperature=0.7, max_tokens=256, stream=False, default=prompt,
            )
            return create_map(state), clar

        sl_env = list(sl)
        if lc and "landcover" not in sl_env:
            sl_env.append("landcover")

        run_env = os.environ.copy()
        run_env["SELECTED_LAYERS"] = ",".join(sl_env)
        run_env["SELECTED_LANDCOVER_CLASSES"] = ",".join(lc)

        cmd = [sys.executable, "-u", os.path.join(APP_ROOT, "workflow", "fetch_predictors.py")]
        proc = _run_subprocess(cmd, cwd=session_root, env=run_env)
        logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        record_step(
            kind="predictor_fetch",
            function="workflow.fetch_predictors",
            inputs={"layers": sl_env, "landcover_classes": lc},
            outputs={"returncode": proc.returncode},
            session_id=(state or {}).get("session_id") if isinstance(state, dict) else None,
        )
        if proc.returncode != 0:
            return create_map(state), f"Fetch failed:\n```\n{logs}\n```"
        ws_msg = ""
        try:
            ws = refresh_workspace_from_rasters()
            if ws.get("ok"):
                added = ws.get("new_predictors", [])
                if added:
                    ws_msg = f"\n\nExtracted values to points for {len(added)} new predictor(s): {', '.join(added)}."
                else:
                    ws_msg = "\n\nExtracted values table is up to date."
        except Exception as exc:
            ws_msg = f"\n\nPredictors fetched, but value extraction failed: {exc}"
        if logs.strip():
            return create_map(state), f"Predictors fetched.\n\n```bash\n{logs}\n```{ws_msg}"
        return create_map(state), f"Predictors fetched.{ws_msg}"
# --- Deterministic “I want …” fetch intent (handles spaces in landcover names) ---
def try_parse_fetch_from_text(user_msg: str):
    text = user_msg.lower()
    tokens = re.findall(r'[a-zA-Z_]+\d*|\d+', text)
    has_fetch_verb = bool(re.search(r'\b(fetch|get|grab|download|want|need|add)\b', user_msg, re.I))
    predictor_like_tokens = {
        t for t in tokens
        if t in VALID_LAYERS
        or t in LANDCOVER_CLASSES
        or re.fullmatch(r"bio(?:[1-9]|1[0-9])", t)
    }
    if not has_fetch_verb and not predictor_like_tokens:
        return None, None
    if not has_fetch_verb:
        non_connector_tokens = [
            t for t in tokens
            if t not in predictor_like_tokens
            and t not in {"and", "or", "plus", "with", "also", "layer", "layers", "predictor", "predictors", "please"}
        ]
        if non_connector_tokens:
            return None, None
    layers, classes = [], []
    bio_nums = set()
    allow_bio_continuation = False
    for t in tokens:
        m_bio = re.fullmatch(r"bio(?:[1-9]|1[0-9])", t)
        if m_bio:
            if t not in layers:
                layers.append(t)
            try:
                bio_nums.add(int(t.replace("bio", "")))
            except Exception:
                pass
            allow_bio_continuation = True
            continue
        if t == "bio":
            allow_bio_continuation = True
            continue
        if allow_bio_continuation and t.isdigit():
            n = int(t)
            if 1 <= n <= 19:
                bio_nums.add(n)
            continue
        if t in {"and", "or", "nd", "n"} and allow_bio_continuation:
            continue
        allow_bio_continuation = False

        if re.fullmatch(r'bio(?:[1-9]|1[0-9])', t):
            if t not in layers: layers.append(t)
            continue
        if t in VALID_LAYERS and t not in layers:
            layers.append(t); continue
        if t in LANDCOVER_CLASSES and t not in classes:
            classes.append(t); continue

    for n in sorted(bio_nums):
        name = f"bio{n}"
        if name not in layers:
            layers.append(name)

    snake_text = re.sub(r'[^a-z0-9]+', '_', text)
    for cls in LANDCOVER_CLASSES:
        if cls in snake_text and cls not in classes:
            classes.append(cls)
    if classes and "landcover" not in layers:
        layers.append("landcover")
    if layers or classes:
        return layers, classes
    return None, None

def run_model(state=None, method=None):
    method_key = _normalize_method_name(method) or "logistic_regression"
    def _run_effect_plot_postprocessing() -> str:
        proc = _run_subprocess(
            [
                sys.executable,
                "-u",
                os.path.join(APP_ROOT, "workflow", "effect_plots.py"),
                "--model",
                method_key,
            ],
            cwd=session_root,
            env=run_env,
        )
        logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        record_step(
            kind="postprocessing",
            function="workflow.effect_plots",
            inputs={"method": method_key},
            outputs={"returncode": proc.returncode, "logs": logs[-2000:]},
            session_id=(state or {}).get("session_id") if isinstance(state, dict) else None,
        )
        if proc.returncode != 0:
            return f"\n\nEffect plot postprocessing failed:\n```\n{logs}\n```"
        return ""

    with _session_cwd(state) as session_root:
        _clean_dir("outputs", state)
        run_env = os.environ.copy()
        try:
            active_preds = get_active_predictors()
            if active_preds:
                run_env["SDM_ACTIVE_PREDICTORS"] = ",".join(active_preds)
        except Exception:
            active_preds = []
        if method_key == "issa":
            if not os.path.exists(MOVEMENT_POINTS_PATH):
                return (
                    create_map(state),
                    "iSSA requires movement data with timestamps. Upload movement tracks first, then rerun with `issa`.",
                    None,
                    None,
                    None,
                )
            proc = _run_subprocess(
                [sys.executable, "-u", os.path.join(APP_ROOT, "methods", "issa.py")],
                cwd=session_root,
                env=run_env,
            )
            logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            if proc.returncode != 0:
                record_step(
                    kind="movement_model",
                    function="methods.issa",
                    inputs={},
                    outputs={"returncode": proc.returncode, "stderr": logs},
                    session_id=(state or {}).get("session_id") if isinstance(state, dict) else None,
                )
                return create_map(state), f"iSSA run failed:\n{logs}", None, None, None

            summary_lines = [
                "iSSA ran successfully. Suitability map was projected to the map when predictor rasters were available.",
                "Detailed iSSA metrics and coefficients are available in the popup viewer.",
            ]
            postprocess_warning = _run_effect_plot_postprocessing()
            if not postprocess_warning:
                summary_lines.append("Predictor effect plots were generated in postprocessing.")

            record_step(
                kind="movement_model",
                function="methods.issa",
                inputs={},
                outputs={"returncode": proc.returncode},
                session_id=(state or {}).get("session_id") if isinstance(state, dict) else None,
            )
            if isinstance(state, dict):
                try:
                    write_session_snapshot(state, os.path.join(session_root, "outputs"), {"status": "ok", "method": "issa"})
                except Exception:
                    pass
            zip_path = zip_results(state)
            return create_map(state), "\n".join(summary_lines) + postprocess_warning, None, None, zip_path

        proc = _run_subprocess(
            [sys.executable, "-u", os.path.join(APP_ROOT, "methods", "logistic_regression.py")],
            cwd=session_root,
            env=run_env,
        )
        if proc.returncode != 0:
            record_step(
                kind="sdm_model",
                function="methods.logistic_regression",
                inputs={},
                outputs={"returncode": proc.returncode, "stderr": proc.stderr},
                session_id=(state or {}).get("session_id") if isinstance(state, dict) else None,
            )
            return create_map(state), f"Model run failed:\n{proc.stderr}", None, None, None
        postprocess_warning = _run_effect_plot_postprocessing()
        perf_df = pd.read_csv("outputs/performance_metrics_cv.csv")
        coef_df = pd.read_csv("outputs/coefficients.csv")
        record_step(
            kind="sdm_model",
            function="methods.logistic_regression",
            inputs={},
            outputs={"returncode": proc.returncode, "n_coefficients": int(len(coef_df)), "n_metrics_rows": int(len(perf_df))},
            session_id=(state or {}).get("session_id") if isinstance(state, dict) else None,
        )
        if isinstance(state, dict):
            try:
                write_session_snapshot(state, os.path.join(session_root, "outputs"), {"status": "ok", "method": "logistic_regression"})
            except Exception:
                pass
        zip_path = zip_results(state)
        return create_map(state), "Model ran successfully. Download the SDM using the button below the map." + postprocess_warning, perf_df, coef_df, zip_path
def _compose_status_with_cv_and_coefs(status_prefix: str, long_names: bool = False, state=None, zip_path=None):
    """Read CV + coef files and return (assistant_txt, download_update)."""
    with _session_cwd(state):
        cv_fp = "outputs/performance_metrics_cv.csv"
        coef_fp = "outputs/coefficients.csv"
        drop_csv = "outputs/dropped_predictors.csv"
        multicol_csv = "outputs/multicollinearity_check.csv"
        prune_report_csv = "outputs/pruned_variables_report.csv"
        effect_manifest_csv = "outputs/predictor_effect_plots_manifest.csv"

        if not os.path.exists(cv_fp) or not os.path.exists(coef_fp):
            return status_prefix, gr.update()

        cv = pd.read_csv(cv_fp)
        if cv.empty:
            return status_prefix, gr.update()

        row = cv.iloc[0]
        perf_md, folds, warn_md = build_cv_section(row, long_names=long_names)

        coef = pd.read_csv(coef_fp).dropna(axis=1, how='all')
        coef_md = coef.to_markdown(index=False)

        cv_method = _getv(row, "cv_method")
        cv_method = str(cv_method) if not pd.isna(cv_method) else "spatial_blocks"
        cv_label = "spatial blocks" if cv_method in {"spatial_blocks", "kmeans_blocks"} else cv_method

        dropped_md = ""
        try:
            if os.path.exists(drop_csv):
                dropped_df = pd.read_csv(drop_csv)
                if not dropped_df.empty:
                    bullets = []
                    for _, r in dropped_df.iterrows():
                        pred = str(r.get("predictor", ""))
                        reason = str(r.get("reason", ""))
                        cval = r.get("constant_value", np.nan)
                        extra = f" (constant={float(cval):.6g})" if pd.notna(cval) else ""
                        bullets.append(f"- **{pred}** - {reason}{extra}")
                    dropped_md = "\n\n**Dropped Predictors (uninformative on training data):**\n\n" + "\n".join(bullets) + "\n"
        except Exception:
            dropped_md = ""

        multicol_md = ""
        try:
            if os.path.exists(multicol_csv):
                mc = pd.read_csv(multicol_csv)
                if mc.empty:
                    multicol_md = "\n\n**Multicollinearity Check:** no numeric predictors to check.\n"
                else:
                    flagged = mc[mc["flag"].fillna("ok") != "ok"].copy()
                    if flagged.empty:
                        multicol_md = "\n\n**Multicollinearity Check:** no predictors exceeded the correlation/VIF thresholds.\n"
                    else:
                        cols = ["predictor", "max_abs_correlation", "max_correlated_predictor", "vif", "flag"]
                        multicol_md = "\n\n**Multicollinearity Check (flagged predictors):**\n\n" + flagged[cols].to_markdown(index=False) + "\n"
        except Exception:
            multicol_md = ""

        prune_report_md = ""
        try:
            if os.path.exists(prune_report_csv):
                pr = pd.read_csv(prune_report_csv)
                if not pr.empty:
                    cols = [c for c in ["predictor", "reason", "details", "correlated_with"] if c in pr.columns]
                    prune_report_md = "\n\n**Pruned Variables Report:**\n\n" + pr[cols].to_markdown(index=False) + "\n"
        except Exception:
            prune_report_md = ""

        effect_note_md = ""
        try:
            if os.path.exists(effect_manifest_csv):
                ef = pd.read_csv(effect_manifest_csv)
                if not ef.empty:
                    effect_note_md = f"\n\n**Predictor Effect Plots:** generated ({len(ef)} plot(s)); open them in the popup viewer."
        except Exception:
            effect_note_md = ""

        warning_block = f"\n\n{warn_md}\n" if warn_md else ""
        assistant_txt = (
            f"{status_prefix}"
            f"{warning_block}\n"
            f"**Cross-validated Performance ({cv_label}; n_folds={folds})**\n\n{perf_md}\n\n"
            f"**Predictor Coefficients:**\n\n{coef_md}"
            f"{dropped_md}"
            f"{multicol_md}"
            f"{prune_report_md}"
            f"{effect_note_md}"
        )

        resolved_zip = zip_path
        if not resolved_zip and os.path.exists("spatchat_results.zip"):
            resolved_zip = os.path.abspath("spatchat_results.zip")
        dl_update = gr.update(value=resolved_zip if resolved_zip and os.path.exists(resolved_zip) else None)
        return assistant_txt, dl_update

def _handle_model_request(user_msg, history, state, requested_method=None, long_names=False, skip_prune_prompt=False):
    with _session_cwd(state):
        has_presence = os.path.exists(PRESENCE_POINTS_PATH)
    if not has_presence:
        fb = [{"role": "system", "content": FALLBACK_PROMPT}, {"role": "user", "content": user_msg}]
        reply = _safe_llm_chat(
            fb,
            temperature=0.7,
            max_tokens=256,
            stream=False,
            default="Please upload point data and fetch predictor layers before running a model.",
        )
        history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}])
        return history, create_map(state), state, gr.update()

    chosen_method = (
        _normalize_method_name(requested_method)
        or _detect_method_from_text(user_msg)
        or _normalize_method_name((state or {}).get("selected_method"))
    )
    if chosen_method is None:
        if isinstance(state, dict):
            state["awaiting_method_choice"] = True
            if _wants_pruning(user_msg):
                state["model_prune_preference"] = "apply"
        prompt = _method_guidance_message(state)
        history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": prompt}])
        return history, create_map(state), state, gr.update()

    if isinstance(state, dict):
        state["awaiting_method_choice"] = False
        state["selected_method"] = chosen_method

    # Validate requested method against current data type before extra prompts/checks.
    if chosen_method == "issa" and not _has_movement_data(state):
        reply = (
            "iSSA is not available for the current dataset because movement tracks with timestamps are not loaded.\n"
            "Use `logistic_regression`, or upload movement data and then choose `issa`."
        )
        history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}])
        return history, create_map(state), state, gr.update()

    auto_check_note = ""
    if not skip_prune_prompt:
        explicit_prune = _wants_pruning(user_msg) or (isinstance(state, dict) and state.get("model_prune_preference") == "apply")
        diag = None
        cached = state.get("last_multicol_diag") if isinstance(state, dict) and isinstance(state.get("last_multicol_diag"), dict) else None
        if cached and float(cached.get("corr_threshold", -1)) == 0.7 and float(cached.get("vif_threshold", -1)) == 5.0:
            diag = {"multicollinearity": cached}
        else:
            with _session_cwd(state):
                diag = run_pre_model_diagnostics(corr_threshold=0.7, vif_threshold=5.0)
        mc = (diag or {}).get("multicollinearity", {}) or {}
        if isinstance(state, dict) and mc.get("ok"):
            state["last_multicol_diag"] = mc
        auto_pruned = mc.get("auto_prune", []) if mc.get("ok") else []
        if auto_pruned:
            with _session_cwd(state):
                set_predictor_active_state(auto_pruned, active=False)
            auto_check_note = (
                "Auto-pruned low-information predictors (all-NA, all-0, or constant in study area): "
                + ", ".join(auto_pruned)
                + "."
            )

        suggested = [s for s in (mc.get("suggested_drop", []) if mc.get("ok") else []) if s not in set(auto_pruned)]
        if suggested:
            if explicit_prune:
                with _session_cwd(state):
                    set_predictor_active_state(suggested, active=False)
                extra_note = (
                    "Applied multicollinearity pruning automatically: "
                    + ", ".join(suggested)
                    + "."
                )
                auto_check_note = f"{auto_check_note}\n{extra_note}".strip()
            else:
                if isinstance(state, dict):
                    state["awaiting_prune_confirmation"] = True
                    state["pending_prune_predictors"] = suggested
                    state["pending_model_request"] = {
                        "user_msg": user_msg,
                        "requested_method": chosen_method,
                        "long_names": bool(long_names),
                    }
                prompt = (
                    "Before running the model, I checked multicollinearity (default thresholds: "
                    "corr>=0.70, VIF>=5.0).\n\n"
                    + (f"Already auto-pruned low-information predictors: {', '.join(auto_pruned)}.\n\n" if auto_pruned else "")
                    + f"Suggested variables to prune: {', '.join(suggested)}.\n"
                    "Do you want me to prune these now? Reply `yes` or `no`."
                )
                history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": prompt}])
                return history, create_map(state), state, gr.update()
        if not suggested and not auto_pruned:
            auto_check_note = "Auto-check complete: no variable pruning suggested at default thresholds (corr>=0.70, VIF>=5.0)."
        if isinstance(state, dict):
            state.pop("model_prune_preference", None)

    m_out, status, perf_df, coef_df, zip_path = run_model(state=state, method=chosen_method)
    if auto_check_note:
        status = f"{status}\n\n{auto_check_note}"
    if perf_df is None:
        assistant_txt = status
        dl_update = gr.update(value=zip_path if zip_path and os.path.exists(zip_path) else None)
    else:
        assistant_txt, dl_update = _compose_status_with_cv_and_coefs(
            status_prefix=status,
            long_names=long_names,
            state=state,
            zip_path=zip_path,
        )

    history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_txt}])
    return history, m_out, state, dl_update


def _status_message(label: str, started_at: Optional[float] = None) -> str:
    _ = started_at
    suffix = "..." if not str(label).endswith("...") else ""
    return f"<span class='spatchat-status'><span class='spatchat-status-dot'></span>{label}{suffix}</span>"


def _status_clear_update():
    return gr.update(value="")


def _typing_frames(text: str, min_step: int = 5, max_step: int = 22):
    msg = str(text or "")
    if not msg:
        yield ""
        return
    idx = 0
    n = len(msg)
    while idx < n:
        step = max(min_step, min(max_step, 8 if msg[idx] == "\n" else 16))
        idx = min(n, idx + step)
        yield msg[:idx]


CHATBOT_HISTORY_MODE = "messages"


def _history_to_messages(history):
    out = []
    for item in list(history or []):
        if isinstance(item, dict):
            role = str(item.get("role", "")).strip().lower()
            if role in {"user", "assistant"}:
                out.append({"role": role, "content": str(item.get("content", "") or "")})
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            if user_msg not in (None, ""):
                out.append({"role": "user", "content": str(user_msg)})
            if assistant_msg not in (None, ""):
                out.append({"role": "assistant", "content": str(assistant_msg)})
    return out


def _messages_to_legacy_chat(messages):
    rows = []
    pending_user = None
    for msg in list(messages or []):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "") or "")
        if role == "user":
            if pending_user is not None:
                rows.append((pending_user, None))
            pending_user = content
        elif role == "assistant":
            rows.append((pending_user, content))
            pending_user = None
    if pending_user is not None:
        rows.append((pending_user, None))
    return rows


def _history_for_chatbot(messages):
    if CHATBOT_HISTORY_MODE == "messages":
        return list(messages or [])
    return _messages_to_legacy_chat(messages)


WELCOME_MESSAGE = (
    "Hello, I'm Spatchat, your SDM assistant. I can help you build a species distribution model "
    "from presence, presence-absence, or movement CSV data and environmental predictors. Upload a CSV to begin."
)

def chat_step(file, user_msg, history, state):
    state = _ensure_session_workspace(state)

    def _fetch_followup(status_text: str) -> str:
        s = str(status_text or "").lower()
        if "predictors fetched" in s:
            if _has_movement_data(state):
                return (
                    f"{status_text}\n\n"
                    "Great! Now you can run the model or fetch more layers.\n\n"
                    "Recommended before modeling movement data:\n"
                    "- Use `issa` (step selection accounts for movement dependence by design)\n"
                    "- Optional: check multicollinearity (`corr 0.7 vif 5`) for predictor pruning"
                )
            return (
                f"{status_text}\n\n"
                "Great! Now you can run the model or fetch more layers.\n\n"
                "Recommended before modeling:\n"
                "- Ask: `check multicollinearity` (or specify thresholds like `corr 0.7 vif 5`)\n"
                "- Ask: `check Moran's I` for spatial autocorrelation"
            )
        return str(status_text or "")

    route = _llm_route_intent(user_msg, state)
    action = route.get("action", "general_chat")
    requested_method_from_text = _detect_method_from_text(user_msg) or _normalize_method_name(user_msg)
    requested_method = route.get("method") or requested_method_from_text

    if isinstance(state, dict) and state.get("awaiting_prune_confirmation"):
        pending = state.get("pending_model_request", {}) if isinstance(state.get("pending_model_request"), dict) else {}
        prune_list = state.get("pending_prune_predictors", []) if isinstance(state.get("pending_prune_predictors"), list) else []
        pending_method = pending.get("requested_method")
        explicit_yes = _is_yes(user_msg)
        explicit_no = _is_no(user_msg)
        effective_action = action
        if explicit_yes:
            effective_action = "prune_yes"
        elif explicit_no:
            effective_action = "prune_no"

        if effective_action == "prune_yes":
            with _session_cwd(state):
                out = set_predictor_active_state(prune_list, active=False)
                active_now = get_active_predictors()
            if not out.get("ok"):
                reply = out.get("message", "I couldn't apply pruning to the suggested predictors.")
                history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}])
                return history, create_map(state), state, gr.update()
            requested_norm = {_norm_col_name(p) for p in prune_list}
            still_active = [
                p for p in (active_now or [])
                if _norm_col_name(p) in requested_norm
            ]
            if still_active:
                reply = (
                    "I received your prune confirmation, but these predictors are still active: "
                    + ", ".join(sorted(still_active))
                    + ". Please try `remove "
                    + ", ".join(sorted(still_active))
                    + "` and rerun."
                )
                history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}])
                return history, create_map(state), state, gr.update()
            state["awaiting_prune_confirmation"] = False
            state["pending_prune_predictors"] = []
            state["pending_model_request"] = {}
            return _handle_model_request(
                user_msg=user_msg,
                history=history,
                state=state,
                requested_method=pending_method,
                long_names=bool(pending.get("long_names", False)),
                skip_prune_prompt=True,
            )
        if effective_action in {"prune_no", "run_model"} or requested_method is not None:
            state["awaiting_prune_confirmation"] = False
            state["pending_prune_predictors"] = []
            state["pending_model_request"] = {}
            return _handle_model_request(
                user_msg=user_msg,
                history=history,
                state=state,
                requested_method=(requested_method or pending_method),
                long_names=bool(pending.get("long_names", False)),
                skip_prune_prompt=True,
            )
        reply = "Please reply `yes` to prune suggested variables, or `no` to keep them and continue."
        history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}])
        return history, create_map(state), state, gr.update()

    if action == "reset_session":
        clear_all(state)
        if isinstance(state, dict):
            state.pop("awaiting_method_choice", None)
            state.pop("selected_method", None)
            state.pop("awaiting_prune_confirmation", None)
            state.pop("pending_prune_predictors", None)
            state.pop("pending_model_request", None)
        new_hist = [{"role": "assistant", "content": "All cleared. Please upload your CSV to begin."}]
        return new_hist, create_map(state), state, gr.update(value=None)

    if route.get("prune_preference") == "apply" and isinstance(state, dict):
        state["model_prune_preference"] = "apply"

    if action == "remove_predictors":
        names = route.get("predictors") or []
        if names:
            with _session_cwd(state):
                out = set_predictor_active_state(names, active=False)
            if out.get("ok"):
                reply = f"Marked as inactive: {', '.join(out.get('matched', []))}. Rasters are kept and can be restored later."
            else:
                reply = out.get("message", "No matching predictors found to remove.")
            history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}])
            return history, create_map(state), state, gr.update()

    if action == "restore_predictors":
        names = route.get("predictors") or []
        if names:
            with _session_cwd(state):
                out = set_predictor_active_state(names, active=True)
            if out.get("ok"):
                reply = f"Re-activated: {', '.join(out.get('matched', []))}."
                history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}])
                return history, create_map(state), state, gr.update()

            fetch_layers = []
            for n in names:
                nn = str(n).strip().lower()
                if re.fullmatch(r"bio(?:[1-9]|1[0-9])", nn) or nn in VALID_LAYERS:
                    fetch_layers.append(nn)
            if fetch_layers:
                m_out, status = run_fetch(fetch_layers, [], state=state)
                assistant_txt = _fetch_followup(status)
                history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_txt}])
                return history, m_out, state, gr.update()

            reply = out.get("message", "No matching predictors found to restore.")
            history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}])
            return history, create_map(state), state, gr.update()

    if action in {"check_both_diagnostics", "check_multicollinearity", "check_morans_i"}:
        c_thr = route.get("corr_threshold")
        v_thr = route.get("vif_threshold")
        c_thr = 0.7 if c_thr is None else float(c_thr)
        v_thr = 5.0 if v_thr is None else float(v_thr)

        if action == "check_both_diagnostics":
            with _session_cwd(state):
                diag = run_multicollinearity_check(corr_threshold=float(c_thr), vif_threshold=float(v_thr))
                mor = run_morans_i_check()

            if not diag.get("ok"):
                mc_msg = diag.get("message", "Unable to run multicollinearity check.")
            else:
                if isinstance(state, dict):
                    state["last_multicol_diag"] = diag
                suggested = diag.get("suggested_drop", [])
                auto_pruned = diag.get("auto_prune", [])
                mc_msg = (
                    f"Multicollinearity check done (corr>={c_thr:.2f}, VIF>={v_thr:.2f}). "
                    + (f"Suggested prune list: {', '.join(suggested)}." if suggested else "No pruning suggested.")
                )
                if auto_pruned:
                    mc_msg += f" Auto-pruned low-information predictors: {', '.join(auto_pruned)}."
                mc_msg += " Detailed VIF summary and correlation matrix are available in the popup viewer."

            if not mor.get("ok"):
                mor_msg = mor.get("message", "Unable to run Moran's I check.")
            else:
                sig = mor.get("significant_predictors", [])
                if sig:
                    mor_msg = f"Moran's I check done. Significant spatial autocorrelation detected in: {', '.join(sig)}."
                else:
                    mor_msg = "Moran's I check done. No significant spatial autocorrelation detected for active predictors."

            reply = mc_msg + "\n\n" + mor_msg
            history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}])
            return history, create_map(state), state, gr.update()

        if action == "check_multicollinearity":
            with _session_cwd(state):
                diag = run_multicollinearity_check(corr_threshold=float(c_thr), vif_threshold=float(v_thr))
            if not diag.get("ok"):
                reply = diag.get("message", "Unable to run multicollinearity check.")
            else:
                if isinstance(state, dict):
                    state["last_multicol_diag"] = diag
                suggested = diag.get("suggested_drop", [])
                auto_pruned = diag.get("auto_prune", [])
                reply = (
                    f"Multicollinearity check done (corr>={c_thr:.2f}, VIF>={v_thr:.2f}). "
                    + (f"Suggested prune list: {', '.join(suggested)}." if suggested else "No pruning suggested.")
                )
                if auto_pruned:
                    reply += f" Auto-pruned low-information predictors: {', '.join(auto_pruned)}."
                reply += " Detailed VIF summary and correlation matrix are available in the popup viewer."
            history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}])
            return history, create_map(state), state, gr.update()

        with _session_cwd(state):
            out = run_morans_i_check()
        if not out.get("ok"):
            reply = out.get("message", "Unable to run Moran's I check.")
        else:
            sig = out.get("significant_predictors", [])
            if sig:
                reply = f"Moran's I check done. Significant spatial autocorrelation detected in: {', '.join(sig)}."
            else:
                reply = "Moran's I check done. No significant spatial autocorrelation detected for active predictors."
        history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}])
        return history, create_map(state), state, gr.update()

    if action == "run_model":
        return _handle_model_request(
            user_msg=user_msg,
            history=history,
            state=state,
            requested_method=requested_method,
            long_names=False,
        )

    if action == "fetch_layers":
        parsed_layers = route.get("layers") or []
        parsed_classes = route.get("landcover") or []
        if parsed_layers:
            m_out, status = run_fetch(parsed_layers, parsed_classes, state=state)
            assistant_txt = _fetch_followup(status)
            history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_txt}])
            return history, m_out, state, gr.update()

    if isinstance(state, dict) and state.get("awaiting_method_choice") and requested_method is not None:
        return _handle_model_request(
            user_msg=user_msg,
            history=history,
            state=state,
            requested_method=requested_method,
            long_names=False,
        )

    assistant_txt = _general_llm_reply(user_msg, history, state)
    history.extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_txt}])
    return history, create_map(state), state, gr.update()


def _legacy_on_upload(f, history, state):
    history2 = history.copy()
    if not isinstance(state, dict):
        state = {}
    state.setdefault("layers_help_shown", False)

    clear_all()
    if f and hasattr(f, "name"):
        shutil.copy(f.name, "inputs/presence_points.csv")
        df = pd.read_csv("inputs/presence_points.csv")
        lat, lon = detect_coords(df)
        if lat and lon:
            # Rename to canonical columns so mapping/modeling stays simple
            df = df.rename(columns={lat: "latitude", lon: "longitude"})
            df.to_csv("inputs/presence_points.csv", index=False)
            msg = "✅ Sweet! I found your `latitude` and `longitude` columns."
            if not state.get("layers_help_shown"):
                msg += "\n\n" + available_layers_markdown()
                state["layers_help_shown"] = True
            history2.append({"role":"assistant","content": msg})
            return (history2, create_map(), state, gr.update(),
                    gr.update(choices=[], visible=False),
                    gr.update(choices=[], visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False))
        else:
            history2.append({"role":"assistant","content":
                "I couldn't detect coordinate columns. Please select them and enter CRS below (e.g., `UTM 10T` or `32610`)."
            })
            cols = list(df.columns)
            return (history2, create_map(), state, gr.update(),
                    gr.update(choices=cols, visible=True, value=None),
                    gr.update(choices=cols, visible=True, value=None),
                    gr.update(visible=True, value="UTM 10T"),
                    gr.update(visible=True))
    return (history2, create_map(), state, gr.update(),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False))

def _legacy_confirm_coords_old(lat_col, lon_col, crs_raw, history, state):
    if not isinstance(state, dict):
        state = {}
    state.setdefault("layers_help_shown", False)

    df = pd.read_csv("inputs/presence_points.csv")
    try:
        src_epsg = resolve_crs(crs_raw) if crs_raw else 4326
    except:
        history.append({"role":"assistant","content":"Sorry, I couldn't recognize that CRS. Try formats like `32610`, `EPSG:32610`, or `UTM 10T`."})
        return (history, create_map(), state, gr.update(),
                gr.update(visible=True), gr.update(visible=True),
                gr.update(visible=True), gr.update(visible=True))

    src_crs = RioCRS.from_epsg(src_epsg)
    dst_crs = RioCRS.from_epsg(4326)
    lon_vals, lat_vals = rio_transform(src_crs, dst_crs, df[lon_col].tolist(), df[lat_col].tolist())
    df['latitude'], df['longitude'] = lat_vals, lon_vals
    df.to_csv("inputs/presence_points.csv", index=False)

    success = f"✅ Coordinates transformed from EPSG:{src_epsg} to WGS84 (lat/lon)."
    if not state.get("layers_help_shown"):
        success += "\n\n" + available_layers_markdown()
        state["layers_help_shown"] = True
    history.append({"role":"assistant","content": success})

    return (history, create_map(), state, gr.update(),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False))

def on_upload(f, history, state):
    history2 = history.copy()
    state = _ensure_session_workspace(state)
    state.setdefault("layers_help_shown", False)

    clear_all(state)
    if f:
        with _session_cwd(state):
            source_path = f.name if hasattr(f, "name") else str(f)
            shutil.copy(source_path, UPLOADED_DATA_PATH)
            df = pd.read_csv(UPLOADED_DATA_PATH)
            lat, lon = detect_coords(df)
            state["pending_upload"] = {
                "path": UPLOADED_DATA_PATH,
                "data_type_choice": "Auto-detect",
            }

            if lat and lon and _coords_look_wgs84(df, lat, lon):
                try:
                    meta = _normalize_upload(df, "Auto-detect", lat_col=lat, lon_col=lon, src_epsg=4326)
                except Exception as exc:
                    history2.append({"role": "assistant", "content": str(exc)})
                    cols = list(df.columns)
                    return (history2, create_map(state), state, gr.update(),
                            gr.update(choices=cols, visible=True, value=lat),
                            gr.update(choices=cols, visible=True, value=lon),
                            gr.update(visible=True, value="4326"),
                            gr.update(visible=True))

                state["data_metadata"] = meta
                state.pop("pending_upload", None)
                history2.append({"role": "assistant", "content": _upload_message(meta, state)})
                return (history2, create_map(state), state, gr.update(),
                        gr.update(choices=[], visible=False),
                        gr.update(choices=[], visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False))

            if lat and lon:
                msg = "I found coordinate-looking columns, but they do not look like WGS84 latitude/longitude. Please confirm the columns and enter the input CRS."
            else:
                msg = "I couldn't detect coordinate columns. Please select them and enter the input CRS."
            history2.append({"role": "assistant", "content": msg})
            cols = list(df.columns)
            return (history2, create_map(state), state, gr.update(),
                    gr.update(choices=cols, visible=True, value=lat),
                    gr.update(choices=cols, visible=True, value=lon),
                    gr.update(visible=True, value="UTM 10T"),
                    gr.update(visible=True))

    return (history2, create_map(state), state, gr.update(),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False))


def confirm_coords(lat_col, lon_col, crs_raw, history, state):
    state = _ensure_session_workspace(state)
    state.setdefault("layers_help_shown", False)

    pending = state.get("pending_upload", {}) if isinstance(state, dict) else {}
    upload_path = pending.get("path", UPLOADED_DATA_PATH)
    data_type_choice = pending.get("data_type_choice", "Auto-detect")

    with _session_cwd(state):
        df = pd.read_csv(upload_path)
        try:
            src_epsg = resolve_crs(crs_raw) if crs_raw else 4326
        except Exception:
            history.append({"role": "assistant", "content": "Sorry, I couldn't recognize that CRS. Try formats like `32610`, `EPSG:32610`, or `UTM 10T`."})
            return (history, create_map(state), state, gr.update(),
                    gr.update(visible=True), gr.update(visible=True),
                    gr.update(visible=True), gr.update(visible=True))

        try:
            meta = _normalize_upload(df, data_type_choice, lat_col=lat_col, lon_col=lon_col, src_epsg=src_epsg)
        except Exception as exc:
            history.append({"role": "assistant", "content": str(exc)})
            return (history, create_map(state), state, gr.update(),
                    gr.update(visible=True), gr.update(visible=True),
                    gr.update(visible=True), gr.update(visible=True))

    state["data_metadata"] = meta
    state.pop("pending_upload", None)
    success = f"Coordinates transformed from EPSG:{src_epsg} to WGS84. " + _upload_message(meta, state)
    history.append({"role": "assistant", "content": success})

    return (history, create_map(state), state, gr.update(),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False))


def on_upload_with_viewer(f, history, state):
    out = on_upload(f, _history_to_messages(history), state)
    st = out[2] if isinstance(out, tuple) and len(out) >= 3 else state
    history_out = _history_for_chatbot(out[0])
    return (history_out, *out[1:], gr.update(value=_render_feature_viewer_html(st)))


def confirm_coords_with_viewer(lat_col, lon_col, crs_raw, history, state):
    out = confirm_coords(lat_col, lon_col, crs_raw, _history_to_messages(history), state)
    st = out[2] if isinstance(out, tuple) and len(out) >= 3 else state
    history_out = _history_for_chatbot(out[0])
    return (history_out, *out[1:], gr.update(value=_render_feature_viewer_html(st)))


def chat_step_with_viewer(file, user_msg, history, state):
    base_history = _history_to_messages(history)
    msg = str(user_msg or "").strip()
    status_label = "Thinking"
    try:
        parsed_layers, parsed_classes = try_parse_fetch_from_text(msg)
        if parsed_layers is not None:
            status_label = "Fetching predictor layers"
        elif _is_model_request(msg) or _detect_method_from_text(msg):
            status_label = "Running model"
    except Exception:
        pass
    visible_history = list(base_history)
    if msg:
        visible_history.append({"role": "user", "content": msg})
    started_at = time.time()
    yield (
        _history_for_chatbot(visible_history) if msg else gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.update(value=_status_message(status_label, started_at)),
    )

    box = {}

    def _target():
        try:
            box["result"] = chat_step(file, user_msg, base_history, state)
        except Exception as exc:
            box["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()

    last_second = -1
    while worker.is_alive():
        sec = int(time.time() - started_at)
        if sec != last_second:
            yield (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.update(value=_status_message(status_label, started_at)),
            )
            last_second = sec
        time.sleep(0.12)

    worker.join()
    if "error" in box:
        err_history = list(visible_history if msg else base_history)
        err_history.append({"role": "assistant", "content": f"Sorry, that failed before I could finish: {box['error']}"})
        yield (
            _history_for_chatbot(err_history),
            create_map(state),
            state,
            gr.update(),
            gr.update(value=_render_feature_viewer_html(state)),
            _status_clear_update(),
        )
        return

    out = box.get("result")
    if not isinstance(out, tuple) or len(out) < 4:
        st = state
        yield (
            _history_for_chatbot(base_history),
            create_map(st),
            st,
            gr.update(),
            gr.update(value=_render_feature_viewer_html(st)),
            _status_clear_update(),
        )
        return

    next_history, map_out, st, dl_update = out[0], out[1], out[2], out[3]
    feature_update = gr.update(value=_render_feature_viewer_html(st))
    next_history = list(next_history or [])

    assistant_index = None
    for i in range(len(next_history) - 1, -1, -1):
        if str(next_history[i].get("role", "")).lower() == "assistant":
            assistant_index = i
            break

    if assistant_index is None:
        yield (_history_for_chatbot(next_history), map_out, st, dl_update, feature_update, _status_clear_update())
        return

    assistant_text = str(next_history[assistant_index].get("content", "") or "")
    if not assistant_text:
        yield (_history_for_chatbot(next_history), map_out, st, dl_update, feature_update, _status_clear_update())
        return

    typing_history = [dict(m) for m in next_history]
    typing_history[assistant_index]["content"] = ""
    yielded_map = False
    yielded_download = False
    yielded_feature = False
    for frame in _typing_frames(assistant_text):
        typing_history[assistant_index]["content"] = frame + "▌"
        yield (
            _history_for_chatbot(typing_history),
            map_out if not yielded_map else gr.skip(),
            st if not yielded_map else gr.skip(),
            dl_update if not yielded_download else gr.skip(),
            feature_update if not yielded_feature else gr.skip(),
            _status_clear_update(),
        )
        yielded_map = True
        yielded_download = True
        yielded_feature = True
        time.sleep(0.015)

    next_history[assistant_index]["content"] = assistant_text
    yield (
        _history_for_chatbot(next_history),
        map_out if not yielded_map else gr.skip(),
        st if not yielded_map else gr.skip(),
        dl_update if not yielded_download else gr.skip(),
        feature_update if not yielded_feature else gr.skip(),
        _status_clear_update(),
    )

# UI

_SPLITTER_HEAD = """
<script>
window.spatchatBeginResize = function(event) {
  event.preventDefault();
  const workarea = document.getElementById("spatchat-workarea");
  const sidebar = document.getElementById("spatchat-sidebar");
  const mapcol = document.getElementById("spatchat-mapcol");
  if (!workarea || !sidebar || !mapcol) return false;

  const rect = workarea.getBoundingClientRect();
  const minWidth = 80;
  const maxWidth = Math.max(minWidth, rect.width - 80);
  let latestClientX = event.clientX;
  let rafId = 0;
  let lastValue = "";

  function apply() {
    rafId = 0;
    const next = Math.min(Math.max(latestClientX - rect.left, minWidth), maxWidth);
    const value = `${Math.round(next)}px`;
    if (value === lastValue) return;
    lastValue = value;
    sidebar.style.width = value;
    sidebar.style.flexBasis = value;
    sidebar.style.maxWidth = value;
    mapcol.style.flex = "1 1 auto";
  }

  function scheduleApply(clientX) {
    latestClientX = clientX;
    if (!rafId) {
      rafId = window.requestAnimationFrame(apply);
    }
  }

  function onMove(moveEvent) {
    scheduleApply(moveEvent.clientX);
  }

  function onUp() {
    document.body.classList.remove("spatchat-resizing");
    if (rafId) {
      window.cancelAnimationFrame(rafId);
      rafId = 0;
    }
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
  }

  document.body.classList.add("spatchat-resizing");
  scheduleApply(event.clientX);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  return false;
};

window.spatchatInitSplitterHandle = function() {
  const handles = document.querySelectorAll("#spatchat-splitter .spatchat-splitter-handle");
  handles.forEach((handle) => {
    if (handle.__spatchatBound) return;
    const start = (event) => {
      if (window.spatchatBeginResize) return window.spatchatBeginResize(event);
      return false;
    };
    handle.addEventListener("mousedown", start);
    handle.__spatchatBound = true;
  });
};

window.addEventListener("DOMContentLoaded", () => {
  if (window.spatchatInitSplitterHandle) window.spatchatInitSplitterHandle();
});
window.addEventListener("load", () => {
  if (window.spatchatInitSplitterHandle) window.spatchatInitSplitterHandle();
});
{
  let splitterInitAttempts = 0;
  const splitterInitTimer = window.setInterval(() => {
    if (window.spatchatInitSplitterHandle) window.spatchatInitSplitterHandle();
    splitterInitAttempts += 1;
    if (splitterInitAttempts >= 20) window.clearInterval(splitterInitTimer);
  }, 500);
}

window.spatchatValuesViewer = (() => {
  const selector = ".spatchat-values-root[data-payload]";
  const layerId = "spatchat-feature-viewer";
  const minWidth = 380;
  const minHeight = 300;
  let activeInteraction = null;

  function clamp(value, min, max) {
    if (!Number.isFinite(value)) return min;
    return Math.min(Math.max(value, min), max);
  }

  function parsePayload(root) {
    try {
      return JSON.parse(root.dataset.payload || "{}");
    } catch (error) {
      return {};
    }
  }

  function refreshStateFromPayload(root) {
    const payload = parsePayload(root);
    const key = root.dataset.payload || "";
    if (root.__spatchatValuesPayload === key && root.__spatchatValuesState) {
      return root.__spatchatValuesState;
    }
    const previous = root.__spatchatValuesState || {};
    const viewportWidth = window.innerWidth || 1280;
    const viewportHeight = window.innerHeight || 800;
    root.__spatchatValuesPayload = key;
    root.__spatchatValuesState = {
      figures: Array.isArray(payload.figures) ? payload.figures : [],
      activeIndex: Number.isFinite(payload.activeIndex) ? payload.activeIndex : (previous.activeIndex || 0),
      // Respect payload isOpen (server sends false by default); never inherit stale open state.
      isOpen: (typeof payload.isOpen === "boolean") ? payload.isOpen : false,
      width: Number.isFinite(payload.width) ? payload.width : (previous.width || clamp(Math.round(viewportWidth * 0.40), minWidth, 720)),
      height: Number.isFinite(payload.height) ? payload.height : (previous.height || clamp(Math.round(viewportHeight * 0.60), minHeight, viewportHeight - 80)),
      x: Number.isFinite(payload.x) ? payload.x : (previous.x ?? null),
      y: Number.isFinite(payload.y) ? payload.y : (previous.y ?? null),
    };
    return root.__spatchatValuesState;
  }

  function syncState(root, state) {
    root.__spatchatValuesState = state;
    root.__spatchatValuesPayload = "";
    root.dataset.payload = JSON.stringify(state);
  }

  function fitToViewport(state) {
    const viewportWidth = window.innerWidth || 1280;
    const viewportHeight = window.innerHeight || 800;
    const maxWidth = Math.max(minWidth, viewportWidth - 8);
    const maxHeight = Math.max(minHeight, viewportHeight - 8);
    const visibleMargin = 72;
    state.width = clamp(state.width || Math.round(viewportWidth * 0.40), minWidth, maxWidth);
    state.height = clamp(state.height || Math.round(viewportHeight * 0.60), minHeight, maxHeight);
    if (!Number.isFinite(state.x) || !Number.isFinite(state.y)) {
      state.x = Math.round((viewportWidth - state.width) / 2);
      state.y = Math.round((viewportHeight - state.height) / 2);
    }
    state.x = clamp(state.x, visibleMargin - state.width, viewportWidth - visibleMargin);
    state.y = clamp(state.y, visibleMargin - state.height, viewportHeight - visibleMargin);
  }

  function applyModalRect(modal, state) {
    if (!modal) return;
    modal.style.width = `${Math.round(state.width)}px`;
    modal.style.height = `${Math.round(state.height)}px`;
    modal.style.left = `${Math.round(state.x)}px`;
    modal.style.top = `${Math.round(state.y)}px`;
  }

  function applyAction(root, action) {
    const state = refreshStateFromPayload(root);
    const figures = Array.isArray(state.figures) ? state.figures : [];
    if (!figures.length && action !== "open") return;
    if (action === "back") state.activeIndex = clamp((state.activeIndex || 0) - 1, 0, figures.length - 1);
    if (action === "forward") state.activeIndex = clamp((state.activeIndex || 0) + 1, 0, figures.length - 1);
    if (action === "open") state.isOpen = true;
    if (action === "close") state.isOpen = false;
    fitToViewport(state);
    syncState(root, state);
    render(root);
  }

  function rootFromElement(element) {
    return element ? element.closest(".spatchat-values-root") : null;
  }

  function stopActiveInteraction() {
    if (!activeInteraction) return;
    window.removeEventListener("mousemove", activeInteraction.onMove);
    window.removeEventListener("mouseup", activeInteraction.onUp);
    if (activeInteraction.bodyClass) document.body.classList.remove(activeInteraction.bodyClass);
    const shield = activeInteraction.root ? activeInteraction.root.querySelector(".spatchat-values-interaction-shield") : null;
    if (shield) shield.classList.remove("is-active");
    activeInteraction = null;
  }

  function startInteraction(root, bodyClass, onMove, onUp) {
    stopActiveInteraction();
    activeInteraction = { root, bodyClass, onMove, onUp };
    if (bodyClass) document.body.classList.add(bodyClass);
    const shield = root ? root.querySelector(".spatchat-values-interaction-shield") : null;
    if (shield) shield.classList.add("is-active");
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  function beginDrag(event, root) {
    if (!event || event.button !== 0) return;
    if (event.target.closest("[data-action]")) return;
    const modal = root.querySelector(".spatchat-values-modal");
    if (!modal) return;
    const state = refreshStateFromPayload(root);
    fitToViewport(state);
    const startX = event.clientX;
    const startY = event.clientY;
    const originX = state.x;
    const originY = state.y;
    event.preventDefault();
    function onMove(moveEvent) {
      state.x = originX + (moveEvent.clientX - startX);
      state.y = originY + (moveEvent.clientY - startY);
      const viewportWidth = window.innerWidth || 1280;
      const viewportHeight = window.innerHeight || 800;
      const visibleMargin = 72;
      state.x = clamp(state.x, visibleMargin - state.width, viewportWidth - visibleMargin);
      state.y = clamp(state.y, visibleMargin - state.height, viewportHeight - visibleMargin);
      applyModalRect(modal, state);
    }
    function onUp() {
      stopActiveInteraction();
      fitToViewport(state);
      syncState(root, state);
      render(root);
    }
    startInteraction(root, "spatchat-values-moving", onMove, onUp);
    return false;
  }

  function beginResize(event, root, direction) {
    if (!event || event.button !== 0) return;
    const modal = root.querySelector(".spatchat-values-modal");
    if (!modal) return;
    const state = refreshStateFromPayload(root);
    fitToViewport(state);
    const startX = event.clientX;
    const startY = event.clientY;
    const originX = state.x;
    const originY = state.y;
    const originWidth = state.width;
    const originHeight = state.height;
    const viewportWidth = window.innerWidth || 1280;
    const viewportHeight = window.innerHeight || 800;
    const visibleMargin = 72;
    event.preventDefault();
    event.stopPropagation();
    function onMove(moveEvent) {
      const dx = moveEvent.clientX - startX;
      const dy = moveEvent.clientY - startY;
      if (direction.includes("e")) state.width = clamp(originWidth + dx, minWidth, viewportWidth + originX - visibleMargin);
      if (direction.includes("s")) state.height = clamp(originHeight + dy, minHeight, viewportHeight + originY - visibleMargin);
      if (direction.includes("w")) {
        const nextX = clamp(originX + dx, visibleMargin - originWidth, originX + originWidth - minWidth);
        state.width = clamp(originWidth - (nextX - originX), minWidth, viewportWidth - visibleMargin - nextX);
        state.x = nextX;
      }
      if (direction.includes("n")) {
        const nextY = clamp(originY + dy, visibleMargin - originHeight, originY + originHeight - minHeight);
        state.height = clamp(originHeight - (nextY - originY), minHeight, viewportHeight - visibleMargin - nextY);
        state.y = nextY;
      }
      state.x = clamp(state.x, visibleMargin - state.width, viewportWidth - visibleMargin);
      state.y = clamp(state.y, visibleMargin - state.height, viewportHeight - visibleMargin);
      applyModalRect(modal, state);
    }
    function onUp() {
      stopActiveInteraction();
      fitToViewport(state);
      syncState(root, state);
      render(root);
    }
    startInteraction(root, "spatchat-values-resizing", onMove, onUp);
    return false;
  }

  function render(root) {
    const state = refreshStateFromPayload(root);
    const figures = Array.isArray(state.figures) ? state.figures : [];
    const launcher = root.querySelector(".spatchat-values-launcher");
    const modal = root.querySelector(".spatchat-values-modal");
    const title = root.querySelector(".spatchat-values-card-title");
    const meta = root.querySelector(".spatchat-values-card-meta");
    const table = root.querySelector(".spatchat-values-card-table");
    const count = root.querySelector(".spatchat-values-modal-count");
    const back = root.querySelector('[data-action="back"]');
    const forward = root.querySelector('[data-action="forward"]');
    if (!figures.length) {
      stopActiveInteraction();
      root.style.display = "";
      if (launcher) launcher.style.display = "inline-flex";
      if (modal) modal.classList.add("is-hidden");
      if (title) title.textContent = "Plots/Tables";
      if (meta) meta.textContent = "No data available yet.";
      if (table) {
        table.innerHTML = "";
        table.style.display = "none";
      }
      if (count) count.textContent = "0 / 0";
      syncState(root, { ...state, isOpen: false, activeIndex: 0, figures: [] });
      return;
    }
    root.style.display = "";
    state.activeIndex = clamp(Number.isFinite(state.activeIndex) ? state.activeIndex : 0, 0, figures.length - 1);
    const active = figures[state.activeIndex];
    if (launcher) launcher.style.display = "inline-flex";
    if (modal) {
      modal.classList.toggle("is-hidden", !state.isOpen);
      if (!state.isOpen) stopActiveInteraction();
      fitToViewport(state);
      applyModalRect(modal, state);
    }
    if (title) title.textContent = active.title || "Table";
    if (meta) meta.textContent = active.subtitle || "";
    if (table) {
      table.innerHTML = active.tableHtml || "";
      table.style.display = active.tableHtml ? "block" : "none";
    }
    if (count) count.textContent = `${state.activeIndex + 1} / ${figures.length}`;
    if (back) back.disabled = state.activeIndex <= 0;
    if (forward) forward.disabled = state.activeIndex >= figures.length - 1;
    syncState(root, state);
  }

  function init(root) {
    if (!root) return;
    if (!root.__spatchatValuesInit) {
      root.__spatchatValuesInit = true;
    }
    render(root);
  }

  function initForLayer() {
    const layer = document.getElementById(layerId);
    if (!layer) return;
    layer.querySelectorAll(selector).forEach((root) => init(root));
  }

  function handleAction(element, event) {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
    const root = rootFromElement(element);
    if (!root) return false;
    const action = element.dataset.action || "";
    applyAction(root, action);
    return false;
  }

  function handleDrag(element, event) {
    const root = rootFromElement(element);
    if (!root) return false;
    beginDrag(event, root);
    return false;
  }

  function handleResize(element, event) {
    const root = rootFromElement(element);
    if (!root) return false;
    beginResize(event, root, String(element.dataset.resize || ""));
    return false;
  }

  const observer = new MutationObserver(() => initForLayer());
  document.addEventListener("click", (event) => {
    const actionEl = event.target && event.target.closest ? event.target.closest(".spatchat-values-root [data-action]") : null;
    if (!actionEl) return;
    handleAction(actionEl, event);
  }, true);
  window.addEventListener("DOMContentLoaded", () => {
    initForLayer();
    const layer = document.getElementById(layerId);
    if (layer) observer.observe(layer, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["data-payload"],
    });
  });
  window.addEventListener("load", initForLayer);
  let initAttempts = 0;
  const initTimer = window.setInterval(() => {
    initForLayer();
    initAttempts += 1;
    if (initAttempts >= 20) window.clearInterval(initTimer);
  }, 500);

  return { initForLayer, handleAction, handleDrag, handleResize };
})();
</script>
"""

with gr.Blocks(title="Spatchat: Species Distribution Model") as demo:
    gr.Image(
        value="logo_long1.png",
        show_label=False,
        type="filepath",
        elem_id="logo-img"
    )
    gr.HTML("""
    <style>
    .gradio-container {
        --spatchat-room-bg: var(--body-background-fill, var(--background-fill-primary, #ffffff));
        max-width: 100% !important;
        padding-left: 4px !important;
        padding-right: 4px !important;
    }

    #spatchat-workarea {
        overflow-x: auto;
        overflow-y: visible;
        padding-bottom: 8px;
    }

    #spatchat-workarea > .gradio-row {
        min-width: 900px;
        align-items: stretch;
        flex-wrap: nowrap;
        gap: 0 !important;
    }

    #spatchat-sidebar {
        min-width: 0 !important;
        max-width: 1200px;
        flex: 0 0 420px !important;
        width: 420px;
        overflow: visible;
        padding-right: 4px !important;
        font-size: 15px;
        scrollbar-color: color-mix(in srgb, var(--background-fill-secondary) 78%, white 22%) var(--background-fill-secondary);
        scrollbar-width: thin;
    }

    #spatchat-sidebar::-webkit-scrollbar {
        width: 12px;
    }

    #spatchat-sidebar::-webkit-scrollbar-track {
        background: var(--background-fill-secondary);
        border-radius: 999px;
    }

    #spatchat-sidebar::-webkit-scrollbar-thumb {
        background: color-mix(in srgb, var(--background-fill-secondary) 78%, white 22%);
        border-radius: 999px;
        border: 2px solid var(--background-fill-secondary);
    }

    #spatchat-sidebar::-webkit-scrollbar-thumb:hover {
        background: color-mix(in srgb, var(--background-fill-secondary) 70%, white 30%);
    }

    #spatchat-sidebar::-webkit-scrollbar-button:single-button {
        display: block;
        height: 12px;
        background-color: color-mix(in srgb, var(--background-fill-secondary) 78%, white 22%);
        border-radius: 999px;
        border: 2px solid var(--background-fill-secondary);
        background-repeat: no-repeat;
        background-position: center;
        background-size: 7px 7px;
    }

    #spatchat-sidebar::-webkit-scrollbar-button:single-button:vertical:decrement {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 6.5 5 3.5 8 6.5' fill='none' stroke='%23606b7a' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }

    #spatchat-sidebar::-webkit-scrollbar-button:single-button:vertical:increment {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 3.5 5 6.5 8 3.5' fill='none' stroke='%23606b7a' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }

    .dark #spatchat-sidebar {
        scrollbar-color: color-mix(in srgb, var(--background-fill-secondary) 86%, white 14%) var(--background-fill-secondary);
    }

    .dark #spatchat-sidebar::-webkit-scrollbar-thumb,
    .dark #spatchat-sidebar::-webkit-scrollbar-button:single-button {
        background-color: color-mix(in srgb, var(--background-fill-secondary) 86%, white 14%);
        background: color-mix(in srgb, var(--background-fill-secondary) 86%, white 14%);
    }

    #spatchat-sidebar > div {
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    #spatchat-chatbot,
    #spatchat-user-input,
    #spatchat-file-input,
    #spatchat-raster-input,
    #spatchat-lat-dropdown,
    #spatchat-lon-dropdown,
    #spatchat-crs-input,
    #spatchat-confirm,
    #spatchat-download {
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    #spatchat-chatbot {
        padding-left: 0 !important;
        padding-right: 0 !important;
        --chatbot-body-text-size: 16px;
        font-size: 16px !important;
        height: 56vh !important;
        min-height: 56vh !important;
        background: var(--spatchat-room-bg) !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: var(--radius-lg) var(--radius-lg) 0 0 !important;
        overflow: hidden !important;
    }

    #spatchat-chatbot > div {
        background: var(--spatchat-room-bg) !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: var(--radius-lg) var(--radius-lg) 0 0 !important;
    }

    #spatchat-status {
        margin: 0 !important;
        padding: 2px 0 6px 0 !important;
        min-height: 20px;
        border: 0 !important;
        box-shadow: none !important;
        outline: none !important;
        background: var(--spatchat-room-bg) !important;
    }

    #spatchat-status > div,
    #spatchat-status .prose,
    #spatchat-status p {
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        box-shadow: none !important;
        outline: none !important;
        background: transparent !important;
    }

    #spatchat-status hr {
        display: none !important;
    }

    #spatchat-status .spatchat-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        line-height: 1;
        color: color-mix(in srgb, var(--body-text-color) 86%, #6eb8ff 14%);
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    #spatchat-status .spatchat-status-dot {
        width: 8px;
        height: 8px;
        flex: 0 0 8px;
        border-radius: 999px;
        background: #6eb8ff;
        box-shadow: 0 0 0 0 rgba(110, 184, 255, 0.7);
        animation: spatchatStatusPulse 1.4s ease-out infinite;
    }

    @keyframes spatchatStatusPulse {
        0% { box-shadow: 0 0 0 0 rgba(110, 184, 255, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(110, 184, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(110, 184, 255, 0); }
    }

    #spatchat-chatbot,
    #spatchat-chatbot > div {
        scrollbar-color: rgba(133, 146, 171, 0.75) var(--spatchat-room-bg);
        scrollbar-width: thin;
    }

    #spatchat-chatbot::-webkit-scrollbar,
    #spatchat-chatbot > div::-webkit-scrollbar {
        width: 12px;
    }

    #spatchat-chatbot::-webkit-scrollbar-track,
    #spatchat-chatbot > div::-webkit-scrollbar-track {
        background: var(--spatchat-room-bg);
        border-radius: 999px;
    }

    #spatchat-chatbot::-webkit-scrollbar-thumb,
    #spatchat-chatbot > div::-webkit-scrollbar-thumb {
        background: #323845;
        border-radius: 999px;
        border: 2px solid var(--spatchat-room-bg);
    }

    #spatchat-chatbot::-webkit-scrollbar-thumb:hover,
    #spatchat-chatbot > div::-webkit-scrollbar-thumb:hover {
        background: #323845;
    }

    #spatchat-chatbot::-webkit-scrollbar-button:single-button,
    #spatchat-chatbot > div::-webkit-scrollbar-button:single-button {
        display: block;
        height: 12px;
        background-color: #323845;
        border-radius: 999px;
        border: 2px solid var(--spatchat-room-bg);
        background-repeat: no-repeat;
        background-position: center;
        background-size: 7px 7px;
    }

    #spatchat-chatbot::-webkit-scrollbar-button:single-button:vertical:decrement,
    #spatchat-chatbot > div::-webkit-scrollbar-button:single-button:vertical:decrement {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 6.5 5 3.5 8 6.5' fill='none' stroke='%23c7ced9' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }

    #spatchat-chatbot::-webkit-scrollbar-button:single-button:vertical:increment,
    #spatchat-chatbot > div::-webkit-scrollbar-button:single-button:vertical:increment {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 3.5 5 6.5 8 3.5' fill='none' stroke='%23c7ced9' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }

    #spatchat-chatbot button[aria-label*="Copy"],
    #spatchat-chatbot button[aria-label*="copy"],
    #spatchat-chatbot button[aria-label*="Share"],
    #spatchat-chatbot button[aria-label*="share"] {
        display: none !important;
    }

    #spatchat-user-input {
        background: color-mix(in srgb, var(--background-fill-secondary) 78%, white 22%) !important;
        border-radius: 0 0 var(--radius-lg) var(--radius-lg) !important;
        padding: 6px 10px !important;
        margin-top: -8px !important;
        border-top: none !important;
    }

    .dark #spatchat-user-input {
        background: color-mix(in srgb, var(--background-fill-secondary) 86%, white 14%) !important;
    }

    #spatchat-user-input > div,
    #spatchat-user-input .wrap,
    #spatchat-user-input label {
        background: transparent !important;
    }

    #spatchat-user-input .block-label,
    #spatchat-user-input .label-wrap,
    #spatchat-user-input [data-testid="block-label"] {
        display: none !important;
    }

    #spatchat-user-input .input-container {
        align-items: center !important;
    }

    #spatchat-user-input textarea,
    #spatchat-user-input input {
        background: color-mix(in srgb, var(--background-fill-secondary) 78%, white 22%) !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: var(--radius-md) !important;
        height: 48px !important;
        min-height: 48px !important;
        box-sizing: border-box !important;
        padding: 13px 12px !important;
        line-height: 22px !important;
    }

    #spatchat-user-input textarea {
        max-height: 240px !important;
        overflow-y: auto !important;
        resize: vertical !important;
    }

    #spatchat-user-input input {
        overflow: hidden !important;
    }

    .dark #spatchat-user-input textarea,
    .dark #spatchat-user-input input {
        background: color-mix(in srgb, var(--background-fill-secondary) 86%, white 14%) !important;
    }

    #spatchat-user-input textarea::placeholder,
    #spatchat-user-input input::placeholder {
        color: var(--body-text-color-subdued) !important;
        text-align: left !important;
    }

    #spatchat-chatbot .message-row,
    #spatchat-chatbot .message-wrap,
    #spatchat-chatbot .bubble-wrap,
    #spatchat-chatbot .wrap {
        background: transparent !important;
    }

    #spatchat-chatbot .message-row.bot,
    #spatchat-chatbot .message-row.assistant,
    #spatchat-chatbot .message-row.bot-row,
    #spatchat-chatbot .bubble.bot-row,
    #spatchat-chatbot [data-testid="chatbot-message-assistant"] {
        background: var(--spatchat-room-bg) !important;
        background-color: var(--spatchat-room-bg) !important;
        justify-content: flex-start !important;
    }

    #spatchat-chatbot .message-row.bot-row,
    #spatchat-chatbot .message-row.bot-row .flex-wrap,
    #spatchat-chatbot .message-row.bot-row .message,
    #spatchat-chatbot .message-row.bot-row .message-bubble-border,
    #spatchat-chatbot .message-row.bot-row .bot,
    #spatchat-chatbot .bubble.bot-row,
    #spatchat-chatbot .bubble.bot-row .message,
    #spatchat-chatbot .bubble.bot-row .message-bubble-border,
    #spatchat-chatbot .bubble.bot-row .bot,
    #spatchat-chatbot .bot,
    #spatchat-chatbot .bot-row .message,
    #spatchat-chatbot .bot-row .message-bubble-border,
    #spatchat-chatbot .bot-row .bot,
    #spatchat-chatbot .bubble.pending,
    #spatchat-chatbot .bubble.pending .message-content {
        background: var(--spatchat-room-bg) !important;
        background-color: var(--spatchat-room-bg) !important;
        border: none !important;
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }

    #spatchat-chatbot .message-row.bot-row .bot,
    #spatchat-chatbot .message-row.bot-row,
    #spatchat-chatbot .bubble.bot-row,
    #spatchat-chatbot .bubble.bot-row .bot,
    #spatchat-chatbot .bot,
    #spatchat-chatbot .bot-row .bot,
    #spatchat-chatbot .bubble.pending {
        padding: 0 !important;
        border-radius: 0 !important;
    }

    #spatchat-chatbot .message-row.bot .bubble-wrap,
    #spatchat-chatbot .message-row.assistant .bubble-wrap,
    #spatchat-chatbot [data-testid="chatbot-message-assistant"] .bubble-wrap,
    #spatchat-chatbot .message-row.bot .bubble,
    #spatchat-chatbot .message-row.assistant .bubble,
    #spatchat-chatbot [data-testid="chatbot-message-assistant"] .bubble,
    #spatchat-chatbot .message-row.bot .avatar-container,
    #spatchat-chatbot .message-row.assistant .avatar-container,
    #spatchat-chatbot [data-testid="chatbot-message-assistant"] .avatar-container,
    #spatchat-chatbot .message-row.bot .message,
    #spatchat-chatbot .message-row.assistant .message,
    #spatchat-chatbot [data-testid="chatbot-message-assistant"] .message {
        background: var(--spatchat-room-bg) !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    #spatchat-chatbot .message-row.bot .message,
    #spatchat-chatbot .message-row.assistant .message,
    #spatchat-chatbot [data-testid="chatbot-message-assistant"] .message {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        width: auto !important;
        color: inherit !important;
    }

    #spatchat-chatbot .message-row.bot .message *,
    #spatchat-chatbot .message-row.assistant .message *,
    #spatchat-chatbot [data-testid="chatbot-message-assistant"] .message * {
        background: transparent !important;
    }

    #spatchat-chatbot .message-row.user,
    #spatchat-chatbot [data-testid="chatbot-message-user"] {
        background: transparent !important;
        display: flex !important;
        justify-content: flex-end !important;
    }

    #spatchat-chatbot .message-row.user .message,
    #spatchat-chatbot [data-testid="chatbot-message-user"] .message {
        margin-left: auto !important;
        margin-right: 0 !important;
        width: auto !important;
        max-width: 90% !important;
        padding: 10px 14px !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 16px 16px 4px 16px !important;
        background: #5b86c5 !important;
        color: #ffffff !important;
    }

    .dark #spatchat-chatbot .message-row.user .message,
    .dark #spatchat-chatbot [data-testid="chatbot-message-user"] .message {
        background: #4d79bd !important;
        color: #ffffff !important;
    }

    #spatchat-chatbot .message-row.user .message *,
    #spatchat-chatbot [data-testid="chatbot-message-user"] .message * {
        color: inherit !important;
        background: transparent !important;
    }

    #spatchat-file-input,
    #spatchat-raster-input,
    #spatchat-lat-dropdown,
    #spatchat-lon-dropdown,
    #spatchat-crs-input,
    #spatchat-confirm {
        margin-top: 10px !important;
    }

    #spatchat-confirm button {
        width: 100%;
    }

    #spatchat-splitter {
        min-width: 10px;
        max-width: 10px;
        width: 10px;
        flex: 0 0 10px !important;
        position: relative;
        padding: 0 !important;
        margin: 0 !important;
        min-height: 78vh;
        align-self: stretch !important;
        background: transparent !important;
        overflow: visible !important;
    }

    #spatchat-splitter > div,
    #spatchat-splitter .gradio-html,
    #spatchat-splitter .gradio-html > div {
        height: 100%;
        min-height: 78vh;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
    }

    #spatchat-splitter .spatchat-splitter-handle {
        position: absolute;
        inset: 0;
        display: block;
        width: 100%;
        height: 100%;
        cursor: col-resize;
        user-select: none;
        background: transparent;
    }

    #spatchat-splitter .spatchat-splitter-handle::before {
        content: "";
        position: absolute;
        top: 8px;
        bottom: 8px;
        left: 50%;
        width: 1px;
        transform: translateX(-50%);
        background: rgba(120, 130, 145, 0.55);
        border-radius: 999px;
        transition: background 120ms ease, width 120ms ease;
    }

    #spatchat-splitter:hover .spatchat-splitter-handle::before,
    body.spatchat-resizing #spatchat-splitter .spatchat-splitter-handle::before {
        width: 2px;
        background: #5b86c5;
    }

    body.spatchat-resizing,
    body.spatchat-resizing * {
        cursor: col-resize !important;
        user-select: none !important;
    }

    #spatchat-mapcol {
        min-width: 0 !important;
        flex: 1 1 auto !important;
        width: auto !important;
    }

    #spatchat-map {
        min-height: 78vh;
        height: 78vh;
        overflow: visible;
        width: 100%;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    #spatchat-map > div {
        height: 100%;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        background: transparent !important;
    }

    #spatchat-map .gradio-html,
    #spatchat-map .gradio-html > div,
    #spatchat-map .block {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        background: transparent !important;
    }

    #spatchat-map iframe {
        width: 100% !important;
        height: 100% !important;
        min-height: 78vh;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
    }

    #spatchat-map iframe:focus,
    #spatchat-map iframe:focus-visible {
        outline: none !important;
        box-shadow: none !important;
        border: none !important;
    }

    body.spatchat-resizing #spatchat-map iframe {
        pointer-events: none !important;
    }

    #spatchat-download {
        display: flex;
        justify-content: center;
        width: 100%;
        margin-top: 10px;
    }

    #spatchat-download button {
        min-width: 220px;
        width: 100%;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px;
        text-align: center;
    }

    #spatchat-feature-viewer {
        margin-top: 10px !important;
    }

    #spatchat-feature-viewer {
        position: relative;
        min-height: 0;
    }
    #spatchat-feature-viewer > div,
    #spatchat-feature-viewer .spatchat-values-root {
        min-height: 0;
    }
    .spatchat-values-root {
        position: relative;
        pointer-events: auto;
    }
    .spatchat-values-launcher {
        position: fixed;
        right: 22px;
        bottom: 22px;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 11px 16px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        background: rgba(9, 13, 21, 0.96);
        color: #f4f7fb;
        pointer-events: auto;
        box-shadow: 0 18px 44px rgba(0, 0, 0, 0.38);
        font-size: 13px;
        font-weight: 700;
        cursor: pointer;
        z-index: 2102;
    }
    .spatchat-values-modal-backdrop {
        position: fixed;
        inset: 0;
        background: transparent;
        pointer-events: none;
    }
    .spatchat-values-interaction-shield {
        position: fixed;
        inset: 0;
        background: transparent;
        pointer-events: none;
        z-index: 2099;
    }
    .spatchat-values-interaction-shield.is-active {
        pointer-events: auto;
        cursor: inherit;
    }
    .spatchat-values-modal {
        position: fixed;
        left: clamp(12px, calc(50vw - 280px), calc(100vw - 572px));
        top: clamp(12px, calc(50vh - 230px), calc(100vh - 472px));
        width: min(560px, calc(100vw - 24px));
        height: min(460px, calc(100vh - 24px));
        max-width: calc(100vw - 24px);
        max-height: calc(100vh - 24px);
        margin: 0;
        display: flex;
        flex-direction: column;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(9, 13, 21, 0.97);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.46);
        color: #f4f7fb;
        overflow: hidden;
        pointer-events: auto;
        transform: none;
        z-index: 2101;
    }
    .spatchat-values-modal.is-hidden {
        display: none;
    }
    .spatchat-values-modal-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        padding: 6px 10px;
        background: linear-gradient(135deg, rgba(36, 51, 82, 0.94), rgba(11, 15, 25, 0.94));
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        cursor: move;
        user-select: none;
    }
    .spatchat-values-modal-copy {
        display: flex;
        flex-direction: column;
        gap: 0;
        min-width: 0;
    }
    .spatchat-values-modal-title {
        font-size: 12px;
        font-weight: 700;
        line-height: 1.15;
    }
    .spatchat-values-modal-count {
        font-size: 10px;
        line-height: 1.1;
        color: rgba(226, 233, 245, 0.74);
    }
    .spatchat-values-modal-nav {
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .spatchat-values-modal-btn {
        width: 24px;
        height: 24px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: none;
        border-radius: 7px;
        background: rgba(255, 255, 255, 0.08);
        color: #f4f7fb;
        font-size: 13px;
        font-weight: 700;
        cursor: pointer;
    }
    .spatchat-values-modal-btn:disabled {
        opacity: 0.35;
        cursor: default;
    }
    .spatchat-values-modal-btn-close {
        font-size: 16px;
        line-height: 0.7;
        padding-bottom: 6px;
    }
    .spatchat-values-modal-body {
        flex: 1 1 auto;
        overflow: auto;
        padding: 18px;
        display: flex;
        flex-direction: column;
    }
    .spatchat-values-resize-handle {
        position: absolute;
        z-index: 3;
        background: transparent;
    }
    .spatchat-values-resize-handle.is-n { left: 12px; right: 12px; top: -4px; height: 8px; cursor: ns-resize; }
    .spatchat-values-resize-handle.is-s { left: 12px; right: 12px; bottom: -4px; height: 8px; cursor: ns-resize; }
    .spatchat-values-resize-handle.is-e { top: 12px; bottom: 12px; right: -4px; width: 8px; cursor: ew-resize; }
    .spatchat-values-resize-handle.is-w { top: 12px; bottom: 12px; left: -4px; width: 8px; cursor: ew-resize; }
    .spatchat-values-resize-handle.is-ne,
    .spatchat-values-resize-handle.is-nw,
    .spatchat-values-resize-handle.is-se,
    .spatchat-values-resize-handle.is-sw { width: 18px; height: 18px; }
    .spatchat-values-resize-handle.is-ne { top: -4px; right: -4px; cursor: nesw-resize; }
    .spatchat-values-resize-handle.is-nw { top: -4px; left: -4px; cursor: nwse-resize; }
    .spatchat-values-resize-handle.is-se { right: -4px; bottom: -4px; cursor: nwse-resize; }
    .spatchat-values-resize-handle.is-sw { left: -4px; bottom: -4px; cursor: nesw-resize; }
    .spatchat-values-resize-handle.is-se::before {
        content: "";
        position: absolute;
        right: 7px;
        bottom: 7px;
        width: 11px;
        height: 11px;
        border-right: 2px solid rgba(244, 247, 251, 0.72);
        border-bottom: 2px solid rgba(244, 247, 251, 0.72);
        border-bottom-right-radius: 2px;
    }
    .spatchat-values-card {
        display: flex;
        flex-direction: column;
        flex: 1 1 auto;
        gap: 10px;
        min-height: 0;
        padding: 16px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .spatchat-values-card-title {
        font-size: 16px;
        font-weight: 700;
        line-height: 1.25;
    }
    .spatchat-values-card-meta {
        font-size: 13px;
        color: rgba(226, 233, 245, 0.74);
        line-height: 1.35;
    }
    .spatchat-values-card-table {
        flex: 1 1 auto;
        min-height: 0;
        overflow: auto;
    }
    .spatchat-values-wrap {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .spatchat-values-note {
        font-size: 11px;
        color: rgba(226, 233, 245, 0.72);
    }
    .spatchat-values-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }
    .spatchat-values-table th,
    .spatchat-values-table td {
        padding: 7px 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        text-align: left;
        white-space: nowrap;
    }
    .spatchat-values-table th {
        position: sticky;
        top: 0;
        background: rgba(13, 18, 29, 0.98);
        z-index: 1;
        font-weight: 700;
    }
    .spatchat-values-empty {
        font-size: 12px;
        color: rgba(226, 233, 245, 0.72);
    }
    body.spatchat-values-moving,
    body.spatchat-values-moving * {
        cursor: move !important;
        user-select: none !important;
    }
    body.spatchat-values-resizing,
    body.spatchat-values-resizing * {
        user-select: none !important;
    }
    @media (max-width: 900px) {
        .spatchat-values-launcher {
            right: 14px;
            bottom: 14px;
        }
        .spatchat-values-modal {
            left: 12px;
            top: 12px;
            width: calc(100vw - 24px);
            height: calc(100vh - 24px);
            max-height: calc(100vh - 24px);
        }
        .spatchat-values-modal-head {
            padding: 14px;
            align-items: flex-start;
            flex-direction: column;
        }
        .spatchat-values-modal-nav {
            width: 100%;
            justify-content: flex-end;
        }
        .spatchat-values-modal-body {
            padding: 14px;
        }
    }


    #logo-img img {
        height: 90px;
        margin: 10px 50px 10px 10px;
        border-radius: 6px;
    }
    </style>
    """)
    gr.Markdown("## 🗺️ Spatchat: Species Distribution Model {sdm}  🐢🐍🦅🦋🦉🦊🐞 ")
    gr.HTML("""
    <div style="margin-top: -10px; margin-bottom: 15px;">
      <input type="text" value="https://spatchat.org/browse/?room=sdm" id="shareLink" readonly style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
      <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">
        📋 Copy Share Link
      </button>
      <div style="margin-top: 10px; font-size: 14px;">
        <b>Share:</b>
        <a href="https://twitter.com/intent/tweet?text=Checkout+Spatchat!&url=https://spatchat.org/browse/?room=sdm" target="_blank">🐦 Twitter</a> |
        <a href="https://www.facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=sdm" target="_blank">📘 Facebook</a>
      </div>
    </div>
    """)
    gr.Markdown("""
                <div style="font-size: 14px;">
                © 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
                If you use Spatchat in research, please cite:<br>
                <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>Spatchat: Species Distribution Model.</i>
                </div>
                """)
    state = gr.State(_fresh_state())
    with gr.Row(elem_id="spatchat-workarea"):
        with gr.Column(scale=3, min_width=0, elem_id="spatchat-sidebar"):
            _chatbot_base_kwargs = dict(
                label="Spatchat",
                show_label=False,
                layout="bubble",
                elem_id="spatchat-chatbot",
            )
            _welcome_messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
            try:
                chat = gr.Chatbot(
                    value=_welcome_messages,
                    **_chatbot_base_kwargs,
                )
            except Exception:
                chat = gr.Chatbot(
                    value=_messages_to_legacy_chat(_welcome_messages),
                    **_chatbot_base_kwargs,
                )
                CHATBOT_HISTORY_MODE = "tuples"
            status_output = gr.HTML(value="", visible=True, elem_id="spatchat-status")
            user_in = gr.Textbox(
                label=None,
                show_label=False,
                placeholder="Ask Spatchat...",
                lines=1,
                elem_id="spatchat-user-input",
            )
            file_input = gr.File(
                label="Upload CSV",
                type="filepath",
                file_types=[".csv"],
                elem_id="spatchat-file-input",
            )
            raster_input = gr.File(
                label="Upload predictor rasters",
                type="filepath",
                file_count="multiple",
                file_types=[".tif", ".tiff"],
                elem_id="spatchat-raster-input",
            )
            lat_dropdown = gr.Dropdown(
                choices=[],
                label="Y / latitude column",
                visible=False,
                elem_id="spatchat-lat-dropdown",
            )
            lon_dropdown = gr.Dropdown(
                choices=[],
                label="X / longitude column",
                visible=False,
                elem_id="spatchat-lon-dropdown",
            )
            crs_input = gr.Textbox(
                label="Input CRS (code, zone, or name)",
                placeholder="e.g. 32610, UTM zone 10N, LCC...",
                visible=False,
                elem_id="spatchat-crs-input",
            )
            confirm_btn = gr.Button(
                "Confirm Coordinates",
                visible=False,
                elem_id="spatchat-confirm",
            )
            download_btn = gr.DownloadButton(
                "Download Results",
                value=None,
                elem_id="spatchat-download",
            )
            feature_viewer = gr.HTML(
                value=_render_feature_viewer_html(None),
                elem_id="spatchat-feature-viewer",
            )
        with gr.Column(scale=0, min_width=14, elem_id="spatchat-splitter"):
            gr.HTML("<div class='spatchat-splitter-handle' title='Drag to resize panels'></div>")
        with gr.Column(scale=5, min_width=0, elem_id="spatchat-mapcol"):
            map_out = gr.HTML(create_map(), label="Map Preview", show_label=False, elem_id="spatchat-map")

    _prune_stale_sessions(max_age_seconds=60 * 60 * 24)
    _start_session_pruner()
    demo.queue(max_size=16)
    demo.unload(_cleanup_current_browser_session)

    file_input.change(on_upload_with_viewer,
        inputs=[file_input, chat, state],
        outputs=[chat, map_out, state, download_btn,
                 lat_dropdown, lon_dropdown, crs_input, confirm_btn, feature_viewer]
    )
    raster_input.change(
        on_raster_upload,
        inputs=[raster_input, chat, state],
        outputs=[chat, map_out, state, download_btn, feature_viewer],
    )
    confirm_btn.click(confirm_coords_with_viewer,
        inputs=[lat_dropdown, lon_dropdown, crs_input, chat, state],
        outputs=[chat, map_out, state, download_btn,
                 lat_dropdown, lon_dropdown, crs_input, confirm_btn, feature_viewer]
    )
    user_in.submit(
        chat_step_with_viewer,
        inputs=[file_input, user_in, chat, state],
        outputs=[chat, map_out, state, download_btn, feature_viewer, status_output],
        show_progress="hidden",
        js="""
        (file, message, chatHistory, state) => {
            const text = message ?? "";
            setTimeout(() => {
                const input = document.querySelector('#spatchat-user-input textarea, #spatchat-user-input input');
                if (input) {
                    input.value = "";
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }, 0);
            return [file, text, chatHistory, state];
        }
        """,
    )

def _launch_demo(blocks: gr.Blocks) -> None:
    launch_kwargs = {"head": _SPLITTER_HEAD}
    try:
        sig = inspect.signature(gr.Blocks.launch)
        params = getattr(sig, "parameters", {}) or {}
    except (ValueError, TypeError):
        params = {}

    available = params.keys() if hasattr(params, "keys") else []
    has_var_kwargs = any(
        getattr(p, "kind", None) == inspect.Parameter.VAR_KEYWORD
        for p in (params.values() if hasattr(params, "values") else [])
    )
    if "head" not in available and not has_var_kwargs:
        launch_kwargs.pop("head", None)
    if "ssr_mode" in available:
        launch_kwargs["ssr_mode"] = False

    blocks.launch(**launch_kwargs)

_launch_demo(demo)
