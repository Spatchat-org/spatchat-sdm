# spatchat_layout.py
"""The standard SpatChat room screen. Every room looks the same; only the analysis differs.

    left   chat, suggestion chips with "Try sample data", message box with "+", results download
    right  tabs in a fixed order:  Map  |  Plots  |  Files  |  (room-specific extras)

    Map    spatial data (points, rasters, results). A room without spatial data still has it, so the screen is
           the same everywhere; with nothing to show it carries the standard "Drop a ... here" banner.
    Plots  every figure the room has made, side by side. Click one to enlarge it.
    Files  what was uploaded and what was produced, with Remove ticks (see spatchat_files.py).

A new room is built like this, inside `with gr.Blocks(...) as demo:`

    ui = spatchat_layout.build(gr, title="...", subtitle="...", room="...", cite="...", welcome="...",
                               chips=[...], placeholder="...", file_types=[".csv"],
                               map_html=spatchat_layout.empty_map("Drop a CSV here, or click + to start", "..."),
                               plots_empty_html=spatchat_ui.empty_state_html("plot", "No plots yet", "..."))

and then wires events to `ui.chatbot`, `ui.file_input`, `ui.sample_btn`, `ui.map_out`, `ui.gallery`, `ui.files`, ...
Map content comes from `points_map(...)` / `raster_map(...)` / `empty_map(...)` below.

Keep the copies in every room byte-identical.
"""
from __future__ import annotations

import html
from types import SimpleNamespace
from typing import Iterable, Optional

import spatchat_files as sfiles
import spatchat_ui

# One qualitative palette, no green. Used when a map or chart needs "a colour per group".
PALETTE = ["#fa6a98", "#e4572e", "#9b3dbf", "#f2a900", "#0891b2", "#d6336c", "#7c4a2d", "#475569", "#4f46e5", "#c2410c"]
POINT_COLOR = PALETTE[0]
MAX_POINTS = 5000

_LAT_NAMES = ("latitude", "lat", "decimallatitude", "decimal_latitude", "lat_dd", "y_wgs84")
_LON_NAMES = ("longitude", "lon", "lng", "long", "decimallongitude", "decimal_longitude", "lon_dd", "x_wgs84")


# ----------------------------------------------------------------------------------------------
# maps
# ----------------------------------------------------------------------------------------------
def _folium():
    import folium  # imported lazily: only rooms that draw maps need it at import time

    return folium


def _base_map(location=(20, 0), zoom=2):
    folium = _folium()
    m = folium.Map(location=list(location), zoom_start=zoom, control_scale=True, tiles=None)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(m)
    folium.TileLayer("CartoDB positron", name="CartoDB Positron", attr="CartoDB", show=False).add_to(m)
    folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", name="Topographic", attr="OpenTopoMap",
                     show=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Satellite", attr="Esri", show=True).add_to(m)
    return m


def _banner(title: str, sub: str) -> str:
    return (
        '<div class="spatchat-map-empty-hint"><div class="spatchat-map-empty-hint-card">'
        '<div class="spatchat-map-empty-hint-icon">&#128205;</div>'
        f"<div>{html.escape(title)}</div>"
        f'<div class="spatchat-map-empty-hint-sub">{html.escape(sub)}</div></div></div>'
    )


def empty_map(title: str, sub: str = "") -> str:
    """World map with the standard banner on top. Used before there is anything to show."""
    folium = _folium()
    m = _base_map()
    folium.LayerControl(collapsed=False).add_to(m)
    return _banner(title, sub) + m._repr_html_()


def detect_lonlat(df) -> Optional[tuple[str, str]]:
    """(lat column, lon column) when the table has WGS84 coordinates, else None."""
    import pandas as pd

    low = {str(c).strip().lower(): c for c in df.columns}
    lat = next((low[n] for n in _LAT_NAMES if n in low), None)
    lon = next((low[n] for n in _LON_NAMES if n in low), None)
    if lat is None or lon is None:
        return None
    la = pd.to_numeric(df[lat], errors="coerce")
    lo = pd.to_numeric(df[lon], errors="coerce")
    ok = la.between(-90, 90) & lo.between(-180, 180)
    return (lat, lon) if ok.sum() >= 1 and ok.mean() >= 0.8 else None


def points_map(df, lat: str, lon: str, label: str = "Points", popup_cols: Optional[Iterable[str]] = None) -> str:
    """Map of the rows of `df` as points, with the standard layer control (the legend that turns layers on and off)."""
    import pandas as pd

    folium = _folium()
    d = df.copy()
    d[lat] = pd.to_numeric(d[lat], errors="coerce")
    d[lon] = pd.to_numeric(d[lon], errors="coerce")
    d = d.dropna(subset=[lat, lon])
    d = d[d[lat].between(-90, 90) & d[lon].between(-180, 180)]
    if d.empty:
        return empty_map("No valid coordinates in this file", "Latitude and longitude must be in decimal degrees (WGS84).")
    total = len(d)
    if total > MAX_POINTS:
        d = d.sample(MAX_POINTS, random_state=0)
        label = f"{label} ({MAX_POINTS:,} of {total:,})"
    cols = [c for c in (popup_cols or [c for c in df.columns if c not in (lat, lon)][:6]) if c in d.columns]
    m = _base_map(location=(float(d[lat].mean()), float(d[lon].mean())), zoom=4)
    layer = folium.FeatureGroup(name=label, show=True)
    for _, row in d.iterrows():
        tip = "<br>".join(f"<b>{html.escape(str(c))}</b>: {html.escape(str(row[c]))}" for c in cols) or None
        folium.CircleMarker((row[lat], row[lon]), radius=4, color=POINT_COLOR, weight=1, fill=True,
                            fill_color=POINT_COLOR, fill_opacity=0.7,
                            popup=folium.Popup(tip, max_width=260) if tip else None).add_to(layer)
    layer.add_to(m)
    m.fit_bounds([[float(d[lat].min()), float(d[lon].min())], [float(d[lat].max()), float(d[lon].max())]],
                 max_zoom=14, padding=(20, 20))
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


def raster_map(path: str, label: str = "Land cover", max_px: int = 1000) -> str:
    """Map of a categorical raster: reprojected to WGS84, one colour per class, with a class legend.
    Raises ValueError when the file has no spatial reference (the caller shows an explanation instead)."""
    import numpy as np
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    folium = _folium()
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("This raster has no spatial reference (CRS), so it cannot be placed on the map.")
        scale = max(1.0, max(src.width, src.height) / max_px)
        h, w = max(1, int(src.height / scale)), max(1, int(src.width / scale))
        data = src.read(1, out_shape=(h, w), masked=True)
        if data.dtype.kind not in "iuf":
            raise ValueError("This raster stores text labels, which cannot be drawn on the map.")
        arr = np.where(np.ma.getmaskarray(data), np.nan, data.filled(0)).astype("float32")
        src_transform = src.transform * Affine.scale(src.width / w, src.height / h)
        dst_t, dw, dh = calculate_default_transform(src.crs, "EPSG:4326", w, h, *src.bounds)
        out = np.full((dh, dw), np.nan, dtype="float32")
        reproject(arr, out, src_transform=src_transform, src_crs=src.crs, dst_transform=dst_t,
                  dst_crs="EPSG:4326", resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    west, north = dst_t.c, dst_t.f
    east, south = west + dw * dst_t.a, north + dh * dst_t.e

    classes = sorted(int(v) for v in np.unique(out[~np.isnan(out)]))
    if not classes:
        raise ValueError("This raster has no data cells to draw.")
    rgba = np.zeros((dh, dw, 4), dtype="uint8")
    legend = []
    for i, cls in enumerate(classes):
        colour = PALETTE[i % len(PALETTE)]
        r, g, b = (int(colour[j:j + 2], 16) for j in (1, 3, 5))
        rgba[out == cls] = (r, g, b, 215)
        legend.append((cls, colour))

    m = _base_map(location=((north + south) / 2, (east + west) / 2), zoom=8)
    folium.raster_layers.ImageOverlay(image=rgba, bounds=[[south, west], [north, east]], name=label, opacity=1.0,
                                      show=True).add_to(m)
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
        f'<span style="width:12px;height:12px;border-radius:3px;background:{c};display:inline-block"></span>'
        f"Class {cls}</div>" for cls, c in legend[:14])
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:28px;left:10px;z-index:9999;background:rgba(255,255,255,.92);'
        'padding:8px 10px;border-radius:8px;font:12px/1.3 system-ui,sans-serif;color:#0f172a;'
        f'box-shadow:0 1px 4px rgba(0,0,0,.25)"><b>{html.escape(label)}</b>{rows}</div>'))
    m.fit_bounds([[south, west], [north, east]], max_zoom=16, padding=(20, 20))
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


# ----------------------------------------------------------------------------------------------
# the screen
# ----------------------------------------------------------------------------------------------
GALLERY_CSS = """
<style>
#spatchat-gallery { width: 100% !important; height: auto !important; max-height: none !important; }
#spatchat-gallery .grid-wrap, #spatchat-gallery .grid-container { height: auto !important; max-height: none !important; overflow: visible !important; }
#spatchat-gallery .thumbnail-item { aspect-ratio: 1 / 1 !important; height: auto !important; }
#spatchat-gallery .thumbnail-item img { object-fit: contain !important; }
#spatchat-plots-pane { overflow-y: auto !important; }
#spatchat-plots-pane > .column { width: 100% !important; flex: 1 1 100% !important; }
</style>
"""


def plots_tab(gr, empty_html: str, label: str = "Plots"):
    """The standard Plots tab. Returns (gallery, empty-state html). Fill it with `spatchat_ui.gallery_updates`."""
    with gr.Tab(label, id="plots", elem_id="spatchat-plots-pane"):
        gr.HTML(GALLERY_CSS)
        empty = gr.HTML(value=empty_html, visible=True)
        gallery = gr.Gallery(label="Figures", show_label=False, value=None, columns=3, height=None,
                             object_fit="contain", allow_preview=True, elem_id="spatchat-gallery")
    return gallery, empty


def build(gr, *, title: str, subtitle: str, room: str, cite: str, welcome: str, chips: list[str], placeholder: str,
          file_types: list[str], map_html: str, plots_empty_html: str, extra_tabs=None,
          sample_label: str = "Try sample data") -> SimpleNamespace:
    """Build the whole standard screen inside an open `gr.Blocks`. Returns the components to wire up."""
    spatchat_ui.build_header(gr, title, subtitle, room)
    with gr.Row(elem_id="spatchat-workarea"):
        with gr.Column(scale=3, min_width=0, elem_id="spatchat-sidebar"):
            chatbot = gr.Chatbot(**spatchat_ui.chatbot_kwargs(gr, welcome))
            sample_btn = spatchat_ui.chip_row_with_sample(gr, chips, sample_label)
            file_input, user_input, send_btn = spatchat_ui.composer(gr, placeholder, file_types)
            download_btn = gr.DownloadButton("Download results", value=None, visible=False, elem_id="spatchat-download")
        with gr.Column(scale=5, min_width=0, elem_id="spatchat-mapcol"):
            with gr.Tabs(elem_id="spatchat-tabs") as tabs:
                with gr.Tab("Map", id="map"):
                    map_out = gr.HTML(value=map_html, show_label=False, elem_id="spatchat-map")
                gallery, gallery_empty = plots_tab(gr, plots_empty_html)
                files = sfiles.build(gr)
                if extra_tabs:
                    extra_tabs()
            session_state = gr.State(None)
    spatchat_ui.build_footer(gr, cite)
    return SimpleNamespace(chatbot=chatbot, sample_btn=sample_btn, file_input=file_input, user_input=user_input,
                           send_btn=send_btn, download_btn=download_btn, tabs=tabs, map_out=map_out, gallery=gallery,
                           gallery_empty=gallery_empty, files=files, session_state=session_state)
