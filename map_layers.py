import os
import re
import sys
import html as html_lib

import folium
import numpy as np
import pandas as pd
import rasterio
from branca.element import MacroElement, Template
from matplotlib import colormaps
from rasterio.enums import Resampling

from map_utils import add_default_basemaps, render_empty_map, fit_map_to_bounds


def _coerce_wgs84_arrays(df, lat_col, lon_col):
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    arr = np.column_stack([lat.values, lon.values]).astype(float)
    mask = np.isfinite(arr).all(axis=1)
    mask &= (arr[:, 0] >= -90) & (arr[:, 0] <= 90) & (arr[:, 1] >= -180) & (arr[:, 1] <= 180)
    return arr[mask, 0], arr[mask, 1]


def _pane_name(label: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(label or "layer"))
    safe = "-".join(part for part in safe.split("-") if part)
    return f"pane-{safe or 'layer'}"


def _ensure_map_pane(m: folium.Map, label: str, z_index: int) -> str:
    name = _pane_name(label)
    folium.map.CustomPane(name=name, z_index=int(z_index)).add_to(m)
    return name


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _rgba_to_hex(rgba) -> str:
    r, g, b, _ = rgba
    return "#{:02x}{:02x}{:02x}".format(
        int(np.clip(round(float(r) * 255), 0, 255)),
        int(np.clip(round(float(g) * 255), 0, 255)),
        int(np.clip(round(float(b) * 255), 0, 255)),
    )


def _read_display_raster(path: str, max_pixels: int = 1_000_000):
    with rasterio.open(path) as src:
        scale = max(1, int(np.ceil(np.sqrt((src.width * src.height) / max_pixels))))
        out_h = max(1, int(np.ceil(src.height / scale)))
        out_w = max(1, int(np.ceil(src.width / scale)))
        arr = src.read(
            1,
            out_shape=(out_h, out_w),
            masked=True,
            resampling=Resampling.nearest,
        )
        arr = arr.astype(np.float32).filled(np.nan)
        return arr, src.bounds


def _add_raster_legend(
    m: folium.Map,
    *,
    name: str,
    cmap: str,
    vmin: float,
    vmax: float,
    pmin: float,
    pmax: float,
    actual_min: float | None = None,
    actual_max: float | None = None,
) -> None:
    root = m.get_root()
    offset = getattr(root, "_spatchat_legend_offset", 0)
    setattr(root, "_spatchat_legend_offset", offset + 120)
    bottom = 20 + offset

    steps = 8
    cmap_obj = colormaps[cmap]
    colors = [_rgba_to_hex(cmap_obj(i / (steps - 1))) for i in range(steps)]
    stops = [f"{color} {int(round(100 * i / (steps - 1)))}%" for i, color in enumerate(colors)]
    gradient = ", ".join(stops)
    stretch_text = (
        "Constant value"
        if np.isfinite(vmin) and np.isfinite(vmax) and abs(vmax - vmin) <= 1e-12
        else f"Display stretch: p{pmin:.1f}-p{pmax:.1f}"
    )

    actual_range_html = ""
    if (
        actual_min is not None
        and actual_max is not None
        and np.isfinite(actual_min)
        and np.isfinite(actual_max)
    ):
        actual_range_html = (
            f'<div style="font-size: 11px; margin-top: 4px;">'
            f'Data range: {_format_value(actual_min)} to {_format_value(actual_max)}'
            f"</div>"
        )

    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div class="spatchat-map-legend" style="
        position: absolute;
        bottom: {bottom}px;
        left: 20px;
        padding: 10px 12px;
        background: rgba(30, 30, 30, 0.85);
        color: #f5f5f5;
        font-size: 13px;
        line-height: 1.3;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35);
        min-width: 200px;
        z-index: 9999;
        pointer-events: none;
    ">
      <div style="font-weight: 600; margin-bottom: 6px;">{name}</div>
      <div style="
            height: 16px;
            border-radius: 4px;
            background: linear-gradient(to right, {gradient});
            margin-bottom: 6px;
        "></div>
      <div style="display: flex; justify-content: space-between; font-size: 12px;">
        <span>{_format_value(vmin)}</span>
        <span>{_format_value(vmax)}</span>
      </div>
      <div style="font-size: 11px; margin-top: 4px;">{stretch_text}</div>
      {actual_range_html}
    </div>
    {{% endmacro %}}
    """
    legend = MacroElement()
    legend._template = Template(legend_html)
    root.add_child(legend)


def _add_floating_ui_patch(m: folium.Map) -> None:
    root = m.get_root()
    if getattr(root, "_spatchat_floating_ui_patch_added", False):
        return
    setattr(root, "_spatchat_floating_ui_patch_added", True)
    patch_html = """
    {% macro html(this, kwargs) %}
    <style>
      .spatchat-legend-dock {
        position: absolute;
        top: 14px;
        left: 14px;
        width: 280px;
        min-width: 220px;
        max-width: calc(100% - 28px);
        min-height: 120px;
        max-height: calc(100% - 28px);
        display: flex;
        flex-direction: column;
        background: rgba(30, 30, 30, 0.9);
        color: #f5f5f5;
        border-radius: 8px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35);
        z-index: 9999;
        overflow: hidden;
      }
      .spatchat-legend-dock-header,
      .spatchat-layer-control-handle {
        padding: 8px 10px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
        cursor: move;
        user-select: none;
        background: rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }
      .spatchat-layer-control-tools {
        display: flex;
        padding: 8px 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.04);
      }
      .spatchat-layer-control-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        appearance: none;
        border: 1px solid rgba(255, 255, 255, 0.18);
        background: rgba(24, 24, 24, 0.92) !important;
        color: #ffffff !important;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: 600;
        line-height: 1.2;
        cursor: pointer;
      }
      .spatchat-layer-control-btn:hover {
        background: rgba(48, 48, 48, 0.98) !important;
      }
      .spatchat-legend-dock-body {
        overflow: auto;
        padding: 8px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        flex: 1 1 auto;
        min-height: 0;
      }
      .spatchat-legend-dock .spatchat-map-legend {
        position: static !important;
        left: auto !important;
        right: auto !important;
        top: auto !important;
        bottom: auto !important;
        margin: 0 !important;
        min-width: 0 !important;
        width: auto !important;
        box-shadow: none !important;
        pointer-events: none !important;
      }
      .leaflet-control-layers,
      .leaflet-control-layers.spatchat-draggable-control {
        background: rgba(30, 30, 30, 0.9) !important;
        color: #f5f5f5 !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35) !important;
      }
      .leaflet-control-layers-toggle {
        background-color: transparent !important;
        border-radius: 8px !important;
        filter: invert(1) brightness(1.25);
      }
      .leaflet-control-layers-expanded {
        padding: 0 !important;
        background: transparent !important;
      }
      .leaflet-control-layers-list,
      .leaflet-control-layers form,
      .leaflet-control-layers .leaflet-control-layers-base,
      .leaflet-control-layers .leaflet-control-layers-overlays,
      .leaflet-control-layers .leaflet-control-layers-separator {
        background: transparent !important;
        color: inherit !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
      }
      .leaflet-control-layers label {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 7px 10px;
        margin: 0;
        color: #f5f5f5 !important;
        font-size: 12px;
      }
      .leaflet-control-layers label:hover {
        background: rgba(255, 255, 255, 0.05);
      }
      .leaflet-control-layers.spatchat-draggable-control {
        z-index: 9998 !important;
        width: 280px;
        min-width: 220px;
        max-width: calc(100% - 28px);
        min-height: 120px;
        max-height: calc(100% - 28px);
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }
      .leaflet-control-layers.spatchat-draggable-control .leaflet-control-layers-list {
        flex: 1 1 auto;
        min-height: 0;
        overflow: auto;
      }
      .spatchat-dock-resize {
        position: absolute;
        right: 0;
        bottom: 0;
        width: 18px;
        height: 18px;
        cursor: nwse-resize;
        z-index: 10000;
      }
      .spatchat-dock-resize::before {
        content: "";
        position: absolute;
        right: 4px;
        bottom: 4px;
        width: 10px;
        height: 10px;
        border-right: 2px solid rgba(255, 255, 255, 0.65);
        border-bottom: 2px solid rgba(255, 255, 255, 0.65);
        border-bottom-right-radius: 2px;
      }
      .spatchat-dock-resize:hover::before {
        border-right-color: #8ec5ff;
        border-bottom-color: #8ec5ff;
      }
    </style>
    <script>
    (function() {
      function getLeafletMap() {
        if (!window.L) return null;
        for (var key in window) {
          try {
            if (window[key] instanceof L.Map) return window[key];
          } catch (err) {}
        }
        return null;
      }

      function clampPosition(container, mapContainer) {
        if (!container || !mapContainer) return;
        var left = parseFloat(container.style.left || "0");
        var top = parseFloat(container.style.top || "0");
        if (!isFinite(left)) left = 0;
        if (!isFinite(top)) top = 0;
        var maxLeft = Math.max(0, mapContainer.clientWidth - container.offsetWidth);
        var maxTop = Math.max(0, mapContainer.clientHeight - container.offsetHeight);
        container.style.left = Math.max(0, Math.min(left, maxLeft)) + "px";
        container.style.top = Math.max(0, Math.min(top, maxTop)) + "px";
      }

      function enableDrag(container, handle, mapContainer) {
        if (!container || !handle || !mapContainer || handle.dataset.spatchatDragBound === "1") return;
        handle.dataset.spatchatDragBound = "1";
        handle.addEventListener("mousedown", function(event) {
          if (event.button !== 0) return;
          event.preventDefault();
          event.stopPropagation();
          var mapRect = mapContainer.getBoundingClientRect();
          var rect = container.getBoundingClientRect();
          if (container.parentElement !== mapContainer) mapContainer.appendChild(container);
          container.style.position = "absolute";
          container.style.left = (rect.left - mapRect.left) + "px";
          container.style.top = (rect.top - mapRect.top) + "px";
          container.style.right = "auto";
          container.style.bottom = "auto";
          var startX = event.clientX;
          var startY = event.clientY;
          var startLeft = rect.left - mapRect.left;
          var startTop = rect.top - mapRect.top;
          function onMove(moveEvent) {
            var nextLeft = startLeft + (moveEvent.clientX - startX);
            var nextTop = startTop + (moveEvent.clientY - startY);
            var maxLeft = Math.max(0, mapContainer.clientWidth - container.offsetWidth);
            var maxTop = Math.max(0, mapContainer.clientHeight - container.offsetHeight);
            container.style.left = Math.max(0, Math.min(nextLeft, maxLeft)) + "px";
            container.style.top = Math.max(0, Math.min(nextTop, maxTop)) + "px";
          }
          function onUp() {
            document.removeEventListener("mousemove", onMove);
            document.removeEventListener("mouseup", onUp);
          }
          document.addEventListener("mousemove", onMove);
          document.addEventListener("mouseup", onUp);
        });
      }

      function enableResize(container, handle, mapContainer, options) {
        if (!container || !handle || !mapContainer || handle.dataset.spatchatResizeBound === "1") return;
        handle.dataset.spatchatResizeBound = "1";
        var minWidth = (options && options.minWidth) || 220;
        var minHeight = (options && options.minHeight) || 120;
        handle.addEventListener("mousedown", function(event) {
          if (event.button !== 0) return;
          event.preventDefault();
          event.stopPropagation();
          var mapRect = mapContainer.getBoundingClientRect();
          var rect = container.getBoundingClientRect();
          if (container.parentElement !== mapContainer) mapContainer.appendChild(container);
          container.style.position = "absolute";
          container.style.left = (rect.left - mapRect.left) + "px";
          container.style.top = (rect.top - mapRect.top) + "px";
          container.style.right = "auto";
          container.style.bottom = "auto";
          var startX = event.clientX;
          var startY = event.clientY;
          var startWidth = rect.width;
          var startHeight = rect.height;
          var startLeft = rect.left - mapRect.left;
          var startTop = rect.top - mapRect.top;
          function onMove(moveEvent) {
            var maxWidth = Math.max(minWidth, mapContainer.clientWidth - startLeft);
            var maxHeight = Math.max(minHeight, mapContainer.clientHeight - startTop);
            var nextWidth = startWidth + (moveEvent.clientX - startX);
            var nextHeight = startHeight + (moveEvent.clientY - startY);
            nextWidth = Math.max(minWidth, Math.min(nextWidth, maxWidth));
            nextHeight = Math.max(minHeight, Math.min(nextHeight, maxHeight));
            container.style.width = Math.round(nextWidth) + "px";
            container.style.height = Math.round(nextHeight) + "px";
            container.style.maxWidth = "none";
            container.style.maxHeight = "none";
          }
          function onUp() {
            document.removeEventListener("mousemove", onMove);
            document.removeEventListener("mouseup", onUp);
            clampPosition(container, mapContainer);
          }
          document.addEventListener("mousemove", onMove);
          document.addEventListener("mouseup", onUp);
        });
      }

      function initLegendDock(map) {
        var mapContainer = map && map.getContainer ? map.getContainer() : null;
        if (!mapContainer) return false;
        var legends = Array.from(document.querySelectorAll(".spatchat-map-legend"));
        if (!legends.length) return false;
        if (mapContainer.querySelector(".spatchat-legend-dock")) return true;
        var dock = document.createElement("div");
        dock.className = "spatchat-legend-dock";
        var header = document.createElement("div");
        header.className = "spatchat-legend-dock-header";
        header.textContent = "Map Legends";
        var body = document.createElement("div");
        body.className = "spatchat-legend-dock-body";
        var resize = document.createElement("div");
        resize.className = "spatchat-dock-resize";
        resize.title = "Resize legend dock";
        legends.forEach(function(legend) { body.appendChild(legend); });
        dock.appendChild(header);
        dock.appendChild(body);
        dock.appendChild(resize);
        mapContainer.appendChild(dock);
        enableDrag(dock, header, mapContainer);
        enableResize(dock, resize, mapContainer, { minWidth: 220, minHeight: 120 });
        return true;
      }

      function initLayerControlDrag(map) {
        var mapContainer = map && map.getContainer ? map.getContainer() : null;
        if (!mapContainer) return false;
        var layerControl = document.querySelector(".leaflet-control-layers");
        if (!layerControl) return false;
        layerControl.classList.add("spatchat-draggable-control");
        if (!layerControl.style.width) layerControl.style.width = "280px";
        var handle = layerControl.querySelector(".spatchat-layer-control-handle");
        if (!handle) {
          handle = document.createElement("div");
          handle.className = "spatchat-layer-control-handle";
          handle.textContent = "Layers";
          layerControl.insertBefore(handle, layerControl.firstChild);
        }
        var overlaysList = layerControl.querySelector(".leaflet-control-layers-overlays");
        var overlayInputs = Array.from(layerControl.querySelectorAll(".leaflet-control-layers-overlays input[type='checkbox']"));
        var tools = layerControl.querySelector(".spatchat-layer-control-tools");
        if (!overlayInputs.length) {
          if (tools) tools.remove();
        } else if (!tools) {
          tools = document.createElement("div");
          tools.className = "spatchat-layer-control-tools";
          var toggleBtn = document.createElement("button");
          toggleBtn.type = "button";
          toggleBtn.className = "spatchat-layer-control-btn";
          toggleBtn.textContent = "All layers on/off";
          toggleBtn.addEventListener("mousedown", function(event) { event.stopPropagation(); });
          toggleBtn.addEventListener("click", function(event) {
            event.preventDefault();
            event.stopPropagation();
            var inputs = Array.from(layerControl.querySelectorAll(".leaflet-control-layers-overlays input[type='checkbox']"));
            var anyUnchecked = inputs.some(function(input) { return !input.checked; });
            inputs.forEach(function(input) {
              if (!!input.checked !== anyUnchecked) input.click();
            });
          });
          tools.appendChild(toggleBtn);
          if (overlaysList) overlaysList.insertAdjacentElement("beforebegin", tools);
          else handle.insertAdjacentElement("afterend", tools);
        }
        var resize = layerControl.querySelector(".spatchat-dock-resize");
        if (!resize) {
          resize = document.createElement("div");
          resize.className = "spatchat-dock-resize";
          resize.title = "Resize layers dock";
          layerControl.appendChild(resize);
        }
        enableDrag(layerControl, handle, mapContainer);
        enableResize(layerControl, resize, mapContainer, { minWidth: 220, minHeight: 120 });
        return true;
      }

      function initFloatingUi() {
        var map = getLeafletMap();
        if (!map) return false;
        var legendsOk = initLegendDock(map);
        var layersOk = initLayerControlDrag(map);
        return legendsOk || layersOk;
      }

      initFloatingUi();
      window.addEventListener("load", initFloatingUi);
      var attempts = 0;
      var timer = setInterval(function() {
        if (initFloatingUi() || attempts >= 20) clearInterval(timer);
        attempts += 1;
      }, 500);
    })();
    </script>
    {% endmacro %}
    """
    patch = MacroElement()
    patch._template = Template(patch_html)
    root.add_child(patch)


def _add_layer_control_sort_patch(m: folium.Map, pane_by_label: dict[str, str] | None = None) -> None:
    root = m.get_root()
    if getattr(root, "_spatchat_layer_sort_patch_added", False):
        return
    setattr(root, "_spatchat_layer_sort_patch_added", True)
    patch_html = """
    {% macro html(this, kwargs) %}
    <style>
      .spatchat-layer-order-handle {
        display: inline-block;
        margin-right: 6px;
        color: #666;
        cursor: grab;
        user-select: none;
        font-size: 11px;
        line-height: 1;
      }
      .spatchat-layer-order-handle:active {
        cursor: grabbing;
      }
      .leaflet-control-layers-overlays label.spatchat-layer-drop-target {
        outline: 1px dashed rgba(0, 123, 255, 0.75);
        outline-offset: 2px;
      }
    </style>
    <script>
    (function() {
      var draggedOverlayLabel = null;
      var suppressOverlaySync = false;

      function getLeafletMapForControl(layerControl) {
        if (!window.L || !layerControl) return null;
        for (var key in window) {
          try {
            var candidate = window[key];
            if (!(candidate instanceof L.Map)) continue;
            var container = candidate.getContainer ? candidate.getContainer() : null;
            if (container && container.contains(layerControl)) return candidate;
          } catch (err) {}
        }
        for (var key2 in window) {
          try {
            if (window[key2] instanceof L.Map) return window[key2];
          } catch (err) {}
        }
        return null;
      }

      function getLayerControlInstanceForDom(layerControl, map) {
        if (!window.L || !layerControl) return null;
        if (map) {
          try {
            var controls = map._controls || [];
            for (var i = 0; i < controls.length; i++) {
              var c = controls[i];
              if (c instanceof L.Control.Layers && c._container === layerControl) return c;
            }
          } catch (err) {}
        }
        for (var key in window) {
          try {
            if (window[key] instanceof L.Control.Layers) return window[key];
          } catch (err) {}
        }
        return null;
      }

      function extractLayerName(label) {
        if (!label) return '';
        var cloned = label.cloneNode(true);
        Array.from(cloned.querySelectorAll('.spatchat-layer-order-handle')).forEach(function(node) { node.remove(); });
        return (cloned.textContent || '').replace(/\\s+/g, ' ').trim();
      }

      function normalizeLayerName(name) {
        return String(name || '')
          .replace(/\\s+/g, ' ')
          .replace(/\u200b/g, '')
          .trim()
          .toLowerCase();
      }

      function paneNameForLabel(name) {
        var paneByLabel = {{ this.pane_by_label | tojson }};
        var normalizedKey = normalizeLayerName(name);
        if (paneByLabel && Object.prototype.hasOwnProperty.call(paneByLabel, normalizedKey)) {
          return paneByLabel[normalizedKey];
        }
        var normalized = String(name || 'layer')
          .replace(/\\s*\\(n=\\d+\\)\\s*$/i, '')
          .trim();
        var cleaned = normalized
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, '-')
          .replace(/^-+|-+$/g, '');
        return 'pane-' + (cleaned || 'layer');
      }

      function panePriorityForLabel(name) {
        var normalized = String(name || '').trim();
        if (!normalized) return 1;
        return 1;
      }

      function findControlLayerByName(layerControlInstance, name) {
        if (!layerControlInstance || !layerControlInstance._layers || !name) return null;
        var target = normalizeLayerName(name);
        for (var i = 0; i < layerControlInstance._layers.length; i++) {
          var item = layerControlInstance._layers[i];
          if (!item || !item.overlay) continue;
          var itemName = normalizeLayerName(item.name || '');
          if (itemName === target) return item.layer;
        }
        return null;
      }

      function retargetLayerToPane(layer, paneName, map) {
        if (!layer || !paneName || !map) return;
        if (layer.eachLayer && typeof layer.eachLayer === 'function') {
          layer.eachLayer(function(child) { retargetLayerToPane(child, paneName, map); });
        }
        if (layer.options) {
          layer.options.pane = paneName;
        }

        if (window.L && layer instanceof L.Path) {
          try {
            layer.options = layer.options || {};
            layer.options.pane = paneName;
            layer.options.renderer = L.svg({ pane: paneName });
            var wasOnMap = map.hasLayer(layer);
            if (wasOnMap) {
              map.removeLayer(layer);
              layer.addTo(map);
            } else if (layer.redraw) {
              layer.redraw();
            }
          } catch (err) {}
          return;
        }

        if (window.L && layer instanceof L.Marker) {
          try {
            var markerOnMap = map.hasLayer(layer);
            if (markerOnMap) {
              map.removeLayer(layer);
              layer.addTo(map);
            }
          } catch (err) {}
          return;
        }

        if (window.L && layer instanceof L.ImageOverlay) {
          try {
            var imageOnMap = map.hasLayer(layer);
            if (imageOnMap) {
              map.removeLayer(layer);
              layer.addTo(map);
            }
          } catch (err) {}
          return;
        }

        try {
          var onMap = map.hasLayer(layer);
          if (onMap) {
            map.removeLayer(layer);
            layer.addTo(map);
          } else if (layer.redraw) {
            layer.redraw();
          }
        } catch (err) {}
      }

      function applyOverlayOrder(layerControl) {
        if (!layerControl) return;
        var map = getLeafletMapForControl(layerControl);
        var layerControlInstance = getLayerControlInstanceForDom(layerControl, map);
        if (layerControlInstance && layerControlInstance._map) {
          map = layerControlInstance._map;
        }
        if (!map || !layerControlInstance) return;
        var labels = Array.from(layerControl.querySelectorAll('.leaflet-control-layers-overlays label'));
        var allEntries = labels.map(function(label) {
          var input = label.querySelector('input[type="checkbox"]');
          return { label: label, input: input, name: extractLayerName(label) };
        });

        allEntries.forEach(function(entry) {
          var layer = findControlLayerByName(layerControlInstance, entry.name);
          if (!layer) return;
          retargetLayerToPane(layer, paneNameForLabel(entry.name), map);
        });

        var paneBase = 2000;
        var paneEntries = [];
        allEntries.forEach(function(entry, index) {
          var paneName = paneNameForLabel(entry.name);
          var paneEl = map && map.getPane ? map.getPane(paneName) : null;
          if (!paneEl && map.createPane) {
            try { map.createPane(paneName); } catch (err) {}
            paneEl = map.getPane ? map.getPane(paneName) : null;
          }
          if (!paneEl) return;
        var zIndex = paneBase + (panePriorityForLabel(entry.name) * 100) + (allEntries.length - index);
          paneEl.style.zIndex = String(zIndex);
          paneEntries.push({ paneEl: paneEl, name: entry.name, zIndex: zIndex });
        });

        if (map && map.getPanes && map.getPanes().mapPane) {
          var mapPane = map.getPanes().mapPane;
          paneEntries
            .slice()
            .sort(function(a, b) { return a.zIndex - b.zIndex; })
            .forEach(function(item) {
              try { mapPane.appendChild(item.paneEl); } catch (err) {}
            });
        }

        var checkedEntries = allEntries.filter(function(entry) {
          return !!(entry.input && entry.input.checked);
        });
        if (!checkedEntries.length) return;

        suppressOverlaySync = true;
        checkedEntries.forEach(function(entry) {
          try { entry.input.click(); } catch (err) {}
        });
        for (var i = checkedEntries.length - 1; i >= 0; i--) {
          try { checkedEntries[i].input.click(); } catch (err) {}
        }
        setTimeout(function() { suppressOverlaySync = false; }, 0);
      }

      function initOverlayOrdering(layerControl) {
        var overlaysList = layerControl ? layerControl.querySelector('.leaflet-control-layers-overlays') : null;
        if (!overlaysList) return;

        Array.from(overlaysList.querySelectorAll('label')).forEach(function(label) {
          if (!label.querySelector('.spatchat-layer-order-handle')) {
            var handleSpan = document.createElement('span');
            handleSpan.className = 'spatchat-layer-order-handle';
            handleSpan.textContent = '↕';
            handleSpan.draggable = true;
            handleSpan.addEventListener('dragstart', function(event) {
              draggedOverlayLabel = label;
              if (event.dataTransfer) {
                event.dataTransfer.effectAllowed = 'move';
                try { event.dataTransfer.setData('text/plain', 'spatchat-layer'); } catch (e) {}
              }
            });
            handleSpan.addEventListener('dragend', function() {
              draggedOverlayLabel = null;
              Array.from(overlaysList.querySelectorAll('label')).forEach(function(item) {
                item.classList.remove('spatchat-layer-drop-target');
              });
            });
            label.insertBefore(handleSpan, label.firstChild);
          }

          if (label.dataset.spatchatDropBound === '1') return;
          label.dataset.spatchatDropBound = '1';
          label.addEventListener('dragover', function(event) {
            if (!draggedOverlayLabel || draggedOverlayLabel === label) return;
            event.preventDefault();
            label.classList.add('spatchat-layer-drop-target');
          });
          label.addEventListener('dragleave', function() {
            label.classList.remove('spatchat-layer-drop-target');
          });
          label.addEventListener('drop', function(event) {
            if (!draggedOverlayLabel || draggedOverlayLabel === label) return;
            event.preventDefault();
            label.classList.remove('spatchat-layer-drop-target');
            var rect = label.getBoundingClientRect();
            var insertAfter = event.clientY > rect.top + rect.height / 2;
            if (insertAfter) overlaysList.insertBefore(draggedOverlayLabel, label.nextSibling);
            else overlaysList.insertBefore(draggedOverlayLabel, label);
            applyOverlayOrder(layerControl);
          });
        });

        if (layerControl.dataset.spatchatOrderBound !== '1') {
          layerControl.dataset.spatchatOrderBound = '1';
          layerControl.addEventListener('change', function() {
            if (suppressOverlaySync) return;
            setTimeout(function() { applyOverlayOrder(layerControl); }, 0);
          });
        }

        applyOverlayOrder(layerControl);
      }

      function tryInitLayerControl() {
        var layerControl = document.querySelector('.leaflet-control-layers');
        if (!layerControl) return false;
        initOverlayOrdering(layerControl);
        return true;
      }

      tryInitLayerControl();
      window.addEventListener('load', tryInitLayerControl);
      var attempts = 0;
      var timer = setInterval(function() {
        tryInitLayerControl();
        attempts += 1;
        if (attempts >= 20) clearInterval(timer);
      }, 500);
    })();
    </script>
    {% endmacro %}
    """
    patch = MacroElement()
    patch._template = Template(patch_html)
    patch.pane_by_label = {re.sub(r"\s+", " ", str(k)).strip().lower(): v for k, v in (pane_by_label or {}).items()}
    root.add_child(patch)


def _add_circle_marker_feature_group_script(
    m: folium.Map,
    *,
    layer_var_name: str,
    pane: str,
    features: list[dict],
) -> None:
    if not features:
        return
    root = m.get_root()
    patch_html = """
    {% macro html(this, kwargs) %}
    <script>
    (function() {
      var features = {{ this.features | tojson }};
      var paneName = {{ this.pane | tojson }};
      var layerVarName = {{ this.layer_var_name | tojson }};

      function getLeafletMap() {
        if (!window.L) return null;
        for (var key in window) {
          try {
            if (window[key] instanceof L.Map) return window[key];
          } catch (err) {}
        }
        return null;
      }

      function bindCircleMarkers() {
        if (!window.L) return false;
        var map = getLeafletMap();
        var layer = window[layerVarName];
        if (!map || !layer) return false;
        if (layer._spatchatCircleMarkersBound) return true;
        try {
          var renderer = L.svg({ pane: paneName });
          features.forEach(function(feature) {
            try {
              var circle = L.circleMarker(feature.location, {
                radius: feature.radius,
                color: feature.color,
                weight: feature.weight,
                fill: true,
                fillColor: feature.fillColor,
                fillOpacity: feature.fillOpacity,
                pane: paneName,
                renderer: renderer,
                bubblingMouseEvents: false,
              });
              circle.addTo(layer);
            } catch (err) {}
          });
          layer._spatchatCircleMarkersBound = true;
          return true;
        } catch (err) {
          return false;
        }
      }

      bindCircleMarkers();
      window.addEventListener('load', bindCircleMarkers);
      var attempts = 0;
      var timer = setInterval(function() {
        if (bindCircleMarkers() || attempts >= 20) clearInterval(timer);
        attempts += 1;
      }, 500);
    })();
    </script>
    {% endmacro %}
    """
    patch = MacroElement()
    patch._template = Template(patch_html)
    patch.features = features
    patch.pane = pane
    patch.layer_var_name = layer_var_name
    root.add_child(patch)


def render_sdm_map(
    *,
    session_root: str | None,
    detect_coords_fn,
    colorbar_base64: str,
    uploaded_absence_path: str = "inputs/absence_points_uploaded.csv",
):
    if not session_root:
        m = render_empty_map()
        return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" style="width:100%; height:100%; min-height:78vh; border:none;"></iframe>'

    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True, tiles=None)
    add_default_basemaps(m)
    pane_by_label = {}
    presence_pane = _ensure_map_pane(m, "Presences", z_index=5000)
    pane_by_label["Presences"] = presence_pane
    background_pane = _ensure_map_pane(m, "Available/Absences", z_index=4900)
    pane_by_label["Available/Absences"] = background_pane

    ppath = os.path.join(session_root, "inputs", "presence_points.csv")
    if os.path.exists(ppath):
        df = pd.read_csv(ppath)
        if {"latitude", "longitude"}.issubset(df.columns):
            lat_col, lon_col = "latitude", "longitude"
        else:
            lat_col, lon_col = detect_coords_fn(df)
        if lat_col and lon_col:
            try:
                lats, lons = _coerce_wgs84_arrays(df, lat_col, lon_col)
            except Exception:
                lats, lons = np.array([]), np.array([])
            if lats.size > 0:
                fg = folium.FeatureGroup(name="Presences")
                fg.add_to(m)
                _add_circle_marker_feature_group_script(
                    m,
                    layer_var_name=fg.get_name(),
                    pane=presence_pane,
                    features=[
                        {
                            "location": [float(lat), float(lon)],
                            "radius": 5,
                            "color": "blue",
                            "weight": 2,
                            "fillColor": "blue",
                            "fillOpacity": 0.85,
                        }
                        for lat, lon in zip(lats, lons)
                    ],
                )
                try:
                    fit_map_to_bounds(m, pd.DataFrame({"latitude": lats, "longitude": lons}))
                except Exception:
                    pass

    abs_fp = os.path.join(session_root, uploaded_absence_path)
    if not os.path.exists(abs_fp):
        abs_fp = os.path.join(session_root, "outputs", "absence_points_coordinates.csv")
    if os.path.exists(abs_fp):
        df_abs = pd.read_csv(abs_fp)
        if {"latitude", "longitude"}.issubset(df_abs.columns):
            try:
                alats, alons = _coerce_wgs84_arrays(df_abs, "latitude", "longitude")
            except Exception:
                alats, alons = np.array([]), np.array([])
            if alats.size > 0:
                fg_abs = folium.FeatureGroup(name="Available/Absences")
                fg_abs.add_to(m)
                _add_circle_marker_feature_group_script(
                    m,
                    layer_var_name=fg_abs.get_name(),
                    pane=background_pane,
                    features=[
                        {
                            "location": [float(lat), float(lon)],
                            "radius": 3,
                            "color": "red",
                            "weight": 1,
                            "fillColor": "red",
                            "fillOpacity": 0.65,
                        }
                        for lat, lon in zip(alats, alons)
                    ],
                )

    rasdir = os.path.join(session_root, "predictor_rasters", "wgs84")
    predictor_cmap = "viridis"
    suitability_cmap = "magma"
    predictor_fns = []
    if os.path.isdir(rasdir):
        predictor_fns = [fn for fn in sorted(os.listdir(rasdir)) if fn.endswith(".tif")]
        n_predictor_layers = len(predictor_fns)
        for idx, fn in enumerate(predictor_fns):
            if not fn.endswith(".tif"):
                continue
            fullfp = os.path.join(rasdir, fn)
            try:
                arr, bnd = _read_display_raster(fullfp)
                finite = np.isfinite(arr)
                if not np.any(finite):
                    continue
                actual_min = float(np.nanmin(arr))
                actual_max = float(np.nanmax(arr))
                pmin = 2.0
                pmax = 98.0
                vmin = float(np.nanpercentile(arr, pmin))
                vmax = float(np.nanpercentile(arr, pmax))
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    continue
                if abs(vmax - vmin) <= 1e-12:
                    normed = np.zeros_like(arr, dtype=np.float32)
                else:
                    normed = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
                rgba = colormaps[predictor_cmap](normed)
                rgba[..., 3] = np.where(finite, 0.78, 0.0)
                layer_name = os.path.splitext(fn)[0]
                # Keep layer control order aligned with map draw order:
                # top item in control gets the highest z-index on map.
                pane = _ensure_map_pane(m, layer_name, z_index=900 - idx)
                pane_by_label[layer_name] = pane
                folium.raster_layers.ImageOverlay(
                    image=rgba,
                    bounds=[[bnd.bottom, bnd.left], [bnd.top, bnd.right]],
                    opacity=1.0,
                    name=layer_name,
                    pane=pane,
                ).add_to(m)
                _add_raster_legend(
                    m,
                    name=layer_name,
                    cmap=predictor_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    pmin=pmin,
                    pmax=pmax,
                    actual_min=actual_min,
                    actual_max=actual_max,
                )
            except Exception as exc:
                print(f"[map] Skipped raster {fn}: {exc}", file=sys.stderr)

    sf = os.path.join(session_root, "outputs", "suitability_map_wgs84.tif")
    if os.path.exists(sf):
        try:
            arr, bnd = _read_display_raster(sf)
            finite = np.isfinite(arr)
            if np.any(finite):
                actual_min = float(np.nanmin(arr))
                actual_max = float(np.nanmax(arr))
                pmin = 1.0
                pmax = 99.0
                vmin = float(np.nanpercentile(arr, pmin))
                vmax = float(np.nanpercentile(arr, pmax))
                if abs(vmax - vmin) <= 1e-12:
                    normed = np.zeros_like(arr, dtype=np.float32)
                else:
                    normed = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
                rgba = colormaps[suitability_cmap](normed)
                rgba[..., 3] = np.where(finite, 0.92, 0.0)
                suitability_name = "Suitability (Prediction)"
                pane = _ensure_map_pane(m, suitability_name, z_index=900 - len(predictor_fns))
                pane_by_label[suitability_name] = pane
                folium.raster_layers.ImageOverlay(
                    image=rgba,
                    bounds=[[bnd.bottom, bnd.left], [bnd.top, bnd.right]],
                    opacity=1.0,
                    name=suitability_name,
                    pane=pane,
                ).add_to(m)
                _add_raster_legend(
                    m,
                    name=suitability_name,
                    cmap=suitability_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    pmin=pmin,
                    pmax=pmax,
                    actual_min=actual_min,
                    actual_max=actual_max,
                )
        except Exception as exc:
            print(f"[map] Skipped suitability raster: {exc}", file=sys.stderr)

    folium.LayerControl(collapsed=False).add_to(m)
    _add_floating_ui_patch(m)
    _add_layer_control_sort_patch(m, pane_by_label=pane_by_label)
    return f'<iframe srcdoc="{html_lib.escape(m.get_root().render())}" style="width:100%; height:100%; min-height:78vh; border:none;"></iframe>'
