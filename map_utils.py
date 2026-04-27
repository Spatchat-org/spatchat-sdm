import folium
import numpy as np


def add_default_basemaps(m: folium.Map) -> folium.Map:
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", attr="CartoDB").add_to(m)
    folium.TileLayer(
        "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="Topographic",
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)
    return m


def render_empty_map():
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True, tiles=None)
    add_default_basemaps(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def fit_map_to_bounds(m, df):
    min_lat, max_lat = df["latitude"].min(), df["latitude"].max()
    min_lon, max_lon = df["longitude"].min(), df["longitude"].max()
    if np.isfinite([min_lat, max_lat, min_lon, max_lon]).all():
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    return m
