#!/usr/bin/env python3
import os
import json
import time
import math
import shutil

import ee
import geemap
import rasterio
import numpy as np
import pandas as pd

from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# ——— Earth Engine auth ————————————————————————————————
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)
print("✅ EE initialized with Service Account")

# ——— Wait for the presence CSV ——————————————————————————
csv_path = "inputs/presence_points.csv"
for i in range(5):
    if os.path.exists(csv_path):
        break
    print(f"⏳ Waiting for {csv_path}… ({i+1}/5)")
    time.sleep(1)
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found")

# ——— Load points & build study‐area bbox —————————————————————
df = pd.read_csv(csv_path)
if not {'latitude','longitude'}.issubset(df.columns):
    raise ValueError("CSV must contain 'latitude' and 'longitude' columns")
min_lat, max_lat = df.latitude.min(), df.latitude.max()
min_lon, max_lon = df.longitude.min(), df.longitude.max()
buf = 0.25
region = ee.Geometry.BBox(
    min_lon - buf, min_lat - buf,
    max_lon + buf, max_lat + buf
)
print(f"📍 {len(df)} points; bbox = "
      f"({min_lat-buf},{min_lon-buf}) → ({max_lat+buf},{max_lon+buf})")

# ——— Grid & reprojection specs ——————————————————————————
# 30 m target resolution
SCALE = 30
# build a WGS84 grid that ~30 m on the ground
crs = CRS.from_epsg(4326)
dx  = (max_lon+buf) - (min_lon-buf)
dy  = (max_lat+buf) - (min_lat-buf)
# convert degrees→meters roughly (at mid‐lat)
mid_lat = (min_lat + max_lat) / 2
m_per_deg_lon = 111320 * math.cos(math.radians(mid_lat))
m_per_deg_lat = 110540
nx = math.ceil((dx * m_per_deg_lon) / SCALE)
ny = math.ceil((dy * m_per_deg_lat) / SCALE)
transform = from_bounds(
    min_lon-buf, min_lat-buf,
    max_lon+buf, max_lat+buf,
    nx, ny
)
print(f"🗺  Grid: {nx}×{ny} @ {SCALE} m → EPSG:4326")

# ——— Layer sources & user picks ——————————————————————————
layer_sources = {
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "slope":     ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("slope"),
    "aspect":    ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("aspect"),
    "ndvi":      ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI").mean(),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1").first(),
}
for i in range(1, 20):
    layer_sources[f"bio{i}"] = ee.Image("WORLDCLIM/V1/BIO")\
                                  .select(f"bio{str(i).zfill(2)}")

selected_layers  = os.environ.get('SELECTED_LAYERS', '').split(',')
selected_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES', '').split(',')

# ——— Prepare output directories ——————————————————————————
raw_dir     = "predictor_rasters/raw"
aligned_dir = "predictor_rasters/wgs84"
shutil.rmtree(raw_dir,    ignore_errors=True)
shutil.rmtree(aligned_dir,ignore_errors=True)
os.makedirs(raw_dir,    exist_ok=True)
os.makedirs(aligned_dir,exist_ok=True)

# ——— Helper: export via geemap with explicit CRS=3857 —————————
def _export(img, out_path, reg):
    geemap.ee_export_image(
        img.clip(reg),
        filename=out_path,
        scale=SCALE,
        crs="EPSG:3857",          # <<– ensure meters
        region=reg,
        file_per_band=False,
        timeout=600
    )

# ——— Core fetch & align routine ——————————————————————————
def fetch_and_align(img, name):
    raw_tif = os.path.join(raw_dir, f"{name}.tif")
    print(f"\n📥 Single‐tile → {name}")
    _export(img, raw_tif, region)

    # if geemap failed to write raw_tif (too big), do two‐tile split + mosaic
    if not os.path.exists(raw_tif):
        print("⚠️ single‐tile export failed → splitting bbox…")
        mid  = (min_lon-buf + max_lon+buf) / 2
        parts = [
          ee.Geometry.BBox(min_lon-buf, min_lat-buf, mid,       max_lat+buf),
          ee.Geometry.BBox(mid,       min_lat-buf, max_lon+buf, max_lat+buf),
        ]
        tiles = []
        for i, subreg in enumerate(parts, 1):
            tp = os.path.join(raw_dir, f"{name}_t{i}.tif")
            print(f"  ▸ tile {i}")
            _export(img, tp, subreg)
            tiles.append(rasterio.open(tp))

        m, out_tf = merge(tiles)
        meta = tiles[0].meta.copy()
        meta.update({
            "height": m.shape[1],
            "width":  m.shape[2],
            "transform": out_tf
        })
        with rasterio.open(raw_tif, 'w', **meta) as dst:
            dst.write(m)
        for t in tiles: t.close()
        print("🔀 Mosaicked →", raw_tif)

    # reproject that raw back into our WGS84 grid
    aligned = os.path.join(aligned_dir, f"{name}.tif")
    with rasterio.open(raw_tif) as src:
        src_arr = src.read(1)
        dst_arr = np.empty((ny, nx), dtype=src_arr.dtype)

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest
        )

        prof = src.profile.copy()
        prof.update({
            "height": ny,
            "width":  nx,
            "transform": transform,
            "crs": crs,
            "driver": "GTiff",
            "count": 1
        })
        with rasterio.open(aligned, 'w', **prof) as dst:
            dst.write(dst_arr, 1)

    print(f"🌐 Aligned → {aligned}")

# ——— Fetch continuous layers ———————————————————————————
for nm in selected_layers:
    if nm == "landcover":
        continue
    if nm not in layer_sources:
        print("⚠️ skipping unknown layer:", nm)
        continue
    fetch_and_align(layer_sources[nm], nm)

# ——— One‐hot encode MODIS landcover ——————————————————————
if "landcover" in selected_layers and selected_classes:
    lc = layer_sources["landcover"]
    code_name = {
        0:"water",1:"evergreen_needleleaf_forest",2:"evergreen_broadleaf_forest",
        3:"deciduous_needleleaf_forest",4:"deciduous_broadleaf_forest",
        5:"mixed_forest",6:"closed_shrublands",7:"open_shrublands",
        8:"woody_savannas",9:"savannas",10:"grasslands",11:"permanent_wetlands",
        12:"croplands",13:"urban_and_built_up",14:"cropland_natural_vegetation_mosaic",
        15:"snow_and_ice",16:"barren_or_sparsely_vegetated"
    }
    print("\n🌱 One‐hot landcover…")
    for code in selected_classes:
        if not code.isdigit(): 
            continue
        ci = int(code)
        label = code_name[ci]
        img   = lc.eq(ci)
        fetch_and_align(img, f"{ci}_{label}")
