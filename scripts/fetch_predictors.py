#!/usr/bin/env python3
import os
import json
import time
import math
import shutil
import ee
import geemap
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import numpy as np
import pandas as pd

# ——— Earth Engine auth (Service Account) —————————————————————
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)

print("✅ EE initialized")

# ——— Wait for presence CSV —————————————————————————————
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
    raise ValueError("CSV needs columns 'latitude' & 'longitude'")
min_lat, max_lat = df.latitude.min(), df.latitude.max()
min_lon, max_lon = df.longitude.min(), df.longitude.max()
buffer = 0.25
region = ee.Geometry.BBox(
    min_lon-buffer, min_lat-buffer,
    max_lon+buffer, max_lat+buffer
)
print("📍 Points:", len(df),
      "|  bbox:", (min_lat-buffer, min_lon-buffer),
                "→", (max_lat+buffer, max_lon+buffer))

# ——— Grid & reprojection specs —————————————————————————
SCALE = 30  # meters
crs = CRS.from_epsg(4326)
# build our target transform from lat/lon bounds @30 m resolution
# we compute approximate degrees per 30 m via the bbox width in degrees
dx = max_lon+buffer - (min_lon-buffer)
dy = max_lat+buffer - (min_lat-buffer)
# number of pixels across/vert
nx = math.ceil((dx * 111320) / SCALE)  # rough: 1° ≈ 111.32 km
ny = math.ceil((dy * 110540) / SCALE)  # at mid‐lat, 1° ≈ 110.54 km
transform = from_bounds(
    min_lon-buffer, min_lat-buffer,
    max_lon+buffer, max_lat+buffer,
    nx, ny
)
print(f"🗺  Grid: {nx}×{ny} @ {SCALE} m → EPSG:4326")

# ——— Layer sources & user selections ——————————————————————
layer_sources = {
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "slope":     ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("slope"),
    "aspect":    ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("aspect"),
    "ndvi":      ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI").mean(),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1").first(),
}
for i in range(1,20):
    layer_sources[f"bio{i}"] = ee.Image("WORLDCLIM/V1/BIO")\
                                    .select(f"bio{str(i).zfill(2)}")

selected_layers = os.environ.get('SELECTED_LAYERS','').split(',')
selected_classes= os.environ.get('SELECTED_LANDCOVER_CLASSES','').split(',')

# ——— Prepare output dirs ———————————————————————————————
raw_dir   = "predictor_rasters/raw"
aligned_dir = "predictor_rasters/wgs84"
shutil.rmtree(raw_dir,    ignore_errors=True)
shutil.rmtree(aligned_dir,ignore_errors=True)
os.makedirs(raw_dir,    exist_ok=True)
os.makedirs(aligned_dir,exist_ok=True)

# ——— Utility: single‐tile export —————————————————————————
def _export_tile(img, name, out_path, region_ee):
    geemap.ee_export_image(
        img.clip(region_ee),
        filename=out_path,
        scale=SCALE,
        region=region_ee,
        file_per_band=False,
        timeout=600
    )

# ——— Export + mosaic + reproject ————————————————————————
def fetch_and_align(img, name):
    raw_single = os.path.join(raw_dir, f"{name}.tif")
    try:
        print(f"📥 Fetching '{name}' at {SCALE} m…")
        _export_tile(img, name, raw_single, region)
    except Exception as e:
        # assume size‐limit; split bbox in two vertical tiles
        print(f"⚠️ Single‐tile export failed ({e}). Splitting into 2 tiles…")
        lon_c = (min_lon-buffer + max_lon+buffer) / 2
        bboxes = [
          ee.Geometry.BBox(min_lon-buffer, min_lat-buffer, lon_c, max_lat+buffer),
          ee.Geometry.BBox(lon_c,             min_lat-buffer, max_lon+buffer, max_lat+buffer),
        ]
        tiles = []
        for i, reg in enumerate(bboxes,1):
            tile_path = os.path.join(raw_dir,f"{name}_tile{i}.tif")
            _export_tile(img, name, tile_path, reg)
            tiles.append(rasterio.open(tile_path))
        # mosaic & write out_single
        m, out_tf = merge(tiles)
        meta = tiles[0].meta.copy()
        meta.update({
          "height": m.shape[1],
          "width":  m.shape[2],
          "transform": out_tf
        })
        with rasterio.open(raw_single,'w',**meta) as dst:
            dst.write(m)
        for t in tiles: t.close()
        print("🔀 Mosaicked 2 tiles →", raw_single)

    # — now reproject → aligned
    aligned = os.path.join(aligned_dir, f"{name}.tif")
    with rasterio.open(raw_single) as src:
        src_data = src.read(1)
        dst    = np.empty((ny,nx),dtype=src_data.dtype)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest
        )
        profile = src.profile.copy()
        profile.update({
            "height": ny,
            "width":  nx,
            "transform": transform,
            "crs": crs,
            "driver": "GTiff",
            "count": 1
        })
        with rasterio.open(aligned,'w',**profile) as out:
            out.write(dst,1)
    print(f"🌐 Aligned → {aligned}")

# ——— Run all requested layers ———————————————————————————
# first the continuous ones:
for name in selected_layers:
    if name=="landcover": continue
    if name not in layer_sources:
        print("⚠️ Skipping unknown:",name)
        continue
    fetch_and_align(layer_sources[name],name)

# ——— One-hot the landcover classes ——————————————————————
if "landcover" in selected_layers and selected_classes:
    land = layer_sources["landcover"]
    # embedded map for naming
    L = {
      0:"water",1:"evergreen_needleleaf_forest",2:"evergreen_broadleaf_forest",
      3:"deciduous_needleleaf_forest",4:"deciduous_broadleaf_forest",
      5:"mixed_forest",6:"closed_shrublands",7:"open_shrublands",
      8:"woody_savannas",9:"savannas",10:"grasslands",11:"permanent_wetlands",
      12:"croplands",13:"urban_and_built_up",14:"cropland_natural_vegetation_mosaic",
      15:"snow_and_ice",16:"barren_or_sparsely_vegetated"
    }
    print("🌱 One-hot encoding landcover…")
    for code in selected_classes:
        if not code.isdigit(): continue
        img = land.eq(int(code))
        fetch_and_align(img, f"{code}_{L[int(code)]}")
