#!/usr/bin/env python3
import os
import json
import time

import ee
import geemap
import rasterio
import numpy as np
import pandas as pd
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# --- Authenticate Earth Engine using Service Account from HF secret ---
service_account = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account['client_email'],
    key_data=json.dumps(service_account)
)
ee.Initialize(credentials)
print("✅ Earth Engine authenticated successfully inside fetch_predictors.py!")

# --- Wait for presence CSV ---
csv_path = "inputs/presence_points.csv"
for i in range(5):
    if os.path.exists(csv_path):
        break
    print(f"⏳ Waiting for presence_points.csv... ({i+1}s)")
    time.sleep(1)
if not os.path.exists(csv_path):
    raise FileNotFoundError("❗ inputs/presence_points.csv not found after 5s wait.")

# --- Load presence points & build EE region ---
df = pd.read_csv(csv_path)
if not {'latitude','longitude'}.issubset(df.columns):
    raise ValueError("❗ CSV must contain 'latitude' and 'longitude' columns.")
min_lat, max_lat = df.latitude.min(), df.latitude.max()
min_lon, max_lon = df.longitude.min(), df.longitude.max()
buffer = 0.25  # degrees
region_ee = ee.Geometry.BBox(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer
)
print(f"📍 Loaded {len(df)} points; study‐area = "
      f"{min_lat-buffer},{min_lon-buffer} → {max_lat+buffer},{max_lon+buffer}")

# --- Build a uniform 30 m WGS84 grid covering that bbox ---
# 30 m ≃ 0.0002695° at the equator; adjust as you wish.
PIXEL_SIZE_DEG = 30 / 111320.0
crs = CRS.from_epsg(4326)
width  = int((max_lon+buffer - (min_lon-buffer))  / PIXEL_SIZE_DEG)
height = int((max_lat+buffer - (min_lat-buffer))  / PIXEL_SIZE_DEG)
transform = from_bounds(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer,
    width, height
)
print(f"🗺  Grid: {width}×{height} @30 m → EPSG:4326")

# --- Where to put your rasters ---
os.makedirs("predictor_rasters/raw",  exist_ok=True)
os.makedirs("predictor_rasters/wgs84", exist_ok=True)

# --- User’s chosen layers (set by app.py) ---
selected_layers  = os.environ.get('SELECTED_LAYERS','').split(',')
selected_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES','').split(',')

# --- EarthEngine source images ---
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

# --- Native export scales (in meters) to keep each tile under 50 MB ---
EXPORT_SCALES = {
    "elevation": 30,
    "slope":     30,
    "aspect":    30,
    "bio1":    30,
    # ... all other BIOs at 30 m, if you like:
    **{f"bio{i}": 30 for i in range(1,20)},
    # but NDVI & MODIS landcover are coarser natively:
    "ndvi":      1000,
    "landcover": 500
}

# --- Hard-coded MODIS LC code → snake_case label map ---
modis_landcover_map = {
    0:"water",1:"evergreen_needleleaf_forest",2:"evergreen_broadleaf_forest",
    3:"deciduous_needleleaf_forest",4:"deciduous_broadleaf_forest",
    5:"mixed_forest",6:"closed_shrublands",7:"open_shrublands",
    8:"woody_savannas",9:"savannas",10:"grasslands",11:"permanent_wetlands",
    12:"croplands",13:"urban_and_built_up",14:"cropland_natural_vegetation_mosaic",
    15:"snow_and_ice",16:"barren_or_sparsely_vegetated"
}

def export_and_reproject(ee_img, name):
    """1) Export a raw tile at its native scale to keep size under 50 MB.
       2) Read it, then reproject+resample into our uniform 30 m grid."""
    scale = EXPORT_SCALES.get(name, 30)
    raw_path = f"predictor_rasters/raw/{name}.tif"
    out_path = f"predictor_rasters/wgs84/{name}.tif"

    # --- 1) export from EE at native scale ---
    print(f"📥 Exporting '{name}' at {scale} m native…")
    geemap.ee_export_image(
        ee_img.clip(region_ee),
        filename=raw_path,
        scale=scale,
        region=region_ee,
        file_per_band=False,
        timeout=600
    )

    # --- 2) reproject / resample to our 30 m grid ---
    with rasterio.open(raw_path) as src:
        src_data = src.read(1)
        dst_data = np.empty((height, width), dtype=src_data.dtype)
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest
        )
        profile = src.profile.copy()
        profile.update({
            "crs":        crs,
            "transform":  transform,
            "height":     height,
            "width":      width,
            "driver":     "GTiff",
            "count":      1
        })
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dst_data, 1)
    print(f"🌐 Reprojected → {out_path}")

# --- Fetch all regular predictors ---
for name in selected_layers:
    if name == "landcover":
        continue
    if name not in layer_sources:
        print(f"⚠️ Skipping unknown layer '{name}'")
        continue
    export_and_reproject(layer_sources[name], name)

# --- One-hot encode any selected MODIS landcover classes ---
if "landcover" in selected_layers and selected_classes:
    lc = layer_sources["landcover"]
    print("🌱 One-hot encoding MODIS landcover…")
    for code in selected_classes:
        if not code.isdigit(): 
            continue
        c = int(code)
        label = modis_landcover_map.get(c, f"class_{c}")
        bin_img = lc.eq(c)
        export_and_reproject(bin_img, f"{c}_{label}")
