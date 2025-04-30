#!/usr/bin/env python
import os
import json
import time

import ee
import geemap
import rasterio
import numpy as np
import pandas as pd
from rasterio.warp import reproject, Resampling

# --- Authenticate Earth Engine using Hugging Face Secret ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)
print("✅ Earth Engine authenticated successfully inside fetch_predictors.py!")

# --- Wait for presence points to appear ---
csv_path = "inputs/presence_points.csv"
for i in range(5):
    if os.path.exists(csv_path):
        break
    print(f"⏳ Waiting for presence_points.csv... ({i+1}s)")
    time.sleep(1)

if not os.path.exists(csv_path):
    raise FileNotFoundError("❗ 'inputs/presence_points.csv' not found after 5s wait.")

# --- Load uploaded presence points ---
df = pd.read_csv(csv_path)
if not {'latitude', 'longitude'}.issubset(df.columns):
    raise ValueError("❗ CSV must contain 'latitude' and 'longitude' columns.")
print(f"📍 Loaded {len(df)} presence points.")

# --- Compute study area bounding box (with buffer) ---
min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
buffer = 0.25
region = ee.Geometry.BBox(min_lon - buffer,
                          min_lat - buffer,
                          max_lon + buffer,
                          max_lat + buffer)

# --- Fetch layer list from environment ---
selected_layers = os.environ.get('SELECTED_LAYERS', '').split(',')
selected_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES', '').split(',')

# --- Prepare output directories ---
os.makedirs("predictor_rasters", exist_ok=True)
os.makedirs("predictor_rasters/wgs84", exist_ok=True)

# --- Define Earth Engine layer sources ---
layer_sources = {
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "slope": ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select('slope'),
    "aspect": ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select('aspect'),
    "ndvi": ee.ImageCollection("MODIS/061/MOD13A2").select('NDVI').mean(),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select('LC_Type1').first()
}
for i in range(1, 20):
    layer_sources[f"bio{i}"] = ee.Image("WORLDCLIM/V1/BIO").select(f"bio{str(i).zfill(2)}")

# --- Hard-coded MODIS landcover code→name map ---
modis_landcover_map = {
    "0": "water",
    "1": "evergreen_needleleaf_forest",
    "2": "evergreen_broadleaf_forest",
    "3": "deciduous_needleleaf_forest",
    "4": "deciduous_broadleaf_forest",
    "5": "mixed_forest",
    "6": "closed_shrublands",
    "7": "open_shrublands",
    "8": "woody_savannas",
    "9": "savannas",
    "10": "grasslands",
    "11": "permanent_wetlands",
    "12": "croplands",
    "13": "urban_and_built_up",
    "14": "cropland_natural_vegetation_mosaic",
    "15": "snow_and_ice",
    "16": "barren_or_sparsely_vegetated"
}

# --- Export & reproject helper ---
def export_and_reproject(ee_image, name, base_profile=None):
    raw_path = f"predictor_rasters/{name}.tif"
    aligned_path = f"predictor_rasters/wgs84/{name}.tif"

    # 1) Fetch raw at 30 m
    geemap.ee_export_image(
        ee_image.clip(region),
        filename=raw_path,
        scale=30,
        region=region,
        file_per_band=False,
        timeout=600
    )
    print(f"✅ Saved raw layer: {raw_path}")

    # 2) Read and, if first layer, capture its exact grid/profile
    with rasterio.open(raw_path) as src:
        data = src.read(1)
        if base_profile is None:
            base_profile = src.profile.copy()
            print(f"🗺 Grid set by {name}.tif → "
                  f"{base_profile['width']}×{base_profile['height']} @ {base_profile['crs']}")

    # 3) Reproject into that grid
    dst = np.empty((base_profile['height'], base_profile['width']),
                   dtype=data.dtype)
    reproject(
        source=data,
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=base_profile['transform'],
        dst_crs=base_profile['crs'],
        resampling=Resampling.nearest
    )

    # 4) Write aligned
    prof = base_profile.copy()
    prof.update(count=1)
    with rasterio.open(aligned_path, 'w', **prof) as dstfile:
        dstfile.write(dst, 1)
    print(f"🌐 Reprojected to: {aligned_path}")

    return base_profile

# --- Fetch & align all regular predictors ---
base_profile = None
for name in selected_layers:
    if name == "landcover":  # handle below
        continue
    if name not in layer_sources:
        print(f"⚠️ Skipping unknown layer: {name}")
        continue
    print(f"📥 Fetching '{name}'...")
    base_profile = export_and_reproject(layer_sources[name], name, base_profile)

# --- One-hot encode & align landcover classes ---
if "landcover" in selected_layers and selected_classes:
    print("🌱 Encoding landcover classes…")
    landcover_img = layer_sources["landcover"]
    for code in selected_classes:
        if not code.isdigit():
            continue
        label = modis_landcover_map.get(code, f"class_{code}")
        binary = landcover_img.eq(int(code))
        base_profile = export_and_reproject(binary, f"{code}_{label}", base_profile)
