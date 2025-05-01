#!/usr/bin/env python3
import os
import json
import time
import math

import ee
import geemap
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# 1) Authenticate Earth Engine with your HF secret
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)
print("✅ Earth Engine authenticated with Service Account!")

# 2) Wait for the presence_points.csv to appear
csv_path = "inputs/presence_points.csv"
for i in range(5):
    if os.path.exists(csv_path):
        break
    print(f"⏳ Waiting for presence_points.csv… ({i+1}s)")
    time.sleep(1)
if not os.path.exists(csv_path):
    raise FileNotFoundError("❗ 'inputs/presence_points.csv' not found after 5s.")

# 3) Load your points & compute buffered bbox
df = pd.read_csv(csv_path)
if not {'latitude','longitude'}.issubset(df.columns):
    raise ValueError("❗ CSV must contain 'latitude' and 'longitude' columns.")
min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
buffer = 0.25  # degrees
region = ee.Geometry.BBox(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer
)
print(f"📍 Loaded {len(df)} points; study area = "
      f"{min_lat-buffer:.4f},{min_lon-buffer:.4f} → "
      f"{max_lat+buffer:.4f},{max_lon+buffer:.4f}")

# 4) Build a “master grid” in EPSG:4326 at 30 m resolution
#    Approx degrees per meter at the equator: 1° ≈ 111 320 m
deg_per_meter = 1.0 / 111_320.0
res_deg = 30 * deg_per_meter
crs = CRS.from_epsg(4326)
x_pixels = math.ceil((max_lon + buffer - (min_lon - buffer)) / res_deg)
y_pixels = math.ceil((max_lat + buffer - (min_lat - buffer)) / res_deg)
transform = from_bounds(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer,
    x_pixels, y_pixels
)
print(f"🗺  Grid: {x_pixels}×{y_pixels} @ {res_deg:.6f}° (~30 m) in EPSG:4326")

# 5) Prepare output dirs
os.makedirs("predictor_rasters/raw", exist_ok=True)
os.makedirs("predictor_rasters/wgs84", exist_ok=True)

# 6) Read your UI’s choices from env vars
selected_layers  = os.environ.get('SELECTED_LAYERS','').split(',')
selected_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES','').split(',')

# 7) Map of EE sources
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

# 8) Hard-coded MODIS landcover code→snake_case map
modis_landcover_map = {
    0:"water",1:"evergreen_needleleaf_forest",2:"evergreen_broadleaf_forest",
    3:"deciduous_needleleaf_forest",4:"deciduous_broadleaf_forest",5:"mixed_forest",
    6:"closed_shrublands",7:"open_shrublands",8:"woody_savannas",9:"savannas",
    10:"grasslands",11:"permanent_wetlands",12:"croplands",13:"urban_and_built_up",
    14:"cropland_natural_vegetation_mosaic",15:"snow_and_ice",16:"barren_or_sparsely_vegetated"
}

def export_and_reproject(img: ee.Image, name: str):
    """Export at 30 m, then reproject into our master 4326 grid."""
    raw_file = f"predictor_rasters/raw/{name}.tif"
    aligned = f"predictor_rasters/wgs84/{name}.tif"

    # export from EE at 30 m
    print(f"📥 Fetching '{name}' at 30 m…")
    geemap.ee_export_image(
        img.clip(region),
        filename=raw_file,
        scale=30,          # meters
        region=region,
        file_per_band=False,
        timeout=600
    )
    print(f"✅ Saved raw: {raw_file}")

    # reproject + resample to our grid
    with rasterio.open(raw_file) as src:
        src_arr = src.read(1)
        dst_arr = np.empty((y_pixels, x_pixels), dtype=src_arr.dtype)

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest
        )

        meta = src.profile.copy()
        meta.update({
            'driver':   'GTiff',
            'height':   y_pixels,
            'width':    x_pixels,
            'transform':transform,
            'crs':      crs,
            'count':    1
        })

        with rasterio.open(aligned, 'w', **meta) as dst:
            dst.write(dst_arr, 1)

    print(f"🌐 Reprojected → {aligned}")

# 9) Loop through your selected layers
for name in selected_layers:
    if name == "landcover":
        continue
    if name not in layer_sources:
        print(f"⚠️ Skipping unknown: {name}")
        continue
    export_and_reproject(layer_sources[name], name)

# 10) And one-hot encode any chosen landcover classes
if "landcover" in selected_layers and selected_classes:
    lc = layer_sources["landcover"]
    print("🌱 One-hot encoding landcover classes…")
    for code in selected_classes:
        if not code.isdigit():
            continue
        code_i = int(code)
        label  = modis_landcover_map.get(code_i, f"class_{code_i}")
        bin_img = lc.eq(code_i)
        export_and_reproject(bin_img, f"{code}_{label}")
