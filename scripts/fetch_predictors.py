#!/usr/bin/env python3
import os
import json
import time
import ee
import geemap
import pandas as pd
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# --- Authenticate Earth Engine using Hugging Face Secret ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)
print("✅ Earth Engine authenticated successfully inside fetch_predictors.py!")

# --- Wait for presence_points.csv to appear ---
csv_path = "inputs/presence_points.csv"
for i in range(5):
    if os.path.exists(csv_path):
        break
    print(f"⏳ Waiting for presence_points.csv... ({i+1}s)")
    time.sleep(1)
if not os.path.exists(csv_path):
    raise FileNotFoundError("❗ 'inputs/presence_points.csv' not found after 5s.")

# --- Load presence points ---
df = pd.read_csv(csv_path)
if not {'latitude','longitude'}.issubset(df.columns):
    raise ValueError("❗ CSV must have 'latitude' and 'longitude' columns.")
print(f"📍 Loaded {len(df)} presence points.")

# --- Compute study-area bbox + buffer ---
min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
buffer = 0.25
region = ee.Geometry.BBox(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer
)

# --- Build the exact WGS84 grid we want ---
res_deg  = 0.01            # ~1 km
crs_epsg = 'EPSG:4326'
width  = int( (max_lon+buffer - (min_lon-buffer)) / res_deg )
height = int( (max_lat+buffer - (min_lat-buffer)) / res_deg )
transform = from_bounds(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer,
    width, height
)
print(f"🗺  Grid: {width}×{height} @ {res_deg}° in {crs_epsg}")

# --- Selections from the UI ---
selected_layers  = os.environ.get('SELECTED_LAYERS','').split(',')
selected_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES','').split(',')

# --- Prepare output directory ---
out_dir = "predictor_rasters/wgs84"
os.makedirs(out_dir, exist_ok=True)

# --- Define your EE sources ---
base_srtm = ee.Image("USGS/SRTMGL1_003")
layer_sources = {
    "elevation": base_srtm,
    "slope":     ee.Terrain.products(base_srtm).select('slope'),
    "aspect":    ee.Terrain.products(base_srtm).select('aspect'),
    "ndvi":      ee.ImageCollection("MODIS/061/MOD13A2").select('NDVI').mean(),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select('LC_Type1').first()
}
for i in range(1,20):
    layer_sources[f"bio{i}"] = ee.Image("WORLDCLIM/V1/BIO").select(f"bio{str(i).zfill(2)}")

# --- Helper: export one layer, server-side reprojection in one step ---
def export_aligned(ee_img, name):
    out_path = os.path.join(out_dir, f"{name}.tif")
    # reproject on the EE server to EXACTLY your grid
    aligned = ee_img.clip(region).reproject(
        crs=crs_epsg,
        crsTransform=[
            transform.a, transform.b, transform.c,
            transform.d, transform.e, transform.f
        ]
    )
    geemap.ee_export_image(
        aligned,
        filename=out_path,
        scale= int(res_deg * 111320),  # ~ meters per degree
        region=region,
        file_per_band=False,
        timeout=600
    )
    print(f"✅ Exported aligned {name} → {out_path}")

# --- Export all numeric predictors ---
for name in selected_layers:
    if name == "landcover":
        continue
    if name not in layer_sources:
        print(f"⚠️ Skipping unknown layer {name}")
        continue
    print(f"📥 Fetching '{name}'...")
    export_aligned(layer_sources[name], name)

# --- One-hot encode and export each landcover class ---
if "landcover" in selected_layers and selected_classes:
    # load your code→label JSON (must be in your repo)
    with open("modis_landcover_code_name.json") as f:
        code_name = json.load(f)
    lc = layer_sources["landcover"]
    print("🌱 One-hot encoding landcover classes...")
    for code in selected_classes:
        if not code.isdigit(): 
            continue
        c = int(code)
        label = code_name.get(str(c), f"class_{c}")
        img = lc.eq(c)
        print(f"  • class {c} → {label}")
        export_aligned(img, f"{c}_{label}")
