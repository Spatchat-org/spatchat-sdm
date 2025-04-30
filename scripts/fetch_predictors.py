#!/usr/bin/env python3
import os
import json
import time
import ee
import geemap
import pandas as pd
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# --- Authenticate Earth Engine ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)
print("✅ Authenticated Earth Engine")

# --- Wait for presence_points.csv ---
csv_path = "inputs/presence_points.csv"
for i in range(5):
    if os.path.exists(csv_path):
        break
    print(f"⏳ Waiting for presence_points.csv... ({i+1}s)")
    time.sleep(1)
if not os.path.exists(csv_path):
    raise FileNotFoundError("❗ inputs/presence_points.csv not found")

# --- Load points & define study area ---
df = pd.read_csv(csv_path)
if not {'latitude','longitude'}.issubset(df.columns):
    raise ValueError("❗ CSV must have latitude,longitude")
print(f"📍 Loaded {len(df)} points")
min_lat, max_lat = df.latitude.min(), df.latitude.max()
min_lon, max_lon = df.longitude.min(), df.longitude.max()
buffer = 0.25
region = ee.Geometry.BBox(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer
)

# --- Build exact WGS84 grid from bbox+buffer ---
res_deg  = 0.01
crs_epsg = 'EPSG:4326'
width  = int((max_lon+buffer - (min_lon-buffer)) / res_deg)
height = int((max_lat+buffer - (min_lat-buffer)) / res_deg)
transform = from_bounds(
    min_lon-buffer, min_lat-buffer,
    max_lon+buffer, max_lat+buffer,
    width, height
)
print(f"🗺  Grid: {width}×{height} @ {res_deg}°")

# --- What the user selected in the UI ---
selected_layers  = os.environ.get('SELECTED_LAYERS','').split(',')
selected_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES','').split(',')

# --- Prepare output dir ---
out_dir = "predictor_rasters/wgs84"
os.makedirs(out_dir, exist_ok=True)

# --- Earth Engine image sources ---
base = ee.Image("USGS/SRTMGL1_003")
sources = {
    "elevation": base,
    "slope":     ee.Terrain.products(base).select('slope'),
    "aspect":    ee.Terrain.products(base).select('aspect'),
    "ndvi":      ee.ImageCollection("MODIS/061/MOD13A2").select('NDVI').mean(),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select('LC_Type1').first()
}
for i in range(1,20):
    sources[f"bio{i}"] = ee.Image("WORLDCLIM/V1/BIO").select(f"bio{str(i).zfill(2)}")

# --- Hard-coded MODIS code→name map (no external file) ---
modis_labels = {
    0: "water",
    1: "evergreen_needleleaf_forest",
    2: "evergreen_broadleaf_forest",
    3: "deciduous_needleleaf_forest",
    4: "deciduous_broadleaf_forest",
    5: "mixed_forest",
    6: "closed_shrublands",
    7: "open_shrublands",
    8: "woody_savannas",
    9: "savannas",
    10: "grasslands",
    11: "permanent_wetlands",
    12: "croplands",
    13: "urban_and_built_up",
    14: "cropland_natural_vegetation_mosaic",
    15: "snow_and_ice",
    16: "barren_or_sparsely_vegetated"
}

# --- Helper to export a layer reprojected server-side to YOUR grid ---
def export_aligned(img, name):
    out_path = os.path.join(out_dir, f"{name}.tif")
    aligned = img.clip(region).reproject(
        crs=crs_epsg,
        crsTransform=[
            transform.a, transform.b, transform.c,
            transform.d, transform.e, transform.f
        ]
    )
    geemap.ee_export_image(
        aligned,
        filename=out_path,
        scale=int(res_deg * 111320),
        region=region,
        file_per_band=False,
        timeout=600
    )
    print(f"✅ {name} → {out_path}")

# --- Export numeric predictors ---
for name in selected_layers:
    if name=="landcover" or name not in sources: continue
    print(f"📥 Fetching {name}…")
    export_aligned(sources[name], name)

# --- Export one-hot landcover masks ---
if "landcover" in selected_layers and selected_classes:
    print("🌱 Encoding landcover…")
    lc = sources["landcover"]
    for code in selected_classes:
        if not code.isdigit(): continue
        c = int(code)
        lbl = modis_labels.get(c, f"class_{c}")
        print(f" • class {c} → {lbl}")
        export_aligned(lc.eq(c), f"{c}_{lbl}")
