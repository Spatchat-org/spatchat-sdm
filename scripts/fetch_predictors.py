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
for _ in range(5):
    if os.path.exists(csv_path):
        break
    print("⏳ Waiting for presence_points.csv...")
    time.sleep(1)
if not os.path.exists(csv_path):
    raise FileNotFoundError("❗ 'inputs/presence_points.csv' not found after wait.")

# --- Load presence points ---
df = pd.read_csv(csv_path)
if not {'latitude', 'longitude'}.issubset(df.columns):
    raise ValueError("❗ CSV must contain 'latitude' and 'longitude' columns.")
print(f"📍 Loaded {len(df)} presence points.")

# --- Compute study area bbox + buffer ---
min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
buffer = 0.25
region_geom = ee.Geometry.BBox(min_lon - buffer, min_lat - buffer,
                               max_lon + buffer, max_lat + buffer)

# --- Define target grid in WGS84 ---
res = 0.01  # deg (~1km)
crs = CRS.from_epsg(4326)
x_size = int((max_lon - min_lon + 2 * buffer) / res)
y_size = int((max_lat - min_lat + 2 * buffer) / res)
transform = from_bounds(
    min_lon - buffer,
    min_lat - buffer,
    max_lon + buffer,
    max_lat + buffer,
    x_size,
    y_size
)

# --- Selection from env ---
selected_layers = os.environ.get('SELECTED_LAYERS', '').split(',')
selected_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES', '').split(',')

# --- Ensure output dirs ---
os.makedirs("predictor_rasters", exist_ok=True)
os.makedirs("predictor_rasters/wgs84", exist_ok=True)

# --- Earth Engine sources ---
layer_sources = {
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "slope": ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select('slope'),
    "aspect": ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select('aspect'),
    "ndvi": ee.ImageCollection("MODIS/061/MOD13A2").select('NDVI').mean(),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select('LC_Type1').first()
}
for i in range(1, 20):
    layer_sources[f"bio{i}"] = ee.Image("WORLDCLIM/V1/BIO").select(f"bio{str(i).zfill(2)}")

# --- Export & reproject function ---
def export_and_reproject(image, name):
    raw_path = f"predictor_rasters/{name}.tif"
    aligned_path = f"predictor_rasters/wgs84/{name}.tif"

    # Export directly in EPSG:4326 using ee.Geometry
    geemap.ee_export_image(
        image.clip(region_geom),
        raw_path,
        scale=1000,
        crs='EPSG:4326',
        region=region_geom,
        file_per_band=False,
        timeout=600
    )
    print(f"✅ Exported raw layer: {raw_path}")

    # Read & reproject to exact grid
    with rasterio.open(raw_path) as src:
        src_arr = src.read(1)
        dst_arr = np.full((y_size, x_size), src.nodata or np.nan, dtype=src_arr.dtype)
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest
        )
        profile = src.profile.copy()
    profile.update({
        'driver': 'GTiff',
        'height': y_size,
        'width': x_size,
        'crs': crs,
        'transform': transform,
        'count': 1
    })
    with rasterio.open(aligned_path, 'w', **profile) as dst:
        dst.write(dst_arr, 1)
    print(f"🌐 Reprojected layer: {aligned_path}")

# --- Regular predictors ---
for name in selected_layers:
    if name == 'landcover':
        continue
    if name not in layer_sources:
        print(f"⚠️ Unknown layer: {name}")
        continue
    print(f"📥 Fetching '{name}'...")
    export_and_reproject(layer_sources[name], name)

# --- One-hot landcover ---
if 'landcover' in selected_layers and selected_classes:
    print("🌱 One-hot encoding landcover...")
    with open('modis_landcover_code_name.json') as f:
        label_map = json.load(f)
    lc_img = layer_sources['landcover']
    for code in selected_classes:
        if not code.isdigit():
            continue
        code_int = int(code)
        label = label_map.get(str(code_int), f'class_{code_int}')
        mask = lc_img.eq(code_int)
        export_and_reproject(mask, f"{code}_{label}")
