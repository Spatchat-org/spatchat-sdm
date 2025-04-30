import os
import time
import ee
import geemap
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import pandas as pd
import json

# --- Authenticate Earth Engine using Hugging Face Secret ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials, project='ee-hoyiwan')
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
region = ee.Geometry.BBox(min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)

# --- Define standard projection grid ---
res = 0.01  # degrees ~1km
crs = CRS.from_epsg(4326)
x_size = int((max_lon + buffer - (min_lon - buffer)) / res)
y_size = int((max_lat + buffer - (min_lat - buffer)) / res)
transform = from_bounds(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer,
    x_size, y_size
)

# --- Fetch layer list from environment ---
selected_layers = os.environ.get('SELECTED_LAYERS', '').split(',')
selected_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES', '').split(',')

# --- Prepare output directories ---
os.makedirs("predictor_rasters", exist_ok=True)
os.makedirs("predictor_rasters/wgs84", exist_ok=True)

# --- Define layer sources ---
layer_sources = {
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "slope": ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select('slope'),
    "aspect": ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select('aspect'),
    "ndvi": ee.ImageCollection("MODIS/061/MOD13A2").select('NDVI').mean(),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select('LC_Type1').first()
}
for i in range(1, 20):
    layer_sources[f"bio{i}"] = ee.Image("WORLDCLIM/V1/BIO").select(f"bio{str(i).zfill(2)}")

# --- Hard-coded MODIS landcover code→label map ---
modis_landcover_map = {
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

# --- Export & reproject helper ---
def export_and_reproject(image, name):
    raw_path = f"predictor_rasters/{name}.tif"
    aligned_path = f"predictor_rasters/wgs84/{name}.tif"

    # export from EE
    geemap.ee_export_image(
        image.clip(region),
        filename=raw_path,
        scale=1000,
        region=region,
        file_per_band=False,
        timeout=600
    )
    print(f"✅ Saved raw layer: {raw_path}")

    # reproject/resample into our uniform grid
    with rasterio.open(raw_path) as src:
        src_data = src.read(1)
        dst_data = np.empty((y_size, x_size), dtype=src_data.dtype)

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
            'height': y_size,
            'width': x_size,
            'transform': transform,
            'crs': crs,
            'driver': 'GTiff',
            'count': 1
        })

    with rasterio.open(aligned_path, 'w', **profile) as dst:
        dst.write(dst_data, 1)
    print(f"🌐 Reprojected to: {aligned_path}")

# --- Fetch all non-landcover predictors ---
for name in selected_layers:
    if name == "landcover": continue
    if name not in layer_sources:
        print(f"⚠️ Skipping unknown layer: {name}")
        continue
    print(f"📥 Fetching '{name}'...")
    export_and_reproject(layer_sources[name], name)

# --- One-hot MODIS landcover classes ---
if "landcover" in selected_layers and selected_classes:
    print("🌱 One-hot encoding MODIS landcover classes...")
    lc_img = layer_sources["landcover"]
    for code in selected_classes:
        if not code.isdigit(): continue
        code_int = int(code)
        label = modis_landcover_map.get(code_int, f"class_{code_int}")
        binary_img = lc_img.eq(code_int)
        export_and_reproject(binary_img, f"{code}_{label}")
