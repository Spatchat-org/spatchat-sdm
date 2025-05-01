#!/usr/bin/env python3
import os
import json
import time

import ee
import geemap
from geemap.common import ee_to_xarray
import pandas as pd
import rioxarray  # registers the `.rio` methods on xarray objects

# --- Authenticate Earth Engine using Service Account from HF secret ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)
print("✅ Earth Engine authenticated with Service Account!")

# --- Wait for presence CSV to land on disk ---
csv_path = "inputs/presence_points.csv"
for i in range(5):
    if os.path.exists(csv_path):
        break
    print(f"⏳ Waiting for presence_points.csv… ({i+1}s)")
    time.sleep(1)
if not os.path.exists(csv_path):
    raise FileNotFoundError("❗ 'inputs/presence_points.csv' not found after 5s.")

# --- Load presence points & build study‐area bbox ---
df = pd.read_csv(csv_path)
if not {'latitude','longitude'}.issubset(df.columns):
    raise ValueError("❗ CSV must contain 'latitude' and 'longitude' columns.")
min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
buffer = 0.25
region_ee = ee.Geometry.BBox(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer
)
region_geojson = region_ee.getInfo()  # for ee_to_xarray

print(f"📍 Loaded {len(df)} points; study area = "
      f"{min_lat-buffer},{min_lon-buffer} → {max_lat+buffer},{max_lon+buffer}")

# --- Prepare output dir ---
os.makedirs("predictor_rasters/wgs84", exist_ok=True)

# --- What layers the UI set in env vars ---
selected_layers  = os.environ.get('SELECTED_LAYERS','').split(',')
selected_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES','').split(',')

# --- Earth Engine source images ---
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

# --- Hard-coded MODIS landcover code→snake_case label map ---
modis_landcover_map = {
    0:"water",1:"evergreen_needleleaf_forest",2:"evergreen_broadleaf_forest",
    3:"deciduous_needleleaf_forest",4:"deciduous_broadleaf_forest",5:"mixed_forest",
    6:"closed_shrublands",7:"open_shrublands",8:"woody_savannas",9:"savannas",
    10:"grasslands",11:"permanent_wetlands",12:"croplands",13:"urban_and_built_up",
    14:"cropland_natural_vegetation_mosaic",15:"snow_and_ice",16:"barren_or_sparsely_vegetated"
}

# --- Desired output resolution in meters ---
SCALE = 30

def fetch_with_xee(img: ee.Image, name: str):
    """Clip to study region, pull into xarray, normalize dims, write GeoTIFF."""
    print(f"📥 Fetching via xee → {name}")
    ds = ee_to_xarray(
        img.clip(region_ee),
        region_geojson,
        crs="EPSG:4326",
        scale=SCALE,
        n_images=1,
        ee_initialize=False
    )

    # Pick out the single data variable
    var = list(ds.data_vars)[0]
    da = ds[var]

    # Drop time if present
    if 'time' in da.dims:
        da = da.isel(time=0)

    # Auto-map whatever lon/lat dims you have to x/y
    mapping = {}
    for dim in da.dims:
        l = dim.lower()
        if l.startswith('lon'):
            mapping[dim] = 'x'
        elif l.startswith('lat'):
            mapping[dim] = 'y'
        elif dim == 'x':
            mapping[dim] = 'x'
        elif dim == 'y':
            mapping[dim] = 'y'
    if mapping:
        da = da.rename(mapping)

    # Ensure correct 2D order
    da = da.transpose('y','x')

    # Attach spatial metadata and write out
    da = da.rio.set_spatial_dims(x_dim='x', y_dim='y')
    da = da.rio.write_crs("EPSG:4326")
    out_path = f"predictor_rasters/wgs84/{name}.tif"
    da.rio.to_raster(out_path)
    print(f"✅  Aligned → {out_path}")

# --- Fetch all selected environmental layers ---
for name in selected_layers:
    if name == "landcover":
        continue
    if name not in layer_sources:
        print(f"⚠️ Skipping unknown: {name}")
        continue
    fetch_with_xee(layer_sources[name], name)

# --- One-hot encode any selected MODIS landcover classes ---
if "landcover" in selected_layers and selected_classes:
    lc = layer_sources["landcover"]
    print("🌱 One-hot encoding landcover…")
    for code in selected_classes:
        if not code.isdigit():
            continue
        code_i = int(code)
        label  = modis_landcover_map.get(code_i, f"class_{code_i}")
        bin_img = lc.eq(code_i)
        fetch_with_xee(bin_img, f"{code}_{label}")
