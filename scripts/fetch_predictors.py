import os
import json
import time

import ee
import xarray as xr
import xee                            # pip install xee
import rioxarray                      # pip install rioxarray
import pandas as pd
from rasterio.crs import CRS

# --- Authenticate Earth Engine with your Service Account Secret ---
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
    raise FileNotFoundError("❗ 'inputs/presence_points.csv' not found after waiting.")

# --- Load points & build EE region ---
df = pd.read_csv(csv_path)
if not {'latitude','longitude'}.issubset(df.columns):
    raise ValueError("❗ CSV needs 'latitude' and 'longitude' columns.")
min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
buffer = 0.25
region_ee      = ee.Geometry.BBox(min_lon-buffer, min_lat-buffer,
                                 max_lon+buffer, max_lat+buffer)
region_geojson = region_ee.getInfo()  # for xarray

print(f"📍 Loaded {len(df)} presence points.")
print(f"🗺  Study area: {min_lat-buffer},{min_lon-buffer} → {max_lat+buffer},{max_lon+buffer}")

# --- Output dir ---
out_dir = "predictor_rasters/wgs84"
os.makedirs(out_dir, exist_ok=True)

# --- What to fetch (from env) ---
selected_layers        = os.environ.get('SELECTED_LAYERS','').split(',')
selected_landcover_codes = os.environ.get('SELECTED_LANDCOVER_CLASSES','').split(',')

# --- EE sources dict ---
layer_sources = {
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "slope":     ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("slope"),
    "aspect":    ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("aspect"),
    "ndvi":      ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI").mean(),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1").first()
}
for i in range(1,20):
    layer_sources[f"bio{i}"] = ee.Image("WORLDCLIM/V1/BIO").select(f"bio{str(i).zfill(2)}")

# --- Hard-coded MODIS code→snake_case map for one-hot labels ---
modis_landcover_map = {
    0:  "water",
    1:  "evergreen_needleleaf_forest",
    2:  "evergreen_broadleaf_forest",
    3:  "deciduous_needleleaf_forest",
    4:  "deciduous_broadleaf_forest",
    5:  "mixed_forest",
    6:  "closed_shrublands",
    7:  "open_shrublands",
    8:  "woody_savannas",
    9:  "savannas",
    10: "grasslands",
    11: "permanent_wetlands",
    12: "croplands",
    13: "urban_and_built_up",
    14: "cropland_natural_vegetation_mosaic",
    15: "snow_and_ice",
    16: "barren_or_sparsely_vegetated"
}

# --- Fixed output resolution (meters) ---
SCALE = 30

def fetch_layer(img: ee.Image, name: str):
    """Clip to region, pull into xarray via Xee, and write single-band GeoTIFF."""
    print(f"📥 Fetching → {name}")
    ds = xr.open_dataset(
        img.clip(region_ee),
        region=region_geojson,
        crs="EPSG:4326",
        scale=SCALE,
        engine=xee.EarthEngineBackendEntrypoint
    )
    # pick the sole band
    band = list(ds.data_vars)[0]
    da   = ds[band]
    out_path = os.path.join(out_dir, f"{name}.tif")
    da.rio.to_raster(out_path)
    print(f"✅ Wrote {out_path}")

# --- Loop through “regular” predictors ---
for name in selected_layers:
    if name == "landcover":
        continue
    if name not in layer_sources:
        print(f"⚠️ Skipping unknown layer: {name}")
        continue
    fetch_layer(layer_sources[name], name)

# --- One-hot encode & fetch each selected landcover class ---
if "landcover" in selected_layers and selected_landcover_codes:
    print("🌱 One-hot encoding MODIS landcover classes…")
    lc = layer_sources["landcover"]
    for code_str in selected_landcover_codes:
        if not code_str.isdigit(): 
            continue
        code = int(code_str)
        label = modis_landcover_map.get(code, f"class_{code}")
        binary = lc.eq(code)
        fetch_layer(binary, f"{code}_{label}")
