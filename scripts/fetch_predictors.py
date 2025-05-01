#!/usr/bin/env python3
import os
import json
import time

import ee
import geemap
from geemap.common import ee_to_xarray
import pandas as pd
import rioxarray   # provides the .rio methods

# --- Authenticate Earth Engine using your HuggingFace secret ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)
print("✅ Earth Engine authenticated with Service Account!")

# --- Wait for the presence CSV ---
csv_path = "inputs/presence_points.csv"
for i in range(5):
    if os.path.exists(csv_path):
        break
    print(f"⏳ Waiting for presence_points.csv ({i+1}s)…")
    time.sleep(1)
if not os.path.exists(csv_path):
    raise FileNotFoundError("❗ inputs/presence_points.csv not found.")

# --- Load points and build study‐area bbox ---
df = pd.read_csv(csv_path)
assert {'latitude','longitude'}.issubset(df.columns), \
       "Presence CSV needs ‘latitude’ & ‘longitude’ columns."
min_lat, max_lat = df.latitude.min(), df.latitude.max()
min_lon, max_lon = df.longitude.min(), df.longitude.max()
buffer = 0.25
region_ee = ee.Geometry.BBox(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer
)
region_geojson = region_ee.getInfo()   # pass this into ee_to_xarray
print(f"📍 Loaded {len(df)} points; study area = "
      f"{min_lat-buffer},{min_lon-buffer} → {max_lat+buffer},{max_lon+buffer}")

# --- Where to drop your rasters ---
os.makedirs("predictor_rasters/wgs84", exist_ok=True)

# --- What user selected via the UI (env vars set in app.py) ---
selected_layers  = os.environ.get('SELECTED_LAYERS','').split(',')
selected_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES','').split(',')

# --- EarthEngine sources ---
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

# --- One-hot map (embed instead of JSON) ---
modis_landcover_map = {
    0:"water",1:"evergreen_needleleaf_forest",2:"evergreen_broadleaf_forest",
    3:"deciduous_needleleaf_forest",4:"deciduous_broadleaf_forest",
    5:"mixed_forest",6:"closed_shrublands",7:"open_shrublands",
    8:"woody_savannas",9:"savannas",10:"grasslands",11:"permanent_wetlands",
    12:"croplands",13:"urban_and_built_up",14:"cropland_natural_vegetation_mosaic",
    15:"snow_and_ice",16:"barren_or_sparsely_vegetated"
}

# --- Desired output resolution ---
SCALE = 30  # meters

def fetch_with_xee(img: ee.Image, name: str):
    """Clip to region, pull into xarray, set dims/CRS, write GeoTIFF."""
    print(f"📥 Fetching via xee → {name}")
    # clip & hand off to geemap
    ds = ee_to_xarray(
        img.clip(region_ee),
        region_geojson,
        crs="EPSG:4326",
        scale=SCALE,
        n_images=1,
        ee_initialize=False
    )
    # grab the single data variable
    var = list(ds.data_vars)[0]
    da  = ds[var].isel(time=0)                      # drop time
    da  = da.rename({"lon":"x","lat":"y"})          # xarray→rioxarray
    da  = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da  = da.rio.write_crs("EPSG:4326")
    out = f"predictor_rasters/wgs84/{name}.tif"
    da.rio.to_raster(out)
    print(f"✅  Aligned → {out}")

# --- Fetch each chosen environmental layer ---
for name in selected_layers:
    if name == "landcover":  continue
    if name not in layer_sources:
        print(f"⚠️ Skipping unknown: {name}")
        continue
    fetch_with_xee(layer_sources[name], name)

# --- One-hot encode each selected landcover class ---
if "landcover" in selected_layers and selected_classes:
    lc = layer_sources["landcover"]
    print("🌱 One-hot encoding landcover…")
    for code in selected_classes:
        if not code.isdigit(): 
            continue
        code_i = int(code)
        label  = modis_landcover_map.get(code_i, f"class_{code_i}")
        bin_im = lc.eq(code_i)
        fetch_with_xee(bin_im, f"{code}_{label}")
