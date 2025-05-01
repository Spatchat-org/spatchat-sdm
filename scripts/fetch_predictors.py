#!/usr/bin/env python3
import os
import json
import time

import ee
import geemap
from geemap.common import ee_to_xarray
import pandas as pd
import rioxarray  # noqa: F401  Provides the .rio accessor

# --- 1) Authenticate Earth Engine via Service Account ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)
print("✅ Earth Engine authenticated with Service Account!")

# --- 2) Wait for the presence-points CSV to land ---
csv_path = "inputs/presence_points.csv"
for i in range(5):
    if os.path.exists(csv_path):
        break
    print(f"⏳ Waiting for presence_points.csv ({i+1})…")
    time.sleep(1)
if not os.path.exists(csv_path):
    raise FileNotFoundError("❗ inputs/presence_points.csv not found.")

# --- 3) Load points, build study-area bbox & GeoJSON region ---
df = pd.read_csv(csv_path)
if not {'latitude','longitude'}.issubset(df.columns):
    raise ValueError("Presence CSV must contain 'latitude' and 'longitude' columns.")

min_lat, max_lat = df.latitude.min(), df.latitude.max()
min_lon, max_lon = df.longitude.min(), df.longitude.max()
buffer = 0.25
region_ee = ee.Geometry.BBox(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer
)
region_geojson = region_ee.getInfo()

print(f"📍 Loaded {len(df)} points; study area = "
      f"{min_lat-buffer},{min_lon-buffer} → {max_lat+buffer},{max_lon+buffer}")

# --- 4) Prepare output folder ---
os.makedirs("predictor_rasters/wgs84", exist_ok=True)

# --- 5) What the user selected via UI (exported into envvars) ---
selected_layers  = os.environ.get('SELECTED_LAYERS','').split(',')
selected_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES','').split(',')

# --- 6) EE sources dictionary ---
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

# --- 7) Inlined MODIS code→snake_case map (no JSON file!) ---
modis_landcover_map = {
     0:"water",1:"evergreen_needleleaf_forest",2:"evergreen_broadleaf_forest",
     3:"deciduous_needleleaf_forest",4:"deciduous_broadleaf_forest",
     5:"mixed_forest",6:"closed_shrublands",7:"open_shrublands",
     8:"woody_savannas",9:"savannas",10:"grasslands",11:"permanent_wetlands",
    12:"croplands",13:"urban_and_built_up",14:"cropland_natural_vegetation_mosaic",
    15:"snow_and_ice",16:"barren_or_sparsely_vegetated"
}

# --- 8) Desired native resolution (meters) ---
SCALE = 30

def fetch_with_xee(img: ee.Image, name: str):
    """
    1) Clip
    2) ee_to_xarray → xarray.Dataset
    3) pick the single var & drop time
    4) rename dims, transpose to (y,x)
    5) write CRS + raster
    """
    print(f"📥 Fetching via xee → {name}")
    ds = ee_to_xarray(
        img.clip(region_ee),
        region_geojson,
        crs="EPSG:4326",
        scale=SCALE,
        n_images=1,
        ee_initialize=False
    )

    # pick the one data variable
    var = list(ds.data_vars)[0]
    da = ds[var]

    # drop time if it exists
    if "time" in da.dims:
        da = da.isel(time=0)

    # rename the EE default coords → x,y
    # (ee_to_xarray names them 'longitude' & 'latitude')
    da = da.rename({"longitude":"x", "latitude":"y"})

    # ensure the dim order is exactly (y, x)
    da = da.transpose("y", "x")

    # tell rioxarray which dims are spatial & write CRS
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da = da.rio.write_crs("EPSG:4326")

    out = f"predictor_rasters/wgs84/{name}.tif"
    da.rio.to_raster(out)
    print(f"✅ Aligned → {out}")

# --- 9) Fetch each regular predictor ---
for name in selected_layers:
    if name == "landcover":
        continue
    if name not in layer_sources:
        print(f"⚠️ Skipping unknown layer: {name}")
        continue
    fetch_with_xee(layer_sources[name], name)

# --- 10) One-hot encode each selected landcover class ---
if "landcover" in selected_layers and selected_classes:
    print("🌱 One-hot encoding landcover…")
    lc_img = layer_sources["landcover"]
    for code in selected_classes:
        if not code.isdigit():
            continue
        code_i = int(code)
        label  = modis_landcover_map.get(code_i, f"class_{code_i}")
        bin_img = lc_img.eq(code_i)
        fetch_with_xee(bin_img, f"{code}_{label}")
