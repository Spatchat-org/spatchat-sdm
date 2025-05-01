#!/usr/bin/env python3
import os
import time
import json

import ee
import pandas as pd
import xarray as xr
import xee                               # Xee backend for xarray
import rioxarray                         # adds the .rio accessor

# ————— USER SETTINGS —————
SCALE = 30     # in meters
BUFFER = 0.25  # degrees buffer around points

# ————— AUTH EARTH ENGINE —————
service_account_info = json.loads(os.environ["GEE_SERVICE_ACCOUNT"])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info["client_email"],
    key_data=json.dumps(service_account_info),
)
ee.Initialize(credentials)
print("✅ Earth Engine authenticated successfully!")

# ————— WAIT FOR UPLOADED CSV —————
csv_path = "inputs/presence_points.csv"
for _ in range(5):
    if os.path.exists(csv_path):
        break
    print("⏳ Waiting for inputs/presence_points.csv…")
    time.sleep(1)
if not os.path.exists(csv_path):
    raise FileNotFoundError("❗ inputs/presence_points.csv not found")

# ————— LOAD PRESENCE POINTS & DEFINE REGION —————
df = pd.read_csv(csv_path)
if not {"latitude", "longitude"}.issubset(df.columns):
    raise ValueError("❗ CSV must contain 'latitude' & 'longitude' columns")

min_lat, max_lat = df["latitude"].min(), df["latitude"].max()
min_lon, max_lon = df["longitude"].min(), df["longitude"].max()

region_ee = ee.Geometry.BBox(
    min_lon - BUFFER, min_lat - BUFFER,
    max_lon + BUFFER, max_lat + BUFFER
)
print(f"📍 Loaded {len(df)} points → region: "
      f"{min_lat-BUFFER},{min_lon-BUFFER} to {max_lat+BUFFER},{max_lon+BUFFER}")

# ————— OUTPUT DIR —————
out_dir = "predictor_rasters/wgs84"
os.makedirs(out_dir, exist_ok=True)

# ————— WHAT TO FETCH —————
selected_layers = os.environ.get("SELECTED_LAYERS", "").split(",")
selected_classes = os.environ.get("SELECTED_LANDCOVER_CLASSES", "").split(",")

# ————— EARTH-ENGINE SOURCES —————
layer_sources = {
    "elevation":  ee.Image("USGS/SRTMGL1_003"),
    "slope":      ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("slope"),
    "aspect":     ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("aspect"),
    "ndvi":       ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI").mean(),
    "landcover":  ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1").first(),
}
for i in range(1, 20):
    layer_sources[f"bio{i}"] = ee.Image("WORLDCLIM/V1/BIO")\
                               .select(f"bio{str(i).zfill(2)}")

# ————— HARD-CODED LANDCOVER MAP —————
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
    16: "barren_or_sparsely_vegetated",
}

def fetch_layer(img: ee.Image, name: str):
    """
    Clip → pull via Xee into xarray → rename dims to x/y → set CRS →
    write aligned GeoTIFF.
    """
    print(f"📥 Fetching → {name}")
    clipped = img.clip(region_ee)

    # open with Xee backend
    ds = xr.open_dataset(
        clipped,
        engine=xee.EarthEngineBackendEntrypoint,
        crs="EPSG:4326",
        scale=SCALE,
    )

    # --- Rename any lon/lat dims to x/y for rioxarray
    rename_dims = {}
    for d in ds.dims:
        dl = d.lower()
        if "lon" in dl or d == "x":
            rename_dims[d] = "x"
        elif "lat" in dl or d == "y":
            rename_dims[d] = "y"
    if not rename_dims:
        raise RuntimeError(f"❗ Could not detect spatial dims in {ds.dims}")
    ds = ds.rename(rename_dims)

    # --- Tell rioxarray which are spatial dims &
    #     write the CRS so .rio.to_raster() works
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds.rio.write_crs("EPSG:4326", inplace=True)

    # --- Export GeoTIFF
    out_path = os.path.join(out_dir, f"{name}.tif")
    ds.rio.to_raster(out_path)
    print(f"✅ Exported aligned {name} → {out_path}")


# ————— STANDARD PREDICTORS —————
for layer in selected_layers:
    if layer == "landcover":
        continue
    if layer not in layer_sources:
        print(f"⚠️ Skipping unknown layer: {layer}")
        continue
    fetch_layer(layer_sources[layer], layer)

# ————— ONE-HOT LANDCOVER —————
if "landcover" in selected_layers and selected_classes:
    print("🌱 One-hot encoding MODIS landcover…")
    lc_img = layer_sources["landcover"]
    for code_str in selected_classes:
        if not code_str.isdigit():
            continue
        code = int(code_str)
        name = modis_landcover_map.get(code)
        if not name:
            print(f"⚠️ Unknown code {code}, skipping")
            continue
        fetch_layer(lc_img.eq(code), f"{code}_{name}")
