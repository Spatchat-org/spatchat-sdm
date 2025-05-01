import os
import time
import json

import ee
import xee
import rioxarray  # for .rio.to_raster()
import pandas as pd

# --- Authenticate Earth Engine using Service Account ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)
# initialize Xee so it reuses the same creds (no interactive flow)
xee.initialize(credentials=credentials)
print("✅ Authenticated Earth Engine & Xee with Service Account!")

# --- Wait for the uploaded CSV to appear ---
csv_path = "inputs/presence_points.csv"
for i in range(5):
    if os.path.exists(csv_path):
        break
    print(f"⏳ Waiting for presence_points.csv... ({i+1}s)")
    time.sleep(1)
if not os.path.exists(csv_path):
    raise FileNotFoundError("❗ 'inputs/presence_points.csv' not found after 5s wait.")

# --- Load points and build EE region ---
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
# For Xee we need GeoJSON
region_geojson = region_ee.getInfo()

print(f"📍 Loaded {len(df)} presence points.")
print(f"🗺  Study area: "
      f"{min_lat-buffer},{min_lon-buffer} → {max_lat+buffer},{max_lon+buffer}")

# --- Prepare output dir ---
os.makedirs("predictor_rasters/wgs84", exist_ok=True)

# --- What layers to fetch? ---
selected_layers = os.environ.get('SELECTED_LAYERS','').split(',')
selected_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES','').split(',')

# --- Earth Engine source images ---
layer_sources = {
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "slope": ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("slope"),
    "aspect": ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("aspect"),
    "ndvi": ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI").mean(),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1").first()
}
for i in range(1,20):
    layer_sources[f"bio{i}"] = ee.Image("WORLDCLIM/V1/BIO").select(f"bio{str(i).zfill(2)}")

# --- Desired resolution (meters) ---
SCALE = 30

def fetch_with_xee(image: ee.Image, name: str):
    """
    Use Xee to pull the EE image at SCALE m,
    clip to region_ee, convert to xarray, then write GeoTIFF.
    """
    print(f"📥 Fetching via Xee → {name}")
    da = xee.image_to_xarray(
        image=image.clip(region_ee),
        region=region_geojson,
        crs="EPSG:4326",
        scale=SCALE
    )
    out_path = f"predictor_rasters/wgs84/{name}.tif"
    da.rio.to_raster(out_path)
    print(f"✅ Exported aligned {name} → {out_path}")

# --- Export non‐landcover predictors ---
for name in selected_layers:
    if name == "landcover":
        continue
    if name not in layer_sources:
        print(f"⚠️ Skipping unknown layer: {name}")
        continue
    fetch_with_xee(layer_sources[name], name)

# --- One‐hot encode selected MODIS landcover classes ---
if "landcover" in selected_layers and selected_classes:
    print("🌱 One-hot encoding MODIS landcover classes…")
    lc_img = layer_sources["landcover"]
    # Embedded mapping code→snake_case
    modis_landcover_map = {
        "0":  "water",
        "1":  "evergreen_needleleaf_forest",
        "2":  "evergreen_broadleaf_forest",
        "3":  "deciduous_needleleaf_forest",
        "4":  "deciduous_broadleaf_forest",
        "5":  "mixed_forest",
        "6":  "closed_shrublands",
        "7":  "open_shrublands",
        "8":  "woody_savannas",
        "9":  "savannas",
        "10": "grasslands",
        "11": "permanent_wetlands",
        "12": "croplands",
        "13": "urban_and_built_up",
        "14": "cropland_natural_vegetation_mosaic",
        "15": "snow_and_ice",
        "16": "barren_or_sparsely_vegetated"
    }
    for code in selected_classes:
        if code not in modis_landcover_map:
            continue
        label = modis_landcover_map[code]
        binary = lc_img.eq(int(code))
        fetch_with_xee(binary, f"{code}_{label}")
