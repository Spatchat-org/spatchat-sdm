import ee
import json
import os
import geemap
import pandas as pd
import time

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

# --- Compute bounding box (expanded) ---
min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
buffer = 0.25
region = ee.Geometry.BBox(min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)

# --- Ensure output folder exists ---
os.makedirs("predictor_rasters", exist_ok=True)

# --- Get selected layers from environment variable ---
selected_layers = os.environ.get('SELECTED_LAYERS', '').split(',')
selected_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES', '').split(',')

# --- Define Earth Engine layer sources ---
layer_sources = {
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "slope": ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select('slope'),
    "aspect": ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select('aspect'),
    "ndvi": ee.ImageCollection("MODIS/061/MOD13A2").select('NDVI').mean(),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select('LC_Type1').first()
}
for i in range(1, 20):
    layer_sources[f"bio{i}"] = ee.Image("WORLDCLIM/V1/BIO").select(f"bio{str(i).zfill(2)}")

# --- Export regular layers ---
for name in selected_layers:
    if name == "landcover":
        continue  # handled below
    if name not in layer_sources:
        print(f"⚠️ Layer {name} not recognized.")
        continue
    image = layer_sources[name]
    out_file = f"predictor_rasters/{name}.tif"
    print(f"📥 Fetching {name}...")
    try:
        geemap.ee_export_image(
            image.clip(region),
            filename=out_file,
            scale=1000,
            region=region,
            file_per_band=False,
            timeout=600
        )
        print(f"✅ Saved {name} to {out_file}")
    except Exception as e:
        print(f"❗ Failed to fetch {name}: {e}")

# --- One-hot encode selected MODIS landcover classes ---
if "landcover" in selected_layers and selected_classes:
    landcover = layer_sources["landcover"]
    print("🌱 One-hot encoding selected MODIS landcover classes...")

    # Load static MODIS class name map
    with open("modis_landcover_code_name.json") as f:
        modis_map = json.load(f)

    for code in selected_classes:
        if not code.isdigit() or int(code) >= 250:
            continue
        code_int = int(code)
        name = modis_map.get(str(code_int), f"class_{code_int}")
        binary_image = landcover.eq(code_int)
        out_file = f"predictor_rasters/{code_int}_{name}.tif"
        try:
            geemap.ee_export_image(
                binary_image.clip(region),
                filename=out_file,
                scale=1000,
                region=region,
                file_per_band=False,
                timeout=600
            )
            print(f"✅ Saved class {code_int}: {out_file}")
        except Exception as e:
            print(f"❗ Failed class {code_int}: {e}")
