import ee
import json
import os
import geemap

# --- Authenticate Earth Engine using Hugging Face Secret ---
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)
print("✅ Earth Engine authenticated successfully inside fetch_predictors.py!")

# --- Ensure output folder exists ---
os.makedirs("predictor_rasters", exist_ok=True)

# --- Get selected layers from environment variable ---
selected_layers = os.environ.get('SELECTED_LAYERS', '')
selected_layers = selected_layers.split(',') if selected_layers else []

# --- Define WorldClim BIOCLIM source ---
bio_image = ee.Image("WORLDCLIM/V1/BIO")

# --- Map of bioclim layer names to WorldClim bands ---
layer_sources = {f"bio{i}": bio_image.select(f"bio{str(i).zfill(2)}") for i in range(1, 20)}

# --- For slope and aspect ---
terrain = ee.Terrain.products(ee.Image("USGS/SRTMGL1_003"))

# Optional additional layers
layer_sources.update({
    "slope": terrain.select("slope"),
    "aspect": terrain.select("aspect"),
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "ndvi": ee.ImageCollection("MODIS/061/MOD13A2").select('NDVI').first(),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select('LC_Type1').first(),
})

# --- Clip region: Adjust as needed ---
default_bbox = ee.Geometry.BBox(-125, 40, -115, 50)  # Example: Oregon region

# --- Export selected layers ---
for layer_name in selected_layers:
    if layer_name not in layer_sources:
        print(f"⚠️ Layer {layer_name} not recognized.")
        continue

    image = layer_sources[layer_name]
    out_file = f"predictor_rasters/{layer_name}.tif"

    print(f"📥 Fetching {layer_name}...")

    try:
        geemap.ee_export_image(
            image.clip(default_bbox),
            filename=out_file,
            scale=1000,
            region=default_bbox,
            file_per_band=False,
            timeout=600
        )
        print(f"✅ Saved {layer_name} to {out_file}")
    except Exception as e:
        print(f"❗ Failed to fetch {layer_name}: {e}")
