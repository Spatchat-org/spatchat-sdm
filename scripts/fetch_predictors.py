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

# --- Parameters ---
selected_layers = os.environ.get('SELECTED_LAYERS', '')
selected_layers = selected_layers.split(',') if selected_layers else []

# --- Define Layer Sources ---
layer_sources = {
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "slope": ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select('slope'),
    "aspect": ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select('aspect'),
    "ndvi": ee.ImageCollection("MODIS/006/MOD13A2").select('NDVI').first(),
    "precipitation": ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").mean(),
    "mean_temperature": ee.ImageCollection("NASA/ORNL/DAYMET_V4").select('tavg').mean(),
    "min_temperature": ee.ImageCollection("NASA/ORNL/DAYMET_V4").select('tmin').mean(),
    "max_temperature": ee.ImageCollection("NASA/ORNL/DAYMET_V4").select('tmax').mean(),
    "landcover": ee.ImageCollection("MODIS/006/MCD12Q1").select('LC_Type1').first()
}

# --- Default Study Area Bounding Box ---
default_bbox = ee.Geometry.BBox(-180, -60, 180, 85)

# --- Create output folder if missing ---
os.makedirs("predictor_rasters", exist_ok=True)

# --- Fetch Each Selected Layer ---
for layer_name in selected_layers:
    if layer_name not in layer_sources:
        print(f"⚠️ Layer {layer_name} not recognized.")
        continue

    image = layer_sources[layer_name]
    out_file = f"predictor_rasters/{layer_name}.tif"

    print(f"📥 Fetching {layer_name}...")

    try:
        geemap.ee_export_image(
            image=image.clip(default_bbox),
            filename=out_file,
            scale=1000,          # 1km resolution
            region=default_bbox,
            file_per_band=False,
            timeout=600           # increase if necessary
        )
        print(f"✅ Saved {layer_name} to {out_file}")
    except Exception as e:
        print(f"❗ Failed to fetch {layer_name}: {e}")
