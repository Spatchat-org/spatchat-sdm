import ee
import os
import geemap.foliumap as geemap

# --- Initialize Earth Engine ---
try:
    ee.Initialize()
except Exception:
    print("🔑 GEE Authentication required... Getting dynamic token...")
    token = geemap.get_ee_token()
    ee.Initialize(token=token)

# --- Output Folder ---
output_folder = "predictor_rasters"
os.makedirs(output_folder, exist_ok=True)

# --- Read Selections from Environment ---
selected_layers = os.environ.get('SELECTED_LAYERS', '').split(',')

# --- Available Layers ---
available_layers = {
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "slope": ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select('slope'),
    "aspect": ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select('aspect'),
    "ndvi": ee.ImageCollection("MODIS/006/MOD13A2").select('NDVI').first().divide(10000),
    "precipitation": ee.ImageCollection("WORLDCLIM/V1/BIO").select('bio12').first(),
    "mean_temperature": ee.ImageCollection("WORLDCLIM/V1/BIO").select('bio1').first().divide(10),
    "min_temperature": ee.ImageCollection("WORLDCLIM/V1/BIO").select('bio6').first().divide(10),
    "max_temperature": ee.ImageCollection("WORLDCLIM/V1/BIO").select('bio5').first().divide(10),
    "landcover": ee.ImageCollection("MODIS/006/MCD12Q1").select('LC_Type1').first(),
}

# --- Fetch Selected Layers ---
for layer in selected_layers:
    if layer in available_layers:
        print(f"📥 Fetching {layer}...")
        path = os.path.join(output_folder, f"{layer}.tif")
        geemap.ee_export_image(
            image=available_layers[layer],
            filename=path,
            scale=500 if layer == "ndvi" else 30,  # ndvi = MODIS = coarser
            region=ee.Geometry.BBox(-125, 25, -66, 50)  # USA bounding box; you can change
        )
        print(f"✅ {layer} saved!")

print("🎯 Done fetching predictors.")
