import ee
import json
import os
import geemap
import pandas as pd

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

# --- Optional: get landcover class IDs to encode ---
landcover_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES', '')
landcover_classes = [int(x) for x in landcover_classes.split(',') if x.strip().isdigit()]

# --- Compute dynamic bounding box from presence points ---
df = pd.read_csv("predictor_rasters/presence_points.csv")
min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
buffer = 0.25  # degree buffer (~25–28 km)
dynamic_bbox = ee.Geometry.BBox(
    min_lon - buffer,
    min_lat - buffer,
    max_lon + buffer,
    max_lat + buffer
)

# --- Define layer sources ---
bio_image = ee.Image("WORLDCLIM/V1/BIO")
terrain = ee.Terrain.products(ee.Image("USGS/SRTMGL1_003"))

layer_sources = {f"bio{i}": bio_image.select(f"bio{str(i).zfill(2)}") for i in range(1, 20)}
layer_sources.update({
    "slope": terrain.select("slope"),
    "aspect": terrain.select("aspect"),
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "ndvi": ee.ImageCollection("MODIS/061/MOD13A2").select('NDVI').first(),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select('LC_Type1').first(),
})

# --- Landcover class labels ---
landcover_labels = {
    0: "water",
    1: "evergreen_needleleaf_forest",
    2: "evergreen_broadleaf_forest",
    3: "deciduous_needleleaf_forest",
    4: "deciduous_broadleaf_forest",
    5: "mixed_forest",
    6: "closed_shrublands",
    7: "open_shrublands",
    8: "woody_savannas",
    9: "savannas",
    10: "grasslands",
    11: "permanent_wetlands",
    12: "croplands",
    13: "urban_and_built_up",
    14: "cropland_natural_vegetation_mosaic",
    15: "snow_and_ice",
    16: "barren_or_sparsely_vegetated"
}

# --- Export selected layers ---
for layer_name in selected_layers:
    if layer_name == "landcover":
        print("🌱 One-hot encoding selected MODIS landcover classes...")
        lc_image = layer_sources["landcover"].clip(dynamic_bbox)

        for class_id in landcover_classes:
            if class_id not in landcover_labels:
                print(f"⏩ Skipping invalid landcover class: {class_id}")
                continue

            class_name = landcover_labels[class_id]
            binary_mask = lc_image.eq(class_id)
            out_file = f"predictor_rasters/{class_id}_{class_name}.tif"

            try:
                geemap.ee_export_image(
                    binary_mask,
                    filename=out_file,
                    scale=500,
                    region=dynamic_bbox,
                    file_per_band=False,
                    timeout=300
                )
                print(f"✅ Saved class {class_id}: {out_file}")
            except Exception as e:
                print(f"❗ Failed to save {class_id} ({class_name}): {e}")

        continue  # skip raw landcover export

    if layer_name not in layer_sources:
        print(f"⚠️ Layer {layer_name} not recognized.")
        continue

    image = layer_sources[layer_name]
    out_file = f"predictor_rasters/{layer_name}.tif"

    print(f"📥 Fetching {layer_name}...")

    try:
        geemap.ee_export_image(
            image.clip(dynamic_bbox),
            filename=out_file,
            scale=1000,
            region=dynamic_bbox,
            file_per_band=False,
            timeout=600
        )
        print(f"✅ Saved {layer_name} to {out_file}")
    except Exception as e:
        print(f"❗ Failed to fetch {layer_name}: {e}")
