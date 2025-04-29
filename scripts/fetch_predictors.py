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

# --- Optional: get landcover class IDs to encode ---
landcover_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES', '')
landcover_classes = [int(x) for x in landcover_classes.split(',') if x.strip().isdigit()]

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

# --- Clip region (Oregon-ish default) ---
default_bbox = ee.Geometry.BBox(-125, 40, -115, 50)

# --- Export selected layers ---
for layer_name in selected_layers:
    if layer_name == "landcover":
        print("🌱 One-hot encoding selected MODIS landcover classes...")
        lc_image = layer_sources["landcover"].clip(default_bbox)

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
                    region=default_bbox,
                    file_per_band=False,
                    timeout=300
                )
                print(f"✅ Saved class {class_id}: {out_file}")
            except Exception as e:
                print(f"❗ Failed to save {class_id} ({class_name}): {e}")

        continue  # skip regular export of the raw landcover layer

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
