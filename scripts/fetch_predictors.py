import os, time, json
import ee
import geemap
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import pandas as pd

# -----------------------------------------------------------------------------
# Authenticate Earth Engine via Service Account
# -----------------------------------------------------------------------------
service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)
print("✅ Earth Engine authenticated successfully!")

# -----------------------------------------------------------------------------
# Wait for presence_points.csv
# -----------------------------------------------------------------------------
csv_path = "inputs/presence_points.csv"
for i in range(5):
    if os.path.exists(csv_path):
        break
    print(f"⏳ Waiting for presence_points.csv… ({i+1}/5)")
    time.sleep(1)
if not os.path.exists(csv_path):
    raise FileNotFoundError("❗ inputs/presence_points.csv not found.")

# -----------------------------------------------------------------------------
# Load points & define study-area region + 30 m lat/lon grid
# -----------------------------------------------------------------------------
df = pd.read_csv(csv_path)
# we know the app has already renamed the columns:
min_lat, max_lat = df.latitude.min(), df.latitude.max()
min_lon, max_lon = df.longitude.min(), df.longitude.max()
buffer = 0.25
region = ee.Geometry.BBox(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer
)

# target CRS/grid
crs = CRS.from_epsg(4326)
RES = 30  # 30 m target resolution
x_size = int((max_lon + buffer - (min_lon - buffer)) * (111320/30))  # approximate
y_size = int((max_lat + buffer - (min_lat - buffer)) * (110540/30))
transform = from_bounds(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer,
    x_size, y_size
)

print(f"📍 Loaded {len(df)} points → region: {min_lat-buffer},{min_lon-buffer} → {max_lat+buffer},{max_lon+buffer}")
print(f"🗺  Grid: {x_size}×{y_size} @ {RES} m in EPSG:4326")

# -----------------------------------------------------------------------------
# Which layers + one-hot codes the UI set (via os.environ)
# -----------------------------------------------------------------------------
layers = os.environ.get('SELECTED_LAYERS','').split(',')
raw    = os.environ.get('SELECTED_LANDCOVER_CLASSES','')
labels = [c for c in raw.split(',') if c]
# convert each snake_case label back to its integer code
codes  = [ str(name_to_code[c]) for c in labels if c in name_to_code ]

# -----------------------------------------------------------------------------
# Earth Engine sources
# -----------------------------------------------------------------------------
sources = {
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "slope":     ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("slope"),
    "aspect":    ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("aspect"),
    "ndvi":      (ee.ImageCollection("MODIS/006/MOD13Q1")
            .filterDate("2022-01-01", "2024-01-01")
            .select("NDVI")
            .mean()),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1").first(),
}
for i in range(1,20):
    sources[f"bio{i}"] = ee.Image("WORLDCLIM/V1/BIO").select(f"bio{str(i).zfill(2)}")

# -----------------------------------------------------------------------------
# Modis landcover code → snake_case label map
# -----------------------------------------------------------------------------
modis_landcover_map = {
    0:"water",1:"evergreen_needleleaf_forest",2:"evergreen_broadleaf_forest",
    3:"deciduous_needleleaf_forest",4:"deciduous_broadleaf_forest",
    5:"mixed_forest",6:"closed_shrublands",7:"open_shrublands",
    8:"woody_savannas",9:"savannas",10:"grasslands",11:"permanent_wetlands",
    12:"croplands",13:"urban_and_built_up",14:"cropland_natural_vegetation_mosaic",
    15:"snow_and_ice",16:"barren_or_sparsely_vegetated"
}
# reverse mapping of label→code
name_to_code = { label: code for code, label in modis_landcover_map.items() }

# -----------------------------------------------------------------------------
# Export scales (native) for each predictor
# -----------------------------------------------------------------------------
EXPORT_SCALES = {
    **{f"bio{i}": 1000 for i in range(1,20)},  # BIOCLIM at 1 km
    "elevation": 30,
    "slope":     30,
    "aspect":    30,
    "ndvi":      250,  # MODIS NDVI is 1 km
    "landcover": 500    # MODIS landcover is 500 m
}

# -----------------------------------------------------------------------------
# Prepare folders
# -----------------------------------------------------------------------------
os.makedirs("predictor_rasters/raw",  exist_ok=True)
os.makedirs("predictor_rasters/wgs84", exist_ok=True)

# -----------------------------------------------------------------------------
# Function: export at native scale, then reproject onto 30 m lat/lon grid
# -----------------------------------------------------------------------------
def export_and_align(img: ee.Image, name: str):
    raw = f"predictor_rasters/raw/{name}.tif"
    out = f"predictor_rasters/wgs84/{name}.tif"
    scale = EXPORT_SCALES.get(name, RES)

    # 1) Force the source image into EPSG:4326 at its native scale:
    ee_img = img.reproject(crs="EPSG:4326", scale=scale)

    # 2) Export from Earth Engine:
    print(f"📥 Exporting '{name}' at {scale} m native…")
    geemap.ee_export_image(
        ee_img.clip(region),
        filename=raw,
        scale=scale,
        region=region,
        crs="EPSG:4326",
        file_per_band=False,
        timeout=600,
    )

    # 3) Reproject+resample to 30 m study-area grid
    with rasterio.open(raw) as src:
        src_arr = src.read(1)
        dst_arr = np.empty((y_size, x_size), dtype=src_arr.dtype)
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
        profile = src.profile.copy()
        profile.update({
            "crs":        crs,
            "transform":  transform,
            "height":     y_size,
            "width":      x_size,
            "count":      1,
        })
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(dst_arr, 1)

    print(f"🌐 Aligned → {out}")

# -----------------------------------------------------------------------------
# 1) Export each regular predictor
# -----------------------------------------------------------------------------
# Remove any empty entries
layers = [l for l in layers if l.strip()]

for name in layers:
    if name == "landcover":
        continue
    if name not in sources:
        print(f"⚠️ Skipping unknown layer '{name}'")
        continue

    raw_fp = f"predictor_rasters/raw/{name}.tif"
    out_fp = f"predictor_rasters/wgs84/{name}.tif"
    scale  = EXPORT_SCALES.get(name, RES)

    try:
        # Download only if missing
        if not os.path.exists(raw_fp):
            print(f"📥 Exporting '{name}' at {scale} m native…")
            geemap.ee_export_image(
                sources[name]
                  .reproject(crs="EPSG:4326", scale=scale)
                  .clip(region),
                filename=raw_fp,
                scale=scale,
                region=region,
                crs="EPSG:4326",
                file_per_band=False,
                timeout=600,
            )
        else:
            print(f"ℹ️ Raw file for '{name}' exists; skipping download.")

        # Reproject & align
        with rasterio.open(raw_fp) as src:
            arr = src.read(1)
            dst = np.empty((y_size, x_size), dtype=arr.dtype)
            reproject(
                source=arr,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.nearest,
            )
            profile = src.profile.copy()
            profile.update({
                "crs":       crs,
                "transform": transform,
                "height":    y_size,
                "width":     x_size,
                "count":     1,
            })
            with rasterio.open(out_fp, "w", **profile) as dst_file:
                dst_file.write(dst, 1)
        print(f"🌐 Aligned → {out_fp}")

    except Exception as e:
        print(f"❌ Failed processing '{name}': {e}")
        continue

# -----------------------------------------------------------------------------
# 2) One-hot encode landcover
# -----------------------------------------------------------------------------
if "landcover" in layers and codes:
    lc = sources["landcover"]
    print("🌱 One-hot encoding MODIS landcover…")
    for code in codes:
        if not code.isdigit(): 
            continue
        ci    = int(code)
        label = modis_landcover_map.get(ci, f"class_{ci}")
        binary = lc.eq(ci)
        export_and_align(binary, f"{ci}_{label}")
