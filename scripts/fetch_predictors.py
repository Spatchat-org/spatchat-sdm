import os, time, json
import ee
import geemap
import math
import rasterio
from rasterio.merge import merge
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
grid_transform = from_bounds(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer,
    x_size, y_size,
)

print(f"📍 Loaded {len(df)} points → region: {min_lat-buffer},{min_lon-buffer} → {max_lat+buffer},{max_lon+buffer}")
print(f"🗺  Grid: {x_size}×{y_size} @ {RES} m in EPSG:4326")



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
name_to_code = {label: str(code) for code, label in modis_landcover_map.items()}

# -----------------------------------------------------------------------------
# Which layers + one‑hot codes the UI set (via os.environ)
# -----------------------------------------------------------------------------
layers = os.environ.get("SELECTED_LAYERS", "").split(",")
raw    = os.environ.get("SELECTED_LANDCOVER_CLASSES", "")
labels = [c for c in raw.split(",") if c]
# convert each snake_case label back to its integer code
codes  = [name_to_code[c] for c in labels if c in name_to_code]

# -----------------------------------------------------------------------------
# Earth Engine sources
# -----------------------------------------------------------------------------
sources = {
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "slope":     ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("slope"),
    "aspect":    ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("aspect"),
    # ← Here’s the only change: use the v061 MOD13Q1 collection
    "ndvi": (ee.ImageCollection("MODIS/061/MOD13Q1")
                 .filterDate("2022-01-01", "2024-01-01")
                 .select("NDVI")
                 .mean()),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1").first(),
}
for i in range(1, 20):
    sources[f"bio{i}"] = ee.Image("WORLDCLIM/V1/BIO").select(f"bio{str(i).zfill(2)}")

# -----------------------------------------------------------------------------
# Export scales (native) for each predictor
# -----------------------------------------------------------------------------
EXPORT_SCALES = {
    **{f"bio{i}": 1000 for i in range(1,20)},  # BIOCLIM at 1 km
    "elevation": 30,
    "slope":     30,
    "aspect":    30,
    "ndvi":      250,  # MODIS NDVI is 250 m
    "landcover": 500    # MODIS landcover is 500 m
}

# -----------------------------------------------------------------------------
# Prepare folders
# -----------------------------------------------------------------------------
os.makedirs("predictor_rasters/raw",  exist_ok=True)
os.makedirs("predictor_rasters/wgs84", exist_ok=True)

# -----------------------------------------------------------------------------
# Chunked export-and-align (never exceeds EE’s 50 MB limit)
# -----------------------------------------------------------------------------
def export_and_align(img: ee.Image, name: str):
    raw_dir = "predictor_rasters/raw"
    os.makedirs(raw_dir, exist_ok=True)
    raw_fp = os.path.join(raw_dir, f"{name}.tif")
    out_fp = f"predictor_rasters/wgs84/{name}.tif"
    scale = EXPORT_SCALES.get(name, RES)

    # 1) Reproject EE image to EPSG:4326 at its native scale
    ee_img = img.reproject(crs="EPSG:4326", scale=scale)

    # 2) Decide between single‐tile vs chunked export
    max_bytes = 50 * 1024**2
    bytes_per_pixel = 2
    total_pixels = x_size * y_size

    if total_pixels * bytes_per_pixel <= max_bytes:
        print(f"📥 Exporting '{name}' in one tile…")
        geemap.ee_export_image(
            ee_img.clip(region),
            filename=raw_fp,
            scale=scale,
            region=region,
            crs="EPSG:4326",
            file_per_band=False,
            timeout=600,
        )
    else:
        # chop into fixed 2000×2000 px tiles
        CHUNK_PX = 2000
        lon_step = (max_lon + buffer - (min_lon - buffer)) / math.ceil(x_size/CHUNK_PX)
        lat_step = (max_lat + buffer - (min_lat - buffer)) / math.ceil(y_size/CHUNK_PX)
        tiles = []
        print(f"📥 Exporting '{name}' in chunks…")
        nx = math.ceil((max_lon + buffer - (min_lon - buffer)) / lon_step)
        ny = math.ceil((max_lat + buffer - (min_lat - buffer)) / lat_step)
        for i in range(nx):
            for j in range(ny):
                sub_region = ee.Geometry.BBox(
                    min_lon - buffer + i * lon_step,
                    min_lat - buffer + j * lat_step,
                    min(min_lon - buffer + (i+1) * lon_step, max_lon + buffer),
                    min(min_lat - buffer + (j+1) * lat_step, max_lat + buffer)
                )
                tile_fp = os.path.join(raw_dir, f"{name}_tile_{i}_{j}.tif")
                geemap.ee_export_image(
                    ee_img.clip(sub_region),
                    filename=tile_fp,
                    scale=scale,
                    region=sub_region,
                    crs="EPSG:4326",
                    file_per_band=False,
                    timeout=600,
                )
                tiles.append(tile_fp)

        # mosaic all tiles
        print(f"🔀 Mosaicking {len(tiles)} tiles…")
        srcs = [rasterio.open(t) for t in tiles]
        mosaic_arr, mosaic_trans = merge(srcs)
        profile = srcs[0].profile.copy()
        profile.update({
            "height": mosaic_arr.shape[1],
            "width":  mosaic_arr.shape[2],
            "transform": mosaic_trans,
            "crs": CRS.from_epsg(4326),
            "count": 1,
        })
        with rasterio.open(raw_fp, "w", **profile) as dst:
            dst.write(mosaic_arr[0], 1)
        for s in srcs:
            s.close()
        for t in tiles:
            os.remove(t)

    # 3) Align the raw .tif to your study grid
    print(f"🌐 Aligning → {out_fp}")
    with rasterio.open(raw_fp) as src:
        arr = src.read(1)
        dst = np.empty((y_size, x_size), dtype=arr.dtype)
        reproject(
            source=arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid_transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
        profile = src.profile.copy()
        profile.update({
            "crs":       crs,
            "transform": grid_transform,
            "height":    y_size,
            "width":     x_size,
            "count":     1,
        })
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        with rasterio.open(out_fp, "w", **profile) as dstf:
            dstf.write(dst, 1)

    print(f"✅ '{name}' ready.")


# -----------------------------------------------------------------------------
# 1) Export each regular predictor
# -----------------------------------------------------------------------------

layers = [l for l in layers if l.strip()]
for name in layers:
    if name == "landcover":
        continue
    try:
        export_and_align(sources[name], name)
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
