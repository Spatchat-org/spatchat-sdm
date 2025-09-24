# scripts/fetch_predictors.py
import os
import time
import json
import math
import rasterio
import numpy as np
import pandas as pd

import ee
import geemap

from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS

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
min_lat, max_lat = df.latitude.min(), df.latitude.max()
min_lon, max_lon = df.longitude.min(), df.longitude.max()
buffer = 0.25  # degrees

region = ee.Geometry.BBox(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer
)

# target CRS/grid
crs = CRS.from_epsg(4326)  # WGS84
RES = 30  # 30 m target resolution (approximate in lat/lon below)

# Approximate meters-to-degrees scaling for a regular lat/lon grid at 30 m
# 1 deg lon ~ 111,320 m at the equator; 1 deg lat ~ 110,540 m
x_size = int((max_lon + buffer - (min_lon - buffer)) * (111320 / RES))
y_size = int((max_lat + buffer - (min_lat - buffer)) * (110540 / RES))

# Avoid zero-sized grids for extremely tiny extents
x_size = max(32, x_size)
y_size = max(32, y_size)

grid_transform = from_bounds(
    min_lon - buffer, min_lat - buffer,
    max_lon + buffer, max_lat + buffer,
    x_size, y_size,
)

print(f"📍 Loaded {len(df)} points → region: "
      f"{min_lat-buffer:.5f},{min_lon-buffer:.5f} → {max_lat+buffer:.5f},{max_lon+buffer:.5f}")
print(f"🗺  Grid: {x_size}×{y_size} @ ~{RES} m in EPSG:4326")

# -----------------------------------------------------------------------------
# MODIS landcover code → snake_case label map
# -----------------------------------------------------------------------------
modis_landcover_map = {
    0:"water",1:"evergreen_needleleaf_forest",2:"evergreen_broadleaf_forest",
    3:"deciduous_needleleaf_forest",4:"deciduous_broadleaf_forest",
    5:"mixed_forest",6:"closed_shrublands",7:"open_shrublands",
    8:"woody_savannas",9:"savannas",10:"grasslands",11:"permanent_wetlands",
    12:"croplands",13:"urban_and_built_up",14:"cropland_natural_vegetation_mosaic",
    15:"snow_and_ice",16:"barren_or_sparsely_vegetated"
}
name_to_code = {label: str(code) for code, label in modis_landcover_map.items()}

# -----------------------------------------------------------------------------
# Which layers + one-hot codes the UI set (via os.environ)
# -----------------------------------------------------------------------------
layers = [l for l in os.environ.get("SELECTED_LAYERS", "").split(",") if l.strip()]
raw    = os.environ.get("SELECTED_LANDCOVER_CLASSES", "")
labels = [c for c in raw.split(",") if c]
codes  = [name_to_code[c] for c in labels if c in name_to_code]

print(f"🎯 Requested layers: {layers or 'none'}")
print(f"🌱 Requested landcover classes: {labels or 'none'} (codes: {codes or 'none'})")

# -----------------------------------------------------------------------------
# Earth Engine sources
# -----------------------------------------------------------------------------
sources = {
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "slope":     ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("slope"),
    "aspect":    ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("aspect"),
    # MODIS NDVI v061, 250 m; mean over 2022
    "ndvi": (ee.ImageCollection("MODIS/061/MOD13Q1")
                 .filterDate("2022-01-01", "2024-01-01")
                 .select("NDVI")
                 .mean()),
    # MODIS landcover v061, 500 m
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
    "landcover": 500   # MODIS landcover is 500 m
}

# -----------------------------------------------------------------------------
# Prepare folders
# -----------------------------------------------------------------------------
os.makedirs("predictor_rasters/raw",  exist_ok=True)
os.makedirs("predictor_rasters/wgs84", exist_ok=True)

# -----------------------------------------------------------------------------
# Chunked export-and-align (never exceeds EE’s 50 MB limit)
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
    bytes_per_pixel = 2  # uint16/float16; geemap uses GeoTIFF defaults—safe estimate
    total_pixels = x_size * y_size

    if total_pixels * bytes_per_pixel <= max_bytes:
        print(f"📥 Exporting '{name}' in one tile… (scale={scale} m)")
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
        # chop into fixed 2000×2000 px tiles
        CHUNK_PX = 2000
        lon_span = (max_lon + buffer) - (min_lon - buffer)
        lat_span = (max_lat + buffer) - (min_lat - buffer)
        nx = max(1, math.ceil(x_size / CHUNK_PX))
        ny = max(1, math.ceil(y_size / CHUNK_PX))
        lon_step = lon_span / nx
        lat_step = lat_span / ny
        tiles = []
        print(f"📥 Exporting '{name}' in {nx*ny} chunks… (scale={scale} m)")
        for i in range(nx):
            for j in range(ny):
                sub_region = ee.Geometry.BBox(
                    (min_lon - buffer) + i * lon_step,
                    (min_lat - buffer) + j * lat_step,
                    min((min_lon - buffer) + (i+1) * lon_step, max_lon + buffer),
                    min((min_lat - buffer) + (j+1) * lat_step, max_lat + buffer)
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
        for s in srcs:
            s.close()
        profile.update({
            "height": mosaic_arr.shape[1],
            "width":  mosaic_arr.shape[2],
            "transform": mosaic_trans,
            "crs": CRS.from_epsg(4326),
            "count": 1,
        })
        with rasterio.open(raw_fp, "w", **profile) as dst:
            dst.write(mosaic_arr[0], 1)
        for t in tiles:
            try:
                os.remove(t)
            except OSError:
                pass

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
for name in layers:
    if name == "landcover":
        continue
    try:
        export_and_align(sources[name], name)
    except Exception as e:
        print(f"❌ Failed processing '{name}': {e}")

# -----------------------------------------------------------------------------
# 2) One-hot encode selected landcover classes
# -----------------------------------------------------------------------------
if "landcover" in layers and codes:
    lc = sources["landcover"]
    print("🌱 One-hot encoding MODIS landcover for requested classes…")
    for code in codes:
        if not code.isdigit():
            continue
        ci    = int(code)
        label = modis_landcover_map.get(ci, f"class_{ci}")
        try:
            binary = lc.eq(ci)
            export_and_align(binary, f"{ci}_{label}")
        except Exception as e:
            print(f"❌ Failed processing landcover class '{label}' ({ci}): {e}")

print("🏁 Done.")
