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
    "ndvi":      250,  # MODIS NDVI is 250 m
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
            dst_transform=grid_transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
        profile = src.profile.copy()
        profile.update({
            "crs":        crs,
            "transform":  grid_transform,
            "height":     y_size,
            "width":      x_size,
            "count":      1,
        })
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(dst_arr, 1)

    print(f"🌐 Aligned → {out}")


# -------------------------------------------------------------------------
# Function: export at native scale with tiling if needed, then reproject onto 30 m grid
# -------------------------------------------------------------------------
def export_and_align(img: ee.Image, name: str):
    raw_dir = "predictor_rasters/raw"
    os.makedirs(raw_dir, exist_ok=True)
    raw_fp = os.path.join(raw_dir, f"{name}.tif")
    out_fp = f"predictor_rasters/wgs84/{name}.tif"
    scale = EXPORT_SCALES.get(name, RES)

    # Estimate max pixels allowed per HTTP export (50 MB limit)
    max_bytes = 50 * 1024**2
    bytes_per_pixel = 2  # assume uint16
    max_pixels = max_bytes // bytes_per_pixel
    total_pixels = x_size * y_size

    # Prepare the base ee.Image reprojection
    ee_img = img.reproject(crs="EPSG:4326", scale=scale)

    if total_pixels <= max_pixels:
        # single export
        print(f"📥 Exporting '{name}' at {scale} m (single tile)…")
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
        # tile the region
        n_tiles = math.ceil(total_pixels / max_pixels)
        n_cols = math.ceil(math.sqrt(n_tiles))
        n_rows = math.ceil(n_tiles / n_cols)
        lon_step = (max_lon + buffer - (min_lon - buffer)) / n_cols
        lat_step = (max_lat + buffer - (min_lat - buffer)) / n_rows

        tile_files = []
        print(f"📥 Exporting '{name}' in {n_cols}×{n_rows} tiles…")
        for i in range(n_cols):
            for j in range(n_rows):
                sub_min_lon = min_lon - buffer + i * lon_step
                sub_max_lon = sub_min_lon + lon_step
                sub_min_lat = min_lat - buffer + j * lat_step
                sub_max_lat = sub_min_lat + lat_step
                sub_region = ee.Geometry.BBox(
                    sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat
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
                tile_files.append(tile_fp)

        # mosaic tiles
        print(f"🔀 Mosaicking {len(tile_files)} tiles…")
        srcs = [rasterio.open(fp) for fp in tile_files]
        mosaic_arr, mosaic_trans = merge(srcs)
        profile = srcs[0].profile.copy()
        profile.update({
            "height": mosaic_arr.shape[1],
            "width":  mosaic_arr.shape[2],
            "transform": mosaic_trans
        })
        with rasterio.open(raw_fp, "w", **profile) as dst:
            dst.write(mosaic_arr)
        # optionally delete tile files
        for fp in tile_files:
            os.remove(fp)

    # now reproject & align to study grid
    print(f"🌐 Aligning → {out_fp}")
    with rasterio.open(raw_fp) as src:
        src_arr = src.read(1)
        dst_arr = np.empty((y_size, x_size), dtype=src_arr.dtype)
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid_transform,
            dst_crs=CRS.from_epsg(4326),
            resampling=Resampling.nearest,
        )
        profile = src.profile.copy()
        profile.update({
            "crs":       CRS.from_epsg(4326),
            "transform": grid_transform,
            "height":    y_size,
            "width":     x_size,
            "count":     1,
        })
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(dst_arr, 1)

    print(f"✅ '{name}' ready.")


# -----------------------------------------------------------------------------
# 1) Export each regular predictor
# -----------------------------------------------------------------------------
layers = [l for l in layers if l.strip()]

for name in layers:
    if name == "landcover":
        continue

    # set output paths
    raw_fp = f"predictor_rasters/raw/{name}.tif"
    out_fp = f"predictor_rasters/wgs84/{name}.tif"
    scale  = EXPORT_SCALES.get(name, RES)

    # Chunk-based export: split into fixed-size tiles (by pixels)
    if not os.path.exists(raw_fp):
        print(f"📥 Chunk-exporting '{name}' at {scale} m native…")
        # compute degrees per pixel
        west  = min_lon - buffer
        east  = max_lon + buffer
        south = min_lat - buffer
        north = max_lat + buffer
        deg_per_px_x = (east - west) / x_size
        deg_per_px_y = (north - south) / y_size
        CHUNK_PIXELS = 2000  # fixed chunk dimension
        chunk_dx = deg_per_px_x * CHUNK_PIXELS
        chunk_dy = deg_per_px_y * CHUNK_PIXELS
        tile_files = []
        nx = math.ceil((east - west) / chunk_dx)
        ny = math.ceil((north - south) / chunk_dy)
        for i in range(nx):
            for j in range(ny):
                # tile bounds
                w = west + i * chunk_dx
                e = min(west + (i+1) * chunk_dx, east)
                s = south + j * chunk_dy
                n = min(south + (j+1) * chunk_dy, north)
                tile_region = ee.Geometry.BBox(w, s, e, n)
                tile_fp = f"predictor_rasters/raw/{name}_tile_{i}_{j}.tif"
                print(f"  • exporting tile ({i},{j}) of '{name}'")
                geemap.ee_export_image(
                    sources[name]
                      .reproject(crs="EPSG:4326", scale=scale)
                      .clip(tile_region),
                    filename=tile_fp,
                    scale=scale,
                    region=tile_region,
                    crs="EPSG:4326",
                    file_per_band=False,
                    timeout=600,
                )
                tile_files.append(tile_fp)
        # merge tiles into single raw
        print(f"🔀 Merging {len(tile_files)} tiles for '{name}'...")
        srcs = [rasterio.open(fp) for fp in tile_files]
        mosaic, out_trans = merge(srcs)
        profile = srcs[0].profile.copy()
        profile.update({
            "height": mosaic.shape[1],
            "width":  mosaic.shape[2],
            "transform": out_trans,
        })
        with rasterio.open(raw_fp, "w", **profile) as dst:
            dst.write(mosaic)
        # cleanup tiles
        for src, fp in zip(srcs, tile_files):
            src.close()
            os.remove(fp)
    else:
        print(f"ℹ️ Raw file for '{name}' exists; skipping chunk export.")

        # Reproject & align
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
