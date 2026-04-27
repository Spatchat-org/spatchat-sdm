# workflow/fetch_predictors.py
import os
import time
import json
import math
import base64
import sys
import zipfile
import urllib.request
import rasterio
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import ee
import geemap

from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS

load_dotenv()
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# -----------------------------------------------------------------------------
# Authenticate Earth Engine (service-account JSON/file or persistent creds)
# -----------------------------------------------------------------------------
def _parse_service_account_json(raw_text):
    text = (raw_text or "").strip()
    if not text:
        return None
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    candidates = [text]
    if "\\n" in text:
        candidates.append(text.replace("\\n", "\n"))
    try:
        decoded = base64.b64decode(text).decode("utf-8", errors="strict").strip()
        candidates.append(decoded)
    except Exception:
        pass

    for candidate in candidates:
        if not candidate or not candidate.lstrip().startswith("{"):
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _load_service_account_from_env():
    for raw in (os.environ.get("GEE_SERVICE_ACCOUNT_JSON", ""), os.environ.get("GEE_SERVICE_ACCOUNT", "")):
        raw = (raw or "").strip()
        if not raw:
            continue
        maybe_path = raw.strip('"').strip("'")
        if os.path.isfile(maybe_path):
            with open(maybe_path, "r", encoding="utf-8") as f:
                return json.load(f)
        svc = _parse_service_account_json(raw)
        if svc:
            return svc
    for path_var in ("GEE_SERVICE_ACCOUNT_FILE", "GOOGLE_APPLICATION_CREDENTIALS"):
        path_val = (os.environ.get(path_var, "") or "").strip().strip('"').strip("'")
        if path_val and os.path.isfile(path_val):
            with open(path_val, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _initialize_ee():
    project = (os.environ.get("GEE_PROJECT", "") or "").strip() or None
    errors = []

    svc = _load_service_account_from_env()
    if svc is not None:
        try:
            client_email = (svc.get("client_email") or "").strip()
            private_key = (svc.get("private_key") or "").strip()
            if not client_email or not private_key:
                raise RuntimeError("Service account JSON missing client_email/private_key.")
            creds = ee.ServiceAccountCredentials(client_email, key_data=json.dumps(svc))
            ee.Initialize(credentials=creds, project=project)
            print("Earth Engine authenticated via service account.")
            return
        except Exception as exc:
            errors.append(f"service-account auth failed: {exc}")

    try:
        ee.Initialize(project=project)
        print("Earth Engine initialized from persistent credentials.")
        return
    except Exception as exc:
        errors.append(f"persistent credentials failed: {exc}")

    raise RuntimeError(" | ".join(errors))


EE_READY = True
EE_INIT_ERROR = None
try:
    _initialize_ee()
except Exception as exc:
    EE_READY = False
    EE_INIT_ERROR = str(exc)
    print(f"[warning] Earth Engine unavailable, fallback mode may be used: {EE_INIT_ERROR}")

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

region = None
if EE_READY:
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

WORLDCLIM_BASE_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/"
WORLDCLIM_CACHE_DIR = "predictor_rasters/worldclim_cache"
os.makedirs(WORLDCLIM_CACHE_DIR, exist_ok=True)


def _download_if_missing(url: str, dst_fp: str):
    if os.path.exists(dst_fp):
        return
    print(f"Downloading {url} -> {dst_fp}")
    urllib.request.urlretrieve(url, dst_fp)


def _worldclim_member_for_layer(layer_name: str):
    if layer_name.startswith("bio"):
        idx = int(layer_name.replace("bio", ""))
        return ("wc2.1_10m_bio.zip", f"wc2.1_10m_bio_{idx}.tif")
    if layer_name in {"elevation", "slope", "aspect"}:
        return ("wc2.1_10m_elev.zip", "wc2.1_10m_elev.tif")
    return (None, None)


def _ensure_worldclim_tif(layer_name: str):
    zip_name, member_name = _worldclim_member_for_layer(layer_name)
    if not zip_name or not member_name:
        raise ValueError(f"Layer '{layer_name}' is not available from WorldClim fallback.")
    zip_fp = os.path.join(WORLDCLIM_CACHE_DIR, zip_name)
    _download_if_missing(WORLDCLIM_BASE_URL + zip_name, zip_fp)
    out_member_fp = os.path.join(WORLDCLIM_CACHE_DIR, member_name)
    if os.path.exists(out_member_fp):
        return out_member_fp
    with zipfile.ZipFile(zip_fp, "r") as zf:
        zf.extract(member_name, WORLDCLIM_CACHE_DIR)
    return out_member_fp


def _align_local_raster_to_grid(src_fp: str, out_name: str, resampling=Resampling.nearest):
    out_fp = f"predictor_rasters/wgs84/{out_name}.tif"
    with rasterio.open(src_fp) as src:
        src_arr = src.read(1).astype(np.float32)
        dst = np.full((y_size, x_size), np.nan, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid_transform,
            dst_crs=crs,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        profile = src.profile.copy()
        profile.update({
            "crs": crs,
            "transform": grid_transform,
            "height": y_size,
            "width": x_size,
            "count": 1,
            "dtype": "float32",
            "nodata": np.nan,
        })
        with rasterio.open(out_fp, "w", **profile) as dstf:
            dstf.write(dst.astype(np.float32), 1)
    return out_fp


def _write_slope_aspect_from_elevation(elev_fp: str, need_slope: bool, need_aspect: bool):
    if not (need_slope or need_aspect):
        return
    with rasterio.open(elev_fp) as src:
        elev = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    lat_center = float((min_lat + max_lat) / 2.0)
    meters_per_deg_lon = 111320.0 * max(1e-6, abs(math.cos(math.radians(lat_center))))
    meters_per_deg_lat = 110540.0
    dx = abs(grid_transform.a) * meters_per_deg_lon
    dy = abs(grid_transform.e) * meters_per_deg_lat
    dy = max(dy, 1e-6)
    dx = max(dx, 1e-6)

    gy, gx = np.gradient(elev, dy, dx)
    slope = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy))).astype(np.float32)
    aspect = np.degrees(np.arctan2(gx, -gy))
    aspect = np.where(aspect < 0, aspect + 360.0, aspect).astype(np.float32)

    profile.update({"dtype": "float32", "nodata": np.nan, "count": 1})
    if need_slope:
        with rasterio.open("predictor_rasters/wgs84/slope.tif", "w", **profile) as dst:
            dst.write(slope, 1)
    if need_aspect:
        with rasterio.open("predictor_rasters/wgs84/aspect.tif", "w", **profile) as dst:
            dst.write(aspect, 1)


if not EE_READY:
    fallback_allowed = {f"bio{i}" for i in range(1, 20)} | {"elevation", "slope", "aspect"}
    unsupported = [l for l in layers if l not in fallback_allowed]
    if "landcover" in layers:
        unsupported.append("landcover")
    if "ndvi" in layers:
        unsupported.append("ndvi")
    if codes:
        unsupported.append("landcover_classes")

    if unsupported:
        raise RuntimeError(
            "Earth Engine is unavailable and requested layers need Earth Engine: "
            + ", ".join(sorted(set(unsupported)))
            + f". Underlying EE error: {EE_INIT_ERROR}"
        )

    need_slope = "slope" in layers
    need_aspect = "aspect" in layers
    need_elev = ("elevation" in layers) or need_slope or need_aspect

    for name in layers:
        if name in {"slope", "aspect"}:
            continue
        src_fp = _ensure_worldclim_tif(name)
        rs = Resampling.bilinear if name == "elevation" else Resampling.nearest
        _align_local_raster_to_grid(src_fp, name, resampling=rs)
        print(f"Fallback prepared layer: {name}")

    if need_elev and not os.path.exists("predictor_rasters/wgs84/elevation.tif"):
        src_fp = _ensure_worldclim_tif("elevation")
        _align_local_raster_to_grid(src_fp, "elevation", resampling=Resampling.bilinear)
        print("Fallback prepared layer: elevation")

    if need_slope or need_aspect:
        _write_slope_aspect_from_elevation(
            elev_fp="predictor_rasters/wgs84/elevation.tif",
            need_slope=need_slope,
            need_aspect=need_aspect,
        )
        if need_slope:
            print("Fallback prepared layer: slope")
        if need_aspect:
            print("Fallback prepared layer: aspect")

    print("Fallback download/processing complete.")
    sys.exit(0)



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
