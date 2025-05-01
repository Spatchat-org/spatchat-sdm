#!/usr/bin/env python3
import os
import json
import time

import ee
import geemap
import rasterio
import numpy as np
import pandas as pd
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# 1) Authenticate EE via Service Account
svc = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
creds = ee.ServiceAccountCredentials(svc['client_email'], key_data=json.dumps(svc))
ee.Initialize(creds)
print("✅ Earth Engine authenticated successfully inside fetch_predictors.py!")

# 2) Wait for the presence CSV
csv_path = "inputs/presence_points.csv"
for i in range(5):
    if os.path.exists(csv_path):
        break
    print(f"⏳ Waiting for presence_points.csv... ({i+1}s)")
    time.sleep(1)
if not os.path.exists(csv_path):
    raise FileNotFoundError("❗ inputs/presence_points.csv not found after 5s wait.")

# 3) Load presence points & compute bbox
df = pd.read_csv(csv_path)
if not {'latitude','longitude'}.issubset(df.columns):
    raise ValueError("❗ CSV must contain 'latitude' and 'longitude' columns.")
min_lat, max_lat = df.latitude.min(), df.latitude.max()
min_lon, max_lon = df.longitude.min(), df.longitude.max()
buffer = 0.25
west, south = min_lon - buffer, min_lat - buffer
east, north = max_lon + buffer, max_lat + buffer

# EarthEngine geometry (for clipping) and GeoJSON array (for export region)
region_ee = ee.Geometry.BBox(west, south, east, north)
geojson = region_ee.getInfo()['coordinates'][0]

print(f"📍 Loaded {len(df)} points; study‐area = {south},{west} → {north},{east}")

# 4) Build uniform 30 m grid in EPSG:4326
PIXEL_DEG = 30/111320.0
width  = int((east - west)/PIXEL_DEG)
height = int((north - south)/PIXEL_DEG)
crs    = CRS.from_epsg(4326)
transform = from_bounds(west, south, east, north, width, height)
print(f"🗺  Grid: {width}×{height} @30 m → EPSG:4326")

# 5) Prepare output dirs
os.makedirs("predictor_rasters/raw",  exist_ok=True)
os.makedirs("predictor_rasters/wgs84", exist_ok=True)

# 6) What user picked
selected_layers  = os.environ.get('SELECTED_LAYERS','').split(',')
selected_classes = os.environ.get('SELECTED_LANDCOVER_CLASSES','').split(',')

# 7) EE sources
layer_sources = {
    "elevation": ee.Image("USGS/SRTMGL1_003"),
    "slope":     ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("slope"),
    "aspect":    ee.Terrain.products(ee.Image("USGS/SRTMGL1_003")).select("aspect"),
    "ndvi":      ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI").mean(),
    "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1").first(),
}
for i in range(1,20):
    layer_sources[f"bio{i}"] = ee.Image("WORLDCLIM/V1/BIO")\
                                  .select(f"bio{str(i).zfill(2)}")

# 8) Native export scales
EXPORT_SCALES = {
    **{f"bio{i}": 30 for i in range(1,20)},
    "elevation": 30,
    "slope":     30,
    "aspect":    30,
    "ndvi":      1000,
    "landcover": 500,
}

# 9) Hard-coded MODIS landcover map
modis_landcover_map = {
     0:"water",1:"evergreen_needleleaf_forest",2:"evergreen_broadleaf_forest",
     3:"deciduous_needleleaf_forest",4:"deciduous_broadleaf_forest",
     5:"mixed_forest",6:"closed_shrublands",7:"open_shrublands",
     8:"woody_savannas",9:"savannas",10:"grasslands",11:"permanent_wetlands",
    12:"croplands",13:"urban_and_built_up",14:"cropland_natural_vegetation_mosaic",
    15:"snow_and_ice",16:"barren_or_sparsely_vegetated"
}

def export_and_reproject(ee_img, name):
    raw_path     = f"predictor_rasters/raw/{name}.tif"
    aligned_path = f"predictor_rasters/wgs84/{name}.tif"
    scale        = EXPORT_SCALES.get(name, 30)

    # --- Try single‐tile export, now passing geojson bbox list! ---
    print(f"\n📥 Exporting '{name}' at {scale} m native…")
    try:
        geemap.ee_export_image(
            ee_img.clip(region_ee),
            filename=raw_path,
            scale=scale,
            region=geojson,         # <<< use pure GeoJSON coords here!
            file_per_band=False,
            timeout=600
        )
    except Exception as e:
        print(f"⚠️ Single‐tile export failed: {e}")

    # --- If that RAW file doesn’t exist, split & mosaic ---
    if not os.path.exists(raw_path):
        print("⚠️ Raw tile missing → splitting bbox in two…")
        midx = (west + east)/2.0
        tiles = []
        for idx,(w,e) in enumerate([(west,midx),(midx,east)], start=1):
            tile_box = [[w, south], [e, south], [e, north], [w, north], [w, south]]
            tile_path = f"predictor_rasters/raw/{name}_t{idx}.tif"
            print(f"  ▸ Tile{idx}: {w,south}→{e,north}")
            geemap.ee_export_image(
                ee_img.clip(ee.Geometry.BBox(w, south, e, north)),
                filename=tile_path,
                scale=scale,
                region=tile_box,    # <<< again pure list
                file_per_band=False,
                timeout=600
            )
            if not os.path.exists(tile_path):
                raise RuntimeError(f"❗ Tile export failed: {tile_path}")
            tiles.append(rasterio.open(tile_path))

        arrs, out_trans = merge(tiles)
        arr = arrs[0]
        prof = tiles[0].profile.copy()
        prof.update({
            "driver":    "GTiff",
            "height":    arr.shape[0],
            "width":     arr.shape[1],
            "transform": out_trans
        })
        with rasterio.open(raw_path, "w", **prof) as dst:
            dst.write(arr, 1)
        for src in tiles: src.close()
        print(f"🔀 Mosaicked → {raw_path}")

    # --- Finally reproject into the EXACT 30 m grid above ---
    with rasterio.open(raw_path) as src:
        src_dat = src.read(1)
        dst_dat = np.empty((height, width), dtype=src_dat.dtype)
        reproject(
            source=src_dat,
            destination=dst_dat,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform,     dst_crs=crs,
            resampling=Resampling.nearest
        )
        prof = src.profile.copy()
        prof.update({
            "driver":    "GTiff",
            "crs":       crs,
            "transform": transform,
            "height":    height,
            "width":     width
        })

    with rasterio.open(aligned_path, "w", **prof) as dst:
        dst.write(dst_dat, 1)
    print(f"🌐 Aligned → {aligned_path}")

# 10) Fetch all regular predictors
for name in selected_layers:
    if name=="landcover": continue
    if name not in layer_sources:
        print(f"⚠️ Skipping unknown '{name}'")
        continue
    export_and_reproject(layer_sources[name], name)

# 11) One-hot encode any landcover classes
if "landcover" in selected_layers and selected_classes:
    lc = layer_sources["landcover"]
    print("\n🌱 One-hot encoding MODIS landcover…")
    for code in selected_classes:
        if not code.isdigit(): continue
        c = int(code)
        label = modis_landcover_map.get(c, f"class_{c}")
        export_and_reproject(lc.eq(c), f"{c}_{label}")

print("\n✅ All predictors fetched & aligned at 30 m.")
