# =============================
# scripts/predictor_fetcher.py
# =============================

import ee, json, os, pandas as pd, geemap, shutil

def fetch_predictors(csv_path, selected_layers, landcover_classes):
    if not ee.data._credentials:
        service_account_info = json.loads(os.environ['GEE_SERVICE_ACCOUNT'])
        credentials = ee.ServiceAccountCredentials(
            email=service_account_info['client_email'],
            key_data=json.dumps(service_account_info)
        )
        ee.Initialize(credentials)

    if os.path.exists("predictor_rasters"):
        for f in os.listdir("predictor_rasters"):
            if f != "presence_points.csv":
                os.remove(os.path.join("predictor_rasters", f))
else:
        os.makedirs("predictor_rasters", exist_ok=True)
    shutil.rmtree("predictor_rasters/wgs84", ignore_errors=True)

    df = pd.read_csv(csv_path)
    min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
    min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
    bbox = ee.Geometry.BBox(min_lon - 0.25, min_lat - 0.25, max_lon + 0.25, max_lat + 0.25)

    bio = ee.Image("WORLDCLIM/V1/BIO")
    terrain = ee.Terrain.products(ee.Image("USGS/SRTMGL1_003"))
    layer_sources = {f"bio{i}": bio.select(f"bio{str(i).zfill(2)}") for i in range(1, 20)}
    layer_sources.update({
        "slope": terrain.select("slope"),
        "aspect": terrain.select("aspect"),
        "elevation": ee.Image("USGS/SRTMGL1_003"),
        "ndvi": ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI").first(),
        "landcover": ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1").first()
    })

    landcover_labels = {
        0: "water", 1: "evergreen_needleleaf_forest", 2: "evergreen_broadleaf_forest",
        3: "deciduous_needleleaf_forest", 4: "deciduous_broadleaf_forest", 5: "mixed_forest",
        6: "closed_shrublands", 7: "open_shrublands", 8: "woody_savannas", 9: "savannas",
        10: "grasslands", 11: "permanent_wetlands", 12: "croplands", 13: "urban_and_built_up",
        14: "cropland_natural_vegetation_mosaic", 15: "snow_and_ice", 16: "barren_or_sparsely_vegetated"
    }

    for layer in selected_layers:
        if layer == "landcover":
            lc = layer_sources["landcover"].clip(bbox)
            for cid in landcover_classes:
                if cid in landcover_labels:
                    geemap.ee_export_image(
                        lc.eq(cid),
                        filename=f"predictor_rasters/{cid}_{landcover_labels[cid]}.tif",
                        scale=500, region=bbox, timeout=300
                    )
        elif layer in layer_sources:
            geemap.ee_export_image(
                layer_sources[layer].clip(bbox),
                filename=f"predictor_rasters/{layer}.tif",
                scale=1000, region=bbox, timeout=600
            )
