# ============================
# scripts/sdm_runner.py
# ============================

import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import rowcol
from sklearn.linear_model import LogisticRegression
import joblib

def run_logistic_sdm(csv_path="predictor_rasters/presence_points.csv", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    suitability_path = os.path.join(output_dir, "suitability_map.tif")
    if os.path.exists(suitability_path):
        os.remove(suitability_path)

    df = pd.read_csv(csv_path)
    print(f"📍 Loaded {len(df)} presence points.")

    input_paths = sorted(glob.glob("predictor_rasters/*.tif"))
    if not input_paths:
        raise RuntimeError("❗ No raster predictors found!")

    reference_path = input_paths[0]
    with rasterio.open(reference_path) as ref:
        ref_profile = ref.profile
        ref_bounds = ref.bounds
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)

        layers = []
        predictor_names = []
        for path in input_paths:
            with rasterio.open(path) as src:
                dst = np.full(ref_shape, np.nan, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest
                )
                layers.append(dst)
                predictor_names.append(os.path.splitext(os.path.basename(path))[0])

        stack = np.stack(layers, axis=-1)
        print(f"🗺️ Stacked predictor shape: {stack.shape}")

    rows, cols = ref_shape
    presence_samples = []
    for _, row in df.iterrows():
        try:
            r, c = rowcol(ref_transform, row['longitude'], row['latitude'])
            if 0 <= r < rows and 0 <= c < cols:
                presence_samples.append(stack[r, c, :])
        except:
            continue

    presence_samples = np.array(presence_samples)
    if len(presence_samples) == 0:
        raise RuntimeError("❗ No valid presence samples found.")

    np.random.seed(42)
    background_samples = [stack[np.random.randint(0, rows), np.random.randint(0, cols), :] for _ in range(len(presence_samples) * 5)]
    background_samples = np.array(background_samples)

    X = np.vstack([presence_samples, background_samples])
    y = np.hstack([np.ones(len(presence_samples)), np.zeros(len(background_samples))])

    non_empty_features = ~(
        np.isnan(X).all(axis=0) |
        (np.nan_to_num(X, nan=0).sum(axis=0) == 0)
    )
    X = X[:, non_empty_features]
    predictor_names = [name for i, name in enumerate(predictor_names) if non_empty_features[i]]

    mask = ~np.isnan(X).any(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]

    if X_clean.shape[0] == 0:
        raise RuntimeError("❗ No valid training data after filtering.")

    model = LogisticRegression(max_iter=500)
    model.fit(X_clean, y_clean)
    joblib.dump(model, os.path.join(output_dir, "sdm_model.joblib"))

    flat_stack = stack.reshape(-1, stack.shape[-1])
    flat_stack = flat_stack[:, non_empty_features]
    mask_flat = ~np.isnan(flat_stack).any(axis=1)

    y_pred = np.full(flat_stack.shape[0], np.nan, dtype=np.float32)
    y_pred[mask_flat] = model.predict_proba(flat_stack[mask_flat])[:, 1]
    y_pred_image = y_pred.reshape(ref_shape)

    with rasterio.open(suitability_path, "w", **ref_profile) as dst:
        dst.write(y_pred_image, 1)

    print(f"🎯 Suitability map saved to {suitability_path}")
    return suitability_path