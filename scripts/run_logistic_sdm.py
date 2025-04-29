import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.linear_model import LogisticRegression
import joblib

# --- Load presence points ---
csv_path = "predictor_rasters/presence_points.csv"
df = pd.read_csv(csv_path)
print(f"📍 Loaded {len(df)} presence points.")

# --- Load and align raster predictors ---
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
            nan_pct = np.isnan(dst).mean()
            print(f"🧪 {os.path.basename(path)} → NaN %: {nan_pct:.2%}")
            layers.append(dst)
            predictor_names.append(os.path.splitext(os.path.basename(path))[0])

    stack = np.stack(layers, axis=-1)  # (rows, cols, bands)
    print(f"🗺️ Stacked predictor shape: {stack.shape}")

# --- Sample presence points from raster stack ---
from rasterio.transform import rowcol

rows, cols = ref_shape
presence_samples = []
for _, row in df.iterrows():
    try:
        r, c = rowcol(ref_transform, row['longitude'], row['latitude'])
        if 0 <= r < rows and 0 <= c < cols:
            sample = stack[r, c, :]
            presence_samples.append(sample)
        else:
            print(f"⚠️ Skipping point ({row['latitude']}, {row['longitude']}) — outside raster bounds")
    except Exception as e:
        print(f"⚠️ Error sampling point ({row['latitude']}, {row['longitude']}): {e}")

presence_samples = np.array(presence_samples)
print(f"📍 Presence samples: {presence_samples.shape}")

if len(presence_samples) == 0:
    raise RuntimeError("❗ No valid presence samples found within raster bounds.")

# --- Generate background (pseudo-absence) samples ---
np.random.seed(42)
background_samples = []
num_background = len(presence_samples) * 5

for _ in range(num_background):
    r = np.random.randint(0, rows)
    c = np.random.randint(0, cols)
    sample = stack[r, c, :]
    background_samples.append(sample)

background_samples = np.array(background_samples)
print(f"🌎 Background samples: {background_samples.shape}")

# --- Combine and clean ---
X = np.vstack([presence_samples, background_samples])
y = np.hstack([np.ones(len(presence_samples)), np.zeros(len(background_samples))])

# --- Remove all-NaN or all-zero columns ---
non_empty_features = ~(
    np.isnan(X).all(axis=0) |
    (np.nan_to_num(X, nan=0).sum(axis=0) == 0)
)
X = X[:, non_empty_features]
predictor_names = [name for i, name in enumerate(predictor_names) if non_empty_features[i]]

print(f"✅ Using {len(predictor_names)} predictors after cleanup:")
for name in predictor_names:
    print(f"   - {name}")

mask = ~np.isnan(X).any(axis=1)
X_clean = X[mask]
y_clean = y[mask]

print(f"🧹 Samples after removing NaN: {X_clean.shape}")

if X_clean.shape[0] == 0:
    raise RuntimeError("❗ No valid training samples remain after NaN filtering.")

# --- Train model ---
model = LogisticRegression(max_iter=500)
model.fit(X_clean, y_clean)
print("🚀 Logistic Regression model trained!")

# --- Save model ---
os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/sdm_model.joblib")
print("💾 Model saved.")

# --- Predict across raster stack ---
flat_stack = stack.reshape(-1, stack.shape[-1])
flat_stack = flat_stack[:, non_empty_features]  # match filtered predictors
mask_flat = ~np.isnan(flat_stack).any(axis=1)

y_pred = np.full(flat_stack.shape[0], np.nan, dtype=np.float32)
y_pred[mask_flat] = model.predict_proba(flat_stack[mask_flat])[:, 1]
y_pred_image = y_pred.reshape(ref_shape)

# --- Save suitability map ---
suitability_path = "outputs/suitability_map.tif"
with rasterio.open(suitability_path, "w", **ref_profile) as dst:
    dst.write(y_pred_image, 1)

print(f"🎯 Suitability map saved at {suitability_path}")
