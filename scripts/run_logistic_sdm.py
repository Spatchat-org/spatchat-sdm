import rasterio
import numpy as np
import pandas as pd
import glob
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# --- Paths ---
predictor_folder = "predictor_rasters"
presence_file = os.path.join(predictor_folder, "presence_points.csv")
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# --- Read Selected Layers ---
with open("scripts/user_layer_selection.txt", "r") as f:
    selected_layers = [line.strip() for line in f.readlines()]

# --- Load Only Selected Predictor Rasters ---
predictor_paths = [os.path.join(predictor_folder, f"{layer}.tif") for layer in selected_layers]

arrays = []
array_names = []

# --- Pick master raster for alignment ---
master_raster = rasterio.open(predictor_paths[0])
master_profile = master_raster.profile

def resample_raster(src_raster, target_profile):
    data = src_raster.read(
        out_shape=(
            src_raster.count,
            target_profile['height'],
            target_profile['width']
        ),
        resampling=rasterio.enums.Resampling.nearest
    )
    return data[0]

for path in predictor_paths:
    name = os.path.splitext(os.path.basename(path))[0]
    raster = rasterio.open(path)
    resampled_array = resample_raster(raster, master_profile)
    arrays.append(resampled_array)
    array_names.append(name)

# --- Stack Predictors ---
stacked_predictors = np.stack(arrays, axis=-1)
height, width, num_predictors = stacked_predictors.shape

print(f"🗺️ Stacked predictors shape: {stacked_predictors.shape}")

# --- Load Presence Points ---
presence = pd.read_csv(presence_file)
coords = list(zip(presence.longitude, presence.latitude))

samples = []
transform = master_profile['transform']
for lon, lat in coords:
    col, row = ~transform * (lon, lat)
    col, row = int(col), int(row)
    if 0 <= row < height and 0 <= col < width:
        samples.append(stacked_predictors[row, col, :])
    else:
        samples.append(np.full((num_predictors,), np.nan))

presence_X = np.stack(samples, axis=0)
presence_y = np.ones(presence_X.shape[0])

print(f"📍 Presence samples: {presence_X.shape}")

# --- Generate Random Background Points ---
np.random.seed(42)
num_background = presence_X.shape[0] * 5
row_idxs = np.random.randint(0, height, num_background)
col_idxs = np.random.randint(0, width, num_background)

background_samples = stacked_predictors[row_idxs, col_idxs, :]
background_y = np.zeros(background_samples.shape[0])

print(f"🌎 Background samples: {background_samples.shape}")

# --- Combine Presence + Background ---
X = np.vstack([presence_X, background_samples])
y = np.concatenate([presence_y, background_y])

print(f"🔀 Total samples before cleaning: {X.shape}")

# --- Remove any NaN samples ---
mask = np.all(np.isfinite(X), axis=1)
X = X[mask]
y = y[mask]

print(f"🧹 Samples after removing NaN: {X.shape}")

# --- Train Logistic Regression Model ---
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("🚀 Logistic Regression model trained!")

# --- Predict Suitability over Full Map ---
flat_predictors = stacked_predictors.reshape(-1, num_predictors)
suitability = model.predict_proba(flat_predictors)[:, 1]
suitability_map = suitability.reshape(height, width)

# --- Save Suitability Map ---
output_suitability = os.path.join(output_folder, "suitability_map.tif")
with rasterio.open(
    output_suitability,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=suitability_map.dtype,
    crs=master_profile['crs'],
    transform=master_profile['transform']
) as dst:
    dst.write(suitability_map, 1)

print(f"🎯 Suitability map saved at {output_suitability}")

# --- Save Trained Model (Optional) ---
joblib.dump(model, os.path.join(output_folder, "logistic_model.pkl"))
print("💾 Model saved!")
