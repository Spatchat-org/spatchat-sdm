import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression
import joblib

# --- Paths ---
csv_path = "inputs/presence_points.csv"
raster_dir = "predictor_rasters/wgs84"
output_map = "outputs/suitability_map.tif"
os.makedirs("outputs", exist_ok=True)

# --- Load presence points ---
df = pd.read_csv(csv_path)
lats = df['latitude'].values
lons = df['longitude'].values
print(f"📍 Loaded {len(df)} presence points.")

# --- Load and stack predictors ---
layers = []
layer_names = []
for fname in sorted(os.listdir(raster_dir)):
    if fname.endswith(".tif"):
        path = os.path.join(raster_dir, fname)
        with rasterio.open(path) as src:
            print(f"🧪 {fname} → NaN %: {np.isnan(src.read(1)).mean() * 100:.2f}%")
            arr = src.read(1)
            layers.append(arr)
            layer_names.append(fname)
            profile = src.profile
stack = np.stack(layers, axis=-1)
print(f"🗺️ Stacked predictor shape: {stack.shape}")

# --- Extract values at presence points ---
transform = profile["transform"]
inv_transform = ~transform
presence_samples = []
for lat, lon in zip(lats, lons):
    col, row = inv_transform * (lon, lat)
    row, col = int(row), int(col)
    if 0 <= row < stack.shape[0] and 0 <= col < stack.shape[1]:
        values = stack[row, col, :]
        if not np.any(np.isnan(values)):
            presence_samples.append(values)
        else:
            print(f"⚠️ Skipping point ({lat}, {lon}) — contains NaN")
    else:
        print(f"⚠️ Skipping point ({lat}, {lon}) — outside raster bounds")
presence_samples = np.array(presence_samples)
print(f"📍 Presence samples: {presence_samples.shape}")

# --- Sample background ---
np.random.seed(42)
flat_stack = stack.reshape(-1, stack.shape[-1])
mask = ~np.any(np.isnan(flat_stack), axis=1)
background_pool = flat_stack[mask]
background_samples = background_pool[np.random.choice(len(background_pool), size=5 * len(presence_samples), replace=False)]
print(f"🌎 Background samples: {background_samples.shape}")

# --- Combine and train model ---
X = np.vstack([presence_samples, background_samples])
y = np.array([1] * len(presence_samples) + [0] * len(background_samples))

mask = ~np.any(np.isnan(X), axis=1)
X_clean = X[mask]
y_clean = y[mask]
print(f"🔀 Total training samples: {X.shape}, After NaN removal: {X_clean.shape}")

model = LogisticRegression(max_iter=1000)
model.fit(X_clean, y_clean)
joblib.dump(model, "outputs/logistic_model.pkl")
print("🧠 Logistic regression model trained.")

# --- Predict across the raster stack ---
flat_pred = model.predict_proba(flat_stack)[:, 1]
map_pred = np.full(flat_stack.shape[0], np.nan)
map_pred[mask] = flat_pred
raster_pred = map_pred.reshape(stack.shape[:2])

# --- Save suitability map ---
profile.update(dtype=rasterio.float32, count=1)
with rasterio.open(output_map, 'w', **profile) as dst:
    dst.write(raster_pred.astype(rasterio.float32), 1)
print(f"🎯 Suitability map saved to {output_map}")
