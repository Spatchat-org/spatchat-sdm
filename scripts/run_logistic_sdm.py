import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from sklearn.linear_model import LogisticRegression
import joblib

# --- Paths ---
csv_path = "inputs/presence_points.csv"
raster_dir = "predictor_rasters"
output_map = "outputs/suitability_map.tif"
os.makedirs("outputs", exist_ok=True)

# --- Load presence points ---
df = pd.read_csv(csv_path)
lats = df['latitude'].values
lons = df['longitude'].values
print(f"📍 Loaded {len(df)} presence points.")

# --- Compute study area bounds ---
buffer = 0.25
min_lat, max_lat = lats.min() - buffer, lats.max() + buffer
min_lon, max_lon = lons.min() - buffer, lons.max() + buffer
print(f"🗺️ Study area bounds: ({min_lat}, {min_lon}) to ({max_lat}, {max_lon})")

# --- Define reference raster specs ---
res = 0.01  # approx 1km resolution
y_size = int((max_lat - min_lat) / res)
x_size = int((max_lon - min_lon) / res)
transform = from_bounds(min_lon, min_lat, max_lon, max_lat, x_size, y_size)
crs = CRS.from_epsg(4326)

# --- Load and reproject predictors ---
raster_paths = sorted([os.path.join(raster_dir, f) for f in os.listdir(raster_dir) if f.endswith(".tif")])
layers = []
layer_names = []
for path in raster_paths:
    with rasterio.open(path) as src:
        arr = src.read(1, out_shape=(y_size, x_size), resampling=Resampling.nearest)
        if src.crs != crs or src.transform != transform:
            print(f"📐 Resampling {os.path.basename(path)} to match reference shape and transform.")
        print(f"🧪 {os.path.basename(path)} → NaN %: {np.isnan(arr).mean() * 100:.2f}%")
        layers.append(arr)
        layer_names.append(os.path.basename(path))

stack = np.stack(layers, axis=-1)
print(f"🗺️ Stacked predictor shape: {stack.shape}")

# --- Extract values at presence points ---
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
valid_mask = ~np.any(np.isnan(flat_stack), axis=1)
background_pool = flat_stack[valid_mask]
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
pred_map = np.full(flat_stack.shape[0], np.nan)
pred_map[valid_mask] = model.predict_proba(flat_stack[valid_mask])[:, 1]
raster_pred = pred_map.reshape(stack.shape[:2])

# --- Save suitability map ---
profile = {
    'driver': 'GTiff',
    'height': raster_pred.shape[0],
    'width': raster_pred.shape[1],
    'count': 1,
    'dtype': rasterio.float32,
    'crs': crs,
    'transform': transform
}
with rasterio.open(output_map, 'w', **profile) as dst:
    dst.write(raster_pred.astype(rasterio.float32), 1)
print(f"🎯 Suitability map saved to {output_map}")
