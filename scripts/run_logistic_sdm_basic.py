import os
import numpy as np
import pandas as pd
import rasterio
import joblib

from rasterio.enums import Resampling
from rasterio.crs import CRS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --- Paths ---
csv_path    = "inputs/presence_points.csv"
raster_dir  = "predictor_rasters/wgs84"
output_map  = "outputs/suitability_map_wgs84.tif"
os.makedirs("outputs", exist_ok=True)

# --- Load presence points ---
df   = pd.read_csv(csv_path)
lats = df['latitude'].values
lons = df['longitude'].values
print(f"📍 Loaded {len(df)} presence points.")

# --- Grab reference grid from the *first* .tif in raster_dir ---
rasters = sorted([
    os.path.join(raster_dir, f)
    for f in os.listdir(raster_dir)
    if f.endswith(".tif")
])
if not rasters:
    raise RuntimeError(f"No .tif found in {raster_dir}")

with rasterio.open(rasters[0]) as ref:
    ref_crs        = ref.crs
    ref_transform = ref.transform
    height, width = ref.height, ref.width
    print(f"🎯 Reference grid: {width}×{height} @ {ref_transform} in {ref_crs}")

# --- Load & resample all predictors to reference grid ---
layers = []
names  = []
for path in rasters:
    name = os.path.splitext(os.path.basename(path))[0]
    with rasterio.open(path) as src:
        if src.crs != ref_crs or src.transform != ref_transform:
            print(f"📐 Resampling {name} to reference grid…")
        arr = src.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.nearest
        )
    print(f"🧪 {name}.tif → NaN%: {np.isnan(arr).mean()*100:.2f}")
    layers.append(arr)
    names.append(name)

# Now names == your layer_names
layer_names = names

stack = np.stack(layers, axis=-1)
print(f"🗺️ Stacked predictor shape: {stack.shape}")

# --- Extract values at presence points ---
inv = ~ref_transform
presence_samples = []
for lat, lon in zip(lats, lons):
    col, row = inv * (lon, lat)
    row, col = int(row), int(col)
    if 0 <= row < height and 0 <= col < width:
        vals = stack[row, col, :]
        if not np.any(np.isnan(vals)):
            presence_samples.append(vals)
        else:
            print(f"⚠️ Skipping ({lat},{lon})—NaNs")
    else:
        print(f"⚠️ Skipping ({lat},{lon})—outside bounds")

presence_samples = np.array(presence_samples)
print(f"📍 Presence samples: {presence_samples.shape}")

# --- Sample background ---
np.random.seed(42)
flat       = stack.reshape(-1, stack.shape[-1])
valid_mask = ~np.any(np.isnan(flat), axis=1)
pool       = flat[valid_mask]
n_bg       = 5 * len(presence_samples)
bg_indices = np.random.choice(len(pool), size=n_bg, replace=False)
background_samples = pool[bg_indices]
print(f"🌎 Background samples: {background_samples.shape}")

# --- Train logistic regression ---
X = np.vstack([presence_samples, background_samples])
y = np.concatenate([np.ones(len(presence_samples)), np.zeros(len(background_samples))])
mask = ~np.any(np.isnan(X), axis=1)
Xc, yc = X[mask], y[mask]
print(f"🔀 Training samples: {X.shape} → after NaN removal: {Xc.shape}")

model = LogisticRegression(max_iter=1000)
model.fit(Xc, yc)
joblib.dump(model, "outputs/logistic_model.pkl")
print("🧠 Model trained.")

# --- Compute AUC on training data ---
y_prob = model.predict_proba(Xc)[:, 1]
auc    = roc_auc_score(yc, y_prob)

# --- Build stats table including AUC and coefficients ---
# summary row for AUC
summary = pd.DataFrame([{
    'predictor': 'AUC',
    'coefficient': auc
}])

# coefficient rows
coef_df = pd.DataFrame({
    'predictor': layer_names,
    'coefficient': model.coef_.flatten()
})

stats_df = pd.concat([summary, coef_df], ignore_index=True)
stats_df.to_csv("outputs/model_stats.csv", index=False)
print("📊 Model stats saved to outputs/model_stats.csv")

# --- Predict over the full grid ---
pred_flat = np.full(flat.shape[0], np.nan)
pred_flat[valid_mask] = model.predict_proba(flat[valid_mask])[:, 1]
pred_map = pred_flat.reshape((height, width))

# --- Save suitability map using the reference profile ---
profile = {
    'driver':    'GTiff',
    'height':    height,
    'width':     width,
    'count':     1,
    'dtype':     rasterio.float32,
    'crs':       ref_crs,
    'transform': ref_transform
}
with rasterio.open(output_map, "w", **profile) as dst:
    dst.write(pred_map.astype(rasterio.float32), 1)
print(f"🎯 Suitability map saved to {output_map}")
