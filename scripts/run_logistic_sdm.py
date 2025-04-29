import os
import pandas as pd
import numpy as np
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from joblib import dump

# --- Ensure output folder exists ---
os.makedirs("outputs", exist_ok=True)

# --- Load presence points ---
presence_path = "predictor_rasters/presence_points.csv"
if not os.path.exists(presence_path):
    print("❗ Presence CSV not found.")
    exit()

df = pd.read_csv(presence_path)
if not {'latitude', 'longitude'}.issubset(df.columns):
    print("❗ CSV missing 'latitude' or 'longitude' columns.")
    exit()

coords = list(zip(df['longitude'], df['latitude']))
print(f"📍 Loaded {len(coords)} presence points.")

# --- Load predictor rasters ---
raster_folder = "predictor_rasters"
raster_paths = [os.path.join(raster_folder, f) for f in os.listdir(raster_folder) if f.endswith(".tif") and "presence" not in f]
print(f"🔍 Found predictor rasters: {raster_paths}")

arrays = []
for path in raster_paths:
    with rasterio.open(path) as src:
        arrays.append(src.read(1))

try:
    stacked = np.stack(arrays, axis=-1)
    print(f"🗺️ Stacked predictor shape: {stacked.shape}")
except Exception as e:
    print(f"❗ Could not stack rasters: {e}")
    exit()

# --- Extract presence pixel values ---
samples = []
for lon, lat in coords:
    row_samples = []
    for path in raster_paths:
        with rasterio.open(path) as src:
            row, col = src.index(lon, lat)
            try:
                val = src.read(1)[row, col]
                row_samples.append(val)
            except:
                row_samples.append(np.nan)
    samples.append(row_samples)

X_pos = np.array(samples)
print(f"📍 Presence samples: {X_pos.shape}")

# --- Sample background points ---
np.random.seed(42)
background_samples = []
height, width = stacked.shape[:2]

for _ in range(len(X_pos) * 5):  # 5x background
    row = np.random.randint(0, height)
    col = np.random.randint(0, width)
    values = stacked[row, col, :]
    background_samples.append(values)

X_neg = np.array(background_samples)
print(f"🌎 Background samples: {X_neg.shape}")

# --- Stack and train ---
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])
X, y = shuffle(X, y, random_state=42)

print(f"🔀 Total training samples: {X.shape}")
X = np.nan_to_num(X)
print(f"🧹 Samples after removing NaN: {X.shape}")

# --- Train model ---
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
print("🚀 Logistic Regression model trained!")

# --- Save model ---
dump(model, "outputs/sdm_model.joblib")
print("💾 Model saved.")

# --- Apply to entire raster ---
flat = stacked.reshape(-1, stacked.shape[-1])
flat = np.nan_to_num(flat)
y_pred = model.predict_proba(flat)[:, 1]
suitability = y_pred.reshape(stacked.shape[:2])

# --- Save suitability map ---
out_path = "outputs/suitability_map.tif"
with rasterio.open(raster_paths[0]) as ref:
    meta = ref.meta.copy()
    meta.update({
        "count": 1,
        "dtype": "float32"
    })

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(suitability.astype(np.float32), 1)

print(f"🎯 Suitability map saved at {out_path}")
