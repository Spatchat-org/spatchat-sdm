import os
import numpy as np
import pandas as pd
import rasterio
import joblib

from rasterio.enums import Resampling
from rasterio.crs import CRS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             cohen_kappa_score)
try:
    import statsmodels.api as sm
    _HAS_SM = True
except ImportError:
    print("⚠️ statsmodels not available; skipping p-values and CIs")
    _HAS_SM = False

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

# --- Reference grid ---
rasters = sorted([os.path.join(raster_dir, f)
                  for f in os.listdir(raster_dir)
                  if f.endswith(".tif")])
if not rasters:
    raise RuntimeError(f"No .tif found in {raster_dir}")
with rasterio.open(rasters[0]) as ref:
    ref_crs        = ref.crs
    ref_transform = ref.transform
    height, width = ref.height, ref.width
    print(f"🎯 Reference grid: {width}×{height} @ {ref_transform} in {ref_crs}")

# --- Stack predictors ---
layers, names = [], []
for path in rasters:
    name = os.path.splitext(os.path.basename(path))[0]
    with rasterio.open(path) as src:
        arr = src.read(1,
                       out_shape=(height, width),
                       resampling=Resampling.nearest)
    print(f"🧪 {name}.tif → NaN%: {np.isnan(arr).mean()*100:.2f}")
    layers.append(arr)
    names.append(name)
stack = np.stack(layers, axis=-1)
print(f"🗺️ Stacked predictor shape: {stack.shape}")

# --- Extract samples ---
inv = ~ref_transform
presence_samples = []
for lat, lon in zip(lats, lons):
    col, row = inv * (lon, lat)
    row, col = int(row), int(col)
    if 0 <= row < height and 0 <= col < width:
        vals = stack[row, col, :]
        if not np.any(np.isnan(vals)):
            presence_samples.append(vals)
presence_samples = np.array(presence_samples)
print(f"📍 Presence samples: {presence_samples.shape}")

# --- Background sampling ---
np.random.seed(42)
flat       = stack.reshape(-1, stack.shape[-1])
valid_mask = ~np.any(np.isnan(flat), axis=1)
pool       = flat[valid_mask]
n_bg       = 5 * len(presence_samples)
bg_indices = np.random.choice(len(pool), size=n_bg, replace=False)
background_samples = pool[bg_indices]
print(f"🌎 Background samples: {background_samples.shape}")

# --- Prepare data ---
X = np.vstack([presence_samples, background_samples])
y = np.concatenate([np.ones(len(presence_samples)), np.zeros(len(background_samples))])
mask = ~np.any(np.isnan(X), axis=1)
Xc, yc = X[mask], y[mask]
print(f"🔀 Training samples: {X.shape} → after NaN removal: {Xc.shape}")

# --- Spatial cross-validation ---
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
# Cluster presence points into spatial blocks
coords = np.column_stack((lats, lons))[mask[:len(presence_samples)]]  # only presence coords
n_blocks = 5
blocks = KMeans(n_clusters=n_blocks, random_state=42).fit_predict(coords)

gkf = GroupKFold(n_splits=n_blocks)
cv_aucs, cv_tsses, cv_kappas = [], [], []
for train_idx, test_idx in gkf.split(presence_samples, yc[:len(presence_samples)], groups=blocks):
    # Train presence and background
    Xp_train = presence_samples[train_idx]
    Xp_test = presence_samples[test_idx]
    # Sample background separately for train and test
    background_pool = pool
    np.random.seed(42)
    n_bg_train = 5 * len(train_idx)
    bg_train_idx = np.random.choice(len(background_pool), size=n_bg_train, replace=False)
    Xb_train = background_pool[bg_train_idx]
    n_bg_test = 5 * len(test_idx)
    bg_test_idx = np.random.choice(len(background_pool), size=n_bg_test, replace=False)
    Xb_test = background_pool[bg_test_idx]
    # Combine
    X_train = np.vstack([Xp_train, Xb_train])
    y_train = np.concatenate([np.ones(len(Xp_train)), np.zeros(len(Xb_train))])
    X_test = np.vstack([Xp_test, Xb_test])
    y_test = np.concatenate([np.ones(len(Xp_test)), np.zeros(len(Xb_test))])
    # Fit & evaluate
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    p_test = clf.predict_proba(X_test)[:, 1]
    # AUC
    auc_cv = roc_auc_score(y_test, p_test)
    # Optimal threshold by TSS
    fpr_cv, tpr_cv, thr_cv = roc_curve(y_test, p_test)
    youden_cv = tpr_cv - fpr_cv
    bt_cv = thr_cv[np.argmax(youden_cv)]
    yhat_cv = (p_test >= bt_cv).astype(int)
    tn_cv, fp_cv, fn_cv, tp_cv = confusion_matrix(y_test, yhat_cv).ravel()
    sens_cv = tp_cv/(tp_cv+fn_cv)
    spec_cv = tn_cv/(tn_cv+fp_cv)
    tss_cv = sens_cv + spec_cv - 1
    kappa_cv = cohen_kappa_score(y_test, yhat_cv)
    cv_aucs.append(auc_cv)
    cv_tsses.append(tss_cv)
    cv_kappas.append(kappa_cv)
# Report CV metrics
print(f"📐 Spatial CV [{n_blocks} folds] AUC: {np.mean(cv_aucs):.3f} ± {np.std(cv_aucs):.3f}")
print(f"📐 Spatial CV TSS: {np.mean(cv_tsses):.3f} ± {np.std(cv_tsses):.3f}")
print(f"📐 Spatial CV Kappa: {np.mean(cv_kappas):.3f} ± {np.std(cv_kappas):.3f}")

# --- Train final model ---
