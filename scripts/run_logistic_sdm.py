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
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
try:
    import statsmodels.api as sm
    _HAS_SM = True
except ImportError:
    print("⚠️ statsmodels not available; skipping p-values and CIs")
    _HAS_SM = False

# --- Paths ---
csv_path    = "inputs/presence_points.csv"
raster_dir  = "predictor_rasters/wgs84"
stats_csv   = "outputs/model_stats.csv"
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

# --- Extract presence samples ---
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
coords = np.column_stack((lats, lons))[:len(presence_samples)]
coords = coords[~np.any(np.isnan(presence_samples), axis=1)]
n_blocks = 5
blocks = KMeans(n_clusters=n_blocks, random_state=42).fit_predict(coords)

gkf = GroupKFold(n_splits=n_blocks)
cv_aucs, cv_tsses, cv_kappas = [], [], []
for train_idx, test_idx in gkf.split(presence_samples, yc[:len(presence_samples)], groups=blocks):
    # split
    Xp_tr, Xp_te = presence_samples[train_idx], presence_samples[test_idx]
    # background
    n_bt = 5 * len(train_idx)
    bt_idx = np.random.choice(len(pool), size=n_bt, replace=False)
    Xb_tr = pool[bt_idx]
    n_bt2 = 5 * len(test_idx)
    bt_idx2 = np.random.choice(len(pool), size=n_bt2, replace=False)
    Xb_te = pool[bt_idx2]
    # combine
    X_tr = np.vstack([Xp_tr, Xb_tr])
    y_tr = np.concatenate([np.ones(len(Xp_tr)), np.zeros(len(Xb_tr))])
    X_te = np.vstack([Xp_te, Xb_te])
    y_te = np.concatenate([np.ones(len(Xp_te)), np.zeros(len(Xb_te))])
    # fit & eval
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:,1]
    auc_cv = roc_auc_score(y_te, p_te)
    fpr_cv, tpr_cv, thr = roc_curve(y_te, p_te)
    bt = thr[np.argmax(tpr_cv - fpr_cv)]
    yhat = (p_te >= bt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, yhat).ravel()
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    tss_cv = sens + spec - 1
    kapp = cohen_kappa_score(y_te, yhat)
    cv_aucs.append(auc_cv)
    cv_tsses.append(tss_cv)
    cv_kappas.append(kapp)
print(f"📐 Spatial CV AUC: {np.mean(cv_aucs):.3f} ± {np.std(cv_aucs):.3f}")
print(f"📐 Spatial CV TSS: {np.mean(cv_tsses):.3f} ± {np.std(cv_tsses):.3f}")
print(f"📐 Spatial CV Kappa: {np.mean(cv_kappas):.3f} ± {np.std(cv_kappas):.3f}")

# --- Final model fit & stats ---
model = LogisticRegression(max_iter=1000).fit(Xc, yc)
joblib.dump(model, "outputs/logistic_model_full.pkl")
y_prob = model.predict_proba(Xc)[:,1]
auc    = roc_auc_score(yc, y_prob)
fpr, tpr, thr = roc_curve(yc, y_prob)
best_thr = thr[np.argmax(tpr - fpr)]
yhat = (y_prob >= best_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(yc, yhat).ravel()
sens = tp/(tp+fn)
spec = tn/(tn+fp)
tss = sens + spec - 1
kappa = cohen_kappa_score(yc, yhat)

# p-values & CIs
enabled_sm = _HAS_SM
if enabled_sm:
    X_sm = sm.add_constant(Xc)
    sm_model = sm.Logit(yc, X_sm).fit(disp=False)
    pvals = sm_model.pvalues
    ci = sm_model.conf_int()
else:
    pvals = pd.Series([np.nan]*(len(names)+1), index=['const']+names)
    ci = pd.DataFrame({0:[np.nan]*(len(names)+1),1:[np.nan]*(len(names)+1)})

# --- Write out performance and coefficients ---
perf_df = pd.DataFrame([{  # performance metrics
    'AUC':          auc,
    'Threshold':    best_thr,
    'Sensitivity':  sens,
    'Specificity':  spec,
    'TSS':          tss,
    'Kappa':        kappa
}])
perf_df.to_csv("outputs/performance_metrics.csv", index=False)
print("📊 Performance metrics saved to outputs/performance_metrics.csv")

coef_df = pd.DataFrame({  # model coefficients
    'predictor':   ['Intercept'] + names,
    'coefficient': np.concatenate([[sm_model.params['const'] if enabled_sm else model.intercept_[0]], model.coef_.flatten()]),
    'p_value':     pvals.values,
    'CI_lower':    ci[0].values,
    'CI_upper':    ci[1].values
})
coef_df.to_csv("outputs/coefficients.csv", index=False)
print("📊 Coefficients saved to outputs/coefficients.csv")

# --- Save final suitability map ---
pred_flat = np.full(flat.shape[0], np.nan)
pred_flat[valid_mask] = model.predict_proba(pool)[:,1]
pred_map = pred_flat.reshape((height, width))
profile = {
    'driver': 'GTiff', 'height': height, 'width': width,
    'count': 1, 'dtype': rasterio.float32,
    'crs': ref_crs, 'transform': ref_transform
}
with rasterio.open(output_map, "w", **profile) as dst:
    dst.write(pred_map.astype(rasterio.float32), 1)
print(f"🎯 Final suitability map saved to {output_map}")
