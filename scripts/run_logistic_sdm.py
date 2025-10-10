# scripts/run_logistic_sdm.py  (patched for constant-feature handling)

import os
import numpy as np
import pandas as pd
import rasterio
import joblib

from rasterio.enums import Resampling  # kept for consistency; not used in reprojection
from rasterio.transform import rowcol
from rasterio.warp import reproject, Resampling as WarpResampling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             cohen_kappa_score)
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Paths ---
csv_path     = "inputs/presence_points.csv"
raster_dir   = "predictor_rasters/wgs84"
perf_cv_csv  = "outputs/performance_metrics_cv.csv"
perf_fit_csv = "outputs/performance_metrics_fitted.csv"
compat_perf  = "outputs/performance_metrics.csv"  # also in ZIP
coef_csv     = "outputs/coefficients.csv"
scaler_pkl   = "outputs/standard_scaler.pkl"
output_map   = "outputs/suitability_map_wgs84.tif"
std_ras_dir  = "outputs/standardized_rasters_wgs84"
os.makedirs("outputs", exist_ok=True)

# --- Load presence points ---
df   = pd.read_csv(csv_path)
lats = df['latitude'].values
lons = df['longitude'].values
print(f"📍 Loaded {len(df)} presence points.")

# --- Reference grid ---
rasters = sorted([os.path.join(raster_dir, f)
                  for f in os.listdir(raster_dir)
                  if f.endswith('.tif')])
if not rasters:
    raise RuntimeError(f"No .tif found in {raster_dir}")
with rasterio.open(rasters[0]) as ref:
    ref_crs       = ref.crs
    ref_transform = ref.transform
    height, width = ref.height, ref.width
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': rasterio.float32,
        'crs': ref_crs,
        'transform': ref_transform
    }
    print(f"🎯 Reference grid: {width}×{height} @ {ref_transform} in {ref_crs}")

# --- Stack predictors (reproject/align every layer to reference grid) ---
def _is_categorical(name: str) -> bool:
    n = name.lower()
    return ("landcover" in n) or n.endswith("_onehot") or ("class" in n)

layers, names = [], []
for path in rasters:
    name = os.path.splitext(os.path.basename(path))[0]
    with rasterio.open(path) as src:
        dst = np.full((height, width), np.nan, dtype=np.float32)
        resamp = WarpResampling.nearest if _is_categorical(name) else WarpResampling.bilinear
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            resampling=resamp, dst_nodata=np.nan
        )
    nan_pct = float(np.isnan(dst).mean() * 100.0)
    print(f"🧪 {name}.tif → NaN%: {nan_pct:.2f}")
    layers.append(dst)
    names.append(name)

stack = np.stack(layers, axis=-1)
print(f"🗺️ Stacked predictor shape: {stack.shape}")

# --- Extract presence samples (only where predictors are all valid) ---
presence_samples = []
kept_coords = []  # (lat, lon) that landed on valid pixels
for lat, lon in zip(lats, lons):
    try:
        r, c = rowcol(ref_transform, lon, lat, op=round)
    except Exception:
        continue
    if 0 <= r < height and 0 <= c < width:
        vals = stack[r, c, :]
        if not np.any(np.isnan(vals)):
            presence_samples.append(vals)
            kept_coords.append((lat, lon))

presence_samples = np.array(presence_samples, dtype=np.float32)
kept_coords = np.array(kept_coords, dtype=np.float32) if len(kept_coords) else np.empty((0, 2), dtype=np.float32)
print(f"📍 Presence samples: {presence_samples.shape}")

# --- Background sampling pool (all valid pixels) ---
flat       = stack.reshape(-1, stack.shape[-1]).astype(np.float32)
valid_mask = ~np.any(np.isnan(flat), axis=1)
valid_idx  = np.flatnonzero(valid_mask)
pool       = flat[valid_mask]
print(f"✅ Valid pixel pool: {len(pool)}")

# Choose background for the *raw sample table* (map plotting, not CV)
np.random.seed(42)
n_bg       = 5 * max(1, len(presence_samples))
n_bg       = min(n_bg, len(pool))  # cap at available pool size
bg_indices = np.random.choice(len(pool), size=n_bg, replace=False)
background_samples = pool[bg_indices].astype(np.float32)
print(f"🌎 Background samples (for tables): {background_samples.shape}")

# Convert chosen background indices back to row/col → lon/lat (for export)
bg_linear  = valid_idx[bg_indices]
bg_rows    = bg_linear // width
bg_cols    = bg_linear %  width
bg_xs, bg_ys = rasterio.transform.xy(ref_transform, bg_rows, bg_cols, offset='center')
bg_lons = np.array(bg_xs); bg_lats = np.array(bg_ys)

# --- Prepare data (combined presence + background) ---
X = np.vstack([presence_samples, background_samples])
y = np.concatenate([
    np.ones(len(presence_samples), dtype=int),
    np.zeros(len(background_samples), dtype=int)
])

# Guard rows: require finite values (this also excludes NaNs and infs)
finite_mask = np.all(np.isfinite(X), axis=1)
Xc, yc = X[finite_mask], y[finite_mask]
print(f"🔀 Training samples: {X.shape} → after finite filter: {Xc.shape}; positives={int((yc==1).sum())}, negatives={int((yc==0).sum())}")

if Xc.shape[0] == 0:
    raise RuntimeError("No usable training rows after filtering. Check predictor coverage & presence coordinates.")

if len(np.unique(yc)) < 2:
    raise RuntimeError("Only one class present in training data (either no presences or no backgrounds after filtering).")

# --- Drop constant predictors based on presence+background combined (Xc) ---
EPS = 1e-8
col_std = np.nanstd(Xc, axis=0)
keep_mask = col_std > EPS
dropped = [n for n, k in zip(names, keep_mask) if not k]
used    = [n for n, k in zip(names, keep_mask) if k]
if not np.any(keep_mask):
    raise RuntimeError("All predictors are constant over training data. Add at least one varying layer.")

if dropped:
    print(f"⚠️ Dropping constant predictors (std≈0 on training rows): {', '.join(dropped)}")

# Slice for modeling/scaling
Xc_model = Xc[:, keep_mask]
names_model = used
used_idx = np.flatnonzero(keep_mask)
drop_idx = np.flatnonzero(~keep_mask)

# --- Export extracted samples: RAW values ---
pres_lats = kept_coords[:, 0] if kept_coords.size else np.array([], dtype=np.float32)
pres_lons = kept_coords[:, 1] if kept_coords.size else np.array([], dtype=np.float32)

presence_df_raw = pd.DataFrame(presence_samples, columns=names)
presence_df_raw.insert(0, 'latitude',  pres_lats[:len(presence_df_raw)])
presence_df_raw.insert(1, 'longitude', pres_lons[:len(presence_df_raw)])
presence_df_raw.insert(2, 'label', 1)

absence_df_raw = pd.DataFrame(background_samples, columns=names)
absence_df_raw.insert(0, 'latitude',  bg_lats[:len(absence_df_raw)])
absence_df_raw.insert(1, 'longitude', bg_lons[:len(absence_df_raw)])
absence_df_raw.insert(2, 'label', 0)

samples_df_raw = pd.concat([presence_df_raw, absence_df_raw], ignore_index=True)
raw_csv = "outputs/sdm_point_samples_raw.csv"
samples_df_raw.to_csv(raw_csv, index=False)
print(f"📄 Point samples (RAW) saved to {raw_csv}")

# Small helper CSV solely for mapping background/absence points
absences_csv = "outputs/absence_points_coordinates.csv"
absence_df_raw[['latitude', 'longitude']].to_csv(absences_csv, index=False)
print(f"🗺️ Absence coordinates saved to {absences_csv}")

# --- Spatial cross-validation (blocked backgrounds, threshold chosen on TRAIN fold) ---
coords = kept_coords  # (n_presences, 2) as [lat, lon]
n_blocks = min(5, len(coords)) if len(coords) > 0 else 0
cv_aucs, cv_tsses, cv_kappas = [], [], []
cv_thresholds, cv_sensitivities, cv_specificities = [], [], []

# Build lon/lat for ALL valid pixels in the pool (for assigning blocks)
pool_rows = valid_idx // width
pool_cols = valid_idx %  width
pool_xs, pool_ys = rasterio.transform.xy(ref_transform, pool_rows, pool_cols, offset='center')
pool_lons_all = np.array(pool_xs); pool_lats_all = np.array(pool_ys)

if n_blocks >= 2:
    try:
        kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init="auto").fit(coords)
        blocks = kmeans.labels_  # presences' block labels
        # Assign every valid pixel in the pool to a block
        pool_blocks = kmeans.predict(np.column_stack([pool_lats_all, pool_lons_all]))

        gkf = GroupKFold(n_splits=n_blocks)
        for train_idx, test_idx in gkf.split(presence_samples,
                                             np.ones(len(presence_samples), dtype=int),
                                             groups=blocks):
            # split presence
            Xp_tr, Xp_te = presence_samples[train_idx], presence_samples[test_idx]
            test_blocks = set(blocks[test_idx])
            train_blocks = [b for b in range(n_blocks) if b not in test_blocks]

            # masks for pool by block
            mask_tr_pool = np.isin(pool_blocks, train_blocks)
            mask_te_pool = np.isin(pool_blocks, list(test_blocks))

            # background for train/test (draw only from their respective blocks)
            n_bt  = min(5 * len(train_idx), int(mask_tr_pool.sum()))
            n_bt2 = min(5 * len(test_idx),  int(mask_te_pool.sum()))

            # fallbacks if a block is tiny
            if n_bt < 1 or n_bt2 < 1:
                bt_idx  = np.random.choice(len(pool), size=max(1, 5*len(train_idx)), replace=False)
                bt_idx2 = np.random.choice(len(pool), size=max(1, 5*len(test_idx)),  replace=False)
            else:
                tr_inds = np.flatnonzero(mask_tr_pool)
                te_inds = np.flatnonzero(mask_te_pool)
                bt_idx  = np.random.choice(tr_inds, size=n_bt,  replace=False)
                bt_idx2 = np.random.choice(te_inds, size=n_bt2, replace=False)

            Xb_tr = pool[bt_idx]
            Xb_te = pool[bt_idx2]

            # combine (NOTE: CV uses raw features; scaling per fold below)
            X_tr = np.vstack([Xp_tr, Xb_tr])
            y_tr = np.concatenate([np.ones(len(Xp_tr), dtype=int), np.zeros(len(Xb_tr), dtype=int)])
            X_te = np.vstack([Xp_te, Xb_te])
            y_te = np.concatenate([np.ones(len(Xp_te), dtype=int), np.zeros(len(Xb_te), dtype=int)])

            # Filter finite rows
            mtr = np.all(np.isfinite(X_tr), axis=1)
            mte = np.all(np.isfinite(X_te), axis=1)
            X_tr, y_tr = X_tr[mtr], y_tr[mtr]
            X_te, y_te = X_te[mte], y_te[mte]

            # Drop constant columns per our *global* keep_mask to ensure consistency
            X_tr_model = X_tr[:, keep_mask]
            X_te_model = X_te[:, keep_mask]

            # scale per fold (fit on train, apply to train & test)
            scaler_cv = StandardScaler().fit(X_tr_model)
            X_tr_s = scaler_cv.transform(X_tr_model)
            X_te_s = scaler_cv.transform(X_te_model)

            # fit
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr_s, y_tr)

            # TRAIN-SET threshold (no peeking)
            p_tr = clf.predict_proba(X_tr_s)[:, 1]
            fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr, p_tr)
            bt = thr_tr[np.argmax(tpr_tr - fpr_tr)]  # Youden's J on TRAIN

            # Test predictions & metrics using that fixed threshold
            p_te = clf.predict_proba(X_te_s)[:, 1]
            auc_cv = roc_auc_score(y_te, p_te)
            yhat = (p_te >= bt).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_te, yhat, labels=[0, 1]).ravel()
            sens = tp/(tp+fn) if (tp+fn) else 0.0
            spec = tn/(tn+fp) if (tn+fp) else 0.0
            tss_cv = sens + spec - 1
            kapp = cohen_kappa_score(y_te, yhat)

            print(
                f"[CV] fold={len(cv_aucs)+1} "
                f"test presences={np.sum(y_te==1)} backgrounds={np.sum(y_te==0)} "
                f"auc={auc_cv:.3f} thr={bt:.3f} sens={sens:.3f} spec={spec:.3f}"
            )
            
            cv_aucs.append(float(auc_cv))
            cv_tsses.append(float(tss_cv))
            cv_kappas.append(float(kapp))
            cv_thresholds.append(float(bt))
            cv_sensitivities.append(float(sens))
            cv_specificities.append(float(spec))

        # Save CV summary (means & sds)
        cv_summary = pd.DataFrame([{
            "n_folds": n_blocks,
            "AUC_mean":   float(np.mean(cv_aucs)),           "AUC_sd":   float(np.std(cv_aucs)),
            "TSS_mean":   float(np.mean(cv_tsses)),          "TSS_sd":   float(np.std(cv_tsses)),
            "Kappa_mean": float(np.mean(cv_kappas)),         "Kappa_sd": float(np.std(cv_kappas)),
            "Threshold_mean":   float(np.mean(cv_thresholds)),"Threshold_sd":   float(np.std(cv_thresholds)),
            "Sensitivity_mean": float(np.mean(cv_sensitivities)), "Sensitivity_sd": float(np.std(cv_sensitivities)),
            "Specificity_mean": float(np.mean(cv_specificities)), "Specificity_sd": float(np.std(cv_specificities)),
        }])
        cv_summary.to_csv(perf_cv_csv, index=False)
        print(f"📊 Cross-validated metrics saved to {perf_cv_csv}")
    except Exception as e:
        print(f"⚠️ Skipping spatial CV due to issue: {e}")
        cv_summary = pd.DataFrame([{
            "n_folds": 0,
            "AUC_mean": np.nan, "AUC_sd": np.nan,
            "TSS_mean": np.nan, "TSS_sd": np.nan,
            "Kappa_mean": np.nan, "Kappa_sd": np.nan,
            "Threshold_mean": np.nan, "Threshold_sd": np.nan,
            "Sensitivity_mean": np.nan, "Sensitivity_sd": np.nan,
            "Specificity_mean": np.nan, "Specificity_sd": np.nan,
        }])
        cv_summary.to_csv(perf_cv_csv, index=False)
        print("⚠️ Wrote NaNs to CV metrics due to CV failure.")
else:
    cv_summary = pd.DataFrame([{
        "n_folds": 0,
        "AUC_mean": np.nan, "AUC_sd": np.nan,
        "TSS_mean": np.nan, "TSS_sd": np.nan,
        "Kappa_mean": np.nan, "Kappa_sd": np.nan,
        "Threshold_mean": np.nan, "Threshold_sd": np.nan,
        "Sensitivity_mean": np.nan, "Sensitivity_sd": np.nan,
        "Specificity_mean": np.nan, "Specificity_sd": np.nan,
    }])
    cv_summary.to_csv(perf_cv_csv, index=False)
    print("⚠️ Not enough presence points for spatial CV (need ≥ 2 blocks). Wrote NaNs to CV metrics.")

# --- Final scaler & model on all filtered samples (scale everything; only kept cols) ---
scaler = StandardScaler().fit(Xc_model)
Xc_s = scaler.transform(Xc_model)
model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xc_s, yc)
joblib.dump(scaler, scaler_pkl)
print(f"💾 Saved scaler to {scaler_pkl}")

# --- Export standardization stats (mark dropped constants) ---
# Map scaler stats back to full predictor list
mean_full = np.zeros(len(names), dtype=float)
std_full  = np.zeros(len(names), dtype=float)
mean_full[used_idx] = scaler.mean_.astype(float)
std_full[used_idx]  = scaler.scale_.astype(float)
# Dropped columns remain 0.0 mean/std (z-score defined as 0 downstream)
stats_csv = "outputs/standardization_stats.csv"
stats_df = pd.DataFrame({
    "predictor": names,
    "mean":  mean_full,
    "std":   std_full,
    "dropped_constant": [not k for k in keep_mask]
})
stats_df.to_csv(stats_csv, index=False)
print(f"📄 Standardization stats saved to {stats_csv}")

# --- Export extracted samples: STANDARDIZED values (reconstruct full columns) ---
def _std_full_from_model(arr_full: np.ndarray) -> np.ndarray:
    """Standardize using kept columns then expand back to full width (dropped → 0)."""
    if arr_full.size == 0:
        return arr_full.reshape((0, len(names)))
    arr_kept = arr_full[:, keep_mask]  # safe: rows known finite from earlier extraction
    arr_kept_s = scaler.transform(arr_kept)
    out = np.zeros((arr_full.shape[0], len(names)), dtype=np.float32)
    out[:, used_idx] = arr_kept_s.astype(np.float32)
    # dropped columns already 0 z-score
    return out

presence_std = _std_full_from_model(presence_samples)
absence_std  = _std_full_from_model(background_samples)

presence_df_std = pd.DataFrame(presence_std, columns=names)
presence_df_std.insert(0, 'latitude',  pres_lats[:len(presence_df_std)])
presence_df_std.insert(1, 'longitude', pres_lons[:len(presence_df_std)])
presence_df_std.insert(2, 'label', 1)

absence_df_std = pd.DataFrame(absence_std, columns=names)
absence_df_std.insert(0, 'latitude',  bg_lats[:len(absence_df_std)])
absence_df_std.insert(1, 'longitude', bg_lons[:len(absence_df_std)])
absence_df_std.insert(2, 'label', 0)

samples_df_std = pd.concat([presence_df_std, absence_df_std], ignore_index=True)
std_csv = "outputs/sdm_point_samples_standardized.csv"
samples_df_std.to_csv(std_csv, index=False)
print(f"📄 Point samples (STANDARDIZED) saved to {std_csv}")

# --- Fitted (in-sample) metrics on standardized training data ---
y_prob = model.predict_proba(Xc_s)[:, 1]
auc    = roc_auc_score(yc, y_prob)
fpr, tpr, thr = roc_curve(yc, y_prob)
best_thr = thr[np.argmax(tpr - fpr)]
yhat = (y_prob >= best_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(yc, yhat, labels=[0, 1]).ravel()
sens  = tp/(tp+fn) if (tp+fn) else 0.0
spec  = tn/(tn+fp) if (tn+fp) else 0.0
tss   = sens + spec - 1
kappa = cohen_kappa_score(yc, yhat)

fitted_df = pd.DataFrame([{
    'AUC_fitted':         float(auc),
    'Threshold_fitted':   float(best_thr),
    'Sensitivity_fitted': float(sens),
    'Specificity_fitted': float(spec),
    'TSS_fitted':         float(tss),
    'Kappa_fitted':       float(kappa)
}])
fitted_df.to_csv(perf_fit_csv, index=False)
print(f"📊 Fitted (in-sample) metrics saved to {perf_fit_csv}")

# Also save a compatibility file for ZIP / legacy readers
fitted_df.to_csv(compat_perf, index=False)
print(f"📊 Fitted (compat) metrics saved to {compat_perf}")

# --- Write out coefficients (map back to full feature list; dropped → 0) ---
intercept = model.intercept_[0]
coef_used = model.coef_.flatten()
full_coef = np.zeros(len(names), dtype=float)
full_coef[used_idx] = coef_used
predictors = ['Intercept'] + names
coef_vals  = np.concatenate([[intercept], full_coef])
coef_df = pd.DataFrame({'predictor': predictors, 'coefficient': coef_vals})
coef_df.to_csv(coef_csv, index=False)
print(f"📊 Coefficients saved to {coef_csv}")

# --- Save final suitability map (using scaled raster pixels; only kept cols) ---
pred_flat = np.full(flat.shape[0], np.nan, dtype=np.float32)
pool_model = pool[:, keep_mask]
pool_s = scaler.transform(pool_model)
pred_flat[valid_mask] = model.predict_proba(pool_s)[:, 1].astype(np.float32)
pred_map = pred_flat.reshape((height, width))
with rasterio.open(output_map, "w", **profile) as dst:
    dst.write(pred_map, 1)
print(f"🎯 Final suitability map saved to {output_map}")

# --- Export standardized rasters (z-scored; dropped → 0) ---
os.makedirs(std_ras_dir, exist_ok=True)
print(f"💾 Writing standardized rasters to {std_ras_dir}")

pool_s_full = np.zeros((pool.shape[0], len(names)), dtype=np.float32)
pool_s_full[:, used_idx] = pool_s.astype(np.float32)  # dropped stay at 0

for j, name in enumerate(names):
    band_flat = np.full(flat.shape[0], np.float32(np.nan), dtype=np.float32)
    band_flat[valid_mask] = pool_s_full[:, j].astype(np.float32)
    band = band_flat.reshape((height, width))
    out_path = os.path.join(std_ras_dir, f"{name}_z.tif")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(band, 1)
    print(f"    • {name}_z.tif")
