# scripts/run_logistic_sdm.py
# Robust SDM pipeline: spatial-blocks CV with K-fold fallback; standardizes ALL used predictors.
# Removes fragile background lon/lat export (writes an empty CSV instead) to avoid run crashes.
# Guarantees 2-D shapes before stacking to prevent vstack errors.

import os
import numpy as np
import pandas as pd
import rasterio
import joblib

from rasterio.transform import rowcol
from rasterio.warp import reproject, Resampling as WarpResampling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             cohen_kappa_score)
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Paths ---
csv_path     = "inputs/presence_points.csv"
raster_dir   = "predictor_rasters/wgs84"
perf_cv_csv  = "outputs/performance_metrics_cv.csv"
perf_fit_csv = "outputs/performance_metrics_fitted.csv"
compat_perf  = "outputs/performance_metrics.csv"   # legacy/zip
coef_csv     = "outputs/coefficients.csv"
scaler_pkl   = "outputs/standard_scaler.pkl"
output_map   = "outputs/suitability_map_wgs84.tif"
std_ras_dir  = "outputs/standardized_rasters_wgs84"
absences_csv = "outputs/absence_points_coordinates.csv"  # map overlay only
os.makedirs("outputs", exist_ok=True)

# ---------- Constants & helpers ----------
EPS = 1e-8  # small value for zero-variance checks

def _is_categorical(name: str) -> bool:
    n = name.lower()
    return ("landcover" in n) or n.endswith("_onehot") or ("class" in n)

def _zscore_train_test(X_tr, X_te):
    """Fit z-score on TRAIN; clamp tiny std to 1.0; apply to TRAIN/TEST."""
    mu = np.nanmean(X_tr, axis=0)
    sd = np.nanstd(X_tr, axis=0)
    sd = np.where(sd < EPS, 1.0, sd)
    return (X_tr - mu) / sd, (X_te - mu) / sd, mu, sd

def _compute_metrics_from_probs(y_true, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    sens = tp/(tp+fn) if (tp+fn) else 0.0
    spec = tn/(tn+fp) if (tn+fp) else 0.0
    tss  = sens + spec - 1
    kapp = cohen_kappa_score(y_true, yhat)
    auc  = roc_auc_score(y_true, p)
    return auc, tss, kapp, sens, spec

def _cv_summary_frame(n_folds, method, note, aucs, tsses, kappas, thrs, sens, spec):
    df_out = pd.DataFrame([{
        "n_folds": int(n_folds),
        "cv_method": method,
        "cv_note": note or "",
        "AUC_mean":   float(np.mean(aucs)) if len(aucs) else np.nan,
        "AUC_sd":     float(np.std(aucs))  if len(aucs) else np.nan,
        "TSS_mean":   float(np.mean(tsses)) if len(tsses) else np.nan,
        "TSS_sd":     float(np.std(tsses))  if len(tsses) else np.nan,
        "Kappa_mean": float(np.mean(kappas)) if len(kappas) else np.nan,
        "Kappa_sd":   float(np.std(kappas))  if len(kappas) else np.nan,
        "Threshold_mean":   float(np.mean(thrs)) if len(thrs) else np.nan,
        "Threshold_sd":     float(np.std(thrs))  if len(thrs) else np.nan,
        "Sensitivity_mean": float(np.mean(sens)) if len(sens) else np.nan,
        "Sensitivity_sd":   float(np.std(sens))  if len(sens) else np.nan,
        "Specificity_mean": float(np.mean(spec)) if len(spec) else np.nan,
        "Specificity_sd":   float(np.std(spec))  if len(spec) else np.nan,
    }])
    # Diagnostics for UI
    df_out["n_input_presences"] = int(len(df))
    df_out["n_kept_presences"]  = int(len(presence_samples))
    df_out["n_predictors_used"] = int(len(used))
    return df_out

# --- Load presence points ---
print("[STEP] load-presences")
df   = pd.read_csv(csv_path)
lats = df['latitude'].values
lons = df['longitude'].values
print(f"📍 Loaded {len(df)} presence points.")

# --- Reference grid ---
print("[STEP] load-reference-grid")
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

# --- Stack predictors (align every layer to reference grid) ---
print("[STEP] stack-predictors")
layers, names = [], []
for path in rasters:
    name = os.path.splitext(os.path.basename(path))[0]
    with rasterio.open(path) as src:
        src_crs = src.crs if src.crs is not None else ref_crs
        if src.crs is None:
            print(f"⚠️  {name}.tif has no CRS; assuming {ref_crs}")
        dst = np.full((height, width), np.nan, dtype=np.float32)
        resamp = WarpResampling.nearest if _is_categorical(name) else WarpResampling.bilinear
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src_crs,
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
print("[STEP] extract-presence-samples")
presence_samples = []
kept_coords = []  # (lat, lon) on valid pixels
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
kept_coords = np.array(kept_coords, dtype=np.float64) if len(kept_coords) else np.empty((0, 2), dtype=np.float64)
print(f"📍 Presence samples: {presence_samples.shape}")

if presence_samples.shape[0] == 0:
    raise RuntimeError(
        "No presence points fell on pixels where all predictors are valid. "
        "Check overlap and predictor coverage."
    )

# --- Background sampling pool (all valid pixels) ---
print("[STEP] build-valid-pool")
flat       = stack.reshape(-1, stack.shape[-1]).astype(np.float32)
valid_mask = ~np.any(np.isnan(flat), axis=1)
valid_idx  = np.flatnonzero(valid_mask)
pool       = flat[valid_mask]
pool_len   = len(pool)
print(f"✅ Valid pixel pool: {pool_len}")

if pool_len == 0:
    raise RuntimeError(
        "No valid pixels where all predictors overlap (stack fully NaN). "
        "Likely a categorical class has no coverage in your AOI."
    )

# Choose background for the *raw sample table* (map plotting, not CV)
print("[STEP] sample-background-for-tables")
np.random.seed(42)
n_bg       = min(5 * max(1, len(presence_samples)), pool_len)
bg_indices = np.random.choice(pool_len, size=n_bg, replace=False)  # OK if n_bg == 0
background_samples = pool[bg_indices].astype(np.float32)
print(f"🌎 Background samples (for tables): {background_samples.shape}")

# Always write a valid (possibly empty) absence CSV so UI never crashes
pd.DataFrame({"latitude": [], "longitude": []}).to_csv(absences_csv, index=False)
print(f"🗺️ Absence coordinates (overlay) skipped — wrote empty file at {absences_csv}")

# --- Prepare data (combined presence + background) ---
print("[STEP] build-train-table")
# Ensure both sample matrices are 2-D with identical feature width
n_features = stack.shape[-1]
presence_samples   = np.asarray(presence_samples,   dtype=np.float32).reshape(-1, n_features)
background_samples = np.asarray(background_samples, dtype=np.float32).reshape(-1, n_features)

X = np.vstack([presence_samples, background_samples])
y = np.concatenate([
    np.ones(len(presence_samples), dtype=int),
    np.zeros(len(background_samples), dtype=int)
])

# Guard rows: require finite values
finite_mask = np.all(np.isfinite(X), axis=1)
Xc, yc = X[finite_mask], y[finite_mask]
print(f"🔀 Training samples: {X.shape} → after finite filter: {Xc.shape}; "
      f"positives={int((yc==1).sum())}, negatives={int((yc==0).sum())}")

if Xc.shape[0] == 0:
    raise RuntimeError("No usable training rows after filtering.")
if len(np.unique(yc)) < 2:
    raise RuntimeError("Only one class present in training data.")

# --- Drop constant predictors based on presence+background combined (Xc) ---
print("[STEP] drop-constant-predictors")
col_std = np.nanstd(Xc, axis=0)
keep_mask = col_std > EPS
dropped = [n for n, k in zip(names, keep_mask) if not k]
used    = [n for n, k in zip(names, keep_mask) if k]
if not np.any(keep_mask):
    raise RuntimeError("All predictors are constant over training data.")
if dropped:
    print(f"⚠️ Dropping constant predictors: {', '.join(dropped)}")

# Slice for modeling/scaling
Xc_model = Xc[:, keep_mask]
names_model = used
used_idx = np.flatnonzero(keep_mask)

# --- Export extracted samples: RAW values ---
print("[STEP] export-sample-tables-raw")
pres_lats = kept_coords[:, 0] if kept_coords.size else np.array([], dtype=np.float32)
pres_lons = kept_coords[:, 1] if kept_coords.size else np.array([], dtype=np.float32)

presence_df_raw = pd.DataFrame(presence_samples, columns=names)
presence_df_raw.insert(0, 'latitude',  pres_lats[:len(presence_df_raw)])
presence_df_raw.insert(1, 'longitude', pres_lons[:len(presence_df_raw)])
presence_df_raw.insert(2, 'label', 1)

absence_df_raw = pd.DataFrame(background_samples, columns=names)
absence_df_raw.insert(0, 'latitude',  np.array([], dtype=np.float64)[:len(absence_df_raw)])
absence_df_raw.insert(1, 'longitude', np.array([], dtype=np.float64)[:len(absence_df_raw)])
absence_df_raw.insert(2, 'label', 0)

samples_df_raw = pd.concat([presence_df_raw, absence_df_raw], ignore_index=True)
samples_df_raw.to_csv("outputs/sdm_point_samples_raw.csv", index=False)
print(f"📄 Point samples (RAW) saved to outputs/sdm_point_samples_raw.csv")

# ---------- Cross-validation ----------
print("[STEP] cross-validation")
coords = kept_coords  # (n_presences, 2) as [lat, lon]
cv_aucs, cv_tsses, cv_kappas = [], [], []
cv_thresholds, cv_sensitivities, cv_specificities = [], [], []

def _run_repeated_kfold_fallback():
    print("ℹ️ Falling back to Repeated Stratified K-fold (non-spatial).")
    Np = len(presence_samples)
    if Np < 3:
        note = "Too few presences; reporting NaNs."
        out = _cv_summary_frame(0, "repeated_kfold", note, [], [], [], [], [], [])
        out.to_csv(perf_cv_csv, index=False)
        print(f"📊 Wrote CV fallback (NaNs) to {perf_cv_csv}")
        return

    n_splits  = max(3, min(5, max(2, Np // 5)))
    n_repeats = 5 if Np >= 10 else 3
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    aucs = []; tsses = []; kappas = []; thrs = []; sensv = []; specv = []
    for fold_i, (tr_idx, te_idx) in enumerate(rskf.split(presence_samples, np.ones(Np, dtype=int)), start=1):
        Xp_tr = presence_samples[tr_idx]; Xp_te = presence_samples[te_idx]
        rng = np.random.default_rng(1000 + fold_i)
        n_bt_tr = min(5 * len(tr_idx), pool_len)
        n_bt_te = min(5 * len(te_idx),  pool_len)
        bt_idx  = rng.choice(pool_len, size=n_bt_tr, replace=False)
        bt_idx2 = rng.choice(pool_len, size=n_bt_te, replace=False)
        Xb_tr = pool[bt_idx]; Xb_te = pool[bt_idx2]

        X_tr = np.vstack([Xp_tr, Xb_tr]); y_tr = np.concatenate([np.ones(len(Xp_tr), dtype=int), np.zeros(len(Xb_tr), dtype=int)])
        X_te = np.vstack([Xp_te, Xb_te]); y_te = np.concatenate([np.ones(len(Xp_te), dtype=int), np.zeros(len(Xb_te), dtype=int)])

        mtr = np.all(np.isfinite(X_tr), axis=1); mte = np.all(np.isfinite(X_te), axis=1)
        X_tr, y_tr = X_tr[mtr], y_tr[mtr]; X_te, y_te = X_te[mte], y_te[mte]
        if np.unique(y_te).size < 2 or np.unique(y_tr).size < 2: continue

        X_tr_s, X_te_s, _, _ = _zscore_train_test(X_tr[:, keep_mask], X_te[:, keep_mask])

        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr_s, y_tr)
        p_tr = clf.predict_proba(X_tr_s)[:, 1]
        fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr, p_tr)
        bt = thr_tr[np.argmax(tpr_tr - fpr_tr)]

        p_te = clf.predict_proba(X_te_s)[:, 1]
        auc, tss, kapp, sens, spec = _compute_metrics_from_probs(y_te, p_te, bt)

        aucs.append(float(auc)); tsses.append(float(tss)); kappas.append(float(kapp))
        thrs.append(float(bt));  sensv.append(float(sens)); specv.append(float(spec))

    note = f"Used non-spatial Repeated Stratified K-fold (splits={n_splits}, repeats={n_repeats})."
    _cv_summary_frame(n_splits * n_repeats, "repeated_kfold", note, aucs, tsses, kappas, thrs, sensv, specv).to_csv(perf_cv_csv, index=False)
    print(f"📊 Wrote CV fallback (Repeated K-fold) to {perf_cv_csv}")

did_spatial = False
if len(coords) >= 2:
    try:
        print("[STEP] spatial-cv")
        n_blocks = min(5, len(coords))
        kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init=10).fit(coords)
        blocks = kmeans.labels_
        gkf = GroupKFold(n_splits=n_blocks)
        for train_idx, test_idx in gkf.split(presence_samples, np.ones(len(presence_samples), dtype=int), groups=blocks):
            Xp_tr, Xp_te = presence_samples[train_idx], presence_samples[test_idx]
            # For robustness with categorical layers, just use global random backgrounds per fold.
            rng = np.random.default_rng(10_000 + int(test_idx[0]))
            n_bt  = min(5 * len(train_idx), pool_len)
            n_bt2 = min(5 * len(test_idx),  pool_len)
            bt_idx  = rng.choice(pool_len, size=n_bt,  replace=False)
            bt_idx2 = rng.choice(pool_len, size=n_bt2, replace=False)
            Xb_tr = pool[bt_idx]; Xb_te = pool[bt_idx2]

            X_tr = np.vstack([Xp_tr, Xb_tr]); y_tr = np.concatenate([np.ones(len(Xp_tr), dtype=int), np.zeros(len(Xb_tr), dtype=int)])
            X_te = np.vstack([Xp_te, Xb_te]); y_te = np.concatenate([np.ones(len(Xp_te), dtype=int), np.zeros(len(Xb_te), dtype=int)])

            mtr = np.all(np.isfinite(X_tr), axis=1); mte = np.all(np.isfinite(X_te), axis=1)
            X_tr, y_tr = X_tr[mtr], y_tr[mtr]; X_te, y_te = X_te[mte], y_te[mte]
            if np.unique(y_te).size < 2 or np.unique(y_tr).size < 2: continue

            X_tr_s, X_te_s, _, _ = _zscore_train_test(X_tr[:, keep_mask], X_te[:, keep_mask])

            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr_s, y_tr)

            p_tr = clf.predict_proba(X_tr_s)[:, 1]
            fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr, p_tr)
            bt = thr_tr[np.argmax(tpr_tr - fpr_tr)]

            p_te = clf.predict_proba(X_te_s)[:, 1]
            auc = roc_auc_score(y_te, p_te)
            yhat = (p_te >= bt).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_te, yhat, labels=[0, 1]).ravel()
            sens = tp/(tp+fn) if (tp+fn) else 0.0
            spec = tn/(tn+fp) if (tn+fp) else 0.0
            tss  = sens + spec - 1
            kapp = cohen_kappa_score(y_te, yhat)

            cv_aucs.append(float(auc)); cv_tsses.append(float(tss)); cv_kappas.append(float(kapp))
            cv_thresholds.append(float(bt)); cv_sensitivities.append(float(sens)); cv_specificities.append(float(spec))

        did_spatial = True
        _cv_summary_frame(len(cv_aucs), "spatial_blocks", "", cv_aucs, cv_tsses, cv_kappas,
                          cv_thresholds, cv_sensitivities, cv_specificities).to_csv(perf_cv_csv, index=False)
        print(f"📊 Cross-validated (spatial blocks) metrics saved to {perf_cv_csv}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"⚠️ Spatial CV failed with error: {e}")
        _run_repeated_kfold_fallback()
else:
    print("ℹ️ Not enough presences for spatial CV (need ≥2 blocks).")
    _run_repeated_kfold_fallback()

# --- Final scaler & model on all filtered samples (only kept cols) ---
print("[STEP] fit-final-model")
scaler = StandardScaler().fit(Xc_model)
Xc_s = scaler.transform(Xc_model)
model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xc_s, yc)
joblib.dump(scaler, scaler_pkl)
print(f"💾 Saved scaler to {scaler_pkl}")

# --- Export standardization stats ---
print("[STEP] export-standardization-stats")
mean_full = np.zeros(len(names), dtype=float)
std_full  = np.zeros(len(names), dtype=float)
mean_full[used_idx] = scaler.mean_.astype(float)
std_full[used_idx]  = scaler.scale_.astype(float)
pd.DataFrame({
    "predictor": names,
    "mean":  mean_full,
    "std":   std_full,
    "dropped_constant": [not k for k in keep_mask]
}).to_csv("outputs/standardization_stats.csv", index=False)
print(f"📄 Standardization stats saved to outputs/standardization_stats.csv")

# --- Export standardized sample tables ---
print("[STEP] export-sample-tables-standardized")
def _std_full(arr_full: np.ndarray) -> np.ndarray:
    if arr_full.size == 0:
        return arr_full.reshape((0, len(names)))
    arr_kept = arr_full[:, keep_mask]
    arr_kept_s = scaler.transform(arr_kept)
    out = np.zeros((arr_full.shape[0], len(names)), dtype=np.float32)
    out[:, used_idx] = arr_kept_s.astype(np.float32)
    return out

presence_std = _std_full(presence_samples)
absence_std  = _std_full(background_samples)

presence_df_std = pd.DataFrame(presence_std, columns=names)
presence_df_std.insert(0, 'latitude',  pres_lats[:len(presence_df_std)] if kept_coords.size else [])
presence_df_std.insert(1, 'longitude', pres_lons[:len(presence_df_std)] if kept_coords.size else [])
presence_df_std.insert(2, 'label', 1)

absence_df_std = pd.DataFrame(absence_std, columns=names)
absence_df_std.insert(0, 'latitude',  [])
absence_df_std.insert(1, 'longitude', [])
absence_df_std.insert(2, 'label', 0)

pd.concat([presence_df_std, absence_df_std], ignore_index=True).to_csv("outputs/sdm_point_samples_standardized.csv", index=False)
print(f"📄 Point samples (STANDARDIZED) saved to outputs/sdm_point_samples_standardized.csv")

# --- Fitted (in-sample) metrics ---
print("[STEP] fitted-metrics")
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

pd.DataFrame([{
    'AUC_fitted':         float(auc),
    'Threshold_fitted':   float(best_thr),
    'Sensitivity_fitted': float(sens),
    'Specificity_fitted': float(spec),
    'TSS_fitted':         float(tss),
    'Kappa_fitted':       float(kappa)
}]).to_csv(perf_fit_csv, index=False)
print(f"📊 Fitted (in-sample) metrics saved to {perf_fit_csv}")
pd.read_csv(perf_fit_csv).to_csv(compat_perf, index=False)
print(f"📊 Fitted (compat) metrics saved to {compat_perf}")

# --- Coefficients ---
print("[STEP] export-coefficients")
intercept = model.intercept_[0]
coef_used = model.coef_.flatten()
full_coef = np.zeros(len(names), dtype=float)
full_coef[used_idx] = coef_used
pd.DataFrame({'predictor': ['Intercept'] + names,
             'coefficient': np.concatenate([[intercept], full_coef])}).to_csv(coef_csv, index=False)
print(f"📊 Coefficients saved to {coef_csv}")

# --- Suitability map ---
print("[STEP] write-suitability-map")
pred_flat = np.full(flat.shape[0], np.nan, dtype=np.float32)
pool_model = pool[:, keep_mask]
pool_s = scaler.transform(pool_model)
pred_flat[valid_mask] = model.predict_proba(pool_s)[:, 1].astype(np.float32)
pred_map = pred_flat.reshape((height, width))
with rasterio.open(output_map, "w", **profile) as dst:
    dst.write(pred_map, 1)
print(f"🎯 Final suitability map saved to {output_map}")

# --- Standardized rasters ---
print("[STEP] write-standardized-rasters")
os.makedirs(std_ras_dir, exist_ok=True)
pool_s_full = np.zeros((pool.shape[0], len(names)), dtype=np.float32)
pool_s_full[:, used_idx] = pool_s.astype(np.float32)
for j, name in enumerate(names):
    band_flat = np.full(flat.shape[0], np.float32(np.nan), dtype=np.float32)
    band_flat[valid_mask] = pool_s_full[:, j].astype(np.float32)
    with rasterio.open(os.path.join(std_ras_dir, f"{name}_z.tif"), "w", **profile) as dst:
        dst.write(band_flat.reshape((height, width)), 1)
    print(f"    • {name}_z.tif")
