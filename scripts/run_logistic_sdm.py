# scripts/run_logistic_sdm.py
# Spatial-blocks CV with K-fold fallback. Standardizes ALL predictors
# (including binary/categorical). Constant features are dropped globally;
# per-fold scaling uses an ε-guard to avoid NaNs when a column is constant in a fold.
# Spatial CV assigns blocks lazily (batch candidates only), avoiding huge kmeans.predict over the full pool.

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
compat_perf  = "outputs/performance_metrics.csv"      # legacy/zip
coef_csv     = "outputs/coefficients.csv"
scaler_pkl   = "outputs/standard_scaler.pkl"
output_map   = "outputs/suitability_map_wgs84.tif"
std_ras_dir  = "outputs/standardized_rasters_wgs84"
os.makedirs("outputs", exist_ok=True)

# ---------- Constants & helpers ----------
EPS = 1e-8  # small value for zero-variance checks

def _is_categorical(name: str) -> bool:
    n = name.lower()
    return ("landcover" in n) or n.endswith("_onehot") or ("class" in n)

def _zscore_train_test(X_tr, X_te):
    """
    Fit z-score on TRAIN; clamp tiny std to 1.0; apply to TRAIN/TEST.
    Returns X_tr_s, X_te_s, mu, sd
    """
    mu = np.nanmean(X_tr, axis=0)
    sd = np.nanstd(X_tr, axis=0)
    sd = np.where(sd < EPS, 1.0, sd)  # avoid divide-by-zero
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

def _lin_to_coords(linear_idx, width, transform):
    """Linear index -> (lon, lat) with integer row/col; safe on empty arrays."""
    linear_idx = np.asarray(linear_idx, dtype=np.int64)
    if linear_idx.size == 0:
        return (np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64))
    rows = linear_idx // width
    cols = linear_idx %  width
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
    # ensure numpy arrays
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)

def _sample_by_blocks(kmeans: KMeans, blocks_wanted, n_needed, rng, pool_len, valid_idx, width, transform, max_batch=100_000, max_tries=30):
    """
    Lazily sample candidate background pixels from the pool (by index),
    predict their block labels, and keep those that match blocks_wanted.
    Avoids predicting labels for the entire pool.
    """
    if n_needed <= 0 or pool_len == 0:
        return np.empty((0,), dtype=np.int64)
    collected = []
    tries = 0
    size_per_try = min(max_batch, pool_len)
    while len(collected) < n_needed and tries < max_tries:
        batch_idx = rng.choice(pool_len, size=size_per_try, replace=False)
        lin = valid_idx[batch_idx]
        xs, ys = _lin_to_coords(lin, width, transform)  # lon, lat
        coords = np.column_stack([ys, xs])  # kmeans was fit on [lat, lon]
        pred = kmeans.predict(coords)
        mask = np.isin(pred, blocks_wanted)
        if mask.any():
            selected = batch_idx[mask]
            collected.extend(selected.tolist())
        tries += 1
        if tries == 10:
            size_per_try = max(10_000, size_per_try // 2)
        if tries == 20:
            size_per_try = max(5_000, size_per_try // 2)
    if len(collected) < n_needed:
        need = n_needed - len(collected)
        fallback = rng.choice(pool_len, size=need, replace=False)
        collected.extend(fallback.tolist())
    return np.asarray(collected[:n_needed], dtype=np.int64)

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
        # Handle missing CRS by assuming reference CRS (common with some GeoTIFF exports)
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
kept_coords = np.array(kept_coords, dtype=np.float32) if len(kept_coords) else np.empty((0, 2), dtype=np.float32)
print(f"📍 Presence samples: {presence_samples.shape}")

if presence_samples.shape[0] == 0:
    raise RuntimeError(
        "No presence points fell on pixels where all predictors are valid. "
        "Check that your points overlap the fetched rasters and that predictor coverage isn’t all-NaN in that area."
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
        "No valid pixels where all predictors overlap (the stack is fully NaN). "
        "This usually means one layer has no data over your area of interest. "
        "Try removing the offending layer(s) or widening the extent."
    )

# Choose background for the *raw sample table* (map plotting, not CV)
print("[STEP] sample-background-for-tables")
np.random.seed(42)
n_bg       = min(5 * max(1, len(presence_samples)), pool_len)
bg_indices = np.random.choice(pool_len, size=n_bg, replace=False)  # size may be 0; that's okay
background_samples = pool[bg_indices].astype(np.float32)
print(f"🌎 Background samples (for tables): {background_samples.shape}")

# Convert chosen background indices back to row/col → lon/lat (for export)
bg_linear = valid_idx[bg_indices]
if bg_linear.size == 0:
    bg_lons = np.empty((0,), dtype=np.float64)
    bg_lats = np.empty((0,), dtype=np.float64)
else:
    bg_xs, bg_ys = _lin_to_coords(bg_linear, width, ref_transform)  # lon, lat
    bg_lons = np.array(bg_xs); bg_lats = np.array(bg_ys)

# --- Prepare data (combined presence + background) ---
print("[STEP] build-train-table")
X = np.vstack([presence_samples, background_samples])
y = np.concatenate([
    np.ones(len(presence_samples), dtype=int),
    np.zeros(len(background_samples), dtype=int)
])

# Guard rows: require finite values (also excludes NaNs and infs)
finite_mask = np.all(np.isfinite(X), axis=1)
Xc, yc = X[finite_mask], y[finite_mask]
print(f"🔀 Training samples: {X.shape} → after finite filter: {Xc.shape}; "
      f"positives={int((yc==1).sum())}, negatives={int((yc==0).sum())}")

if Xc.shape[0] == 0:
    raise RuntimeError("No usable training rows after filtering. Check predictor coverage & presence coordinates.")
if len(np.unique(yc)) < 2:
    raise RuntimeError("Only one class present in training data (either no presences or no backgrounds after filtering).")

# --- Drop constant predictors based on presence+background combined (Xc) ---
print("[STEP] drop-constant-predictors")
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

# --- Export extracted samples: RAW values ---
print("[STEP] export-sample-tables-raw")
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

# ---------- Cross-validation ----------
print("[STEP] cross-validation")
coords = kept_coords  # (n_presences, 2) as [lat, lon]
cv_aucs, cv_tsses, cv_kappas = [], [], []
cv_thresholds, cv_sensitivities, cv_specificities = [], [], []

def _run_repeated_kfold_fallback():
    """Repeated Stratified K-fold fallback (non-spatial)."""
    print("ℹ️ Falling back to Repeated Stratified K-fold CV (non-spatial).")
    Np = len(presence_samples)
    if Np < 3:
        note = "Too few presences for K-fold; reporting NaNs."
        print(f"⚠️ {note}")
        out = _cv_summary_frame(0, "repeated_kfold", note, [], [], [], [], [], [])
        out.to_csv(perf_cv_csv, index=False)
        print(f"📊 Wrote CV fallback (NaNs) to {perf_cv_csv}")
        return

    n_splits  = max(3, min(5, max(2, Np // 5)))
    n_repeats = 5 if Np >= 10 else 3

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    aucs = []; tsses = []; kappas = []; thrs = []; sensv = []; specv = []

    # We re-sample background each fold to keep priors aligned with train/test sizes
    for fold_i, (tr_idx, te_idx) in enumerate(rskf.split(presence_samples, np.ones(Np, dtype=int)), start=1):
        Xp_tr = presence_samples[tr_idx]
        Xp_te = presence_samples[te_idx]

        # global background sampling (non-spatial)
        rng = np.random.default_rng(1000 + fold_i)
        n_bt_tr = min(5 * len(tr_idx), pool_len)
        n_bt_te = min(5 * len(te_idx),  pool_len)
        bt_idx  = rng.choice(pool_len, size=n_bt_tr, replace=False)
        bt_idx2 = rng.choice(pool_len, size=n_bt_te, replace=False)
        Xb_tr = pool[bt_idx]
        Xb_te = pool[bt_idx2]

        # Combine
        X_tr = np.vstack([Xp_tr, Xb_tr])
        y_tr = np.concatenate([np.ones(len(Xp_tr), dtype=int), np.zeros(len(Xb_tr), dtype=int)])
        X_te = np.vstack([Xp_te, Xb_te])
        y_te = np.concatenate([np.ones(len(Xp_te), dtype=int), np.zeros(len(Xb_te), dtype=int)])

        # Finite filter
        mtr = np.all(np.isfinite(X_tr), axis=1)
        mte = np.all(np.isfinite(X_te), axis=1)
        X_tr, y_tr = X_tr[mtr], y_tr[mtr]
        X_te, y_te = X_te[mte], y_te[mte]

        # Ensure both classes exist in TEST and TRAIN
        if np.unique(y_te).size < 2:
            print("[Kfold] Skipping fold: test set collapsed to a single class after filtering.")
            continue
        if np.unique(y_tr).size < 2:
            print("[Kfold] Skipping fold: train set collapsed to a single class after filtering.")
            continue

        # Drop to kept columns; scale per fold (ε-guarded z-score)
        X_tr_model = X_tr[:, keep_mask]
        X_te_model = X_te[:, keep_mask]
        X_tr_s, X_te_s, _, _ = _zscore_train_test(X_tr_model, X_te_model)

        # Fit, pick threshold on TRAIN (Youden), score on TEST
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr_s, y_tr)
        p_tr = clf.predict_proba(X_tr_s)[:, 1]
        fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr, p_tr)
        bt = thr_tr[np.argmax(tpr_tr - fpr_tr)]

        p_te = clf.predict_proba(X_te_s)[:, 1]
        auc, tss, kapp, sens, spec = _compute_metrics_from_probs(y_te, p_te, bt)

        print(f"[Kfold] fold={len(aucs)+1}/{n_splits*n_repeats} "
              f"test presences={np.sum(y_te==1)} backgrounds={np.sum(y_te==0)} "
              f"auc={auc:.3f} thr={bt:.3f} sens={sens:.3f} spec={spec:.3f}")

        aucs.append(float(auc)); tsses.append(float(tss)); kappas.append(float(kapp))
        thrs.append(float(bt));  sensv.append(float(sens)); specv.append(float(spec))

    note = f"Spatial blocking unavailable; used non-spatial Repeated Stratified K-fold (splits={n_splits}, repeats={n_repeats})."
    out = _cv_summary_frame(n_splits * n_repeats, "repeated_kfold", note, aucs, tsses, kappas, thrs, sensv, specv)
    out.to_csv(perf_cv_csv, index=False)
    print(f"📊 Wrote CV fallback (Repeated K-fold) to {perf_cv_csv}")

# ---- Try spatial blocks; else fall back to K-fold ----
did_spatial = False
if len(coords) >= 2:
    try:
        print("[STEP] spatial-cv")
        # n_init=10 for sklearn<=1.3 compatibility
        n_blocks = min(5, len(coords))
        kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init=10).fit(coords)
        blocks = kmeans.labels_  # presences' block labels

        gkf = GroupKFold(n_splits=n_blocks)
        for fold_id, (train_idx, test_idx) in enumerate(
            gkf.split(presence_samples, np.ones(len(presence_samples), dtype=int), groups=blocks),
            start=1
        ):
            print(f"[STEP] spatial-cv-fold {fold_id}/{n_blocks}")
            # split presence
            Xp_tr, Xp_te = presence_samples[train_idx], presence_samples[test_idx]
            test_blocks = set(blocks[test_idx])
            train_blocks = [b for b in np.unique(blocks) if b not in test_blocks]

            rng = np.random.default_rng(10_000 + fold_id)

            # Decide how many backgrounds
            n_bt  = min(5 * len(train_idx), pool_len)
            n_bt2 = min(5 * len(test_idx),  pool_len)

            # Lazily sample backgrounds by blocks (batch candidates only)
            bt_idx  = _sample_by_blocks(kmeans, train_blocks, n_bt,  rng, pool_len, valid_idx, width, ref_transform)
            bt_idx2 = _sample_by_blocks(kmeans, list(test_blocks), n_bt2, rng, pool_len, valid_idx, width, ref_transform)

            Xb_tr = pool[bt_idx]
            Xb_te = pool[bt_idx2]

            # combine; scaling per fold below
            X_tr = np.vstack([Xp_tr, Xb_tr])
            y_tr = np.concatenate([np.ones(len(Xp_tr), dtype=int), np.zeros(len(Xb_tr), dtype=int)])
            X_te = np.vstack([Xp_te, Xb_te])
            y_te = np.concatenate([np.ones(len(Xp_te), dtype=int), np.zeros(len(Xb_te), dtype=int)])

            # Filter finite rows
            mtr = np.all(np.isfinite(X_tr), axis=1)
            mte = np.all(np.isfinite(X_te), axis=1)
            X_tr, y_tr = X_tr[mtr], y_tr[mtr]
            X_te, y_te = X_te[mte], y_te[mte]

            # Ensure both classes exist in TEST and TRAIN
            if np.unique(y_te).size < 2:
                print("[CV] Skipping fold: test set collapsed to a single class after filtering.")
                continue
            if np.unique(y_tr).size < 2:
                print("[CV] Skipping fold: train set collapsed to a single class after filtering.")
                continue

            # Drop to kept columns; scale per fold (ε-guarded z-score)
            X_tr_model = X_tr[:, keep_mask]
            X_te_model = X_te[:, keep_mask]
            X_tr_s, X_te_s, _, _ = _zscore_train_test(X_tr_model, X_te_model)

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

        did_spatial = True
        note = ""
        n_cv_folds = len(cv_aucs)
        cv_summary = _cv_summary_frame(n_cv_folds, "spatial_blocks", note,
                                       cv_aucs, cv_tsses, cv_kappas,
                                       cv_thresholds, cv_sensitivities, cv_specificities)
        cv_summary.to_csv(perf_cv_csv, index=False)
        print(f"📊 Cross-validated (spatial blocks) metrics saved to {perf_cv_csv}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"⚠️ Spatial CV failed with error: {e}")
        _run_repeated_kfold_fallback()
else:
    print("ℹ️ Not enough presence points/blocks for spatial CV (need ≥2 blocks).")
    _run_repeated_kfold_fallback()

# --- Final scaler & model on all filtered samples (scale everything; only kept cols) ---
print("[STEP] fit-final-model")
scaler = StandardScaler().fit(Xc_model)
Xc_s = scaler.transform(Xc_model)
model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xc_s, yc)
joblib.dump(scaler, scaler_pkl)
print(f"💾 Saved scaler to {scaler_pkl}")

# --- Export standardization stats (mark dropped constants) ---
print("[STEP] export-standardization-stats")
mean_full = np.zeros(len(names), dtype=float)
std_full  = np.zeros(len(names), dtype=float)
mean_full[used_idx] = scaler.mean_.astype(float)
std_full[used_idx]  = scaler.scale_.astype(float)
# Dropped columns remain 0.0 mean/std (their z-scores are 0 downstream)
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
print("[STEP] export-sample-tables-standardized")
def _std_full_from_model(arr_full: np.ndarray) -> np.ndarray:
    """Standardize kept columns using fitted scaler, expand back to full width (dropped → 0)."""
    if arr_full.size == 0:
        return arr_full.reshape((0, len(names)))
    arr_kept = arr_full[:, keep_mask]  # rows known finite from earlier extraction
    arr_kept_s = scaler.transform(arr_kept)
    out = np.zeros((arr_full.shape[0], len(names)), dtype=np.float32)
    out[:, used_idx] = arr_kept_s.astype(np.float32)
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

std_csv = "outputs/sdm_point_samples_standardized.csv"
pd.concat([presence_df_std, absence_df_std], ignore_index=True).to_csv(std_csv, index=False)
print(f"📄 Point samples (STANDARDIZED) saved to {std_csv}")

# --- Fitted (in-sample) metrics on standardized training data ---
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
print("[STEP] export-coefficients")
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
print("[STEP] write-suitability-map")
pred_flat = np.full(flat.shape[0], np.nan, dtype=np.float32)
pool_model = pool[:, keep_mask]
pool_s = scaler.transform(pool_model)
pred_flat[valid_mask] = model.predict_proba(pool_s)[:, 1].astype(np.float32)
pred_map = pred_flat.reshape((height, width))
with rasterio.open(output_map, "w", **profile) as dst:
    dst.write(pred_map, 1)
print(f"🎯 Final suitability map saved to {output_map}")

# --- Export standardized rasters (z-scored; dropped → 0) ---
print("[STEP] write-standardized-rasters")
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
