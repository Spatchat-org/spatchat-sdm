# scripts/run_logistic_sdm.py
# Logistic SDM with spatial-block CV (k-fold fallback), robust handling of binary landcover,
# presence-row imputation from pool means, and export of background coordinates.

import os
import re
import numpy as np
import pandas as pd
import rasterio
import joblib

from rasterio.transform import rowcol
from rasterio.warp import reproject, Resampling as WarpResampling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
from sklearn.cluster import KMeans

# ──────────────────────────────────────────────────────────────
# Paths & setup
# ──────────────────────────────────────────────────────────────
csv_path     = "inputs/presence_points.csv"
raster_dir   = "predictor_rasters/wgs84"
os.makedirs("outputs", exist_ok=True)
perf_cv_csv  = "outputs/performance_metrics_cv.csv"
perf_fit_csv = "outputs/performance_metrics_fitted.csv"
compat_perf  = "outputs/performance_metrics.csv"
coef_csv     = "outputs/coefficients.csv"
scaler_pkl   = "outputs/standard_scaler.pkl"
output_map   = "outputs/suitability_map_wgs84.tif"
std_ras_dir  = "outputs/standardized_rasters_wgs84"
raw_csv      = "outputs/sdm_point_samples_raw.csv"
std_csv      = "outputs/sdm_point_samples_standardized.csv"
drop_csv     = "outputs/dropped_predictors.csv"
absences_csv = "outputs/absence_points_coordinates.csv"  # optional for the map

EPS = 1e-8
np.random.seed(42)

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _is_categorical(name: str) -> bool:
    """
    Heuristic: MODIS one-hot layers look like '5_mixed_forest', or include 'landcover'/'onehot'.
    """
    n = name.lower()
    return bool(re.match(r"^\d+_", n) or "landcover" in n or "onehot" in n)

def _compute_metrics_from_probs(y_true, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    sens = tp/(tp+fn) if (tp+fn) else 0.0
    spec = tn/(tn+fp) if (tn+fp) else 0.0
    tss  = sens + spec - 1
    kapp = cohen_kappa_score(y_true, yhat)
    auc  = roc_auc_score(y_true, p)
    return auc, tss, kapp, sens, spec

def _cv_summary_frame(n_folds, method, note, aucs, tsses, kappas, thrs, sens, spec,
                      n_input_presences, n_kept_presences, n_predictors_used):
    return pd.DataFrame([{
        "n_folds": int(n_folds),
        "cv_method": method,
        "cv_note": note or "",
        "AUC_mean": float(np.mean(aucs)) if len(aucs) else np.nan,
        "AUC_sd":   float(np.std(aucs))  if len(aucs) else np.nan,
        "TSS_mean": float(np.mean(tsses)) if len(tsses) else np.nan,
        "TSS_sd":   float(np.std(tsses))  if len(tsses) else np.nan,
        "Kappa_mean": float(np.mean(kappas)) if len(kappas) else np.nan,
        "Kappa_sd":   float(np.std(kappas))  if len(kappas) else np.nan,
        "Threshold_mean": float(np.mean(thrs)) if len(thrs) else np.nan,
        "Threshold_sd":   float(np.std(thrs))  if len(thrs) else np.nan,
        "Sensitivity_mean": float(np.mean(sens)) if len(sens) else np.nan,
        "Sensitivity_sd":   float(np.std(sens))  if len(sens) else np.nan,
        "Specificity_mean": float(np.mean(spec)) if len(spec) else np.nan,
        "Specificity_sd":   float(np.std(spec))  if len(spec) else np.nan,
        "n_input_presences": int(n_input_presences),
        "n_kept_presences":  int(n_kept_presences),
        "n_predictors_used": int(n_predictors_used),
    }])

def _zscore_train_test(X_tr, X_te, is_cat_kept):
    """
    Fit scaling on TRAIN:
      - continuous -> z-score (mu, sd)
      - categorical (binary) -> center-only (sd=1)
      - guard tiny variance for continuous (sd<EPS -> 1)
    """
    mu = np.nanmean(X_tr, axis=0)
    sd = np.nanstd(X_tr, axis=0)
    sd = np.where(is_cat_kept, 1.0, sd)          # center-only for binary
    sd = np.where(sd < EPS, 1.0, sd)             # guard
    return (X_tr - mu) / sd, (X_te - mu) / sd, mu, sd

# ──────────────────────────────────────────────────────────────
# Load presence points
# ──────────────────────────────────────────────────────────────
df_in = pd.read_csv(csv_path)
lats = df_in["latitude"].values
lons = df_in["longitude"].values
print(f"📍 Loaded {len(df_in)} presence points.")

# ──────────────────────────────────────────────────────────────
# Reference grid & aligned stack
# ──────────────────────────────────────────────────────────────
rasters = sorted([os.path.join(raster_dir, f) for f in os.listdir(raster_dir) if f.endswith(".tif")])
if not rasters:
    raise RuntimeError(f"No .tif found in {raster_dir}")

with rasterio.open(rasters[0]) as ref:
    ref_crs       = ref.crs
    ref_transform = ref.transform
    height, width = ref.height, ref.width
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": rasterio.float32,
        "crs": ref_crs,
        "transform": ref_transform,
        "compress": "DEFLATE",
        "predictor": 2,
        "zlevel": 4,
    }
    print(f"🎯 Reference grid: {width}×{height} @ {ref_transform} in {ref_crs}")

layers, names, is_cat = [], [], []
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
    layers.append(dst)
    names.append(name)
    is_cat.append(_is_categorical(name))
    print(f"🧩 Added layer '{name}' (categorical={is_cat[-1]})")

is_cat = np.array(is_cat, dtype=bool)
stack = np.stack(layers, axis=-1)
print(f"🗺️ Stacked predictor shape: {stack.shape}")

# ──────────────────────────────────────────────────────────────
# Extract presence samples (allow NaNs; we'll impute later)
# ──────────────────────────────────────────────────────────────
presence_samples = []
kept_coords = []
for lat, lon in zip(lats, lons):
    try:
        r, c = rowcol(ref_transform, lon, lat, op=round)
    except Exception:
        continue
    if 0 <= r < height and 0 <= c < width:
        vals = stack[r, c, :].astype(np.float32, copy=False)
        presence_samples.append(vals)
        kept_coords.append((lat, lon))

presence_samples = np.array(presence_samples, dtype=np.float32)
kept_coords = np.array(kept_coords, dtype=np.float32) if len(kept_coords) else np.empty((0, 2), dtype=np.float32)
print(f"📍 Presence samples extracted (may include NaNs): {presence_samples.shape[0]}")
if presence_samples.shape[0] == 0:
    raise RuntimeError("No presence points fell inside the raster extent. Check overlap and predictor coverage.")

# ──────────────────────────────────────────────────────────────
# Background pool (valid pixels only) & sampling
# ──────────────────────────────────────────────────────────────
flat       = stack.reshape(-1, stack.shape[-1]).astype(np.float32)
valid_mask = ~np.any(np.isnan(flat), axis=1)
valid_idx  = np.flatnonzero(valid_mask)
pool       = flat[valid_mask]
pool_len   = len(pool)
print(f"✅ Valid pixel pool: {pool_len}")
if pool_len == 0:
    raise RuntimeError("No valid pixels in predictor stack. Check predictor coverage/NaNs.")

# Impute missing predictor values at presence rows using pool column means
col_means = np.nanmean(pool, axis=0)
col_means = np.where(np.isfinite(col_means), col_means, 0.0)
nan_mask_pres = ~np.isfinite(presence_samples)
if np.any(nan_mask_pres):
    # indices of cols per-row where NaN occurs
    row_idx, col_idx = np.nonzero(nan_mask_pres)
    presence_samples[row_idx, col_idx] = col_means[col_idx]
    print(f"🧽 Imputed NaNs in presence rows with pool means (cols affected: {int(nan_mask_pres.any(axis=0).sum())})")

# choose background count and sample
n_bg = min(max(5 * presence_samples.shape[0], 1), pool_len)
bg_indices = np.random.choice(pool_len, size=n_bg, replace=False)
background_samples = pool[bg_indices].astype(np.float32)

# background locations for exports/map
bg_linear = valid_idx[bg_indices].astype(np.int64)
bg_rows   = (bg_linear // width).astype(np.int64)
bg_cols   = (bg_linear %  width).astype(np.int64)
# rasterio.transform.xy returns (x=lon, y=lat)
bg_lons, bg_lats = rasterio.transform.xy(ref_transform, bg_rows, bg_cols, offset="center")
bg_lats = np.asarray(bg_lats, dtype=float)
bg_lons = np.asarray(bg_lons, dtype=float)
pd.DataFrame({"latitude": bg_lats, "longitude": bg_lons}).to_csv(absences_csv, index=False)
print(f"🗺️ Absence coordinates saved to {absences_csv} ({len(bg_lats)} points)")

# ──────────────────────────────────────────────────────────────
# Build training matrix; drop uninformative predictors safely
# ──────────────────────────────────────────────────────────────
X = np.vstack([presence_samples, background_samples])
y = np.concatenate([
    np.ones(presence_samples.shape[0], dtype=int),
    np.zeros(background_samples.shape[0], dtype=int)
])

finite_mask = np.all(np.isfinite(X), axis=1)
Xc, yc = X[finite_mask], y[finite_mask]
print(f"🔀 Train rows after finite filter: {X.shape} → {Xc.shape} (pos={int((yc==1).sum())}, neg={int((yc==0).sum())})")
if Xc.shape[0] == 0:
    raise RuntimeError("No usable training rows after filtering. Check predictor coverage & presence coordinates.")
if len(np.unique(yc)) < 2:
    raise RuntimeError("Only one class present after filtering. Need both presences and backgrounds.")

# Determine drops
col_std = np.nanstd(Xc, axis=0)  # std over combined train rows
keep_mask = np.ones(len(names), dtype=bool)
drop_reasons = []

for j, n in enumerate(names):
    if is_cat[j]:
        # Binary: check if both classes appear in combined rows
        minv, maxv = float(np.nanmin(Xc[:, j])), float(np.nanmax(Xc[:, j]))
        if (minv >= 1.0 - EPS) and (maxv <= 1.0 + EPS):
            keep_mask[j] = False
            drop_reasons.append((n, "categorical: all 1 in training rows", 1.0))
        elif (maxv <= 0.0 + EPS) and (minv >= 0.0 - EPS):
            keep_mask[j] = False
            drop_reasons.append((n, "categorical: all 0 in training rows (class absent in AOI samples)", 0.0))
        # else informative → keep
    else:
        if col_std[j] <= EPS:
            keep_mask[j] = False
            drop_reasons.append((n, "continuous: near-constant on training rows", float(np.nanmean(Xc[:, j]))))

if not np.any(keep_mask):
    raise RuntimeError("All predictors are uninformative on training rows. Add at least one varying layer.")

used_idx = np.flatnonzero(keep_mask)
names_used = [names[i] for i in used_idx]
is_cat_kept = is_cat[used_idx]
print(f"✅ Using {len(names_used)}/{len(names)} predictors: {', '.join(names_used)}")

# Export drop report for the UI
pd.DataFrame([{"predictor": n, "reason": r, "constant_value": v} for n, r, v in drop_reasons]).to_csv(drop_csv, index=False)

# ──────────────────────────────────────────────────────────────
# Export point samples (RAW)
# ──────────────────────────────────────────────────────────────
pres_lats = kept_coords[:, 0] if kept_coords.size else np.array([], dtype=float)
pres_lons = kept_coords[:, 1] if kept_coords.size else np.array([], dtype=float)

presence_df_raw = pd.DataFrame(presence_samples, columns=names)
presence_df_raw.insert(0, "latitude",  pres_lats[:len(presence_df_raw)])
presence_df_raw.insert(1, "longitude", pres_lons[:len(presence_df_raw)])
presence_df_raw.insert(2, "label", 1)

absence_df_raw = pd.DataFrame(background_samples, columns=names)
absence_df_raw.insert(0, "latitude",  bg_lats[:len(absence_df_raw)])
absence_df_raw.insert(1, "longitude", bg_lons[:len(absence_df_raw)])
absence_df_raw.insert(2, "label", 0)

samples_df_raw = pd.concat([presence_df_raw, absence_df_raw], ignore_index=True)
samples_df_raw.to_csv(raw_csv, index=False)
print(f"📄 Point samples (RAW) saved to {raw_csv}")

# ──────────────────────────────────────────────────────────────
# Spatial CV (blocked) with per-fold scaling; fallback to RepeatedStratifiedKFold
# ──────────────────────────────────────────────────────────────
coords = kept_coords  # presences only (lat, lon)
n_blocks = min(5, len(coords)) if len(coords) > 0 else 0

cv_aucs, cv_tsses, cv_kappas = [], [], []
cv_thresholds, cv_sensitivities, cv_specificities = [], [], []

# lon/lat for all valid pool pixels (to assign blocks)
pool_rows = valid_idx // width
pool_cols = valid_idx %  width
pool_lons_all, pool_lats_all = rasterio.transform.xy(ref_transform, pool_rows, pool_cols, offset="center")
pool_lats_all = np.asarray(pool_lats_all, dtype=float)
pool_lons_all = np.asarray(pool_lons_all, dtype=float)

def _run_repeated_kfold_fallback():
    print("ℹ️ Falling back to Repeated Stratified K-fold CV (non-spatial).")
    Np = presence_samples.shape[0]
    if Np < 3:
        out = _cv_summary_frame(
            0, "repeated_kfold",
            "Too few presences for K-fold; reporting NaNs.",
            [], [], [], [], [], [],
            len(df_in), len(presence_samples), len(names_used)
        )
        out.to_csv(perf_cv_csv, index=False)
        print(f"📊 Wrote CV fallback (NaNs) to {perf_cv_csv}")
        return

    n_splits  = max(3, min(5, max(2, Np // 5)))
    n_repeats = 5 if Np >= 10 else 3
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    aucs = []; tsses = []; kappas = []; thrs = []; sensv = []; specv = []

    for fold_i, (tr_i, te_i) in enumerate(rskf.split(presence_samples, np.ones(Np, dtype=int)), start=1):
        Xp_tr, Xp_te = presence_samples[tr_i], presence_samples[te_i]

        rng = np.random.default_rng(1000 + fold_i)
        n_bt_tr = min(5 * len(tr_i), pool_len)
        n_bt_te = min(5 * len(te_i),  pool_len)
        bt_idx  = rng.choice(pool_len, size=n_bt_tr, replace=False)
        bt_idx2 = rng.choice(pool_len, size=n_bt_te, replace=False)
        Xb_tr, Xb_te = pool[bt_idx], pool[bt_idx2]

        X_tr = np.vstack([Xp_tr, Xb_tr]); y_tr = np.concatenate([np.ones(len(Xp_tr), dtype=int), np.zeros(len(Xb_tr), dtype=int)])
        X_te = np.vstack([Xp_te, Xb_te]); y_te = np.concatenate([np.ones(len(Xp_te), dtype=int), np.zeros(len(Xb_te), dtype=int)])

        mtr = np.all(np.isfinite(X_tr), axis=1); mte = np.all(np.isfinite(X_te), axis=1)
        X_tr, y_tr = X_tr[mtr], y_tr[mtr]
        X_te, y_te = X_te[mte], y_te[mte]

        X_tr_k = X_tr[:, used_idx]; X_te_k = X_te[:, used_idx]
        X_tr_s, X_te_s, _, _ = _zscore_train_test(X_tr_k, X_te_k, is_cat_kept)

        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr_s, y_tr)
        p_tr = clf.predict_proba(X_tr_s)[:, 1]
        fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr, p_tr)
        bt = thr_tr[np.argmax(tpr_tr - fpr_tr)]

        p_te = clf.predict_proba(X_te_s)[:, 1]
        auc, tss, kapp, sens, spec = _compute_metrics_from_probs(y_te, p_te, bt)

        aucs.append(float(auc)); tsses.append(float(tss)); kappas.append(float(kapp))
        thrs.append(float(bt));  sensv.append(float(sens)); specv.append(float(spec))

    out = _cv_summary_frame(
        n_splits * n_repeats, "repeated_kfold",
        f"Spatial blocking unavailable; used non-spatial Repeated Stratified K-fold (splits={n_splits}, repeats={n_repeats}).",
        aucs, tsses, kappas, thrs, sensv, specv,
        len(df_in), len(presence_samples), len(names_used)
    )
    out.to_csv(perf_cv_csv, index=False)
    print(f"📊 Wrote CV fallback (Repeated K-fold) to {perf_cv_csv}")

did_spatial = False
if n_blocks >= 2:
    try:
        kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init="auto").fit(coords)
        blocks = kmeans.labels_
        pool_blocks = kmeans.predict(np.column_stack([pool_lats_all, pool_lons_all]))

        gkf = GroupKFold(n_splits=n_blocks)
        for tr_i, te_i in gkf.split(presence_samples, np.ones(len(presence_samples), dtype=int), groups=blocks):
            Xp_tr, Xp_te = presence_samples[tr_i], presence_samples[te_i]
            test_blocks = set(blocks[te_i])
            train_blocks = [b for b in range(n_blocks) if b not in test_blocks]

            mask_tr_pool = np.isin(pool_blocks, train_blocks)
            mask_te_pool = np.isin(pool_blocks, list(test_blocks))

            n_bt  = min(5 * len(tr_i), int(mask_tr_pool.sum()))
            n_bt2 = min(5 * len(te_i),  int(mask_te_pool.sum()))

            if n_bt < 1 or n_bt2 < 1:
                bt_idx  = np.random.choice(pool_len, size=max(1, 5 * len(tr_i)), replace=False)
                bt_idx2 = np.random.choice(pool_len, size=max(1, 5 * len(te_i)),  replace=False)
            else:
                tr_inds = np.flatnonzero(mask_tr_pool)
                te_inds = np.flatnonzero(mask_te_pool)
                bt_idx  = np.random.choice(tr_inds, size=n_bt,  replace=False)
                bt_idx2 = np.random.choice(te_inds, size=n_bt2, replace=False)

            Xb_tr, Xb_te = pool[bt_idx], pool[bt_idx2]

            X_tr = np.vstack([Xp_tr, Xb_tr]); y_tr = np.concatenate([np.ones(len(Xp_tr), dtype=int), np.zeros(len(Xb_tr), dtype=int)])
            X_te = np.vstack([Xp_te, Xb_te]); y_te = np.concatenate([np.ones(len(Xp_te), dtype=int), np.zeros(len(Xb_te), dtype=int)])

            mtr = np.all(np.isfinite(X_tr), axis=1); mte = np.all(np.isfinite(X_te), axis=1)
            X_tr, y_tr = X_tr[mtr], y_tr[mtr]
            X_te, y_te = X_te[mte], y_te[mte]

            X_tr_k = X_tr[:, used_idx]; X_te_k = X_te[:, used_idx]
            X_tr_s, X_te_s, _, _ = _zscore_train_test(X_tr_k, X_te_k, is_cat_kept)

            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr_s, y_tr)

            p_tr = clf.predict_proba(X_tr_s)[:, 1]
            fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr, p_tr)
            bt = thr_tr[np.argmax(tpr_tr - fpr_tr)]

            p_te = clf.predict_proba(X_te_s)[:, 1]
            auc  = roc_auc_score(y_te, p_te)
            yhat = (p_te >= bt).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_te, yhat, labels=[0, 1]).ravel()
            sens = tp/(tp+fn) if (tp+fn) else 0.0
            spec = tn/(tn+fp) if (tn+fp) else 0.0
            tss  = sens + spec - 1
            kapp = cohen_kappa_score(y_te, yhat)

            cv_aucs.append(float(auc))
            cv_tsses.append(float(tss))
            cv_kappas.append(float(kapp))
            cv_thresholds.append(float(bt))
            cv_sensitivities.append(float(sens))
            cv_specificities.append(float(spec))

        did_spatial = True
        out = _cv_summary_frame(
            n_blocks, "spatial_blocks", "",
            cv_aucs, cv_tsses, cv_kappas, cv_thresholds, cv_sensitivities, cv_specificities,
            len(df_in), len(presence_samples), len(names_used)
        )
        out.to_csv(perf_cv_csv, index=False)
        print(f"📊 Cross-validated (spatial blocks) metrics saved to {perf_cv_csv}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"⚠️ Spatial CV failed with error: {e}")
        _run_repeated_kfold_fallback()
else:
    print("ℹ️ Not enough presence points/blocks for spatial CV (need ≥2 blocks).")
    _run_repeated_kfold_fallback()

# ──────────────────────────────────────────────────────────────
# Final fit with center-only on categorical
# ──────────────────────────────────────────────────────────────
print("[STEP] fit-final-model")

Xc_model = Xc[:, used_idx]

mu_final = np.nanmean(Xc_model, axis=0)
sd_final = np.nanstd(Xc_model, axis=0)
sd_final = np.where(is_cat_kept, 1.0, sd_final)     # center-only for categorical
sd_final = np.where(sd_final < EPS, 1.0, sd_final)  # tiny variance guard

def _apply_standardize(arr_kept):
    return (arr_kept - mu_final) / sd_final

Xc_s = _apply_standardize(Xc_model)

model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xc_s, yc)
joblib.dump({"mu": mu_final, "sd": sd_final, "keep_mask": keep_mask, "is_cat": is_cat}, scaler_pkl)
print(f"💾 Saved scaler stats to {scaler_pkl}")

# ── Export standardization stats mapped back to full list
mean_full = np.zeros(len(names), dtype=float)
std_full  = np.zeros(len(names), dtype=float)
mean_full[used_idx] = mu_final
std_full[used_idx]  = sd_final
pd.DataFrame({
    "predictor": names,
    "mean": mean_full,
    "std":  std_full,
    "dropped_constant": [not k for k in keep_mask]
}).to_csv("outputs/standardization_stats.csv", index=False)

# ──────────────────────────────────────────────────────────────
# Export point samples (STANDARDIZED)
# ──────────────────────────────────────────────────────────────
def _std_full(arr_full: np.ndarray) -> np.ndarray:
    if arr_full.size == 0:
        return arr_full.reshape((0, len(names)))
    arr_kept_s = _apply_standardize(arr_full[:, used_idx])
    out = np.zeros((arr_full.shape[0], len(names)), dtype=np.float32)
    out[:, used_idx] = arr_kept_s.astype(np.float32)
    return out

presence_std = _std_full(presence_samples)
absence_std  = _std_full(background_samples)

presence_df_std = pd.DataFrame(presence_std, columns=names)
presence_df_std.insert(0, "latitude",  pres_lats[:len(presence_df_std)])
presence_df_std.insert(1, "longitude", pres_lons[:len(presence_df_std)])
presence_df_std.insert(2, "label", 1)

absence_df_std = pd.DataFrame(absence_std, columns=names)
absence_df_std.insert(0, "latitude",  bg_lats[:len(absence_df_std)])
absence_df_std.insert(1, "longitude", bg_lons[:len(absence_df_std)])
absence_df_std.insert(2, "label", 0)

samples_df_std = pd.concat([presence_df_std, absence_df_std], ignore_index=True)
samples_df_std.to_csv(std_csv, index=False)
print(f"📄 Point samples (STANDARDIZED) saved to {std_csv}")

# ──────────────────────────────────────────────────────────────
# Fitted (in-sample) metrics on standardized training data
# ──────────────────────────────────────────────────────────────
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
    "AUC_fitted":         float(auc),
    "Threshold_fitted":   float(best_thr),
    "Sensitivity_fitted": float(sens),
    "Specificity_fitted": float(spec),
    "TSS_fitted":         float(tss),
    "Kappa_fitted":       float(kappa)
}])
fitted_df.to_csv(perf_fit_csv, index=False)
fitted_df.to_csv(compat_perf, index=False)
print(f"📊 Fitted metrics saved to {perf_fit_csv}")

# ──────────────────────────────────────────────────────────────
# Coefficients mapped back to full feature list
# ──────────────────────────────────────────────────────────────
intercept = model.intercept_[0]
coef_used = model.coef_.flatten()
full_coef = np.zeros(len(names), dtype=float)
full_coef[used_idx] = coef_used
coef_df = pd.DataFrame({"predictor": ["Intercept"] + names, "coefficient": np.concatenate([[intercept], full_coef])})
coef_df.to_csv(coef_csv, index=False)
print(f"📊 Coefficients saved to {coef_csv}")

# ──────────────────────────────────────────────────────────────
# Suitability map (predict on full grid)
# ──────────────────────────────────────────────────────────────
pred_flat = np.full(flat.shape[0], np.nan, dtype=np.float32)
pool_model = pool[:, used_idx]
pool_s = _apply_standardize(pool_model).astype(np.float32)
pred_flat[valid_mask] = model.predict_proba(pool_s)[:, 1].astype(np.float32)
pred_map = pred_flat.reshape((height, width))

with rasterio.open(output_map, "w", **profile) as dst:
    dst.write(pred_map, 1)
print(f"🎯 Final suitability map saved to {output_map}")

# ──────────────────────────────────────────────────────────────
# Standardized rasters (z-scored for continuous, centered for categorical)
# ──────────────────────────────────────────────────────────────
os.makedirs(std_ras_dir, exist_ok=True)
print(f"💾 Writing standardized rasters to {std_ras_dir}")

pool_s_full = np.zeros((pool.shape[0], len(names)), dtype=np.float32)
pool_s_full[:, used_idx] = pool_s  # already standardized according to type

for j, name in enumerate(names):
    band_flat = np.full(flat.shape[0], np.float32(np.nan), dtype=np.float32)
    band_flat[valid_mask] = pool_s_full[:, j]
    band = band_flat.reshape((height, width))
    out_path = os.path.join(std_ras_dir, f"{name}_z.tif")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(band, 1)
    print(f"    • {name}_z.tif")
