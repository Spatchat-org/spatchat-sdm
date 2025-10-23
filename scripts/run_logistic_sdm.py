# scripts/run_logistic_sdm.py
# Logistic SDM: numeric standardization only, categorical 0/1, random spatial backgrounds.
# Spatial-block CV with repeated K-fold fallback. Robust dropping of uninformative predictors.
# Exports raw & standardized samples (with coords), coefficients with p-values, and suitability map.

import os
import sys
import numpy as np
import pandas as pd
import rasterio
import joblib
import warnings

from rasterio.transform import rowcol
from rasterio.warp import Resampling as WarpResampling, reproject
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             cohen_kappa_score)
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Optional statsmodels for p-values
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import statsmodels.api as sm
        HAVE_SM = True
    except Exception:
        HAVE_SM = False

# ---------------- Paths ----------------
csv_path     = "inputs/presence_points.csv"
raster_dir   = "predictor_rasters/wgs84"
out_dir      = "outputs"
perf_cv_csv  = os.path.join(out_dir, "performance_metrics_cv.csv")
perf_fit_csv = os.path.join(out_dir, "performance_metrics_fitted.csv")
compat_perf  = os.path.join(out_dir, "performance_metrics.csv")
coef_csv     = os.path.join(out_dir, "coefficients.csv")
drop_csv     = os.path.join(out_dir, "dropped_predictors.csv")
scaler_pkl   = os.path.join(out_dir, "standard_scaler.pkl")
stats_csv    = os.path.join(out_dir, "standardization_stats.csv")
raw_csv      = os.path.join(out_dir, "sdm_point_samples_raw.csv")
std_csv      = os.path.join(out_dir, "sdm_point_samples_standardized.csv")
abs_csv      = os.path.join(out_dir, "absence_points_coordinates.csv")  # legacy / optional
output_map   = os.path.join(out_dir, "suitability_map_wgs84.tif")
std_ras_dir  = os.path.join(out_dir, "standardized_rasters_wgs84")

os.makedirs(out_dir, exist_ok=True)

# ---------------- Helpers ----------------
def _is_categorical_name(name: str) -> bool:
    n = name.lower()
    # Treat MODIS class one-hots like "5_mixed_forest" as categorical.
    return ("landcover" in n) or n.endswith("_onehot") or ("_mixed_forest" in n) or n.split("_", 1)[0].isdigit()

def _read_stack_align(raster_paths, ref_profile=None):
    """Read rasters, reproject/align to the reference grid, return stack [H,W,C], names, and profile of ref."""
    names = []
    aligned = []

    # Reference grid: first raster
    if ref_profile is None:
        if not raster_paths:
            raise RuntimeError(f"No .tif found in {raster_dir}")
        with rasterio.open(raster_paths[0]) as ref:
            ref_crs = ref.crs
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
            }
    else:
        profile = dict(ref_profile)
        ref_crs = profile["crs"]; ref_transform = profile["transform"]
        height = profile["height"]; width = profile["width"]

    print(f"🗺  Grid: {width}×{height} @ {ref_transform} in {ref_crs}")

    for p in raster_paths:
        nm = os.path.splitext(os.path.basename(p))[0]
        with rasterio.open(p) as src:
            dst = np.full((height, width), np.nan, dtype=np.float32)
            resamp = WarpResampling.nearest if _is_categorical_name(nm) else WarpResampling.bilinear
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                resampling=resamp, dst_nodata=np.nan,
            )
        nan_pct = float(np.isnan(dst).mean() * 100.0)
        print(f"🧪 {nm}.tif → NaN%: {nan_pct:.2f}")
        names.append(nm)
        aligned.append(dst)

    stack = np.stack(aligned, axis=-1)  # H W C
    return stack, names, profile

def _lin_to_coords(lin_idx, width, transform):
    """Convert linear indices → (lon, lat) arrays. Safe for empty input."""
    arr = np.asarray(lin_idx).reshape(-1)
    if arr.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    rows = (arr // width).astype(int)
    cols = (arr %  width).astype(int)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    # rasterio returns lists; coerce to float arrays
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)

def _compute_metrics_from_probs(y_true, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    tss = sens + spec - 1
    kapp = cohen_kappa_score(y_true, yhat)
    auc = roc_auc_score(y_true, p)
    return auc, tss, kapp, sens, spec

def _cv_summary_frame(n_folds, method, note, aucs, tsses, kappas, thrs, sens, spec, n_input_pres, n_kept_pres, n_used_preds):
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
        "n_input_presences": int(n_input_pres),
        "n_kept_presences":  int(n_kept_pres),
        "n_predictors_used": int(n_used_preds),
    }])
    return df_out

# ---------------- Load data ----------------
df = pd.read_csv(csv_path)
if not {"latitude", "longitude"}.issubset(df.columns):
    raise RuntimeError("presence_points.csv must contain 'latitude' and 'longitude' (WGS84).")

lats = df["latitude"].astype(float).values
lons = df["longitude"].astype(float).values
print(f"📍 Loaded {len(df)} presence points.")

# ---------------- Find rasters ----------------
rasters = sorted([os.path.join(raster_dir, f) for f in os.listdir(raster_dir) if f.endswith(".tif")])
if not rasters:
    raise RuntimeError(f"No .tif found in {raster_dir}")

# Build stack
stack, names, profile = _read_stack_align(rasters)
H, W, C = stack.shape

# ---------------- Extract presence samples (valid predictors) ----------------
presence_samples = []
kept_coords = []
for lat, lon in zip(lats, lons):
    try:
        r, c = rowcol(profile["transform"], lon, lat, op=round)
    except Exception:
        continue
    if 0 <= r < H and 0 <= c < W:
        vals = stack[r, c, :]
        if not np.any(np.isnan(vals)):
            presence_samples.append(vals)
            kept_coords.append((lat, lon))

presence_samples = np.array(presence_samples, dtype=np.float32)
kept_coords = np.array(kept_coords, dtype=np.float32) if len(kept_coords) else np.empty((0, 2), dtype=np.float32)
print(f"📍 Presence samples: {presence_samples.shape}")

if presence_samples.shape[0] == 0:
    raise RuntimeError("No presence points fell on pixels where all predictors are valid. Check coverage or extent.")

# ---------------- Background pool & random sampling ----------------
flat       = stack.reshape(-1, C).astype(np.float32)
valid_mask = ~np.any(np.isnan(flat), axis=1)
valid_idx  = np.flatnonzero(valid_mask)
pool       = flat[valid_mask]
pool_len   = int(pool.shape[0])
print(f"✅ Valid pixel pool: {pool_len}")

if pool_len == 0:
    raise RuntimeError("No valid pixels across predictors (all-NaN after alignment). Check rasters/AOI.")

# Purely random backgrounds (uniform on valid pixels)
np.random.seed(42)
n_bg = min(max(5 * max(1, len(presence_samples)), 1), pool_len)
bg_indices = np.random.choice(pool_len, size=n_bg, replace=False)
background_samples = pool[bg_indices].astype(np.float32)
print(f"🌎 Background samples (for tables): {background_samples.shape}")

# Convert chosen background pool indices back to global linear → coords
bg_linear = valid_idx[bg_indices]
bg_lons, bg_lats = _lin_to_coords(bg_linear, W, profile["transform"])  # lon, lat

# ---------------- Prepare X/y ----------------
X = np.vstack([presence_samples, background_samples])
y = np.concatenate([
    np.ones(len(presence_samples), dtype=int),
    np.zeros(len(background_samples), dtype=int)
])

finite_mask = np.all(np.isfinite(X), axis=1)
Xc, yc = X[finite_mask], y[finite_mask]
if Xc.shape[0] == 0:
    raise RuntimeError("No usable training rows after filtering. Check predictor coverage & presence coordinates.")
if len(np.unique(yc)) < 2:
    raise RuntimeError("Only one class present after filtering (no variation between presence and background).")

# ---------------- Identify numeric vs categorical ----------------
is_cat = np.array([_is_categorical_name(nm) for nm in names], dtype=bool)
num_idx = np.flatnonzero(~is_cat)
cat_idx = np.flatnonzero(is_cat)

# ---------------- Drop uninformative predictors ----------------
dropped = []
keep_mask = np.ones(C, dtype=bool)

# (a) Drop constants over training rows
EPS = 1e-8
col_std = np.nanstd(Xc, axis=0)
const_cols = np.flatnonzero(col_std <= EPS)
for j in const_cols:
    keep_mask[j] = False
    dropped.append((names[j], "constant_over_training"))

# (b) For categoricals: drop if all-0 or all-1 over training rows
for j in cat_idx:
    if not keep_mask[j]:
        continue
    col = Xc[:, j]
    all0 = np.all(col < 0.5)
    all1 = np.all(col >= 0.5)
    if all0:
        keep_mask[j] = False
        dropped.append((names[j], "categorical_all_0"))
    elif all1:
        keep_mask[j] = False
        dropped.append((names[j], "categorical_all_1"))

if not np.any(keep_mask):
    raise RuntimeError("All predictors are uninformative over training data. Add varying layers.")

used_idx = np.flatnonzero(keep_mask)
names_used = [names[j] for j in used_idx]
is_cat_used = is_cat[used_idx]

print(f"✅ Using {len(names_used)} predictors: {', '.join(names_used)}")
if dropped:
    dd = pd.DataFrame(dropped, columns=["predictor", "reason"])
    dd.to_csv(drop_csv, index=False)
    print(f"📄 Dropped predictors saved to {drop_csv}")

# ---------------- Export raw samples (with coords) ----------------
# Presence coords (kept only)
pres_lats = kept_coords[:, 0] if kept_coords.size else np.array([], dtype=np.float64)
pres_lons = kept_coords[:, 1] if kept_coords.size else np.array([], dtype=np.float64)

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

# Also keep a simple coordinate list (legacy)
pd.DataFrame({"latitude": bg_lats, "longitude": bg_lons}).to_csv(abs_csv, index=False)
print(f"🗺️ Absence coordinates saved to {abs_csv}")

# ---------------- CV (spatial blocks else K-fold) ----------------
coords = kept_coords  # presence coords (lat, lon)
n_blocks = min(5, len(coords)) if len(coords) > 0 else 0
cv_aucs, cv_tsses, cv_kappas = [], [], []
cv_thresholds, cv_sensitivities, cv_specificities = [], [], []

# For assigning pool to blocks if spatial CV is possible
if n_blocks >= 2:
    pool_lons_all, pool_lats_all = _lin_to_coords(valid_idx, W, profile["transform"])

def _run_kfold_fallback():
    print("ℹ️ Falling back to Repeated Stratified K-fold CV (non-spatial).")
    Np = len(presence_samples)
    if Np < 3:
        note = "Too few presences for K-fold; reporting NaNs."
        out = _cv_summary_frame(0, "repeated_kfold", note, [], [], [], [], [], [],
                                len(df), len(presence_samples), len(names_used))
        out.to_csv(perf_cv_csv, index=False)
        print(f"📊 Wrote CV fallback (NaNs) to {perf_cv_csv}")
        return

    n_splits = max(3, min(5, max(2, Np // 5)))
    n_repeats = 5 if Np >= 10 else 3

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    for fold_i, (tr_idx, te_idx) in enumerate(rskf.split(presence_samples, np.ones(Np, dtype=int)), start=1):
        Xp_tr = presence_samples[tr_idx]
        Xp_te = presence_samples[te_idx]

        rng = np.random.default_rng(1234 + fold_i)
        n_bt_tr = min(5 * len(tr_idx), pool_len)
        n_bt_te = min(5 * len(te_idx),  pool_len)
        bt_idx  = rng.choice(pool_len, size=n_bt_tr, replace=False)
        bt_idx2 = rng.choice(pool_len, size=n_bt_te, replace=False)
        Xb_tr = pool[bt_idx]
        Xb_te = pool[bt_idx2]

        X_tr = np.vstack([Xp_tr, Xb_tr])
        y_tr = np.concatenate([np.ones(len(Xp_tr), dtype=int), np.zeros(len(Xb_tr), dtype=int)])
        X_te = np.vstack([Xp_te, Xb_te])
        y_te = np.concatenate([np.ones(len(Xp_te), dtype=int), np.zeros(len(Xb_te), dtype=int)])

        mtr = np.all(np.isfinite(X_tr), axis=1)
        mte = np.all(np.isfinite(X_te), axis=1)
        X_tr, y_tr = X_tr[mtr], y_tr[mtr]
        X_te, y_te = X_te[mte], y_te[mte]

        # slice to used columns
        X_tr_u = X_tr[:, used_idx]
        X_te_u = X_te[:, used_idx]

        # scale numerics only
        scaler_cv = StandardScaler().fit(X_tr_u[:, ~is_cat_used]) if np.any(~is_cat_used) else None
        if scaler_cv:
            X_tr_u[:, ~is_cat_used] = scaler_cv.transform(X_tr_u[:, ~is_cat_used])
            X_te_u[:, ~is_cat_used] = scaler_cv.transform(X_te_u[:, ~is_cat_used])

        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr_u, y_tr)

        p_tr = clf.predict_proba(X_tr_u)[:, 1]
        fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr, p_tr)
        bt = thr_tr[np.argmax(tpr_tr - fpr_tr)]

        p_te = clf.predict_proba(X_te_u)[:, 1]
        auc, tss, kapp, sens, spec = _compute_metrics_from_probs(y_te, p_te, bt)

        cv_aucs.append(float(auc))
        cv_tsses.append(float(tss))
        cv_kappas.append(float(kapp))
        cv_thresholds.append(float(bt))
        cv_sensitivities.append(float(sens))
        cv_specificities.append(float(spec))

    note = f"Spatial blocking unavailable; used non-spatial Repeated Stratified K-fold (splits={n_splits}, repeats={n_repeats})."
    out = _cv_summary_frame(n_splits * n_repeats, "repeated_kfold", note,
                            cv_aucs, cv_tsses, cv_kappas, cv_thresholds,
                            cv_sensitivities, cv_specificities,
                            len(df), len(presence_samples), len(names_used))
    out.to_csv(perf_cv_csv, index=False)
    print(f"📊 Wrote CV fallback (Repeated K-fold) to {perf_cv_csv}")

# Try spatial blocks
did_spatial = False
if n_blocks >= 2:
    try:
        kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init="auto").fit(coords)
        blocks = kmeans.labels_

        # Assign pool points to blocks
        pool_blocks = kmeans.predict(np.column_stack([pool_lats_all, pool_lons_all]))

        gkf = GroupKFold(n_splits=n_blocks)
        for tr_i, te_i in gkf.split(presence_samples,
                                    np.ones(len(presence_samples), dtype=int),
                                    groups=blocks):
            Xp_tr, Xp_te = presence_samples[tr_i], presence_samples[te_i]
            test_blocks = set(blocks[te_i])
            train_blocks = [b for b in range(n_blocks) if b not in test_blocks]

            mask_tr_pool = np.isin(pool_blocks, train_blocks)
            mask_te_pool = np.isin(pool_blocks, list(test_blocks))

            n_bt  = min(5 * len(tr_i), int(mask_tr_pool.sum()))
            n_bt2 = min(5 * len(te_i),  int(mask_te_pool.sum()))

            rng = np.random.default_rng(1000 + len(cv_aucs) + 1)
            if n_bt < 1 or n_bt2 < 1:
                bt_idx  = rng.choice(pool_len, size=max(1, 5 * len(tr_i)), replace=False)
                bt_idx2 = rng.choice(pool_len, size=max(1, 5 * len(te_i)),  replace=False)
            else:
                tr_inds = np.flatnonzero(mask_tr_pool)
                te_inds = np.flatnonzero(mask_te_pool)
                bt_idx  = rng.choice(tr_inds, size=n_bt,  replace=False)
                bt_idx2 = rng.choice(te_inds, size=n_bt2, replace=False)

            Xb_tr = pool[bt_idx]
            Xb_te = pool[bt_idx2]

            X_tr = np.vstack([Xp_tr, Xb_tr])
            y_tr = np.concatenate([np.ones(len(Xp_tr), dtype=int), np.zeros(len(Xb_tr), dtype=int)])
            X_te = np.vstack([Xp_te, Xb_te])
            y_te = np.concatenate([np.ones(len(Xp_te), dtype=int), np.zeros(len(Xb_te), dtype=int)])

            mtr = np.all(np.isfinite(X_tr), axis=1)
            mte = np.all(np.isfinite(X_te), axis=1)
            X_tr, y_tr = X_tr[mtr], y_tr[mtr]
            X_te, y_te = X_te[mte], y_te[mte]

            X_tr_u = X_tr[:, used_idx]
            X_te_u = X_te[:, used_idx]

            scaler_cv = StandardScaler().fit(X_tr_u[:, ~is_cat_used]) if np.any(~is_cat_used) else None
            if scaler_cv:
                X_tr_u[:, ~is_cat_used] = scaler_cv.transform(X_tr_u[:, ~is_cat_used])
                X_te_u[:, ~is_cat_used] = scaler_cv.transform(X_te_u[:, ~is_cat_used])

            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr_u, y_tr)

            p_tr = clf.predict_proba(X_tr_u)[:, 1]
            fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr, p_tr)
            bt = thr_tr[np.argmax(tpr_tr - fpr_tr)]

            p_te = clf.predict_proba(X_te_u)[:, 1]
            auc, tss, kapp, sens, spec = _compute_metrics_from_probs(y_te, p_te, bt)

            cv_aucs.append(float(auc))
            cv_tsses.append(float(tss))
            cv_kappas.append(float(kapp))
            cv_thresholds.append(float(bt))
            cv_sensitivities.append(float(sens))
            cv_specificities.append(float(spec))

        did_spatial = True
        out = _cv_summary_frame(n_blocks, "spatial_blocks", "",
                                cv_aucs, cv_tsses, cv_kappas, cv_thresholds,
                                cv_sensitivities, cv_specificities,
                                len(df), len(presence_samples), len(names_used))
        out.to_csv(perf_cv_csv, index=False)
        print(f"📊 Cross-validated (spatial blocks) metrics saved to {perf_cv_csv}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"⚠️ Spatial CV failed with error: {e}")
        _run_kfold_fallback()
else:
    print("ℹ️ Not enough presence points/blocks for spatial CV (need ≥2 blocks).")
    _run_kfold_fallback()

# ---------------- Final fit on all filtered rows ----------------
Xc_used = Xc[:, used_idx].copy()

# Standardize numerics only
scaler = StandardScaler().fit(Xc_used[:, ~is_cat_used]) if np.any(~is_cat_used) else None
if scaler:
    Xc_used[:, ~is_cat_used] = scaler.transform(Xc_used[:, ~is_cat_used])
    joblib.dump(scaler, scaler_pkl)
    print(f"💾 Saved scaler to {scaler_pkl}")

model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xc_used, yc)

# Fitted metrics
y_prob = model.predict_proba(Xc_used)[:, 1]
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
fitted_df.to_csv(compat_perf, index=False)
print(f"📊 Fitted metrics saved to {perf_fit_csv} and {compat_perf}")

# ---------------- Coefficients + p-values (statsmodels) ----------------
coef_used = model.coef_.reshape(1, -1)[0]
intercept = model.intercept_[0]

coef_full = np.zeros(C, dtype=float)
coef_full[used_idx] = coef_used

# Statsmodels p-values
std_err_full = np.full(C, np.nan, dtype=float)
pval_full = np.full(C, np.nan, dtype=float)

if HAVE_SM:
    try:
        X_sm = Xc_used.copy()
        X_sm = sm.add_constant(X_sm, has_constant='add')
        logit = sm.Logit(yc, X_sm)
        res = logit.fit(disp=False, maxiter=200)
        params = res.params  # [const, used...]
        bse = res.bse
        pvals = res.pvalues
        intercept = float(params[0])
        coef_used = params[1:].astype(float)

        # Map back to full length
        coef_full[:] = 0.0
        coef_full[used_idx] = coef_used
        # Std errors / pvals per used column
        for k, j in enumerate(used_idx):
            std_err_full[j] = float(bse[1 + k])
            pval_full[j]    = float(pvals[1 + k])
    except Exception as e:
        print(f"[pval] statsmodels failed: {e}", file=sys.stderr)
        # keep defaults: NaNs for se/p, sklearn coefficients already set
else:
    print("[pval] statsmodels not available; std_error and p_value will be NaN.", file=sys.stderr)

# Build coefficient table
coef_rows = [("Intercept", float(intercept), np.nan, np.nan)]
for j, name in enumerate(names):
    coef_rows.append((name, float(coef_full[j]), float(std_err_full[j]) if not np.isnan(std_err_full[j]) else np.nan,
                      float(pval_full[j]) if not np.isnan(pval_full[j]) else np.nan))
coef_df = pd.DataFrame(coef_rows, columns=["predictor", "coefficient", "std_error", "p_value"])
coef_df.to_csv(cof_csv := coef_csv, index=False)
print(f"📊 Coefficients saved to {cof_csv}")

# ---------------- Export standardization stats ----------------
# For numeric used columns → scaler stats; for numeric dropped or categorical → NaN stats
mean_full = np.full(C, np.nan, dtype=float)
std_full  = np.full(C, np.nan, dtype=float)
if scaler:
    # used_idx[~is_cat_used] selects the numeric-used columns in model order
    mean_full[used_idx[~is_cat_used]] = scaler.mean_.astype(float)
    std_full[used_idx[~is_cat_used]]  = scaler.scale_.astype(float)

stats_df = pd.DataFrame({
    "predictor": names,
    "is_categorical": is_cat.astype(int),
    "mean": mean_full,
    "std":  std_full,
    "dropped": ~keep_mask
})
stats_df.to_csv(stats_csv, index=False)
print(f"📄 Standardization stats saved to {stats_csv}")

# ---------------- Export standardized samples (numerics z-scored, cats unchanged) ----------------
def _standardize_rows_full(arr_full: np.ndarray) -> np.ndarray:
    if arr_full.size == 0:
        return arr_full.reshape((0, C))
    out = arr_full.copy().astype(np.float32)
    if scaler:
        out[:, used_idx[~is_cat_used]] = scaler.transform(out[:, used_idx[~is_cat_used]])
    return out

presence_std = _standardize_rows_full(presence_samples)
absence_std  = _standardize_rows_full(background_samples)

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

# ---------------- Suitability map ----------------
# Build predictions over pool
pred_flat = np.full(flat.shape[0], np.nan, dtype=np.float32)
pool_used = pool[:, used_idx].copy()
if scaler:
    pool_used[:, ~is_cat_used] = scaler.transform(pool_used[:, ~is_cat_used])
pred_flat[valid_mask] = model.predict_proba(pool_used)[:, 1].astype(np.float32)
pred_map = pred_flat.reshape((H, W))

with rasterio.open(output_map, "w", **profile) as dst:
    dst.write(pred_map, 1)
print(f"🎯 Final suitability map saved to {output_map}")

# ---------------- Standardized rasters (numerics z-scored; categoricals pass-through) ----------------
os.makedirs(std_ras_dir, exist_ok=True)
print(f"💾 Writing standardized rasters to {std_ras_dir}")

# Prepare standardized pool (full C), numerics scaled if scaler exists; categoricals unchanged
pool_s_full = pool.copy()
if scaler:
    pool_s_full[:, used_idx[~is_cat_used]] = scaler.transform(pool_s_full[:, used_idx[~is_cat_used]])

for j, name in enumerate(names):
    band_flat = np.full(flat.shape[0], np.float32(np.nan), dtype=np.float32)
    band_flat[valid_mask] = pool_s_full[:, j].astype(np.float32)
    band = band_flat.reshape((H, W))
    out_path = os.path.join(std_ras_dir, f"{name}_z.tif")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(band, 1)
    print(f"    • {name}_z.tif")

print("🏁 Done.")
