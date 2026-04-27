import os
import sys
import numpy as np
import pandas as pd
import rasterio
import joblib
import warnings

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

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
uploaded_abs_csv = os.path.join("inputs", "absence_points_uploaded.csv")
raster_dir   = "predictor_rasters/wgs84"
out_dir      = "outputs"
perf_cv_csv  = os.path.join(out_dir, "performance_metrics_cv.csv")
perf_fit_csv = os.path.join(out_dir, "performance_metrics_fitted.csv")
compat_perf  = os.path.join(out_dir, "performance_metrics.csv")
coef_csv     = os.path.join(out_dir, "coefficients.csv")
drop_csv     = os.path.join(out_dir, "dropped_predictors.csv")
multicol_csv = os.path.join(out_dir, "multicollinearity_check.csv")
high_corr_csv = os.path.join(out_dir, "high_correlation_pairs.csv")
pruned_report_csv = os.path.join(out_dir, "pruned_variables_report.csv")
scaler_pkl   = os.path.join(out_dir, "standard_scaler.pkl")
stats_csv    = os.path.join(out_dir, "standardization_stats.csv")
raw_csv      = os.path.join(out_dir, "sdm_point_samples_raw.csv")
std_csv      = os.path.join(out_dir, "sdm_point_samples_standardized.csv")
abs_csv      = os.path.join(out_dir, "absence_points_coordinates.csv")  # legacy / optional
output_map   = os.path.join(out_dir, "suitability_map_wgs84.tif")
std_ras_dir  = os.path.join(out_dir, "standardized_rasters_wgs84")
effect_plot_dir = os.path.join(out_dir, "predictor_effect_plots")
effect_plot_manifest_csv = os.path.join(out_dir, "predictor_effect_plots_manifest.csv")

os.makedirs(out_dir, exist_ok=True)

# ---------------- Helpers ----------------
def _is_categorical_name(name: str) -> bool:
    n = name.lower()
    # Treat MODIS class one-hots like "5_mixed_forest" as categorical.
    return (
        ("landcover" in n)
        or n.endswith("_onehot")
        or ("_mixed_forest" in n)
        or n.split("_", 1)[0].isdigit()
    )

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
        is_cat = _is_categorical_name(nm)

        # For categorical layers, fill with 0 and use dst_nodata=0
        # For numeric layers, fill with NaN and use dst_nodata=NaN
        if is_cat:
            dst = np.zeros((height, width), dtype=np.float32)
            resamp = WarpResampling.nearest
            dst_nodata = 0.0
        else:
            dst = np.full((height, width), np.nan, dtype=np.float32)
            resamp = WarpResampling.bilinear
            dst_nodata = np.nan

        with rasterio.open(p) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                resampling=resamp, dst_nodata=dst_nodata,
            )

        # For logging: % NaN after reprojection (should be ~0 for categoricals now)
        nan_pct = float(np.isnan(dst).mean() * 100.0)
        print(f"🧪 {nm}.tif → NaN%: {nan_pct:.2f}  ({'categorical' if is_cat else 'numeric'})")

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

def _extract_point_samples(points_df, stack, profile):
    if not {"latitude", "longitude"}.issubset(points_df.columns):
        raise RuntimeError("Point CSV must contain 'latitude' and 'longitude' (WGS84).")

    H, W, _ = stack.shape
    samples = []
    kept = []
    for lat, lon in zip(points_df["latitude"].astype(float).values, points_df["longitude"].astype(float).values):
        try:
            r, c = rowcol(profile["transform"], lon, lat, op=round)
        except Exception:
            continue
        if 0 <= r < H and 0 <= c < W:
            vals = stack[r, c, :]
            if np.any(np.isfinite(vals)):
                samples.append(vals)
                kept.append((lat, lon))

    samples = np.array(samples, dtype=np.float32)
    kept = np.array(kept, dtype=np.float32) if len(kept) else np.empty((0, 2), dtype=np.float32)
    return samples, kept

def _write_multicollinearity_check(X_used_raw, predictor_names, numeric_mask, summary_csv, pairs_csv,
                                   corr_threshold=0.70, vif_threshold=5.0):
    numeric_names = [name for name, is_numeric in zip(predictor_names, numeric_mask) if is_numeric]
    X_num = X_used_raw[:, numeric_mask].astype(float) if len(numeric_names) else np.empty((X_used_raw.shape[0], 0))

    summary_cols = ["predictor", "max_abs_correlation", "max_correlated_predictor", "vif", "flag"]
    pairs_cols = ["predictor_1", "predictor_2", "correlation", "abs_correlation"]

    if X_num.shape[1] == 0:
        pd.DataFrame(columns=summary_cols).to_csv(summary_csv, index=False)
        pd.DataFrame(columns=pairs_cols).to_csv(pairs_csv, index=False)
        print(f"Multicollinearity check skipped: no numeric predictors. Wrote {summary_csv}")
        return

    corr = np.eye(X_num.shape[1], dtype=float)
    if X_num.shape[1] > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(X_num, rowvar=False)
        corr = np.asarray(corr, dtype=float)

    pair_rows = []
    for i in range(len(numeric_names)):
        for j in range(i + 1, len(numeric_names)):
            c = corr[i, j]
            if np.isfinite(c) and abs(c) >= corr_threshold:
                pair_rows.append((numeric_names[i], numeric_names[j], float(c), float(abs(c))))

    vifs = np.full(len(numeric_names), np.nan, dtype=float)
    if X_num.shape[1] >= 2 and X_num.shape[0] > X_num.shape[1]:
        for j in range(X_num.shape[1]):
            yj = X_num[:, j]
            others = np.delete(X_num, j, axis=1)
            design = np.column_stack([np.ones(others.shape[0]), others])
            try:
                beta = np.linalg.lstsq(design, yj, rcond=None)[0]
                pred = design @ beta
                ss_res = float(np.sum((yj - pred) ** 2))
                ss_tot = float(np.sum((yj - np.mean(yj)) ** 2))
                if ss_tot > 1e-12:
                    r2 = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))
                    vifs[j] = np.inf if (1.0 - r2) <= 1e-12 else 1.0 / (1.0 - r2)
            except Exception:
                vifs[j] = np.nan

    summary_rows = []
    for i, name in enumerate(numeric_names):
        if len(numeric_names) > 1:
            abs_corr = np.abs(corr[i, :]).astype(float)
            abs_corr[i] = np.nan
            if np.all(np.isnan(abs_corr)):
                max_corr = np.nan
                partner = ""
            else:
                max_idx = int(np.nanargmax(abs_corr))
                max_corr = float(abs_corr[max_idx])
                partner = numeric_names[max_idx]
        else:
            max_corr = np.nan
            partner = ""

        flags = []
        if np.isfinite(max_corr) and max_corr >= corr_threshold:
            flags.append("high_pairwise_correlation")
        if np.isfinite(vifs[i]) and vifs[i] >= vif_threshold:
            flags.append("high_vif")
        summary_rows.append((name, max_corr, partner, float(vifs[i]) if np.isfinite(vifs[i]) else np.nan,
                             ";".join(flags) if flags else "ok"))

    pd.DataFrame(summary_rows, columns=summary_cols).to_csv(summary_csv, index=False)
    pd.DataFrame(pair_rows, columns=pairs_cols).to_csv(pairs_csv, index=False)
    print(f"Multicollinearity check saved to {summary_csv}; high-correlation pairs saved to {pairs_csv}")


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _safe_png_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text or "plot"))
    cleaned = cleaned.strip("_")
    return cleaned or "plot"


def _write_predictor_effect_plots(
    x_used: np.ndarray,
    names_used: list[str],
    is_cat_used: np.ndarray,
    intercept: float,
    coef_used: np.ndarray,
    cov_params: np.ndarray | None = None,
    scaler=None,
) -> pd.DataFrame:
    os.makedirs(effect_plot_dir, exist_ok=True)
    records = []
    n_rows, n_cols = x_used.shape
    if n_rows == 0 or n_cols == 0:
        pd.DataFrame(columns=["predictor", "plot_path", "plot_type", "ci_available", "x_units", "n_points"]).to_csv(
            effect_plot_manifest_csv, index=False
        )
        return pd.DataFrame()

    x_ref = np.zeros(n_cols, dtype=float)
    for j in range(n_cols):
        col = np.asarray(x_used[:, j], dtype=float)
        finite = np.isfinite(col)
        if not finite.any():
            x_ref[j] = 0.0
            continue
        if bool(is_cat_used[j]):
            rounded = np.round(col[finite], 6)
            vals, counts = np.unique(rounded, return_counts=True)
            x_ref[j] = float(vals[int(np.argmax(counts))])
        else:
            x_ref[j] = float(np.nanmedian(col[finite]))

    numeric_positions = np.flatnonzero(~is_cat_used)
    scaler_mean = getattr(scaler, "mean_", None) if scaler is not None else None
    scaler_scale = getattr(scaler, "scale_", None) if scaler is not None else None
    used_numeric_lookup = {}
    if scaler_mean is not None and scaler_scale is not None:
        for pos, col_idx in enumerate(numeric_positions):
            used_numeric_lookup[int(col_idx)] = (float(scaler_mean[pos]), float(scaler_scale[pos]))

    cov_ok = cov_params is not None and np.asarray(cov_params).shape == (n_cols + 1, n_cols + 1)
    cov = np.asarray(cov_params, dtype=float) if cov_ok else None

    for j, pred_name in enumerate(names_used):
        base = x_ref.copy()
        col = np.asarray(x_used[:, j], dtype=float)
        finite = np.isfinite(col)
        if not finite.any():
            continue

        if bool(is_cat_used[j]):
            levels = np.unique(np.round(col[finite], 6))
            levels = np.asarray(levels, dtype=float)
            if levels.size == 0:
                continue
            if levels.size > 12:
                levels = np.linspace(float(np.nanmin(levels)), float(np.nanmax(levels)), 12)
            x_grid = levels
            x_label = f"{pred_name} (categorical/binary)"
            x_units = "categorical"
            plot_type = "categorical_effect"
        else:
            lo = float(np.nanpercentile(col[finite], 1.0))
            hi = float(np.nanpercentile(col[finite], 99.0))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(np.nanmin(col[finite]))
                hi = float(np.nanmax(col[finite]))
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            if hi <= lo:
                hi = lo + 1e-6
            x_grid = np.linspace(lo, hi, 120, dtype=float)
            if j in used_numeric_lookup:
                m, s = used_numeric_lookup[j]
                x_label = f"{pred_name} (original units)"
                x_display = x_grid * s + m
                x_units = "original"
            else:
                x_label = f"{pred_name} (standardized units)"
                x_display = x_grid
                x_units = "standardized"
            plot_type = "numeric_effect"

        design = np.repeat(base.reshape(1, -1), len(x_grid), axis=0)
        design[:, j] = x_grid
        eta = float(intercept) + design @ np.asarray(coef_used, dtype=float)
        p_hat = _sigmoid(eta)

        ci_available = False
        p_lo = None
        p_hi = None
        if cov is not None:
            x_design = np.column_stack([np.ones(len(x_grid), dtype=float), design])
            var_eta = np.einsum("ij,jk,ik->i", x_design, cov, x_design)
            var_eta = np.maximum(var_eta, 0.0)
            se_eta = np.sqrt(var_eta)
            eta_lo = eta - 1.96 * se_eta
            eta_hi = eta + 1.96 * se_eta
            p_lo = _sigmoid(eta_lo)
            p_hi = _sigmoid(eta_hi)
            ci_available = True

        if bool(is_cat_used[j]):
            x_display = x_grid

        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=120)
        if bool(is_cat_used[j]):
            if ci_available:
                yerr = np.vstack([p_hat - p_lo, p_hi - p_hat])
                ax.errorbar(x_display, p_hat, yerr=yerr, fmt="o", capsize=4, color="#1f77b4", linewidth=1.2)
            else:
                ax.plot(x_display, p_hat, "o", color="#1f77b4")
            ax.set_xticks(x_display)
        else:
            ax.plot(x_display, p_hat, color="#1f77b4", linewidth=2.0, label="Predicted probability")
            if ci_available:
                ax.fill_between(x_display, p_lo, p_hi, color="#1f77b4", alpha=0.22, label="95% CI")

        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Predicted presence probability")
        ax.set_xlabel(x_label)
        subtitle = "Marginal effect with 95% CI" if ci_available else "Marginal effect (CI unavailable)"
        ax.set_title(f"{pred_name}\n{subtitle}", fontsize=11)
        ax.grid(True, alpha=0.25)
        if not bool(is_cat_used[j]) and ci_available:
            ax.legend(loc="best", frameon=False)
        fig.tight_layout()

        png_name = f"{_safe_png_name(pred_name)}_effect.png"
        png_path = os.path.join(effect_plot_dir, png_name)
        fig.savefig(png_path, dpi=120)
        plt.close(fig)

        records.append(
            {
                "predictor": pred_name,
                "plot_path": os.path.relpath(png_path, "."),
                "plot_type": plot_type,
                "ci_available": int(ci_available),
                "x_units": x_units,
                "n_points": int(len(x_grid)),
            }
        )

    out_df = pd.DataFrame(records, columns=["predictor", "plot_path", "plot_type", "ci_available", "x_units", "n_points"])
    out_df.to_csv(effect_plot_manifest_csv, index=False)
    print(f"📈 Predictor effect plots manifest saved to {effect_plot_manifest_csv}")
    return out_df

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

active_env = (os.environ.get("SDM_ACTIVE_PREDICTORS", "") or "").strip()
if active_env:
    requested = [s.strip() for s in active_env.split(",") if s.strip()]
    requested_set = set(requested)
    rasters = [p for p in rasters if os.path.splitext(os.path.basename(p))[0] in requested_set]
    if not rasters:
        raise RuntimeError("No active predictors available after applying SDM_ACTIVE_PREDICTORS.")
    print(f"Using active predictors from SDM_ACTIVE_PREDICTORS: {', '.join(sorted(requested_set))}")

# Build stack
stack, names, profile = _read_stack_align(rasters)
H, W, C = stack.shape

# ---------------- Extract presence samples ----------------
presence_samples = []
kept_coords = []
for lat, lon in zip(lats, lons):
    try:
        r, c = rowcol(profile["transform"], lon, lat, op=round)
    except Exception:
        continue
    if 0 <= r < H and 0 <= c < W:
        vals = stack[r, c, :]
        if np.any(np.isfinite(vals)):
            presence_samples.append(vals)
            kept_coords.append((lat, lon))

presence_samples = np.array(presence_samples, dtype=np.float32)
kept_coords = np.array(kept_coords, dtype=np.float32) if len(kept_coords) else np.empty((0, 2), dtype=np.float32)
print(f"📍 Presence samples: {presence_samples.shape}")

if presence_samples.shape[0] == 0:
    raise RuntimeError("No presence points fell on pixels with any predictor values. Check coverage or extent.")

# ---------------- Background pool & random sampling ----------------
flat       = stack.reshape(-1, C).astype(np.float32)
valid_mask = np.any(np.isfinite(flat), axis=1)
valid_idx  = np.flatnonzero(valid_mask)
pool       = flat[valid_mask]
pool_len   = int(pool.shape[0])
print(f"✅ Valid pixel pool: {pool_len}")

if pool_len == 0:
    raise RuntimeError("No pixels with any predictor values after alignment. Check rasters/AOI.")

# Purely random backgrounds (uniform on valid pixels)
np.random.seed(42)
n_bg = min(max(5 * max(1, len(presence_samples)), 1), pool_len)
bg_indices = np.random.choice(pool_len, size=n_bg, replace=False)
background_samples = pool[bg_indices].astype(np.float32)
print(f"🌎 Background samples (for tables): {background_samples.shape}")

# Convert chosen background pool indices back to global linear → coords
bg_linear = valid_idx[bg_indices]
bg_lons, bg_lats = _lin_to_coords(bg_linear, W, profile["transform"])  # lon, lat

has_uploaded_absences = os.path.exists(uploaded_abs_csv)
if has_uploaded_absences:
    abs_df = pd.read_csv(uploaded_abs_csv)
    background_samples, kept_abs_coords = _extract_point_samples(abs_df, stack, profile)
    if background_samples.shape[0] == 0:
        raise RuntimeError("No uploaded absence points fell on pixels with any predictor values. Check coverage or extent.")
    bg_lats = kept_abs_coords[:, 0] if kept_abs_coords.size else np.array([], dtype=np.float64)
    bg_lons = kept_abs_coords[:, 1] if kept_abs_coords.size else np.array([], dtype=np.float64)
    print(f"Uploaded absence samples: {background_samples.shape}")

# ---------------- Prepare X/y ----------------
X = np.vstack([presence_samples, background_samples])
y = np.concatenate([
    np.ones(len(presence_samples), dtype=int),
    np.zeros(len(background_samples), dtype=int)
])

finite_mask = np.all(np.isfinite(X), axis=1)
is_cat = np.array([_is_categorical_name(nm) for nm in names], dtype=bool)
num_idx = np.flatnonzero(~is_cat)
cat_idx = np.flatnonzero(is_cat)

# ---------------- Drop uninformative predictors over presences + available/absences ----------------
dropped = []
keep_mask = np.ones(C, dtype=bool)
EPS = 1e-8

for j, name in enumerate(names):
    col = X[:, j].astype(float)
    finite = np.isfinite(col)
    if not np.any(finite):
        keep_mask[j] = False
        dropped.append((name, "all_na_over_training"))
        continue
    finite_col = col[finite]
    if np.all(np.abs(finite_col) <= EPS):
        keep_mask[j] = False
        dropped.append((name, "all_0_over_training"))
        continue
    if float(np.nanstd(finite_col)) <= EPS:
        keep_mask[j] = False
        dropped.append((name, "constant_over_training"))

for j in cat_idx:
    if not keep_mask[j]:
        continue
    col = X[:, j].astype(float)
    finite_col = col[np.isfinite(col)]
    if finite_col.size == 0:
        keep_mask[j] = False
        dropped.append((names[j], "categorical_all_na"))
        continue
    all0 = np.all(finite_col < 0.5)
    all1 = np.all(finite_col >= 0.5)
    if all0:
        keep_mask[j] = False
        dropped.append((names[j], "categorical_all_0_over_training"))
    elif all1:
        keep_mask[j] = False
        dropped.append((names[j], "categorical_all_1_over_training"))

if not np.any(keep_mask):
    raise RuntimeError("All predictors are uninformative over combined presence and available/absence data. Add varying layers.")

used_idx = np.flatnonzero(keep_mask)
finite_mask = np.all(np.isfinite(X[:, used_idx]), axis=1)
Xc, yc = X[finite_mask], y[finite_mask]
if Xc.shape[0] == 0:
    raise RuntimeError("No usable training rows after pruning and filtering. Check predictor coverage & point coordinates.")
if len(np.unique(yc)) < 2:
    raise RuntimeError("Only one class present after pruning/filtering (no variation between presence and available/absence data).")

# Re-check constants after row filtering.
for j in list(used_idx):
    col = Xc[:, j].astype(float)
    if float(np.nanstd(col)) <= EPS:
        keep_mask[j] = False
        dropped.append((names[j], "constant_after_row_filtering"))

if not np.any(keep_mask):
    raise RuntimeError("All predictors are uninformative after row filtering. Add varying layers.")

used_idx = np.flatnonzero(keep_mask)
finite_mask = np.all(np.isfinite(X[:, used_idx]), axis=1)
Xc, yc = X[finite_mask], y[finite_mask]
if Xc.shape[0] == 0 or len(np.unique(yc)) < 2:
    raise RuntimeError("No usable two-class training rows after final predictor pruning.")

is_cat_used = is_cat[used_idx]
names_used = [names[j] for j in used_idx]

print(f"✅ Using {len(names_used)} predictors: {', '.join(names_used)}")
if dropped:
    dd = pd.DataFrame(dropped, columns=["predictor", "reason"])
    dd.to_csv(drop_csv, index=False)
    pruned_rows = []
    for pred, reason in dropped:
        if reason == "all_na_over_training":
            details = "All combined presence and available/absence training values are NA."
        elif reason == "all_0_over_training":
            details = "All combined presence and available/absence training values are 0."
        elif reason.startswith("categorical_all_0"):
            details = "All combined presence and available/absence training values are categorical 0."
        elif reason.startswith("categorical_all_1"):
            details = "All combined presence and available/absence training values are categorical 1."
        else:
            details = "Predictor is constant across the combined presence and available/absence training sample."
        pruned_rows.append(
            {
                "predictor": pred,
                "reason": reason,
                "details": details,
                "metric": np.nan,
                "threshold": np.nan,
                "correlated_with": "",
                "iteration": 0,
                "auto_pruned": 1,
            }
        )
    pd.DataFrame(pruned_rows).to_csv(pruned_report_csv, index=False)
    print(f"📄 Dropped predictors saved to {drop_csv}")

_write_multicollinearity_check(
    Xc[:, used_idx],
    names_used,
    ~is_cat_used,
    multicol_csv,
    high_corr_csv,
)
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

def _run_presence_absence_kfold():
    print("Using uploaded absences with Repeated Stratified K-fold CV.")
    class_counts = np.bincount(yc.astype(int), minlength=2)
    min_class = int(class_counts.min())
    if min_class < 2:
        note = "Too few rows in one class for K-fold; reporting NaNs."
        out = _cv_summary_frame(0, "presence_absence_repeated_kfold", note, [], [], [], [], [], [],
                                len(df), len(presence_samples), len(names_used))
        out.to_csv(perf_cv_csv, index=False)
        print(f"Wrote presence-absence CV fallback (NaNs) to {perf_cv_csv}")
        return

    n_splits = min(5, min_class)
    n_repeats = 5 if min_class >= 10 else 3
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    for tr_idx, te_idx in rskf.split(Xc, yc):
        X_tr_u = Xc[tr_idx][:, used_idx].copy()
        X_te_u = Xc[te_idx][:, used_idx].copy()
        y_tr = yc[tr_idx]
        y_te = yc[te_idx]

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

    note = f"Used uploaded absence rows with repeated stratified K-fold (splits={n_splits}, repeats={n_repeats})."
    out = _cv_summary_frame(n_splits * n_repeats, "presence_absence_repeated_kfold", note,
                            cv_aucs, cv_tsses, cv_kappas, cv_thresholds,
                            cv_sensitivities, cv_specificities,
                            len(df), len(presence_samples), len(names_used))
    out.to_csv(perf_cv_csv, index=False)
    print(f"Presence-absence CV metrics saved to {perf_cv_csv}")

# Try spatial blocks
did_spatial = False
if has_uploaded_absences:
    _run_presence_absence_kfold()
elif n_blocks >= 2:
    try:
        coords64 = np.asarray(coords, dtype=np.float64)
        kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init="auto").fit(coords64)
        blocks = kmeans.labels_

        # Assign pool points to blocks
        pool_coords = np.column_stack([pool_lats_all, pool_lons_all]).astype(np.float64, copy=False)
        pool_blocks = kmeans.predict(pool_coords)

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
sm_cov_params = None

if HAVE_SM:
    try:
        # Refit using statsmodels for p-values
        X_sm = Xc_used.copy()
        X_sm = sm.add_constant(X_sm, has_constant='add')
        logit = sm.Logit(yc, X_sm)
        res = logit.fit(disp=False, maxiter=200)

        # Extract parameters
        params = res.params
        bse = res.bse
        pvals = res.pvalues
        sm_cov_params = np.asarray(res.cov_params(), dtype=float)

        intercept = float(params[0])
        coef_used = params[1:]  # same order as used_idx

        # Map back correctly
        for k, j in enumerate(used_idx):
            coef_full[j] = float(coef_used[k])
            std_err_full[j] = float(bse[1 + k])
            pval_full[j] = float(pvals[1 + k])

        print("📊 p-values successfully computed with statsmodels.")
    except Exception as e:
        print(f"[pval] statsmodels failed: {e}", file=sys.stderr)
else:
    print("[pval] statsmodels not available; std_error and p_value will be NaN.", file=sys.stderr)

# Build coefficient table
coef_rows = [("Intercept", float(intercept), np.nan, np.nan)]
for j, name in enumerate(names):
    coef_rows.append((
        name,
        float(coef_full[j]),
        float(std_err_full[j]) if np.isfinite(std_err_full[j]) else np.nan,
        float(pval_full[j]) if np.isfinite(pval_full[j]) else np.nan
    ))
coef_df = pd.DataFrame(coef_rows, columns=["predictor", "coefficient", "std_error", "p_value"])
coef_df.to_csv(coef_csv, index=False)
print(f"📊 Coefficients saved to {coef_csv}")


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
