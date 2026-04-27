import argparse
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression

try:
    import statsmodels.api as sm
    HAVE_SM = True
except Exception:
    HAVE_SM = False


OUT_DIR = "outputs"
RASTER_DIR = os.path.join("predictor_rasters", "wgs84")
EFFECT_PLOT_DIR = os.path.join(OUT_DIR, "predictor_effect_plots")
EFFECT_PLOT_MANIFEST_CSV = os.path.join(OUT_DIR, "predictor_effect_plots_manifest.csv")


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -709, 709)
    return 1.0 / (1.0 + np.exp(-z))


def _safe_png_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text or "plot"))
    return cleaned.strip("_") or "plot"


def _is_binary_col(col: np.ndarray) -> bool:
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return False
    vals = np.unique(np.round(finite, 6))
    return vals.size <= 2 and set(vals).issubset({0.0, 1.0})


def _fit_covariance(x: np.ndarray, y: np.ndarray):
    if not HAVE_SM:
        return None
    try:
        model = sm.Logit(y, sm.add_constant(x, has_constant="add")).fit(disp=False, maxiter=200)
        return np.asarray(model.cov_params(), dtype=float)
    except Exception:
        return None


def _write_effect_plots(
    *,
    x_model: np.ndarray,
    y: np.ndarray,
    names: list[str],
    intercept: float,
    coef: np.ndarray,
    y_label: str,
    title_prefix: str,
    x_display_transform: dict[str, tuple[float, float]] | None = None,
    categorical_names: set[str] | None = None,
) -> pd.DataFrame:
    os.makedirs(EFFECT_PLOT_DIR, exist_ok=True)
    records = []
    x_model = np.asarray(x_model, dtype=float)
    y = np.asarray(y, dtype=int)
    coef = np.asarray(coef, dtype=float)
    x_display_transform = x_display_transform or {}
    categorical_names = categorical_names or set()

    if x_model.ndim != 2 or x_model.shape[1] == 0 or not names:
        return _write_manifest(records)

    x_ref = np.nanmedian(x_model, axis=0)
    x_ref = np.where(np.isfinite(x_ref), x_ref, 0.0)
    cov = _fit_covariance(x_model, y)
    cov_ok = cov is not None and cov.shape == (len(names) + 1, len(names) + 1)

    for j, pred_name in enumerate(names):
        col = np.asarray(x_model[:, j], dtype=float)
        finite = np.isfinite(col)
        if not finite.any():
            continue
        is_cat = pred_name in categorical_names or _is_binary_col(col)
        if is_cat:
            levels = np.unique(np.round(col[finite], 6)).astype(float)
            if levels.size == 0:
                continue
            if levels.size > 12:
                levels = np.linspace(float(np.nanmin(levels)), float(np.nanmax(levels)), 12)
            x_grid = levels
            plot_type = "categorical_effect"
            x_label = f"{pred_name} (categorical/binary)"
            x_display = x_grid
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
            plot_type = "numeric_effect"
            if pred_name in x_display_transform:
                mean, std = x_display_transform[pred_name]
                x_display = x_grid * std + mean
                x_label = f"{pred_name} (original units)"
                x_units = "original"
            else:
                x_display = x_grid
                x_label = pred_name
                x_units = "original"

        design = np.repeat(x_ref.reshape(1, -1), len(x_grid), axis=0)
        design[:, j] = x_grid
        eta = float(intercept) + design @ coef
        p_hat = _sigmoid(eta)

        ci_available = False
        p_lo = p_hi = None
        if cov_ok:
            x_design = np.column_stack([np.ones(len(x_grid), dtype=float), design])
            var_eta = np.einsum("ij,jk,ik->i", x_design, cov, x_design)
            if np.all(np.isfinite(var_eta)):
                se_eta = np.sqrt(np.maximum(var_eta, 0.0))
                p_lo = _sigmoid(eta - 1.96 * se_eta)
                p_hi = _sigmoid(eta + 1.96 * se_eta)
                ci_available = True

        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=120)
        if is_cat:
            if ci_available:
                yerr = np.vstack([p_hat - p_lo, p_hi - p_hat])
                ax.errorbar(x_display, p_hat, yerr=yerr, fmt="o", capsize=4, color="#1f77b4", linewidth=1.2)
            else:
                ax.plot(x_display, p_hat, "o", color="#1f77b4")
            ax.set_xticks(x_display)
            x_units = "categorical"
        else:
            ax.plot(x_display, p_hat, color="#1f77b4", linewidth=2.0, label=y_label)
            if ci_available:
                ax.fill_between(x_display, p_lo, p_hi, color="#1f77b4", alpha=0.22, label="95% CI")
                ax.legend(loc="best", frameon=False)

        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        subtitle = f"{title_prefix} with 95% CI" if ci_available else f"{title_prefix} (CI unavailable)"
        ax.set_title(f"{pred_name}\n{subtitle}", fontsize=11)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        png_name = f"{_safe_png_name(pred_name)}_effect.png"
        png_path = os.path.join(EFFECT_PLOT_DIR, png_name)
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

    return _write_manifest(records)


def _write_manifest(records: list[dict]) -> pd.DataFrame:
    cols = ["predictor", "plot_path", "plot_type", "ci_available", "x_units", "n_points"]
    out = pd.DataFrame(records, columns=cols)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(EFFECT_PLOT_MANIFEST_CSV, index=False)
    return out


def _logistic_effect_plots() -> pd.DataFrame:
    samples_fp = os.path.join(OUT_DIR, "sdm_point_samples_standardized.csv")
    coef_fp = os.path.join(OUT_DIR, "coefficients.csv")
    stats_fp = os.path.join(OUT_DIR, "standardization_stats.csv")
    if not os.path.exists(samples_fp) or not os.path.exists(coef_fp):
        return _write_manifest([])

    samples = pd.read_csv(samples_fp)
    coef_df = pd.read_csv(coef_fp)
    stats = pd.read_csv(stats_fp) if os.path.exists(stats_fp) else pd.DataFrame()
    names = [
        str(v)
        for v in coef_df.get("predictor", [])
        if str(v).strip().lower() != "intercept"
    ]
    names = [n for n in names if n in samples.columns]
    if not names or "label" not in samples.columns:
        return _write_manifest([])

    coef_lookup = {
        str(r["predictor"]): float(r["coefficient"])
        for _, r in coef_df.iterrows()
        if str(r.get("predictor", "")).strip().lower() != "intercept"
    }
    intercept_rows = coef_df[coef_df["predictor"].astype(str).str.lower() == "intercept"]
    intercept = float(intercept_rows.iloc[0]["coefficient"]) if not intercept_rows.empty else 0.0
    coef = np.asarray([coef_lookup.get(n, 0.0) for n in names], dtype=float)
    x = samples[names].to_numpy(dtype=float)
    y = samples["label"].to_numpy(dtype=int)

    display = {}
    categorical = set()
    if not stats.empty and "predictor" in stats.columns:
        for _, row in stats.iterrows():
            pred = str(row.get("predictor", ""))
            if pred not in names:
                continue
            if int(row.get("is_categorical", 0) or 0) == 1:
                categorical.add(pred)
                continue
            mean = row.get("mean", np.nan)
            std = row.get("std", np.nan)
            if pd.notna(mean) and pd.notna(std) and float(std) != 0.0:
                display[pred] = (float(mean), float(std))

    return _write_effect_plots(
        x_model=x,
        y=y,
        names=names,
        intercept=intercept,
        coef=coef,
        y_label="Predicted presence probability",
        title_prefix="Marginal effect",
        x_display_transform=display,
        categorical_names=categorical,
    )


def _list_predictor_rasters() -> list[str]:
    if not os.path.isdir(RASTER_DIR):
        return []
    return sorted(os.path.join(RASTER_DIR, fn) for fn in os.listdir(RASTER_DIR) if fn.lower().endswith(".tif"))


def _sample_rasters(points: pd.DataFrame, raster_paths: list[str]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    names = [os.path.splitext(os.path.basename(p))[0] for p in raster_paths]
    out = pd.DataFrame(index=points.index)
    lats = pd.to_numeric(points["latitude"], errors="coerce").to_numpy(dtype=float)
    lons = pd.to_numeric(points["longitude"], errors="coerce").to_numpy(dtype=float)
    finite_points = np.isfinite(lats) & np.isfinite(lons)
    for name, path in zip(names, raster_paths):
        vals = np.full(len(points), np.nan, dtype=float)
        with rasterio.open(path) as src:
            coords = [(float(lons[i]), float(lats[i])) for i in np.flatnonzero(finite_points)]
            nodata = src.nodata
            for idx, sample in zip(np.flatnonzero(finite_points), src.sample(coords, indexes=1, masked=True)):
                try:
                    v = sample.item() if hasattr(sample, "item") else sample[0]
                except Exception:
                    continue
                if np.ma.is_masked(v):
                    continue
                if nodata is not None and np.isfinite(nodata) and float(v) == float(nodata):
                    continue
                vals[idx] = float(v) if np.isfinite(v) else np.nan
        out[name] = vals
    finite = np.all(np.isfinite(out.to_numpy(dtype=float)), axis=1) if names else np.array([], dtype=bool)
    return out, finite, names


def _issa_effect_plots() -> pd.DataFrame:
    samples_fp = os.path.join(OUT_DIR, "issa_steps_samples.csv")
    if not os.path.exists(samples_fp):
        return _write_manifest([])
    samples = pd.read_csv(samples_fp)
    required = {"end_latitude", "end_longitude", "used"}
    if not required.issubset(samples.columns):
        return _write_manifest([])
    raster_paths = _list_predictor_rasters()
    if not raster_paths:
        return _write_manifest([])

    points = pd.DataFrame(
        {
            "latitude": pd.to_numeric(samples["end_latitude"], errors="coerce"),
            "longitude": pd.to_numeric(samples["end_longitude"], errors="coerce"),
            "label": pd.to_numeric(samples["used"], errors="coerce"),
        }
    ).dropna(subset=["latitude", "longitude", "label"]).copy()
    if points.empty:
        return _write_manifest([])
    features, finite, names = _sample_rasters(points, raster_paths)
    if not names:
        return _write_manifest([])
    y = points["label"].to_numpy(dtype=int)[finite]
    x = features.to_numpy(dtype=float)[finite]
    if x.size == 0 or len(np.unique(y)) < 2:
        return _write_manifest([])
    clf = LogisticRegression(max_iter=700, solver="lbfgs", class_weight="balanced")
    clf.fit(x, y)
    return _write_effect_plots(
        x_model=x,
        y=y,
        names=names,
        intercept=float(clf.intercept_[0]),
        coef=np.asarray(clf.coef_[0], dtype=float),
        y_label="Predicted use probability",
        title_prefix="Projection effect",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate predictor effect plots after model fitting.")
    parser.add_argument("--model", choices=["auto", "logistic_regression", "issa"], default="auto")
    args = parser.parse_args()

    if args.model == "logistic_regression":
        out = _logistic_effect_plots()
    elif args.model == "issa":
        out = _issa_effect_plots()
    elif os.path.exists(os.path.join(OUT_DIR, "sdm_point_samples_standardized.csv")):
        out = _logistic_effect_plots()
    else:
        out = _issa_effect_plots()

    print(f"Effect plot postprocessing complete: {len(out)} plot(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
