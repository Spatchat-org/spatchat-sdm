import math
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


MOVEMENT_CSV = os.path.join("inputs", "movement_points.csv")
RASTER_DIR = os.path.join("predictor_rasters", "wgs84")
OUT_DIR = "outputs"

SUMMARY_CSV = os.path.join(OUT_DIR, "issa_summary.csv")
COEF_CSV = os.path.join(OUT_DIR, "issa_coefficients.csv")
SAMPLES_CSV = os.path.join(OUT_DIR, "issa_steps_samples.csv")

PROJ_METRICS_CSV = os.path.join(OUT_DIR, "issa_projection_metrics.csv")
PROJ_COEF_CSV = os.path.join(OUT_DIR, "issa_projection_coefficients.csv")
SUITABILITY_MAP = os.path.join(OUT_DIR, "suitability_map_wgs84.tif")
ABSENCE_COORDS_CSV = os.path.join(OUT_DIR, "absence_points_coordinates.csv")


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2) + math.cos(p1) * math.cos(p2) * (math.sin(dlmb / 2.0) ** 2)
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))


def _bearing_rad(lat1, lon1, lat2, lon2):
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return math.atan2(y, x)


def _angle_wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _destination_point(lat_deg, lon_deg, bearing_rad, distance_m):
    """Great-circle destination from start point + bearing + distance."""
    r = 6371000.0
    if distance_m <= 0:
        return float(lat_deg), float(lon_deg)
    lat1 = math.radians(float(lat_deg))
    lon1 = math.radians(float(lon_deg))
    d = float(distance_m) / r
    brg = float(bearing_rad)
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brg))
    lon2 = lon1 + math.atan2(
        math.sin(brg) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    lon2 = (lon2 + math.pi) % (2.0 * math.pi) - math.pi
    return math.degrees(lat2), math.degrees(lon2)


def _prepare_steps(df):
    rows = []
    for aid, grp in df.groupby("animal_id", sort=False):
        g = grp.sort_values("timestamp").reset_index(drop=True)
        prev_bearing = None
        for i in range(1, len(g)):
            p0 = g.iloc[i - 1]
            p1 = g.iloc[i]
            step_len = _haversine_m(p0.latitude, p0.longitude, p1.latitude, p1.longitude)
            if not np.isfinite(step_len) or step_len <= 0:
                continue
            bearing = _bearing_rad(p0.latitude, p0.longitude, p1.latitude, p1.longitude)
            # First step per animal has undefined turning angle; skip it for TA-based iSSA.
            if prev_bearing is None:
                prev_bearing = bearing
                continue
            turn = _angle_wrap(bearing - prev_bearing)
            rows.append(
                {
                    "animal_id": str(aid),
                    "stratum_id": f"{aid}_{i}",
                    "start_latitude": float(p0.latitude),
                    "start_longitude": float(p0.longitude),
                    "end_latitude": float(p1.latitude),
                    "end_longitude": float(p1.longitude),
                    "start_bearing_rad": float(prev_bearing),
                    "step_length_m": float(step_len),
                    "turn_angle_rad": float(turn),
                    "used": 1,
                }
            )
            prev_bearing = bearing
    return pd.DataFrame(rows)


def _build_used_available(used_steps, n_available=10, seed=42):
    if used_steps.empty:
        raise RuntimeError("No valid movement steps could be derived from movement_points.csv")
    rng = np.random.default_rng(seed)
    empirical_lengths = used_steps["step_length_m"].to_numpy()
    empirical_turns = used_steps["turn_angle_rad"].to_numpy()
    available_rows = []
    for _, row in used_steps.iterrows():
        sampled_lengths = rng.choice(empirical_lengths, size=n_available, replace=True)
        sampled_turns = rng.choice(empirical_turns, size=n_available, replace=True)
        for j in range(n_available):
            d = float(sampled_lengths[j])
            ta = float(sampled_turns[j])
            end_bearing = _angle_wrap(float(row["start_bearing_rad"]) + ta)
            end_lat, end_lon = _destination_point(
                row["start_latitude"],
                row["start_longitude"],
                end_bearing,
                d,
            )
            available_rows.append(
                {
                    "animal_id": row["animal_id"],
                    "stratum_id": row["stratum_id"],
                    "start_latitude": float(row["start_latitude"]),
                    "start_longitude": float(row["start_longitude"]),
                    "end_latitude": float(end_lat),
                    "end_longitude": float(end_lon),
                    "start_bearing_rad": float(row["start_bearing_rad"]),
                    "step_length_m": d,
                    "turn_angle_rad": ta,
                    "used": 0,
                }
            )
    available_df = pd.DataFrame(available_rows)
    return pd.concat([used_steps, available_df], ignore_index=True)


def _step_feature_frame(samples):
    return pd.DataFrame(
        {
            "log_step_length": np.log(np.clip(samples["step_length_m"].to_numpy(dtype=float), 1e-6, None)),
            "step_length_m": samples["step_length_m"].to_numpy(dtype=float),
            "cos_turn_angle": np.cos(samples["turn_angle_rad"].to_numpy(dtype=float)),
        }
    )


def _cv_auc(x: np.ndarray, y: np.ndarray, n_splits=5, n_repeats=2):
    if len(np.unique(y)) < 2 or len(y) < 20:
        return np.nan, np.nan, 0
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    splits = min(max(2, n_splits), n_pos, n_neg)
    if splits < 2:
        return np.nan, np.nan, 0
    rkf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=n_repeats, random_state=42)
    aucs = []
    for tr, te in rkf.split(x, y):
        clf = LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced")
        clf.fit(x[tr], y[tr])
        p = clf.predict_proba(x[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], p)))
    return float(np.mean(aucs)), float(np.std(aucs)), int(len(aucs))


def _list_predictor_rasters():
    if not os.path.isdir(RASTER_DIR):
        return []
    rasters = sorted(
        os.path.join(RASTER_DIR, fn)
        for fn in os.listdir(RASTER_DIR)
        if fn.lower().endswith(".tif")
    )
    active_env = (os.environ.get("SDM_ACTIVE_PREDICTORS", "") or "").strip()
    if active_env:
        wanted = {s.strip() for s in active_env.split(",") if s.strip()}
        rasters = [p for p in rasters if os.path.splitext(os.path.basename(p))[0] in wanted]
    return rasters


def _extract_predictors_at_points(points_df: pd.DataFrame, raster_paths: list[str]):
    if not raster_paths:
        return pd.DataFrame(), np.array([], dtype=bool), None, None
    names = [os.path.splitext(os.path.basename(p))[0] for p in raster_paths]
    with rasterio.open(raster_paths[0]) as ref:
        profile = ref.profile.copy()
        transform = ref.transform
        h, w = ref.height, ref.width

    out = {nm: np.full(len(points_df), np.nan, dtype=float) for nm in names}
    lats = pd.to_numeric(points_df["latitude"], errors="coerce").to_numpy(dtype=float)
    lons = pd.to_numeric(points_df["longitude"], errors="coerce").to_numpy(dtype=float)

    row_col = []
    for lat, lon in zip(lats, lons):
        if not np.isfinite(lat) or not np.isfinite(lon):
            row_col.append((-1, -1))
            continue
        try:
            r, c = rowcol(transform, lon, lat, op=round)
            row_col.append((int(r), int(c)))
        except Exception:
            row_col.append((-1, -1))

    for path, nm in zip(raster_paths, names):
        with rasterio.open(path) as src:
            arr = src.read(1)
        vals = out[nm]
        for i, (r, c) in enumerate(row_col):
            if 0 <= r < h and 0 <= c < w:
                v = arr[r, c]
                vals[i] = float(v) if np.isfinite(v) else np.nan
        out[nm] = vals

    feat_df = pd.DataFrame(out)
    finite = np.all(np.isfinite(feat_df.to_numpy(dtype=float)), axis=1)
    return feat_df, finite, profile, names


def _fit_projection_model(step_samples: pd.DataFrame, raster_paths):
    if not raster_paths:
        return {"ok": False, "message": "No predictor rasters found for iSSA projection."}
    if step_samples.empty:
        return {"ok": False, "message": "No iSSA step samples available for projection model."}
    all_points = pd.DataFrame(
        {
            "latitude": pd.to_numeric(step_samples["end_latitude"], errors="coerce"),
            "longitude": pd.to_numeric(step_samples["end_longitude"], errors="coerce"),
            "label": pd.to_numeric(step_samples["used"], errors="coerce"),
        }
    ).dropna(subset=["latitude", "longitude", "label"]).copy()
    if all_points.empty:
        return {"ok": False, "message": "No valid endpoint coordinates in iSSA samples for projection."}
    all_points["label"] = all_points["label"].astype(int)
    # Map overlay points should reflect iSSA available endpoints for each used step.
    avail_points = all_points.loc[all_points["label"] == 0, ["latitude", "longitude"]].copy()
    avail_points.to_csv(ABSENCE_COORDS_CSV, index=False)

    feat_df, finite_mask, profile, names = _extract_predictors_at_points(all_points, raster_paths)
    if feat_df.empty:
        return {"ok": False, "message": "No predictor feature table could be built for iSSA projection."}
    y = all_points["label"].to_numpy(dtype=int)[finite_mask]
    x = feat_df.to_numpy(dtype=float)[finite_mask]
    if len(np.unique(y)) < 2:
        return {"ok": False, "message": "iSSA projection training set has only one class after filtering."}

    cv_mean, cv_sd, cv_n = _cv_auc(x, y, n_splits=5, n_repeats=2)
    clf = LogisticRegression(max_iter=700, solver="lbfgs", class_weight="balanced")
    clf.fit(x, y)
    p_train = clf.predict_proba(x)[:, 1]
    train_auc = float(roc_auc_score(y, p_train))

    coef_df = pd.DataFrame(
        {
            "term": ["intercept"] + names,
            "coefficient": [float(clf.intercept_[0])] + [float(v) for v in clf.coef_[0]],
        }
    )
    coef_df.to_csv(PROJ_COEF_CSV, index=False)
    pd.DataFrame(
        [
            {
                "n_samples": int(len(y)),
                "n_presences": int((y == 1).sum()),
                "n_available": int((y == 0).sum()),
                "training_auc": train_auc,
                "cv_auc_mean": float(cv_mean) if np.isfinite(cv_mean) else np.nan,
                "cv_auc_sd": float(cv_sd) if np.isfinite(cv_sd) else np.nan,
                "cv_folds_total": int(cv_n),
                "n_predictors": int(len(names)),
            }
        ]
    ).to_csv(PROJ_METRICS_CSV, index=False)

    # Project to raster
    arrays = []
    for path in raster_paths:
        with rasterio.open(path) as src:
            arrays.append(src.read(1).astype(np.float32))
            profile = src.profile.copy()
    stack = np.stack(arrays, axis=-1)  # H W C
    h, w, c = stack.shape
    flat = stack.reshape(-1, c).astype(float)
    valid = np.all(np.isfinite(flat), axis=1)
    out = np.full(flat.shape[0], np.nan, dtype=np.float32)
    if valid.any():
        out[valid] = clf.predict_proba(flat[valid])[:, 1].astype(np.float32)
    out_r = out.reshape(h, w)
    profile.update(dtype=rasterio.float32, count=1, nodata=np.nan, compress="deflate")
    with rasterio.open(SUITABILITY_MAP, "w", **profile) as dst:
        dst.write(out_r, 1)
    return {"ok": True, "train_auc": train_auc, "cv_auc_mean": cv_mean, "cv_auc_sd": cv_sd, "cv_n": cv_n}


def main():
    if not os.path.exists(MOVEMENT_CSV):
        raise RuntimeError("Missing inputs/movement_points.csv. Upload movement data first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(MOVEMENT_CSV)
    required = {"animal_id", "timestamp", "latitude", "longitude"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"movement_points.csv is missing required columns: {', '.join(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["animal_id", "timestamp", "latitude", "longitude"]).copy()
    if df.empty:
        raise RuntimeError("No valid rows remain after parsing movement timestamps and coordinates.")

    used_steps = _prepare_steps(df)
    if len(used_steps) < 10:
        raise RuntimeError("Not enough movement steps for iSSA (need at least 10 valid steps).")
    samples = _build_used_available(used_steps, n_available=10, seed=42)
    samples.to_csv(SAMPLES_CSV, index=False)

    x_df = _step_feature_frame(samples)
    x = x_df.to_numpy(dtype=float)
    y = samples["used"].to_numpy(dtype=int)
    cv_mean, cv_sd, cv_n = _cv_auc(x, y, n_splits=5, n_repeats=2)
    model = LogisticRegression(max_iter=500, solver="lbfgs")
    model.fit(x, y)
    p = model.predict_proba(x)[:, 1]
    train_auc = float(roc_auc_score(y, p))
    coef_df = pd.DataFrame(
        {
            "term": ["intercept"] + list(x_df.columns),
            "coefficient": [float(model.intercept_[0])] + [float(v) for v in model.coef_[0]],
        }
    )
    coef_df.to_csv(COEF_CSV, index=False)

    proj = _fit_projection_model(samples, _list_predictor_rasters())
    summary_df = pd.DataFrame(
        [
            {
                "n_animals": int(df["animal_id"].nunique()),
                "n_used_steps": int((samples["used"] == 1).sum()),
                "n_available_steps": int((samples["used"] == 0).sum()),
                "used_available_ratio": f"{int((samples['used'] == 1).sum())}:{int((samples['used'] == 0).sum())}",
                "n_training_rows": int(len(samples)),
                "training_auc": train_auc,
                "cv_auc_mean": float(cv_mean) if np.isfinite(cv_mean) else np.nan,
                "cv_auc_sd": float(cv_sd) if np.isfinite(cv_sd) else np.nan,
                "cv_folds_total": int(cv_n),
                "projection_ok": bool(proj.get("ok", False)),
                "projection_train_auc": float(proj.get("train_auc")) if isinstance(proj.get("train_auc"), (float, int)) else np.nan,
                "projection_cv_auc_mean": float(proj.get("cv_auc_mean")) if isinstance(proj.get("cv_auc_mean"), (float, int, np.floating)) else np.nan,
                "projection_cv_auc_sd": float(proj.get("cv_auc_sd")) if isinstance(proj.get("cv_auc_sd"), (float, int, np.floating)) else np.nan,
            }
        ]
    )
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("iSSA completed successfully.")
    if proj.get("ok"):
        print("Projection raster created:", SUITABILITY_MAP)
    else:
        print("Projection skipped:", proj.get("message", "unknown reason"))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
