import html
import base64
import json
import os
import re
import uuid
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from sklearn.neighbors import NearestNeighbors


FEATURE_WORKSPACE_CSV = os.path.join("inputs", "feature_workspace.csv")
FEATURE_WORKSPACE_META_JSON = os.path.join("inputs", "feature_workspace_meta.json")
MULTICOL_SUMMARY_CSV = os.path.join("outputs", "multicollinearity_workspace_summary.csv")
MULTICOL_PAIRS_CSV = os.path.join("outputs", "multicollinearity_workspace_pairs.csv")
MULTICOL_PRUNE_CSV = os.path.join("outputs", "multicollinearity_workspace_prune_candidates.csv")
MULTICOL_MATRIX_CSV = os.path.join("outputs", "multicollinearity_workspace_correlation_matrix.csv")
PRUNED_VARIABLES_REPORT_CSV = os.path.join("outputs", "pruned_variables_report.csv")
MORAN_CSV = os.path.join("outputs", "spatial_independence_morans_i.csv")
PREMODEL_DIAG_JSON = os.path.join("outputs", "pre_model_diagnostics.json")
EFFECT_PLOTS_MANIFEST_CSV = os.path.join("outputs", "predictor_effect_plots_manifest.csv")


def _norm_name(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text or "").strip().lower()).strip("_")


def _is_meta_col(col: str) -> bool:
    return _norm_name(col) in {"point_id", "latitude", "longitude", "label"}


def _ensure_dirs() -> None:
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


def _load_json(path: str, default: Any) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _list_raster_paths(raster_dir: str) -> list[str]:
    if not os.path.isdir(raster_dir):
        return []
    return sorted(
        os.path.join(raster_dir, fn)
        for fn in os.listdir(raster_dir)
        if fn.lower().endswith(".tif")
    )


def _sample_raster_at_points(raster_path: str, points: pd.DataFrame) -> np.ndarray:
    vals = np.full(len(points), np.nan, dtype=float)
    if points.empty:
        return vals
    with rasterio.open(raster_path) as src:
        lats = pd.to_numeric(points["latitude"], errors="coerce").to_numpy(dtype=float)
        lons = pd.to_numeric(points["longitude"], errors="coerce").to_numpy(dtype=float)
        valid_idx = [i for i, (lat, lon) in enumerate(zip(lats, lons)) if np.isfinite(lat) and np.isfinite(lon)]
        if not valid_idx:
            return vals
        coords = [(float(lons[i]), float(lats[i])) for i in valid_idx]
        nodata = src.nodata
        for idx, sample in zip(valid_idx, src.sample(coords, indexes=1, masked=True)):
            try:
                v = sample.item() if hasattr(sample, "item") else sample[0]
            except Exception:
                continue
            if np.ma.is_masked(v):
                continue
            if nodata is not None and np.isfinite(nodata) and float(v) == float(nodata):
                continue
            vals[idx] = float(v) if np.isfinite(v) else np.nan
    return vals


def _default_meta(all_predictors: list[str]) -> dict:
    return {
        "active_predictors": sorted(all_predictors),
        "inactive_predictors": [],
        "last_multicollinearity": {},
        "last_spatial_independence": {},
    }


def _load_meta(all_predictors: list[str]) -> dict:
    meta = _load_json(FEATURE_WORKSPACE_META_JSON, _default_meta(all_predictors))
    active = [p for p in meta.get("active_predictors", []) if p in all_predictors]
    inactive = [p for p in meta.get("inactive_predictors", []) if p in all_predictors]
    missing = [p for p in all_predictors if p not in active and p not in inactive]
    active.extend(missing)
    meta["active_predictors"] = sorted(dict.fromkeys(active))
    meta["inactive_predictors"] = sorted(dict.fromkeys(inactive))
    return meta


def _save_meta(meta: dict) -> None:
    _ensure_dirs()
    _write_json(FEATURE_WORKSPACE_META_JSON, meta)


def load_workspace_df() -> pd.DataFrame:
    if os.path.exists(FEATURE_WORKSPACE_CSV):
        try:
            return pd.read_csv(FEATURE_WORKSPACE_CSV)
        except Exception:
            pass
    return pd.DataFrame(columns=["point_id", "latitude", "longitude", "label"])


def get_workspace_predictors(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not _is_meta_col(c)]


def get_active_predictors() -> list[str]:
    df = load_workspace_df()
    preds = get_workspace_predictors(df)
    meta = _load_meta(preds)
    return [p for p in meta.get("active_predictors", []) if p in preds]


def refresh_workspace_from_rasters(
    points_csv: str = os.path.join("inputs", "presence_points.csv"),
    raster_dir: str = os.path.join("predictor_rasters", "wgs84"),
) -> dict:
    _ensure_dirs()
    if not os.path.exists(points_csv):
        return {"ok": False, "message": "presence_points.csv not found.", "rows": 0, "new_predictors": []}
    points = pd.read_csv(points_csv).copy()
    if not {"latitude", "longitude"}.issubset(points.columns):
        return {"ok": False, "message": "presence_points.csv missing latitude/longitude.", "rows": 0, "new_predictors": []}

    points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
    points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
    points = points.dropna(subset=["latitude", "longitude"]).copy()
    points = points.reset_index(drop=True)
    points.insert(0, "point_id", np.arange(len(points), dtype=int))
    if "label" not in points.columns:
        points["label"] = 1
    base = points[["point_id", "latitude", "longitude", "label"]].copy()

    existing = load_workspace_df()
    if not existing.empty:
        same_points = (
            len(existing) == len(base)
            and np.allclose(
                pd.to_numeric(existing["latitude"], errors="coerce").to_numpy(dtype=float),
                base["latitude"].to_numpy(dtype=float),
                equal_nan=True,
            )
            and np.allclose(
                pd.to_numeric(existing["longitude"], errors="coerce").to_numpy(dtype=float),
                base["longitude"].to_numpy(dtype=float),
                equal_nan=True,
            )
        )
        if same_points:
            keep_cols = [c for c in existing.columns if _is_meta_col(c)]
            if "point_id" not in keep_cols:
                keep_cols = ["point_id", "latitude", "longitude", "label"]
            base = existing[keep_cols].copy()

    raster_paths = _list_raster_paths(raster_dir)
    if not raster_paths:
        base.to_csv(FEATURE_WORKSPACE_CSV, index=False)
        meta = _load_meta([])
        _save_meta(meta)
        return {"ok": True, "message": "No predictor rasters found yet.", "rows": len(base), "new_predictors": []}

    new_predictors = []
    for p in raster_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        if name in base.columns:
            continue
        base[name] = _sample_raster_at_points(p, base)
        new_predictors.append(name)

    ordered = ["point_id", "latitude", "longitude", "label"] + sorted([c for c in base.columns if not _is_meta_col(c)])
    base = base[ordered].copy()
    base.to_csv(FEATURE_WORKSPACE_CSV, index=False)

    predictors = get_workspace_predictors(base)
    meta = _load_meta(predictors)
    _save_meta(meta)
    return {
        "ok": True,
        "message": f"Workspace updated with {len(predictors)} predictor columns.",
        "rows": len(base),
        "new_predictors": new_predictors,
        "all_predictors": predictors,
    }


def set_predictor_active_state(names: list[str], active: bool) -> dict:
    df = load_workspace_df()
    predictors = set(get_workspace_predictors(df))
    requested = [n for n in (_norm_name(x) for x in names) if n]
    matched = sorted([p for p in predictors if _norm_name(p) in set(requested) or p in set(names)])
    if not matched:
        return {"ok": False, "matched": [], "message": "No matching predictors found in workspace."}

    meta = _load_meta(sorted(predictors))
    active_set = set(meta.get("active_predictors", []))
    inactive_set = set(meta.get("inactive_predictors", []))
    for m in matched:
        if active:
            active_set.add(m)
            inactive_set.discard(m)
        else:
            active_set.discard(m)
            inactive_set.add(m)
    meta["active_predictors"] = sorted(active_set)
    meta["inactive_predictors"] = sorted(inactive_set)
    _save_meta(meta)
    return {
        "ok": True,
        "matched": matched,
        "active_predictors": meta["active_predictors"],
        "inactive_predictors": meta["inactive_predictors"],
    }


def _vif_for_matrix(x: np.ndarray) -> np.ndarray:
    if x.shape[1] <= 1 or x.shape[0] <= x.shape[1]:
        return np.full(x.shape[1], np.nan, dtype=float)
    out = np.full(x.shape[1], np.nan, dtype=float)
    for j in range(x.shape[1]):
        yj = x[:, j]
        others = np.delete(x, j, axis=1)
        design = np.column_stack([np.ones(others.shape[0]), others])
        try:
            beta = np.linalg.lstsq(design, yj, rcond=None)[0]
            pred = design @ beta
            ss_res = float(np.sum((yj - pred) ** 2))
            ss_tot = float(np.sum((yj - np.mean(yj)) ** 2))
            if ss_tot > 1e-12:
                r2 = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))
                out[j] = np.inf if (1.0 - r2) <= 1e-12 else (1.0 / (1.0 - r2))
        except Exception:
            out[j] = np.nan
    return out


def run_multicollinearity_check(corr_threshold: float = 0.7, vif_threshold: float = 5.0) -> dict:
    _ensure_dirs()
    df = load_workspace_df()
    predictors = get_workspace_predictors(df)
    report_cols = ["predictor", "reason", "details", "metric", "threshold", "correlated_with", "iteration", "auto_pruned"]

    def _write_empty_outputs() -> None:
        pd.DataFrame(columns=["predictor", "max_abs_correlation", "max_correlated_predictor", "vif", "flag"]).to_csv(
            MULTICOL_SUMMARY_CSV, index=False
        )
        pd.DataFrame(columns=["predictor_1", "predictor_2", "correlation", "abs_correlation"]).to_csv(
            MULTICOL_PAIRS_CSV, index=False
        )
        pd.DataFrame(columns=["predictor", "reason", "metric", "threshold", "iteration", "correlated_with"]).to_csv(
            MULTICOL_PRUNE_CSV, index=False
        )
        pd.DataFrame(columns=["predictor"]).to_csv(MULTICOL_MATRIX_CSV, index=False)

    if not predictors:
        _write_empty_outputs()
        pd.DataFrame(columns=report_cols).to_csv(PRUNED_VARIABLES_REPORT_CSV, index=False)
        return {"ok": False, "message": "No predictors in workspace.", "suggested_drop": []}

    meta = _load_meta(predictors)
    active = [p for p in meta.get("active_predictors", []) if p in predictors]
    if not active:
        return {"ok": False, "message": "No active predictors selected.", "suggested_drop": []}

    x_all = df[active].apply(pd.to_numeric, errors="coerce")
    low_info_rows = []
    auto_prune = []
    active_for_multicol = active.copy()

    x = x_all[active_for_multicol]
    keep_rows = np.all(np.isfinite(x.to_numpy(dtype=float)), axis=1)
    x = x.loc[keep_rows].copy()
    if x.empty:
        _write_empty_outputs()
        report_df = pd.DataFrame(low_info_rows, columns=report_cols)
        report_df.to_csv(PRUNED_VARIABLES_REPORT_CSV, index=False)
        return {"ok": False, "message": "No finite rows for active predictors.", "suggested_drop": [], "auto_prune": auto_prune}

    corr = x.corr(method="spearman")
    corr_out = corr.copy()
    corr_out.insert(0, "predictor", corr_out.index.astype(str))
    corr_out.to_csv(MULTICOL_MATRIX_CSV, index=False)
    pair_rows = []
    for i, a in enumerate(active_for_multicol):
        for j in range(i + 1, len(active_for_multicol)):
            b = active_for_multicol[j]
            c = corr.loc[a, b]
            if pd.notna(c) and abs(float(c)) >= corr_threshold:
                pair_rows.append((a, b, float(c), float(abs(c))))
    pd.DataFrame(pair_rows, columns=["predictor_1", "predictor_2", "correlation", "abs_correlation"]).to_csv(
        MULTICOL_PAIRS_CSV, index=False
    )

    # Summary table
    vif_vals = _vif_for_matrix(x.to_numpy(dtype=float))
    summary_rows = []
    for i, p in enumerate(active_for_multicol):
        partners = corr.loc[p].abs().drop(index=p, errors="ignore")
        max_corr = float(partners.max()) if len(partners) else np.nan
        partner = ""
        if len(partners) and pd.notna(partners.max()):
            partner = str(partners.idxmax())
        flags = []
        if pd.notna(max_corr) and float(max_corr) >= corr_threshold:
            flags.append("high_pairwise_correlation")
        v = float(vif_vals[i]) if np.isfinite(vif_vals[i]) else np.nan
        if np.isfinite(v) and v >= vif_threshold:
            flags.append("high_vif")
        summary_rows.append((p, max_corr, partner, v, "|".join(flags) if flags else "ok"))
    pd.DataFrame(summary_rows, columns=["predictor", "max_abs_correlation", "max_correlated_predictor", "vif", "flag"]).to_csv(
        MULTICOL_SUMMARY_CSV, index=False
    )

    # Iterative prune suggestion
    work = active_for_multicol.copy()
    prune_rows = []
    iter_no = 0
    while len(work) >= 2:
        iter_no += 1
        xw = x[work].to_numpy(dtype=float)
        cw = pd.DataFrame(xw, columns=work).corr(method="spearman").abs()
        np.fill_diagonal(cw.values, 0.0)
        max_corr = float(np.nanmax(cw.values)) if cw.size else 0.0
        vif_w = _vif_for_matrix(xw)
        max_vif = float(np.nanmax(vif_w)) if len(vif_w) else np.nan
        if max_corr < corr_threshold and (not np.isfinite(max_vif) or max_vif < vif_threshold):
            break

        drop_candidate = None
        reason = "high_vif"
        metric = max_vif
        threshold = vif_threshold
        correlated_with = ""
        if np.isfinite(max_vif) and max_vif >= vif_threshold:
            drop_candidate = work[int(np.nanargmax(vif_w))]
        else:
            reason = "high_pairwise_correlation"
            metric = max_corr
            threshold = corr_threshold
            ij = np.unravel_index(np.nanargmax(cw.values), cw.shape)
            p1, p2 = work[int(ij[0])], work[int(ij[1])]
            m1 = float(cw[p1].mean())
            m2 = float(cw[p2].mean())
            drop_candidate = p1 if m1 >= m2 else p2
            correlated_with = p2 if drop_candidate == p1 else p1

        prune_rows.append((drop_candidate, reason, float(metric), float(threshold), int(iter_no), correlated_with))
        work = [w for w in work if w != drop_candidate]
        if len(work) < 2:
            break

    prune_df = pd.DataFrame(prune_rows, columns=["predictor", "reason", "metric", "threshold", "iteration", "correlated_with"])
    prune_df.to_csv(MULTICOL_PRUNE_CSV, index=False)
    suggested = [str(v) for v in prune_df["predictor"].dropna().tolist()] if not prune_df.empty else []

    report_rows = list(low_info_rows)
    for _, row in prune_df.iterrows():
        pred = str(row.get("predictor", "") or "")
        reason = str(row.get("reason", "") or "")
        metric = row.get("metric", np.nan)
        threshold = row.get("threshold", np.nan)
        partner = str(row.get("correlated_with", "") or "")
        if reason == "high_pairwise_correlation":
            details = f"High pairwise correlation with {partner or 'another predictor'} (|rho|={float(metric):.3f} >= {float(threshold):.3f})."
        else:
            details = f"High VIF ({float(metric):.3f} >= {float(threshold):.3f})."
        report_rows.append(
            {
                "predictor": pred,
                "reason": reason,
                "details": details,
                "metric": metric,
                "threshold": threshold,
                "correlated_with": partner,
                "iteration": int(row.get("iteration", 0) or 0),
                "auto_pruned": 0,
            }
        )
    pd.DataFrame(report_rows, columns=report_cols).to_csv(PRUNED_VARIABLES_REPORT_CSV, index=False)

    meta["last_multicollinearity"] = {
        "corr_threshold": float(corr_threshold),
        "vif_threshold": float(vif_threshold),
        "suggested_drop": suggested,
        "auto_prune": auto_prune,
        "n_active_predictors": len(active),
    }
    _save_meta(meta)
    msg = "Multicollinearity diagnostics completed."
    if auto_prune:
        msg += f" Auto-pruned low-information predictors: {', '.join(auto_prune)}."
    return {
        "ok": True,
        "message": msg,
        "active_predictors": active,
        "suggested_drop": suggested,
        "auto_prune": auto_prune,
        "corr_threshold": float(corr_threshold),
        "vif_threshold": float(vif_threshold),
        "summary_csv": MULTICOL_SUMMARY_CSV,
        "matrix_csv": MULTICOL_MATRIX_CSV,
        "pairs_csv": MULTICOL_PAIRS_CSV,
        "prune_csv": MULTICOL_PRUNE_CSV,
        "prune_report_csv": PRUNED_VARIABLES_REPORT_CSV,
    }


def run_morans_i_check(k_neighbors: int = 8, permutations: int = 199) -> dict:
    _ensure_dirs()
    df = load_workspace_df()
    predictors = get_workspace_predictors(df)
    if not predictors:
        pd.DataFrame(columns=["predictor", "n_points", "morans_i", "p_value", "z_score", "is_significant"]).to_csv(
            MORAN_CSV, index=False
        )
        return {"ok": False, "message": "No predictors in workspace.", "csv": MORAN_CSV}

    meta = _load_meta(predictors)
    active = [p for p in meta.get("active_predictors", []) if p in predictors]
    if not active:
        return {"ok": False, "message": "No active predictors selected.", "csv": MORAN_CSV}

    coords = df[["longitude", "latitude"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid_xy = np.all(np.isfinite(coords), axis=1)
    coords = coords[valid_xy]
    if len(coords) < max(6, k_neighbors + 1):
        return {"ok": False, "message": "Not enough points for Moran's I.", "csv": MORAN_CSV}

    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(coords)), metric="euclidean")
    nn.fit(coords)
    neigh_idx = nn.kneighbors(coords, return_distance=False)[:, 1:]
    w = np.zeros((len(coords), len(coords)), dtype=float)
    for i in range(len(coords)):
        w[i, neigh_idx[i]] = 1.0
    w = np.maximum(w, w.T)
    s0 = float(w.sum())
    if s0 <= 0:
        return {"ok": False, "message": "Neighbor graph is empty for Moran's I.", "csv": MORAN_CSV}

    rows = []
    rng = np.random.default_rng(42)
    for pred in active:
        vals = pd.to_numeric(df[pred], errors="coerce").to_numpy(dtype=float)[valid_xy]
        ok = np.isfinite(vals)
        if ok.sum() < max(6, k_neighbors + 1):
            continue
        x = vals[ok]
        idx = np.flatnonzero(ok)
        w_sub = w[np.ix_(idx, idx)]
        s0_sub = float(w_sub.sum())
        if s0_sub <= 0:
            continue
        x_center = x - x.mean()
        denom = float(np.dot(x_center, x_center))
        if denom <= 1e-12:
            continue
        num = float(x_center @ w_sub @ x_center)
        i_obs = (len(x) / s0_sub) * (num / denom)

        perms = []
        for _ in range(max(19, int(permutations))):
            xp = rng.permutation(x_center)
            perms.append((len(x) / s0_sub) * float((xp @ w_sub @ xp) / denom))
        perms_arr = np.asarray(perms, dtype=float)
        mu = float(np.mean(perms_arr))
        sd = float(np.std(perms_arr))
        z = (i_obs - mu) / sd if sd > 1e-12 else np.nan
        p = (float((np.abs(perms_arr) >= abs(i_obs)).sum()) + 1.0) / (len(perms_arr) + 1.0)
        rows.append((pred, int(len(x)), float(i_obs), float(p), float(z) if np.isfinite(z) else np.nan, bool(p < 0.05)))

    out = pd.DataFrame(rows, columns=["predictor", "n_points", "morans_i", "p_value", "z_score", "is_significant"])
    out.to_csv(MORAN_CSV, index=False)
    significant = out.loc[out["is_significant"] == True, "predictor"].tolist() if not out.empty else []
    meta["last_spatial_independence"] = {
        "k_neighbors": int(k_neighbors),
        "permutations": int(permutations),
        "n_significant": int(len(significant)),
        "significant_predictors": significant,
    }
    _save_meta(meta)
    return {
        "ok": True,
        "csv": MORAN_CSV,
        "n_tested": int(len(out)),
        "n_significant": int(len(significant)),
        "significant_predictors": significant,
    }


def run_pre_model_diagnostics(corr_threshold: float = 0.7, vif_threshold: float = 5.0) -> dict:
    multicol = run_multicollinearity_check(corr_threshold=corr_threshold, vif_threshold=vif_threshold)
    moran = run_morans_i_check()
    payload = {
        "multicollinearity": multicol,
        "spatial_independence": moran,
    }
    _ensure_dirs()
    _write_json(PREMODEL_DIAG_JSON, payload)
    return payload


def _to_table_html(df: pd.DataFrame, max_rows: int = 20, max_cols: int = 12) -> str:
    if df is None or df.empty:
        return "<div class='spatchat-values-empty'>No rows available.</div>"
    view = df.head(max_rows).copy()
    cols = list(view.columns)[:max_cols]
    rows_html = []
    for _, row in view.iterrows():
        tds = "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in cols)
        rows_html.append(f"<tr>{tds}</tr>")
    ths = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    note = ""
    if len(df) > len(view):
        note = f"<div class='spatchat-values-note'>Showing first {len(view)} of {len(df)} rows.</div>"
    return (
        "<div class='spatchat-values-wrap'>"
        + note
        + "<table class='spatchat-values-table'><thead><tr>"
        + ths
        + "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>"
    )


def _to_image_html(path: str, alt: str = "") -> str:
    try:
        with open(path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("ascii")
        return (
            "<div class='spatchat-values-wrap'>"
            "<div style='font-size:11px;color:rgba(226,233,245,0.72);margin-bottom:8px;'>"
            "Marginal predictor-response curve after model fitting"
            "</div>"
            f"<img alt='{html.escape(alt)}' src='data:image/png;base64,{b64}' "
            "style='display:block;width:100%;height:auto;border-radius:10px;border:1px solid rgba(255,255,255,0.12);background:#0d121d;' />"
            "</div>"
        )
    except Exception:
        return "<div class='spatchat-values-empty'>Effect plot image unavailable.</div>"


def _effect_plot_sections() -> list[tuple[str, str, str]]:
    sections = []
    seen = set()

    def add_plot(pred: str, plot_fp: str, ci_available: bool | None = None) -> None:
        if not plot_fp:
            return
        norm_fp = os.path.normpath(plot_fp)
        if norm_fp in seen or not os.path.exists(norm_fp):
            return
        seen.add(norm_fp)
        label = str(pred or "").strip() or os.path.splitext(os.path.basename(norm_fp))[0]
        if ci_available is None:
            subtitle = "Predictor-response curve after model fitting"
        else:
            subtitle = "95% CI shown" if ci_available else "CI unavailable"
        sections.append((f"Predictor Effect: {label}", subtitle, _to_image_html(norm_fp, alt=label)))

    if os.path.exists(EFFECT_PLOTS_MANIFEST_CSV):
        try:
            pdf = pd.read_csv(EFFECT_PLOTS_MANIFEST_CSV)
            if not pdf.empty:
                for _, row in pdf.iterrows():
                    pred = str(row.get("predictor", "") or "").strip()
                    rel = str(row.get("plot_path", "") or "").strip()
                    if not rel:
                        continue
                    rel = rel.replace("\\", os.sep).replace("/", os.sep)
                    plot_fp = rel if os.path.isabs(rel) else os.path.join(".", rel)
                    try:
                        ci_available = bool(int(row.get("ci_available", 0) or 0))
                    except Exception:
                        ci_available = None
                    add_plot(pred, plot_fp, ci_available)
        except Exception:
            pass

    plot_dir = os.path.join("outputs", "predictor_effect_plots")
    if os.path.isdir(plot_dir):
        for fn in sorted(os.listdir(plot_dir)):
            if not fn.lower().endswith(".png"):
                continue
            pred = re.sub(r"_effect$", "", os.path.splitext(fn)[0])
            add_plot(pred, os.path.join(plot_dir, fn), None)

    return sections


def render_feature_popup_html(auto_open: bool = False) -> str:
    df = load_workspace_df()
    meta = _load_meta(get_workspace_predictors(df))
    active = meta.get("active_predictors", [])
    inactive = meta.get("inactive_predictors", [])
    table_html = _to_table_html(df, max_rows=24, max_cols=14)
    sections = [("Extracted Predictor Values", f"Rows={len(df)} Active={len(active)} Inactive={len(inactive)}", table_html)]
    sections.extend(_effect_plot_sections())
    model_tables = [
        ("Cross-validated Performance", os.path.join("outputs", "performance_metrics_cv.csv")),
        ("Fitted Performance", os.path.join("outputs", "performance_metrics_fitted.csv")),
        ("Model Coefficients", os.path.join("outputs", "coefficients.csv")),
        ("Dropped Predictors", os.path.join("outputs", "dropped_predictors.csv")),
        ("Standardization Stats", os.path.join("outputs", "standardization_stats.csv")),
        ("iSSA Summary", os.path.join("outputs", "issa_summary.csv")),
        ("iSSA Step Coefficients", os.path.join("outputs", "issa_coefficients.csv")),
        ("iSSA Projection Metrics", os.path.join("outputs", "issa_projection_metrics.csv")),
        ("iSSA Projection Coefficients", os.path.join("outputs", "issa_projection_coefficients.csv")),
        ("Multicollinearity Summary", MULTICOL_SUMMARY_CSV),
        ("Multicollinearity Correlation Matrix (Spearman)", MULTICOL_MATRIX_CSV),
        ("Multicollinearity Pairwise Flags", MULTICOL_PAIRS_CSV),
        ("Multicollinearity Prune Candidates", MULTICOL_PRUNE_CSV),
        ("Pruned Variables Report", PRUNED_VARIABLES_REPORT_CSV),
        ("Spatial Independence (Moran's I)", MORAN_CSV),
    ]
    for title, fp in model_tables:
        if os.path.exists(fp):
            try:
                tdf = pd.read_csv(fp)
                if not tdf.empty:
                    sections.append((title, os.path.basename(fp), _to_table_html(tdf, max_rows=40, max_cols=12)))
            except Exception:
                continue
    figures = [{"id": f"values-{idx}", "title": title, "subtitle": subtitle, "tableHtml": body} for idx, (title, subtitle, body) in enumerate(sections)]
    payload = html.escape(
        json.dumps(
            {
                "figures": figures,
                "activeIndex": 0,
                "isOpen": bool(auto_open),
                "isMinimized": False,
                "width": 560,
                "height": 460,
                "x": None,
                "y": None,
                "renderNonce": uuid.uuid4().hex,
            }
        ),
        quote=True,
    )
    back_disabled = "disabled" if len(figures) <= 1 else ""
    forward_disabled = "disabled" if len(figures) <= 1 else ""
    return f"""
<div class="spatchat-values-root" data-payload="{payload}">
  <button class="spatchat-values-launcher" type="button" data-action="open" aria-label="Open plots and tables" onclick="return window.spatchatValuesViewer ? window.spatchatValuesViewer.handleAction(this, event) : false;">
    Plots/Tables ({len(figures)})
  </button>
  <div class="spatchat-values-modal-backdrop"></div>
  <div class="spatchat-values-interaction-shield"></div>
  <section class="spatchat-values-modal is-hidden" role="dialog" aria-modal="true" aria-label="Plots and tables viewer">
    <header class="spatchat-values-modal-head" onmousedown="return window.spatchatValuesViewer ? window.spatchatValuesViewer.handleDrag(this, event) : false;">
      <div class="spatchat-values-modal-copy">
        <div class="spatchat-values-modal-title">Plots/Tables</div>
        <div class="spatchat-values-modal-count">1 / {len(figures)}</div>
      </div>
      <div class="spatchat-values-modal-nav">
        <button class="spatchat-values-modal-btn" type="button" data-action="back" aria-label="Previous table" onclick="return window.spatchatValuesViewer ? window.spatchatValuesViewer.handleAction(this, event) : false;" {back_disabled}>&larr;</button>
        <button class="spatchat-values-modal-btn" type="button" data-action="forward" aria-label="Next table" onclick="return window.spatchatValuesViewer ? window.spatchatValuesViewer.handleAction(this, event) : false;" {forward_disabled}>&rarr;</button>
        <button class="spatchat-values-modal-btn spatchat-values-modal-btn-close" type="button" data-action="close" aria-label="Minimize extracted values viewer" onclick="return window.spatchatValuesViewer ? window.spatchatValuesViewer.handleAction(this, event) : false;">_</button>
      </div>
    </header>
    <div class="spatchat-values-modal-body">
      <article class="spatchat-values-card">
        <div class="spatchat-values-card-title">{html.escape(figures[0]['title'])}</div>
        <div class="spatchat-values-card-meta">{html.escape(figures[0]['subtitle'])}</div>
        <div class="spatchat-values-card-table">{figures[0]['tableHtml']}</div>
      </article>
    </div>
    <div class="spatchat-values-resize-handle is-n" data-resize="n" onmousedown="return window.spatchatValuesViewer ? window.spatchatValuesViewer.handleResize(this, event) : false;"></div>
    <div class="spatchat-values-resize-handle is-e" data-resize="e" onmousedown="return window.spatchatValuesViewer ? window.spatchatValuesViewer.handleResize(this, event) : false;"></div>
    <div class="spatchat-values-resize-handle is-s" data-resize="s" onmousedown="return window.spatchatValuesViewer ? window.spatchatValuesViewer.handleResize(this, event) : false;"></div>
    <div class="spatchat-values-resize-handle is-w" data-resize="w" onmousedown="return window.spatchatValuesViewer ? window.spatchatValuesViewer.handleResize(this, event) : false;"></div>
    <div class="spatchat-values-resize-handle is-ne" data-resize="ne" onmousedown="return window.spatchatValuesViewer ? window.spatchatValuesViewer.handleResize(this, event) : false;"></div>
    <div class="spatchat-values-resize-handle is-nw" data-resize="nw" onmousedown="return window.spatchatValuesViewer ? window.spatchatValuesViewer.handleResize(this, event) : false;"></div>
    <div class="spatchat-values-resize-handle is-se" data-resize="se" title="Resize extracted values viewer" onmousedown="return window.spatchatValuesViewer ? window.spatchatValuesViewer.handleResize(this, event) : false;"></div>
    <div class="spatchat-values-resize-handle is-sw" data-resize="sw" onmousedown="return window.spatchatValuesViewer ? window.spatchatValuesViewer.handleResize(this, event) : false;"></div>
  </section>
</div>
"""
