"""Candidate alignment diagnostics for TECH14.

These helpers intentionally use only already-computed candidate tables. They do
not run solvers and do not use truth when constructing any selector decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


GROUP_KEYS = ["dataset_type", "scenario", "position_init_label", "velocity_init_label", "beta_prior_profile"]
S7_SCENARIO = "S7_fast_north_no_bias"
VALIDATION_METRICS = [
    "raw_validation_rmse_mps",
    "trimmed_validation_rmse_mps",
    "inlier_validation_rmse_mps",
]
RISK_METRICS = ["condition_number", "position_cov_sqrt_trace_m", "position_bootstrap_spread_m"]
DIRECT_GIR_MODELS = {"M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale"}
BE_MODELS = {"M5_be_b0_only", "M6_be_full", "M8_robust_be_b0_only", "M9_robust_be_full"}
CORE_FINAL_MODELS = {
    "M0_static_position",
    "M1_static_position_bias",
    "M2_ctd_full",
    "M3_ctd_no_drift",
    "M4_ctd_no_bias",
    "M7_robust_ctd_full",
    "M12_ctd_full_plus_gir_refine",
    "M13_robust_ctd_full_plus_gir_refine",
}


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def bool_value(value: Any, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def model_family(model: str) -> str:
    model = str(model)
    if model in {"M0_static_position", "M1_static_position_bias"}:
        return "static family"
    if model == "M2_ctd_full":
        return "CTD full"
    if model == "M3_ctd_no_drift":
        return "no-drift"
    if model == "M4_ctd_no_bias":
        return "no-bias"
    if model in {"M7_robust_ctd_full", "M13_robust_ctd_full_plus_gir_refine"}:
        return "robust CTD"
    if model in BE_MODELS:
        return "BE branch"
    if model in {"M10_gir_tr_fixed_scale", "M11_gir_tr_mad_scale", "M12_ctd_full_plus_gir_refine", "M14_static_plus_gir_refine"}:
        return "GIR refine branch"
    return "other"


def validation_score(row: pd.Series | dict[str, Any]) -> float:
    raw = finite_float(row.get("raw_validation_rmse_mps"), np.nan)
    trimmed = finite_float(row.get("trimmed_validation_rmse_mps"), raw)
    inlier = finite_float(row.get("inlier_validation_rmse_mps"), raw)
    vals = [v for v in [raw, trimmed, inlier] if np.isfinite(v)]
    if not vals:
        return 1.0e12
    if not np.isfinite(raw):
        raw = max(vals)
    if not np.isfinite(trimmed):
        trimmed = raw
    if not np.isfinite(inlier):
        inlier = raw
    return float(0.20 * raw + 0.45 * trimmed + 0.35 * inlier)


def risk_score(row: pd.Series | dict[str, Any]) -> float:
    cond = finite_float(row.get("condition_number"), np.nan)
    cov = finite_float(row.get("position_cov_sqrt_trace_m"), np.nan)
    boot = finite_float(row.get("position_bootstrap_spread_m"), np.nan)
    cond_term = np.log10(cond) if np.isfinite(cond) and cond > 0 else 15.0
    cov_term = np.log10(cov + 1.0) if np.isfinite(cov) and cov >= 0 else 6.0
    boot_term = np.log10(boot + 1.0) if np.isfinite(boot) and boot >= 0 else 6.0
    return float(cond_term + cov_term + boot_term)


def spearman_rank_corr(x: pd.Series, y: pd.Series) -> float:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(x_num) & np.isfinite(y_num)
    if int(mask.sum()) < 3:
        return np.nan
    return float(x_num[mask].rank(method="average").corr(y_num[mask].rank(method="average")))


def selected_model_map(selected: pd.DataFrame, column: str) -> dict[tuple[Any, ...], str]:
    if selected.empty or column not in selected.columns:
        return {}
    out: dict[tuple[Any, ...], str] = {}
    for _, row in selected.iterrows():
        key = tuple(row.get(k) for k in GROUP_KEYS)
        out[key] = str(row.get(column, ""))
    return out


def best_by_metric(group: pd.DataFrame, metric: str) -> pd.Series:
    values = pd.to_numeric(group[metric], errors="coerce") if metric in group.columns else pd.Series(np.nan, index=group.index)
    idx = values.replace([np.inf, -np.inf], np.nan).idxmin()
    return group.loc[idx]


def best_risk_candidate(group: pd.DataFrame) -> pd.Series:
    tmp = group.copy()
    tmp["_risk_score"] = tmp.apply(risk_score, axis=1)
    return tmp.sort_values(["_risk_score", "candidate_model"]).iloc[0]


def v72_selected_from_v71(v71_selected: pd.DataFrame) -> pd.DataFrame:
    out = v71_selected.copy()
    rename = {
        "selected_model_v71": "selected_model_v72",
        "selection_reason_v71": "selection_reason_v72",
    }
    out = out.rename(columns=rename)
    if "selected_model_v72" not in out.columns:
        out["selected_model_v72"] = ""
    if "selection_reason_v72" not in out.columns:
        out["selection_reason_v72"] = ""
    out["selection_reason_v72"] = out["selection_reason_v72"].astype(str) + ";v7.2_no_new_selector_rule_s7_declared_limitation"
    return out


def compare_v71_v72(v71_selected: pd.DataFrame, v72_selected: pd.DataFrame) -> pd.DataFrame:
    merged = v71_selected.merge(v72_selected, on=GROUP_KEYS, how="inner", suffixes=("_v71", "_v72"))
    return pd.DataFrame(
        {
            "dataset_type": merged["dataset_type"],
            "scenario": merged["scenario"],
            "position_init_label": merged["position_init_label"],
            "velocity_init_label": merged["velocity_init_label"],
            "beta_prior_profile": merged["beta_prior_profile"],
            "selected_model_v71": merged.get("selected_model_v71", ""),
            "selected_model_v72": merged.get("selected_model_v72", ""),
            "v71_final_position_error_m": merged.get("final_position_error_m_v71", np.nan),
            "v72_final_position_error_m": merged.get("final_position_error_m_v72", np.nan),
            "improvement_m": pd.to_numeric(merged.get("final_position_error_m_v71", np.nan), errors="coerce")
            - pd.to_numeric(merged.get("final_position_error_m_v72", np.nan), errors="coerce"),
            "v71_trimmed_validation_rmse_mps": merged.get("trimmed_validation_rmse_mps_v71", np.nan),
            "v72_trimmed_validation_rmse_mps": merged.get("trimmed_validation_rmse_mps_v72", np.nan),
            "v72_selection_reason": merged.get("selection_reason_v72", ""),
        }
    )


def model_selection_counts_v72(selected_v72: pd.DataFrame) -> pd.DataFrame:
    counts = selected_v72.groupby(["dataset_type", "scenario", "selected_model_v72"]).size().reset_index(name="count")
    totals = counts.groupby(["dataset_type", "scenario"])["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals
    return counts
