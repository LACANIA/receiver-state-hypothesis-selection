"""Family-level validation evidence helpers for MA-BGTR-v7."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .risk_veto import finite_float, model_dof


STATIC_MODELS = {"M0_static_position", "M1_static_position_bias"}
NO_BIAS_MODELS = {"M4_ctd_no_bias"}
NO_DRIFT_MODELS = {"M3_ctd_no_drift"}
FULL_DRIFT_MODELS = {"M2_ctd_full", "M7_robust_ctd_full", "M12_ctd_full_plus_gir_refine", "M13_robust_ctd_full_plus_gir_refine"}
BIAS_CAPABLE_MODELS = {"M2_ctd_full", "M3_ctd_no_drift", "M7_robust_ctd_full", "M12_ctd_full_plus_gir_refine", "M13_robust_ctd_full_plus_gir_refine"}
ROBUST_MODELS = {"M7_robust_ctd_full", "M8_robust_be_b0_only", "M9_robust_be_full", "M13_robust_ctd_full_plus_gir_refine"}
REFINE_MODELS = {"M12_ctd_full_plus_gir_refine", "M13_robust_ctd_full_plus_gir_refine", "M14_static_plus_gir_refine"}

# The final hierarchical selector treats BE and direct GIR branches as
# diagnostics unless no core candidate survives. This keeps the model pool
# unchanged while preventing projection-only branches from steering hard cases.
CORE_DYNAMIC_MODELS = {"M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias", "M7_robust_ctd_full", "M12_ctd_full_plus_gir_refine", "M13_robust_ctd_full_plus_gir_refine"}
CORE_FINAL_MODELS = STATIC_MODELS | CORE_DYNAMIC_MODELS

MAIN_METRICS = ["raw_validation_rmse_mps", "trimmed_validation_rmse_mps", "inlier_validation_rmse_mps"]


def validation_values(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    return {name: finite_float(row.get(name), np.nan) for name in MAIN_METRICS}


def validation_score(row: pd.Series | dict[str, Any]) -> float:
    """Small scalar used only after gates, not as a global risk-weighted score."""
    vals = validation_values(row)
    raw = vals["raw_validation_rmse_mps"]
    trimmed = vals["trimmed_validation_rmse_mps"]
    inlier = vals["inlier_validation_rmse_mps"]
    if not np.isfinite(trimmed):
        trimmed = raw
    if not np.isfinite(inlier):
        inlier = raw
    if not np.isfinite(raw):
        raw = max(trimmed, inlier)
    if not np.isfinite(raw):
        return 1.0e9
    return float(0.45 * trimmed + 0.35 * inlier + 0.20 * raw)


def relative_gain(reference: float, candidate: float) -> float:
    if not (np.isfinite(reference) and np.isfinite(candidate)) or abs(reference) < 1.0e-12:
        return 0.0
    return float((reference - candidate) / abs(reference))


def metric_gains(reference: pd.Series | dict[str, Any], candidate: pd.Series | dict[str, Any]) -> dict[str, float]:
    ref = validation_values(reference)
    cand = validation_values(candidate)
    return {name: relative_gain(ref[name], cand[name]) for name in MAIN_METRICS}


def no_worse(candidate: pd.Series | dict[str, Any], reference: pd.Series | dict[str, Any], eps: float = 0.01) -> bool:
    cand = validation_values(candidate)
    ref = validation_values(reference)
    for name in MAIN_METRICS:
        if np.isfinite(cand[name]) and np.isfinite(ref[name]) and cand[name] > ref[name] * (1.0 + eps):
            return False
    return True


def clearly_better(candidate: pd.Series | dict[str, Any], reference: pd.Series | dict[str, Any], min_gain: float = 0.02, eps: float = 0.01) -> bool:
    if not no_worse(candidate, reference, eps):
        return False
    gains = metric_gains(reference, candidate)
    return bool(max(gains.values()) >= min_gain)


def all_metrics_support(candidate: pd.Series | dict[str, Any], reference: pd.Series | dict[str, Any], min_gain: float = 0.0) -> bool:
    gains = metric_gains(reference, candidate)
    return all(gain > min_gain for gain in gains.values())


def best_candidate(group: pd.DataFrame, models: set[str] | None = None) -> pd.Series | None:
    if group.empty:
        return None
    subset = group.copy()
    if models is not None:
        subset = subset[subset["candidate_model"].isin(models)].copy()
    if subset.empty:
        return None
    subset["_validation_score"] = subset.apply(validation_score, axis=1)
    subset["_risk_condition"] = pd.to_numeric(subset.get("condition_number", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0e15)
    subset["_dof"] = subset.apply(lambda row: model_dof(str(row.get("candidate_model")), row), axis=1)
    subset = subset.sort_values(["_validation_score", "_risk_condition", "_dof", "candidate_model"])
    return subset.iloc[0]


def near_tie_group(group: pd.DataFrame, reference: pd.Series, eps: float = 0.01) -> pd.DataFrame:
    keep = []
    ref_vals = validation_values(reference)
    for _, row in group.iterrows():
        vals = validation_values(row)
        keep.append(all(np.isfinite(vals[m]) and np.isfinite(ref_vals[m]) and vals[m] <= ref_vals[m] * (1.0 + eps) for m in MAIN_METRICS))
    out = group.loc[keep].copy()
    return out if not out.empty else group.loc[[reference.name]].copy()


def choose_near_tie_by_risk(group: pd.DataFrame, eps: float = 0.01) -> pd.Series:
    best = best_candidate(group)
    if best is None:
        raise ValueError("empty near-tie group")
    near = near_tie_group(group, best, eps)
    if len(near) == 1:
        return near.iloc[0]
    near["_cov"] = pd.to_numeric(near.get("position_cov_sqrt_trace_m", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0e9)
    near["_boot"] = pd.to_numeric(near.get("position_bootstrap_spread_m", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0e9)
    near["_cond"] = pd.to_numeric(near.get("condition_number", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0e15)
    near["_dof"] = near.apply(lambda row: model_dof(str(row.get("candidate_model")), row), axis=1)
    near = near.sort_values(["_cov", "_boot", "_cond", "_dof", "candidate_model"])
    return near.iloc[0]
