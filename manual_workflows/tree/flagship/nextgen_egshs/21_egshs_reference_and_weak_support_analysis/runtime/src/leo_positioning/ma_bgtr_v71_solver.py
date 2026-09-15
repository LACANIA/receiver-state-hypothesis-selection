"""MA-BGTR-v7.1 gate bugfix selector."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .bias_gate_fix import evaluate_bias_gate_v71
from .family_evidence import (
    CORE_DYNAMIC_MODELS,
    CORE_FINAL_MODELS,
    FULL_DRIFT_MODELS,
    NO_DRIFT_MODELS,
    REFINE_MODELS,
    ROBUST_MODELS,
    STATIC_MODELS,
    best_candidate,
    choose_near_tie_by_risk,
    metric_gains,
    no_worse,
)
from .hierarchical_gates import HierarchicalGateConfig, _base_filter, _dynamic_gate, _refine_gate, _name
from .risk_veto import bool_value, finite_float
from .robust_gates import evaluate_robust_gate_v71


GROUP_KEYS = ["dataset_type", "scenario", "position_init_label", "velocity_init_label", "beta_prior_profile"]
PENALTY_ERROR_M = 1.0e7


def _safe_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite_series(series: pd.Series, penalty: float = PENALTY_ERROR_M) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(values), values, penalty)


def _first_error(group: pd.DataFrame, model: str) -> float:
    subset = group[group["candidate_model"] == model]
    if subset.empty:
        return np.nan
    return finite_float(subset.iloc[0].get("final_position_error_m"), np.nan)


def select_group_v71(group: pd.DataFrame, cfg: HierarchicalGateConfig | None = None) -> tuple[pd.Series, dict[str, Any], dict[str, Any], list[dict[str, Any]], str, bool]:
    cfg = cfg or HierarchicalGateConfig()
    rows = group.copy()
    dataset_type = str(rows["dataset_type"].iloc[0])
    keys = {col: rows[col].iloc[0] for col in GROUP_KEYS if col in rows.columns}
    best_static_all = best_candidate(rows, STATIC_MODELS)
    eligible, _base_reasons = _base_filter(rows, best_static_all, cfg)
    family_diag: dict[str, Any] = dict(keys)
    robust_diag: list[dict[str, Any]] = []
    low_quality = False
    reason_parts: list[str] = []

    if eligible.empty:
        fallback = best_candidate(rows)
        if fallback is None:
            raise ValueError("candidate group is empty")
        bias_diag = dict(keys)
        bias_diag.update(
            {
                "best_no_bias_model": "",
                "best_bias_capable_model": "",
                "raw_gain_no_bias_vs_bias": 0.0,
                "trimmed_gain_no_bias_vs_bias": 0.0,
                "inlier_gain_no_bias_vs_bias": 0.0,
                "best_bias_capable_beta0_estimated_mps": np.nan,
                "best_bias_capable_condition_number": np.nan,
                "nontrivial_bias_evidence": False,
                "no_bias_gate_decision": "fallback",
                "no_bias_gate_reason": "no_candidate_after_base_filter",
            }
        )
        family_diag.update({"final_family_decision": "fallback", "family_gate_reason": "no_candidate_after_base_filter"})
        return fallback, family_diag, bias_diag, robust_diag, "no_candidate_after_base_filter;fallback_best_validation", True

    core = eligible[eligible["candidate_model"].isin(CORE_FINAL_MODELS)].copy()
    if core.empty:
        core = eligible.copy()
        reason_parts.append("no_core_candidate;using_all_eligible")
    best_static = best_candidate(core, STATIC_MODELS)
    dynamic_core = core[core["candidate_model"].isin(CORE_DYNAMIC_MODELS)].copy()
    best_dynamic = best_candidate(dynamic_core)
    dynamic_pass, dyn_gain, dynamic_reason = _dynamic_gate(best_static, best_dynamic, dataset_type, cfg)
    family_diag.update(
        {
            **keys,
            "best_static_model": _name(best_static),
            "best_dynamic_model": _name(best_dynamic),
            "dynamic_gain": dyn_gain,
            "dynamic_gate_pass": bool(dynamic_pass),
        }
    )
    if not dynamic_pass and best_static is not None:
        static_pool = core[core["candidate_model"].isin(STATIC_MODELS)].copy()
        selected = choose_near_tie_by_risk(static_pool, cfg.near_tie_eps)
        bias_diag = dict(keys)
        bias_diag.update(
            {
                "best_no_bias_model": "",
                "best_bias_capable_model": "",
                "raw_gain_no_bias_vs_bias": 0.0,
                "trimmed_gain_no_bias_vs_bias": 0.0,
                "inlier_gain_no_bias_vs_bias": 0.0,
                "best_bias_capable_beta0_estimated_mps": np.nan,
                "best_bias_capable_condition_number": np.nan,
                "nontrivial_bias_evidence": False,
                "no_bias_gate_decision": "not_reached_static_selected",
                "no_bias_gate_reason": dynamic_reason,
            }
        )
        family_diag.update(
            {
                "best_no_bias_model": "",
                "best_bias_capable_model": "",
                "bias_gain": 0.0,
                "estimated_bias_evidence": 0.0,
                "bias_gate_decision": "not_reached_static_selected",
                "best_no_drift_model": "",
                "best_full_drift_model": "",
                "drift_gain": 0.0,
                "estimated_drift_evidence": 0.0,
                "drift_gate_decision": "not_reached_static_selected",
                "final_family_decision": "static",
                "family_gate_reason": dynamic_reason,
            }
        )
        return selected, family_diag, bias_diag, robust_diag, f"static_dynamic_gate:{dynamic_reason};static_family_selected", False

    candidates = dynamic_core.copy()
    reason_parts.append(f"static_dynamic_gate:{dynamic_reason}")
    candidates, bias_diag = evaluate_bias_gate_v71(candidates)
    bias_diag.update(keys)
    family_diag.update(
        {
            "best_no_bias_model": bias_diag.get("best_no_bias_model", ""),
            "best_bias_capable_model": bias_diag.get("best_bias_capable_model", ""),
            "bias_gain": max(
                float(bias_diag.get("raw_gain_no_bias_vs_bias", 0.0)),
                float(bias_diag.get("trimmed_gain_no_bias_vs_bias", 0.0)),
                float(bias_diag.get("inlier_gain_no_bias_vs_bias", 0.0)),
            ),
            "estimated_bias_evidence": abs(finite_float(bias_diag.get("best_bias_capable_beta0_estimated_mps"), 0.0)),
            "bias_gate_decision": bias_diag.get("no_bias_gate_decision", ""),
        }
    )
    reason_parts.append(f"bias_gate:{bias_diag.get('no_bias_gate_decision')}:{bias_diag.get('no_bias_gate_reason')}")

    best_no_drift = best_candidate(candidates, NO_DRIFT_MODELS)
    best_full = best_candidate(candidates, FULL_DRIFT_MODELS)
    drift_gain = max(metric_gains(best_no_drift, best_full).values()) if best_no_drift is not None and best_full is not None else 0.0
    estimated_drift = abs(finite_float(best_full.get("beta_dot_estimated_mps2"), 0.0)) if best_full is not None else 0.0
    drift_decision = "not_applicable"
    if best_no_drift is not None and best_full is not None:
        full_no_worse = no_worse(best_full, best_no_drift, cfg.near_tie_eps)
        if drift_gain >= cfg.drift_gain_min and estimated_drift <= cfg.bdot_physical_limit:
            candidates = candidates[candidates["candidate_model"].isin(FULL_DRIFT_MODELS)].copy()
            drift_decision = "full_drift_allowed_validation_gain"
        elif full_no_worse and estimated_drift >= cfg.drift_small_mps2 and estimated_drift <= cfg.bdot_physical_limit:
            candidates = candidates[candidates["candidate_model"].isin(FULL_DRIFT_MODELS)].copy()
            drift_decision = "full_drift_allowed_no_worse_nontrivial_drift"
        else:
            candidates = candidates[candidates["candidate_model"].isin(NO_DRIFT_MODELS)].copy()
            drift_decision = "no_drift_selected_insufficient_drift_evidence"
    family_diag.update(
        {
            "best_no_drift_model": _name(best_no_drift),
            "best_full_drift_model": _name(best_full),
            "drift_gain": float(drift_gain),
            "estimated_drift_evidence": float(estimated_drift),
            "drift_gate_decision": drift_decision,
        }
    )
    reason_parts.append(f"drift_gate:{drift_decision}")

    refine_keep = []
    for idx, row in candidates.iterrows():
        ok, refine_reason = _refine_gate(row, eligible)
        if ok:
            refine_keep.append(idx)
        else:
            reason_parts.append(f"{row.get('candidate_model')}_refine_blocked:{refine_reason}")
    candidates = candidates.loc[refine_keep].copy()

    nonrobust_reference = best_candidate(candidates[~candidates["candidate_model"].isin(ROBUST_MODELS)].copy())
    robust_keep = []
    for idx, row in candidates.iterrows():
        ok, diag = evaluate_robust_gate_v71(row.to_dict(), None if nonrobust_reference is None else nonrobust_reference.to_dict())
        diag.update(keys)
        diag["candidate_model"] = row.get("candidate_model")
        robust_diag.append(diag)
        if ok:
            robust_keep.append(idx)
        elif str(row.get("candidate_model")) in ROBUST_MODELS:
            reason_parts.append(f"{row.get('candidate_model')}_robust_blocked:{diag.get('robust_gate_reason')}")
    candidates = candidates.loc[robust_keep].copy()

    if candidates.empty:
        fallback = best_candidate(dynamic_core if not dynamic_core.empty else core)
        if fallback is None:
            fallback = best_candidate(rows)
        selected = fallback
        low_quality = True
        final_decision = "fallback"
        reason_parts.append("empty_after_family_gates;fallback_best_validation")
    else:
        selected = choose_near_tie_by_risk(candidates, cfg.near_tie_eps)
        final_decision = "dynamic"
    family_diag.update({"final_family_decision": final_decision, "family_gate_reason": ";".join(reason_parts)})
    return selected, family_diag, bias_diag, robust_diag, ";".join(reason_parts), low_quality


def _selected_result(keys: tuple[Any, ...], group: pd.DataFrame, selected: pd.Series, reason: str, low_quality: bool) -> dict[str, Any]:
    selected_error = finite_float(selected.get("final_position_error_m"), np.nan)
    oracle_errors = _finite_series(group["final_position_error_m"]) if "final_position_error_m" in group.columns else np.array([PENALTY_ERROR_M])
    oracle_error = float(np.min(oracle_errors)) if oracle_errors.size else np.nan
    ctd_error = _first_error(group, "M2_ctd_full")
    static_error = _first_error(group, "M0_static_position")
    return {
        "dataset_type": keys[0],
        "scenario": keys[1],
        "position_init_label": keys[2],
        "velocity_init_label": keys[3],
        "beta_prior_profile": keys[4],
        "selected_model_v71": selected.get("candidate_model"),
        "selection_reason_v71": reason,
        "selected_low_quality": bool(low_quality),
        "true_speed_mps": selected.get("true_speed_mps", np.nan),
        "true_b0_mps": selected.get("true_b0_mps", np.nan),
        "true_bdot_mps2": selected.get("true_bdot_mps2", np.nan),
        "final_position_error_m": selected.get("final_position_error_m", np.nan),
        "mean_position_error_m": selected.get("mean_position_error_m", np.nan),
        "max_position_error_m": selected.get("max_position_error_m", np.nan),
        "velocity_error_mps": selected.get("velocity_error_mps", np.nan),
        "estimated_speed_mps": selected.get("estimated_speed_mps", np.nan),
        "beta0_estimated_mps": selected.get("beta0_estimated_mps", np.nan),
        "beta_dot_estimated_mps2": selected.get("beta_dot_estimated_mps2", np.nan),
        "beta0_error_mps": selected.get("beta0_error_mps", np.nan),
        "beta_dot_error_mps2": selected.get("beta_dot_error_mps2", np.nan),
        "raw_validation_rmse_mps": selected.get("raw_validation_rmse_mps", np.nan),
        "trimmed_validation_rmse_mps": selected.get("trimmed_validation_rmse_mps", np.nan),
        "inlier_validation_rmse_mps": selected.get("inlier_validation_rmse_mps", np.nan),
        "full_residual_rmse_mps": selected.get("full_residual_rmse_mps", selected.get("residual_rmse_mps", np.nan)),
        "condition_number": selected.get("condition_number", np.nan),
        "quality_pass": selected.get("quality_pass", False),
        "physical_plausible": selected.get("physical_plausible", False),
        "oracle_best_position_error_m": oracle_error,
        "selected_minus_oracle_error_m": selected_error - oracle_error if np.isfinite(selected_error) and np.isfinite(oracle_error) else np.nan,
        "selected_beats_ctd": bool(np.isfinite(selected_error) and np.isfinite(ctd_error) and selected_error <= ctd_error),
        "selected_beats_static_lm": bool(np.isfinite(selected_error) and np.isfinite(static_error) and selected_error <= static_error),
    }


def select_v71_results(candidate_df: pd.DataFrame, config: HierarchicalGateConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or HierarchicalGateConfig()
    selected_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    robust_rows: list[dict[str, Any]] = []
    for keys, group in candidate_df.groupby(GROUP_KEYS, dropna=False, sort=True):
        selected, family_diag, bias_diag, robust_diag, reason, low_quality = select_group_v71(group, cfg)
        selected_rows.append(_selected_result(keys, group, selected, reason, low_quality))
        family_rows.append(family_diag)
        bias_rows.append(bias_diag)
        robust_rows.extend(robust_diag)
    return pd.DataFrame(selected_rows), pd.DataFrame(family_rows), pd.DataFrame(bias_rows), pd.DataFrame(robust_rows)


def entropy(counts: pd.Series) -> float:
    values = counts.to_numpy(dtype=float)
    total = float(np.sum(values))
    if total <= 0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def aggregate_v71_selected(selected_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in selected_df.groupby(["dataset_type", "scenario"], sort=True):
        counts = group["selected_model_v71"].value_counts()
        errors = _finite_series(group["final_position_error_m"])
        rows.append(
            {
                "dataset_type": keys[0],
                "scenario": keys[1],
                "runs": int(len(group)),
                "quality_pass_rate": float(group["quality_pass"].map(_safe_bool).mean()),
                "median_final_position_error_m": float(np.median(errors)),
                "p95_final_position_error_m": float(np.quantile(errors, 0.95)),
                "median_mean_position_error_m": float(pd.to_numeric(group["mean_position_error_m"], errors="coerce").median()),
                "median_velocity_error_mps": float(pd.to_numeric(group["velocity_error_mps"], errors="coerce").median()),
                "median_beta0_error_mps": float(pd.to_numeric(group["beta0_error_mps"], errors="coerce").median()),
                "median_beta_dot_error_mps2": float(pd.to_numeric(group["beta_dot_error_mps2"], errors="coerce").median()),
                "median_trimmed_validation_rmse_mps": float(pd.to_numeric(group["trimmed_validation_rmse_mps"], errors="coerce").median()),
                "most_selected_model_v71": str(counts.index[0]) if len(counts) else "",
                "selected_model_entropy_v71": entropy(counts),
                "oracle_best_position_error_m": float(pd.to_numeric(group["oracle_best_position_error_m"], errors="coerce").median()),
                "selected_minus_oracle_error_m": float(pd.to_numeric(group["selected_minus_oracle_error_m"], errors="coerce").median()),
                "selected_beats_ctd_rate": float(group["selected_beats_ctd"].map(_safe_bool).mean()),
                "selected_beats_static_lm_rate": float(group["selected_beats_static_lm"].map(_safe_bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def model_selection_counts_v71(selected_df: pd.DataFrame) -> pd.DataFrame:
    counts = selected_df.groupby(["dataset_type", "scenario", "selected_model_v71"]).size().reset_index(name="count")
    totals = counts.groupby(["dataset_type", "scenario"])["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals
    return counts


def compare_v7_v71(v7_selected: pd.DataFrame, v71_selected: pd.DataFrame) -> pd.DataFrame:
    if v7_selected.empty or v71_selected.empty:
        return pd.DataFrame()
    merged = v7_selected.merge(v71_selected, on=GROUP_KEYS, how="inner", suffixes=("_v7", "_v71"))
    return pd.DataFrame(
        {
            "dataset_type": merged["dataset_type"],
            "scenario": merged["scenario"],
            "position_init_label": merged["position_init_label"],
            "velocity_init_label": merged["velocity_init_label"],
            "beta_prior_profile": merged["beta_prior_profile"],
            "selected_model_v7": merged["selected_model_v7"],
            "selected_model_v71": merged["selected_model_v71"],
            "v7_final_position_error_m": merged["final_position_error_m_v7"],
            "v71_final_position_error_m": merged["final_position_error_m_v71"],
            "improvement_m": merged["final_position_error_m_v7"] - merged["final_position_error_m_v71"],
            "v7_trimmed_validation_rmse_mps": merged["trimmed_validation_rmse_mps_v7"],
            "v71_trimmed_validation_rmse_mps": merged["trimmed_validation_rmse_mps_v71"],
            "v71_selection_reason": merged["selection_reason_v71"],
        }
    )
