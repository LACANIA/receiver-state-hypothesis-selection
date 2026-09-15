"""Hierarchical evidence gates for MA-BGTR-v7.

TECH13 keeps this v7 implementation as the frozen comparison baseline. The
v7.1 bugfix selector lives in ``ma_bgtr_v71_solver`` and reuses the helpers in
this module while swapping in the repaired bias and robust gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .family_evidence import (
    BIAS_CAPABLE_MODELS,
    CORE_DYNAMIC_MODELS,
    CORE_FINAL_MODELS,
    FULL_DRIFT_MODELS,
    NO_BIAS_MODELS,
    NO_DRIFT_MODELS,
    REFINE_MODELS,
    ROBUST_MODELS,
    STATIC_MODELS,
    best_candidate,
    choose_near_tie_by_risk,
    clearly_better,
    metric_gains,
    no_worse,
    relative_gain,
    validation_score,
)
from .risk_veto import BE_FULL_MODELS, DIRECT_GIR_MODELS, REAL_FORBIDDEN_GIR_MODELS, bool_value, finite_float, is_dynamic_model
from .robust_gates import evaluate_robust_gate_v7


@dataclass(frozen=True)
class HierarchicalGateConfig:
    dynamic_gain_min: float = 0.02
    real_dynamic_gain_min: float = 0.05
    real_speed_strong_gain: float = 0.10
    bias_nontrivial_mps: float = 0.3
    bias_condition_limit: float = 1.0e12
    no_bias_margin: float = 0.02
    no_bias_eps: float = 0.01
    bias_trimmed_guard_eps: float = 0.015
    drift_gain_min: float = 0.02
    drift_small_mps2: float = 0.005
    bdot_physical_limit: float = 1.0
    near_tie_eps: float = 0.01


def _name(row: pd.Series | None) -> str:
    return "" if row is None else str(row.get("candidate_model", ""))


def _trim(row: pd.Series | None) -> float:
    return np.nan if row is None else finite_float(row.get("trimmed_validation_rmse_mps"), np.nan)


def _raw(row: pd.Series | None) -> float:
    return np.nan if row is None else finite_float(row.get("raw_validation_rmse_mps"), np.nan)


def _inlier(row: pd.Series | None) -> float:
    return np.nan if row is None else finite_float(row.get("inlier_validation_rmse_mps"), np.nan)


def _gain(ref: pd.Series | None, cand: pd.Series | None, metric: str) -> float:
    if ref is None or cand is None:
        return 0.0
    return relative_gain(finite_float(ref.get(metric), np.nan), finite_float(cand.get(metric), np.nan))


def _base_filter(group: pd.DataFrame, best_static: pd.Series | None, cfg: HierarchicalGateConfig) -> tuple[pd.DataFrame, dict[int, str]]:
    reasons: dict[int, str] = {}
    keep_indices: list[int] = []
    best_static_trim = _trim(best_static)
    for idx, row in group.iterrows():
        model = str(row.get("candidate_model", ""))
        row_reasons: list[str] = []
        if not bool_value(row.get("numerical_success"), False):
            row_reasons.append("numerical_success_false")
        if not bool_value(row.get("physical_plausible"), False):
            row_reasons.append("physical_plausible_false")
        if not bool_value(row.get("quality_pass"), False):
            row_reasons.append("quality_pass_false")
        if bool_value(row.get("severe_risk_veto"), False):
            row_reasons.append("severe_risk_veto")
        if model in BE_FULL_MODELS:
            retention = finite_float(row.get("retention_trace_ratio"), np.nan)
            rank_loss = finite_float(row.get("rank_loss"), 0.0)
            if np.isfinite(retention) and retention < 0.7:
                row_reasons.append("be_full_retention_lt_0_7")
            if np.isfinite(rank_loss) and rank_loss > 0:
                row_reasons.append("be_full_rank_loss")
        if model in DIRECT_GIR_MODELS:
            row_reasons.append("direct_gir_not_local_refine")
        if str(row.get("dataset_type", "")) == "real" and model in REAL_FORBIDDEN_GIR_MODELS:
            row_reasons.append("real_gir_or_static_refine_forbidden")
        if str(row.get("dataset_type", "")) == "real" and is_dynamic_model(model):
            dyn_gain = relative_gain(best_static_trim, finite_float(row.get("trimmed_validation_rmse_mps"), np.nan))
            speed = finite_float(row.get("estimated_speed_mps"), 0.0)
            bdot = abs(finite_float(row.get("beta_dot_estimated_mps2"), 0.0))
            if dyn_gain < cfg.real_dynamic_gain_min:
                row_reasons.append("real_dynamic_gain_lt_0_05")
            if speed > 2.0 and dyn_gain <= cfg.real_speed_strong_gain:
                row_reasons.append("real_speed_without_10pct_gain")
            if bdot > 0.02 and dyn_gain <= cfg.real_speed_strong_gain:
                row_reasons.append("real_bdot_without_10pct_gain")
        reasons[int(idx)] = ";".join(row_reasons)
        if not row_reasons:
            keep_indices.append(int(idx))
    return group.loc[keep_indices].copy(), reasons


def _dynamic_gate(best_static: pd.Series | None, best_dynamic: pd.Series | None, dataset_type: str, cfg: HierarchicalGateConfig) -> tuple[bool, float, str]:
    if best_static is None:
        return True, np.nan, "no_static_reference"
    if best_dynamic is None:
        return False, 0.0, "no_dynamic_candidate"
    raw_gain = _gain(best_static, best_dynamic, "raw_validation_rmse_mps")
    trim_gain = _gain(best_static, best_dynamic, "trimmed_validation_rmse_mps")
    inlier_gain = _gain(best_static, best_dynamic, "inlier_validation_rmse_mps")
    threshold = cfg.real_dynamic_gain_min if dataset_type == "real" else cfg.dynamic_gain_min
    reasons = []
    if trim_gain < threshold:
        reasons.append(f"dynamic_trimmed_gain_lt_{threshold:g}")
    if raw_gain <= 0:
        reasons.append("raw_not_support_dynamic")
    if inlier_gain <= 0:
        reasons.append("inlier_not_support_dynamic")
    if dataset_type == "real" and finite_float(best_dynamic.get("estimated_speed_mps"), 0.0) > 2.0 and trim_gain <= cfg.real_speed_strong_gain:
        reasons.append("real_speed_without_strong_gain")
    return not reasons, float(trim_gain), "pass" if not reasons else ";".join(reasons)


def _refine_gate(row: pd.Series, all_rows: pd.DataFrame) -> tuple[bool, str]:
    model = str(row.get("candidate_model", ""))
    if model not in REFINE_MODELS:
        return True, "not_refine"
    if "refine_gate_pass" in row.index and pd.notna(row.get("refine_gate_pass")) and not bool_value(row.get("refine_gate_pass"), True):
        return False, "stored_refine_gate_false"
    base_model = str(row.get("base_model_for_refine") or "")
    if not base_model:
        base_model = {"M12_ctd_full_plus_gir_refine": "M2_ctd_full", "M13_robust_ctd_full_plus_gir_refine": "M7_robust_ctd_full", "M14_static_plus_gir_refine": "M0_static_position"}.get(model, "")
    base = best_candidate(all_rows, {base_model}) if base_model else None
    if base is None:
        return False, "missing_refine_base"
    raw_ratio = finite_float(row.get("raw_validation_rmse_mps"), np.inf) / max(finite_float(base.get("raw_validation_rmse_mps"), np.nan), 1.0e-12)
    gains = metric_gains(base, row)
    cond_ratio = finite_float(row.get("condition_number"), np.inf) / max(finite_float(base.get("condition_number"), np.nan), 1.0e-12)
    speed = finite_float(row.get("estimated_speed_mps"), 0.0)
    beta = abs(finite_float(row.get("beta0_estimated_mps"), 0.0))
    bdot = abs(finite_float(row.get("beta_dot_estimated_mps2"), 0.0))
    info = finite_float(row.get("information_retention_ratio"), 1.0)
    reasons: list[str] = []
    if raw_ratio > 1.01:
        reasons.append("raw_worse_gt_1pct")
    if max(gains["trimmed_validation_rmse_mps"], gains["inlier_validation_rmse_mps"]) < 0.02:
        reasons.append("no_trimmed_or_inlier_gain_2pct")
    if np.isfinite(cond_ratio) and cond_ratio > 10.0:
        reasons.append("condition_worse_gt_10x")
    if speed > 300.0 or beta > 30.0 or bdot > 1.0:
        reasons.append("physical_limit_fail")
    if np.isfinite(info) and info < 0.5:
        reasons.append("information_retention_lt_0_5")
    return not reasons, "pass" if not reasons else ";".join(reasons)


def select_group_v7(group: pd.DataFrame, cfg: HierarchicalGateConfig | None = None) -> tuple[pd.Series, dict[str, Any], list[dict[str, Any]], str, bool]:
    cfg = cfg or HierarchicalGateConfig()
    rows = group.copy()
    dataset_type = str(rows["dataset_type"].iloc[0])
    keys = {col: rows[col].iloc[0] for col in ["dataset_type", "scenario", "position_init_label", "velocity_init_label", "beta_prior_profile"] if col in rows.columns}
    best_static_all = best_candidate(rows, STATIC_MODELS)
    eligible, base_reasons = _base_filter(rows, best_static_all, cfg)
    family_diag: dict[str, Any] = dict(keys)
    robust_diag: list[dict[str, Any]] = []
    low_quality = False
    reason_parts: list[str] = []

    if eligible.empty:
        fallback = best_candidate(rows)
        if fallback is None:
            raise ValueError("candidate group is empty")
        family_diag.update(
            {
                "best_static_model": _name(best_static_all),
                "best_dynamic_model": "",
                "dynamic_gain": 0.0,
                "dynamic_gate_pass": False,
                "final_family_decision": "fallback",
                "family_gate_reason": "no_candidate_after_base_filter",
            }
        )
        return fallback, family_diag, robust_diag, "no_candidate_after_base_filter;fallback_best_validation", True

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
            "best_static_model": _name(best_static),
            "best_dynamic_model": _name(best_dynamic),
            "dynamic_gain": dyn_gain,
            "dynamic_gate_pass": bool(dynamic_pass),
        }
    )
    if not dynamic_pass and best_static is not None:
        static_pool = core[core["candidate_model"].isin(STATIC_MODELS)].copy()
        selected = choose_near_tie_by_risk(static_pool, cfg.near_tie_eps)
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
        return selected, family_diag, robust_diag, f"static_dynamic_gate:{dynamic_reason};static_family_selected", False

    candidates = dynamic_core.copy()
    reason_parts.append(f"static_dynamic_gate:{dynamic_reason}")

    best_no_bias = best_candidate(candidates, NO_BIAS_MODELS)
    best_bias = best_candidate(candidates, BIAS_CAPABLE_MODELS)
    bias_gain = max(metric_gains(best_no_bias, best_bias).values()) if best_no_bias is not None and best_bias is not None else 0.0
    estimated_bias = abs(finite_float(best_bias.get("beta0_estimated_mps"), 0.0)) if best_bias is not None else 0.0
    bias_decision = "not_applicable"
    if best_no_bias is not None and best_bias is not None:
        no_bias_clearly = clearly_better(best_no_bias, best_bias, cfg.no_bias_margin, cfg.no_bias_eps)
        bias_condition = finite_float(best_bias.get("condition_number"), np.inf)
        bias_nontrivial = estimated_bias > cfg.bias_nontrivial_mps and bias_condition < cfg.bias_condition_limit
        bias_trim = finite_float(best_bias.get("trimmed_validation_rmse_mps"), np.inf)
        no_bias_trim = finite_float(best_no_bias.get("trimmed_validation_rmse_mps"), np.inf)
        if no_bias_clearly and not (bias_nontrivial and bias_trim <= no_bias_trim * (1.0 + cfg.bias_trimmed_guard_eps)):
            candidates = candidates[candidates["candidate_model"].isin(NO_BIAS_MODELS)].copy()
            bias_decision = "no_bias_allowed_clear_validation_win"
        else:
            candidates = candidates[~candidates["candidate_model"].isin(NO_BIAS_MODELS)].copy()
            bias_decision = "bias_capable_required_or_no_bias_not_clear"
    family_diag.update(
        {
            "best_no_bias_model": _name(best_no_bias),
            "best_bias_capable_model": _name(best_bias),
            "bias_gain": float(bias_gain),
            "estimated_bias_evidence": float(estimated_bias),
            "bias_gate_decision": bias_decision,
        }
    )
    reason_parts.append(f"bias_gate:{bias_decision}")

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
        ok, diag = evaluate_robust_gate_v7(row.to_dict(), None if nonrobust_reference is None else nonrobust_reference.to_dict())
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
    family_diag.update(
        {
            "final_family_decision": final_decision,
            "family_gate_reason": ";".join(reason_parts),
        }
    )
    return selected, family_diag, robust_diag, ";".join(reason_parts), low_quality
