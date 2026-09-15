"""Pareto-frontier selector for MA-BGTR-v6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .dominance_rules import DominanceConfig, dominates
from .risk_veto import (
    FULL_DRIFT_MODELS,
    ROBUST_MODELS,
    compute_group_references,
    drift_gain,
    dynamic_gain,
    finite_float,
    hard_filter,
    has_outlier_evidence,
    is_dynamic_model,
    is_full_drift_model,
    is_robust_model,
    model_dof,
    robust_gain,
)


@dataclass(frozen=True)
class ParetoSelectorConfig:
    dynamic_gain_min: float = 0.02
    drift_gain_min: float = 0.02
    robust_gain_min: float = 0.05
    near_tie_residual_eps: float = 0.01
    dominance: DominanceConfig = DominanceConfig()


def _safe_log(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0e6
    return float(np.log(value + 1.0e-12))


def evidence_score(row: dict[str, Any] | pd.Series) -> float:
    raw = finite_float(row.get("raw_validation_rmse_mps"), np.nan)
    trimmed = finite_float(row.get("trimmed_validation_rmse_mps"), np.nan)
    inlier = finite_float(row.get("inlier_validation_rmse_mps"), np.nan)
    robust = finite_float(row.get("robust_validation_cost"), np.nan)
    parts = [_safe_log(raw), _safe_log(trimmed), _safe_log(inlier)]
    if np.isfinite(robust) and robust > 0:
        parts.append(0.25 * _safe_log(robust))
    return float(np.mean(parts))


def risk_tie_break_key(row: dict[str, Any] | pd.Series) -> tuple[float, float, float, int]:
    condition = finite_float(row.get("condition_number"), np.nan)
    cov = finite_float(row.get("position_cov_sqrt_trace_m"), np.nan)
    bootstrap = finite_float(row.get("position_bootstrap_spread_m"), np.nan)
    return (
        _safe_log(condition) if np.isfinite(condition) and condition > 0 else 1.0e3,
        cov if np.isfinite(cov) else 1.0e9,
        bootstrap if np.isfinite(bootstrap) else 1.0e9,
        model_dof(str(row.get("candidate_model", "")), row),
    )


def _near_best(group: pd.DataFrame, best: pd.Series, eps: float) -> pd.DataFrame:
    keep = []
    for _, row in group.iterrows():
        raw_ok = finite_float(row.get("raw_validation_rmse_mps"), np.inf) <= finite_float(best.get("raw_validation_rmse_mps"), np.inf) * (1.0 + eps)
        trim_ok = finite_float(row.get("trimmed_validation_rmse_mps"), np.inf) <= finite_float(best.get("trimmed_validation_rmse_mps"), np.inf) * (1.0 + eps)
        inlier_ok = finite_float(row.get("inlier_validation_rmse_mps"), np.inf) <= finite_float(best.get("inlier_validation_rmse_mps"), np.inf) * (1.0 + eps)
        keep.append(bool(raw_ok and trim_ok and inlier_ok))
    near = group.loc[keep]
    return near if not near.empty else group.loc[[best.name]]


def _choose_by_evidence_then_risk(group: pd.DataFrame, config: ParetoSelectorConfig) -> tuple[pd.Series, str]:
    scored = group.copy()
    scored["_evidence_score"] = scored.apply(evidence_score, axis=1)
    scored = scored.sort_values(["_evidence_score", "candidate_model"])
    best = scored.iloc[0]
    near = _near_best(scored, best, config.near_tie_residual_eps).copy()
    if len(near) > 1:
        near["_risk_key"] = near.apply(risk_tie_break_key, axis=1)
        near = near.sort_values("_risk_key")
        chosen = near.iloc[0]
        return chosen, "near_tie_risk_tie_breaker"
    return best, "best_validation_evidence"


def select_from_frontier(frontier: pd.DataFrame, all_group: pd.DataFrame, refs: dict[str, float], config: ParetoSelectorConfig | None = None) -> tuple[pd.Series, str]:
    cfg = config or ParetoSelectorConfig()
    if frontier.empty:
        raise ValueError("frontier is empty")
    current = frontier.copy()
    reason_parts: list[str] = []

    dynamic_mask = current["candidate_model"].map(lambda m: is_dynamic_model(str(m)))
    dynamic_candidates = current.loc[dynamic_mask].copy()
    dynamic_candidates["_dynamic_gain"] = dynamic_candidates.apply(lambda row: dynamic_gain(row, refs), axis=1)
    strong_dynamic = dynamic_candidates[dynamic_candidates["_dynamic_gain"] >= cfg.dynamic_gain_min]
    if not strong_dynamic.empty:
        current = strong_dynamic.drop(columns=["_dynamic_gain"])
        reason_parts.append("dynamic_gain_pass")

    full_drift = current[current["candidate_model"].isin(FULL_DRIFT_MODELS)].copy()
    if not full_drift.empty:
        full_drift["_drift_gain"] = full_drift.apply(lambda row: drift_gain(row, refs), axis=1)
        allowed = full_drift[
            (full_drift["_drift_gain"] >= cfg.drift_gain_min)
            & (pd.to_numeric(full_drift.get("beta_dot_estimated_mps2", 0.0), errors="coerce").abs().fillna(0.0) <= 1.0)
        ]
        if not allowed.empty:
            current = allowed.drop(columns=["_drift_gain"])
            reason_parts.append("full_drift_gain_pass")

    robust = current[current["candidate_model"].isin(ROBUST_MODELS)].copy()
    if not robust.empty:
        robust["_robust_gain"] = robust.apply(lambda row: robust_gain(row, refs), axis=1)
        robust["_outlier_evidence"] = robust.apply(has_outlier_evidence, axis=1)
        allowed = robust[(robust["_outlier_evidence"]) & (robust["_robust_gain"] >= cfg.robust_gain_min)]
        if not allowed.empty:
            current = allowed.drop(columns=["_robust_gain", "_outlier_evidence"])
            reason_parts.append("robust_outlier_gain_pass")

    chosen, evidence_reason = _choose_by_evidence_then_risk(current, cfg)
    reason_parts.append(evidence_reason)
    return chosen, ";".join(reason_parts)


def select_group(group: pd.DataFrame, config: ParetoSelectorConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, str, bool]:
    cfg = config or ParetoSelectorConfig()
    rows = group.copy().reset_index(drop=True)
    refs = compute_group_references(rows)

    hard_ok: list[bool] = []
    hard_reasons: list[str] = []
    severe_flags: list[bool] = []
    severe_reasons: list[str] = []
    for _, row in rows.iterrows():
        ok, reason = hard_filter(row, refs)
        hard_ok.append(ok)
        hard_reasons.append(reason)
        severe = "severe_risk:" in reason
        severe_flags.append(severe)
        severe_reasons.append(reason.split("severe_risk:", 1)[1] if severe and "severe_risk:" in reason else "")
    rows["hard_filter_pass"] = hard_ok
    rows["hard_filter_reason"] = hard_reasons
    rows["severe_risk_veto"] = severe_flags
    rows["severe_risk_reason"] = severe_reasons
    rows["pareto_frontier_member"] = False
    rows["dominated_by"] = ""
    rows["selected_by_v6"] = False

    eligible = rows[rows["hard_filter_pass"]].copy()
    dominance_pairs: list[dict[str, Any]] = []
    dominated: dict[int, list[str]] = {int(idx): [] for idx in eligible.index}
    for idx_a, row_a in eligible.iterrows():
        for idx_b, row_b in eligible.iterrows():
            if idx_a == idx_b:
                continue
            ok, gains, reason = dominates(row_a, row_b, cfg.dominance)
            if ok:
                dominated[int(idx_b)].append(str(row_a.get("candidate_model")))
                dominance_pairs.append(
                    {
                        "dominator_model": row_a.get("candidate_model"),
                        "dominated_model": row_b.get("candidate_model"),
                        "raw_gain": gains["raw_gain"],
                        "trimmed_gain": gains["trimmed_gain"],
                        "inlier_gain": gains["inlier_gain"],
                        "dominance_reason": reason,
                    }
                )

    frontier_indices = [idx for idx in eligible.index if not dominated.get(int(idx))]
    frontier = eligible.loc[frontier_indices].copy()
    if not frontier.empty:
        frontier["frontier_reason"] = "not_dominated_after_hard_filter"
        rows.loc[frontier_indices, "pareto_frontier_member"] = True
        for idx, dominators in dominated.items():
            rows.loc[idx, "dominated_by"] = ";".join(dominators)
        selected, reason = select_from_frontier(frontier, rows, refs, cfg)
        low_quality = False
    else:
        fallback_pool = rows[
            rows["numerical_success"].astype(str).str.lower().isin(["true", "1"])
            & rows["physical_plausible"].astype(str).str.lower().isin(["true", "1"])
        ].copy()
        if fallback_pool.empty:
            fallback_pool = rows.copy()
        fallback_pool["_evidence_score"] = fallback_pool.apply(evidence_score, axis=1)
        selected = fallback_pool.sort_values(["_evidence_score", "candidate_model"]).iloc[0]
        reason = "no_pareto_candidate;fallback_best_validation_evidence"
        low_quality = True
        frontier = pd.DataFrame(columns=[*rows.columns, "frontier_reason"])

    selected_idx = int(selected.name)
    rows.loc[selected_idx, "selected_by_v6"] = True
    return rows, frontier, pd.DataFrame(dominance_pairs), rows.loc[selected_idx], reason, low_quality
