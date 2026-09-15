"""Bias/no-bias gate fix for MA-BGTR-v7.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .family_evidence import BIAS_CAPABLE_MODELS, NO_BIAS_MODELS, best_candidate, metric_gains
from .risk_veto import finite_float


@dataclass(frozen=True)
class BiasGateFixConfig:
    nontrivial_bias_mps: float = 0.3
    bias_condition_limit: float = 1.0e12
    near_tie_eps: float = 0.01
    validation_gain_min: float = 0.02


def _gain_no_bias_vs_bias(no_bias: pd.Series, bias: pd.Series, metric: str) -> float:
    nb = finite_float(no_bias.get(metric), np.nan)
    bc = finite_float(bias.get(metric), np.nan)
    if not (np.isfinite(nb) and np.isfinite(bc)) or abs(bc) < 1.0e-12:
        return 0.0
    return float((bc - nb) / abs(bc))


def evaluate_bias_gate_v71(candidates: pd.DataFrame, config: BiasGateFixConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return candidate subset after the v7.1 no-bias/bias-capable gate.

    Positive raw/trimmed/inlier gain means the no-bias candidate is better
    than the best bias-capable candidate. Bias-capable candidates may override
    only when there is nontrivial bias evidence and condition is acceptable.
    """
    cfg = config or BiasGateFixConfig()
    best_no_bias = best_candidate(candidates, NO_BIAS_MODELS)
    best_bias = best_candidate(candidates, BIAS_CAPABLE_MODELS)
    diag: dict[str, Any] = {
        "best_no_bias_model": "" if best_no_bias is None else best_no_bias.get("candidate_model"),
        "best_bias_capable_model": "" if best_bias is None else best_bias.get("candidate_model"),
        "raw_gain_no_bias_vs_bias": 0.0,
        "trimmed_gain_no_bias_vs_bias": 0.0,
        "inlier_gain_no_bias_vs_bias": 0.0,
        "best_bias_capable_beta0_estimated_mps": np.nan,
        "best_bias_capable_condition_number": np.nan,
        "nontrivial_bias_evidence": False,
        "no_bias_gate_decision": "not_applicable",
        "no_bias_gate_reason": "missing_no_bias_or_bias_candidate",
    }
    if best_no_bias is None or best_bias is None:
        return candidates.copy(), diag

    raw_gain = _gain_no_bias_vs_bias(best_no_bias, best_bias, "raw_validation_rmse_mps")
    trimmed_gain = _gain_no_bias_vs_bias(best_no_bias, best_bias, "trimmed_validation_rmse_mps")
    inlier_gain = _gain_no_bias_vs_bias(best_no_bias, best_bias, "inlier_validation_rmse_mps")
    gains = {
        "raw_validation_rmse_mps": raw_gain,
        "trimmed_validation_rmse_mps": trimmed_gain,
        "inlier_validation_rmse_mps": inlier_gain,
    }
    bias_beta0 = abs(finite_float(best_bias.get("beta0_estimated_mps"), 0.0))
    bias_condition = finite_float(best_bias.get("condition_number"), np.inf)
    bias_vs_no_bias = metric_gains(best_no_bias, best_bias)
    bias_improves_trimmed = bias_vs_no_bias["trimmed_validation_rmse_mps"] >= cfg.validation_gain_min
    bias_improves_inlier = bias_vs_no_bias["inlier_validation_rmse_mps"] >= cfg.validation_gain_min
    nontrivial_bias = bool(bias_beta0 >= cfg.nontrivial_bias_mps or bias_improves_trimmed or bias_improves_inlier)
    condition_ok = bool(np.isfinite(bias_condition) and bias_condition <= cfg.bias_condition_limit)
    no_bias_not_worse_count = sum(gain >= -cfg.near_tie_eps for gain in gains.values())
    no_bias_clear = max(gains.values()) >= cfg.validation_gain_min and no_bias_not_worse_count == 3

    if no_bias_clear:
        decision = "keep_no_bias"
        reason = "no_bias_clear_validation_win"
        selected_models = NO_BIAS_MODELS
    elif no_bias_not_worse_count >= 2 and not nontrivial_bias:
        decision = "keep_no_bias"
        reason = "near_tie_without_nontrivial_bias"
        selected_models = NO_BIAS_MODELS
    elif nontrivial_bias and condition_ok:
        decision = "keep_bias_capable"
        reason = "nontrivial_bias_evidence_condition_ok"
        selected_models = BIAS_CAPABLE_MODELS
    elif no_bias_not_worse_count >= 2:
        decision = "keep_no_bias"
        reason = "bias_evidence_insufficient_or_condition_bad"
        selected_models = NO_BIAS_MODELS
    else:
        decision = "keep_bias_capable"
        reason = "bias_capable_validation_better"
        selected_models = BIAS_CAPABLE_MODELS

    diag.update(
        {
            "raw_gain_no_bias_vs_bias": float(raw_gain),
            "trimmed_gain_no_bias_vs_bias": float(trimmed_gain),
            "inlier_gain_no_bias_vs_bias": float(inlier_gain),
            "best_bias_capable_beta0_estimated_mps": float(finite_float(best_bias.get("beta0_estimated_mps"), np.nan)),
            "best_bias_capable_condition_number": float(bias_condition) if np.isfinite(bias_condition) else np.nan,
            "nontrivial_bias_evidence": bool(nontrivial_bias and condition_ok),
            "no_bias_gate_decision": decision,
            "no_bias_gate_reason": reason,
        }
    )
    return candidates[candidates["candidate_model"].isin(selected_models)].copy(), diag
