"""Dominance rules for MA-BGTR-v6 validation evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .risk_veto import finite_float


MAIN_METRICS = ["raw_validation_rmse_mps", "trimmed_validation_rmse_mps", "inlier_validation_rmse_mps"]


@dataclass(frozen=True)
class DominanceConfig:
    eps_raw: float = 0.01
    eps_trim: float = 0.01
    eps_inlier: float = 0.01
    min_gain: float = 0.02


def metric_vector(row: dict[str, Any] | pd.Series) -> dict[str, float]:
    return {metric: finite_float(row.get(metric), np.nan) for metric in [*MAIN_METRICS, "robust_validation_cost"]}


def metric_gain(a_value: float, b_value: float) -> float:
    if not (np.isfinite(a_value) and np.isfinite(b_value)) or abs(b_value) < 1.0e-12:
        return 0.0
    return float((b_value - a_value) / abs(b_value))


def dominates(
    candidate_a: dict[str, Any] | pd.Series,
    candidate_b: dict[str, Any] | pd.Series,
    config: DominanceConfig | None = None,
) -> tuple[bool, dict[str, float], str]:
    """Return whether A dominates B under validation evidence metrics.

    A must be no worse within a small tolerance on raw, trimmed, and inlier
    validation residuals, and must improve at least one main metric by the
    configured minimum gain. Robust cost is intentionally diagnostic only.
    """
    cfg = config or DominanceConfig()
    if bool(candidate_a.get("severe_risk_veto", False)):
        return False, {"raw_gain": 0.0, "trimmed_gain": 0.0, "inlier_gain": 0.0}, "dominator_has_severe_risk"

    a = metric_vector(candidate_a)
    b = metric_vector(candidate_b)
    eps = {
        "raw_validation_rmse_mps": cfg.eps_raw,
        "trimmed_validation_rmse_mps": cfg.eps_trim,
        "inlier_validation_rmse_mps": cfg.eps_inlier,
    }
    gains: dict[str, float] = {}
    for metric in MAIN_METRICS:
        if not (np.isfinite(a[metric]) and np.isfinite(b[metric])):
            return False, {"raw_gain": 0.0, "trimmed_gain": 0.0, "inlier_gain": 0.0}, f"missing_{metric}"
        if a[metric] > b[metric] * (1.0 + eps[metric]):
            return False, {"raw_gain": 0.0, "trimmed_gain": 0.0, "inlier_gain": 0.0}, f"{metric}_worse"
        gains[metric] = metric_gain(a[metric], b[metric])

    raw_gain = gains["raw_validation_rmse_mps"]
    trimmed_gain = gains["trimmed_validation_rmse_mps"]
    inlier_gain = gains["inlier_validation_rmse_mps"]
    if max(raw_gain, trimmed_gain, inlier_gain) < cfg.min_gain:
        return False, {"raw_gain": raw_gain, "trimmed_gain": trimmed_gain, "inlier_gain": inlier_gain}, "no_metric_exceeds_min_gain"

    reason = (
        f"validation_dominance raw_gain={raw_gain:.4f};"
        f"trimmed_gain={trimmed_gain:.4f};inlier_gain={inlier_gain:.4f}"
    )
    return True, {"raw_gain": raw_gain, "trimmed_gain": trimmed_gain, "inlier_gain": inlier_gain}, reason
