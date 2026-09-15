"""Quality and physical-plausibility gates for MA-BGTR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    synthetic_speed_limit_mps: float = 300.0
    real_speed_limit_mps: float = 20.0
    beta0_limit_mps: float = 20.0
    bdot_limit_mps2: float = 0.5
    real_bdot_limit_mps2: float = 0.1
    synthetic_error_limit_m: float = 100_000.0
    real_error_limit_m: float = 200_000.0


def physical_plausibility(
    dataset_type: str,
    estimated_speed_mps: float,
    beta0_mps: float,
    bdot_mps2: float,
    config: QualityConfig | None = None,
) -> tuple[bool, str]:
    cfg = config or QualityConfig()
    reasons: list[str] = []
    speed_limit = cfg.real_speed_limit_mps if dataset_type == "real" else cfg.synthetic_speed_limit_mps
    if not np.isfinite(estimated_speed_mps) or estimated_speed_mps > speed_limit:
        reasons.append("speed_limit")
    if np.isfinite(beta0_mps) and abs(beta0_mps) > cfg.beta0_limit_mps:
        reasons.append("beta0_limit")
    bdot_limit = cfg.real_bdot_limit_mps2 if dataset_type == "real" else cfg.bdot_limit_mps2
    if np.isfinite(bdot_mps2) and abs(bdot_mps2) > bdot_limit:
        reasons.append("bdot_limit")
    return len(reasons) == 0, ";".join(reasons)


def quality_gate(
    dataset_type: str,
    numerical_success: bool,
    physical_plausible: bool,
    final_position_error_m: float | None = None,
    config: QualityConfig | None = None,
) -> tuple[bool, str]:
    cfg = config or QualityConfig()
    reasons: list[str] = []
    if not numerical_success:
        reasons.append("numerical_failure")
    if not physical_plausible:
        reasons.append("physical_failure")
    if final_position_error_m is not None and np.isfinite(final_position_error_m):
        limit = cfg.real_error_limit_m if dataset_type == "real" else cfg.synthetic_error_limit_m
        if final_position_error_m > limit:
            reasons.append("error_quality_limit")
    return len(reasons) == 0, ";".join(reasons)
