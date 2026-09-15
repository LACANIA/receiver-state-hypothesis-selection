"""Robust weighting schedules for BE-GTR."""

from __future__ import annotations

import numpy as np


def cauchy_weights(residual: np.ndarray, c_scale_mps: float) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    scaled = residual / float(c_scale_mps)
    return 1.0 / (1.0 + scaled * scaled)


def graduated_cauchy_schedule(robust: bool) -> list[float]:
    return [10.0, 5.0, 2.0] if robust else [np.inf]


def weights_for_scale(residual: np.ndarray, c_scale_mps: float) -> np.ndarray:
    if not np.isfinite(c_scale_mps):
        return np.ones_like(np.asarray(residual, dtype=float))
    return cauchy_weights(residual, c_scale_mps)
