"""Projection-loss diagnostics for BE-GTR repair experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .be_gtr_solver import h_and_jacobian_x
from .variable_projection import beta_prior_profiles, bias_design_matrix, project_residual_and_jacobian


@dataclass
class ProjectionDiagnostics:
    beta_basis: str
    retention_trace_ratio: float
    retention_logdet_ratio: float
    rank_before: int
    rank_after: int
    rank_loss: int
    subspace_coherence_max: float
    subspace_coherence_mean: float
    velocity_column_loss_ratio: float
    position_column_loss_ratio: float
    beta_normal_condition: float
    reduced_condition_number: float
    diagnostic_status: str


def _weighted_matrix(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sqrt(np.asarray(weights, dtype=float))[:, None] * np.asarray(matrix, dtype=float)


def _matrix_rank(matrix: np.ndarray) -> int:
    s = np.linalg.svd(matrix, compute_uv=False)
    if s.size == 0:
        return 0
    tol = max(matrix.shape) * np.finfo(float).eps * max(float(s[0]), 1.0)
    return int(np.sum(s > tol))


def _logdet_ratio(before: np.ndarray, after: np.ndarray) -> float:
    eps = 1e-12
    s_before = np.linalg.svd(before, compute_uv=False)
    s_after = np.linalg.svd(after, compute_uv=False)
    return float(np.sum(np.log(s_after + eps)) / max(np.sum(np.log(s_before + eps)), eps))


def _subspace_coherence(b_matrix: np.ndarray, j_x: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if b_matrix.shape[1] == 0:
        return 0.0, 0.0
    wb = _weighted_matrix(b_matrix, weights)
    wj = _weighted_matrix(j_x, weights)
    qb, _rb = np.linalg.qr(wb, mode="reduced")
    qj, _rj = np.linalg.qr(wj, mode="reduced")
    if qb.size == 0 or qj.size == 0:
        return 0.0, 0.0
    s = np.linalg.svd(qb.T @ qj, compute_uv=False)
    return float(np.max(s)), float(np.mean(s))


def compute_projection_diagnostics(
    x: np.ndarray,
    time_s: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    t0_s: float,
    weights: np.ndarray,
    beta_basis: str,
    beta_prior_profile: str = "B0_none",
) -> ProjectionDiagnostics:
    h, j_x = h_and_jacobian_x(x, time_s, sat_pos_m, sat_vel_mps, t0_s)
    a = np.asarray(meas_mps, dtype=float) - h
    b_matrix = bias_design_matrix(time_s, t0_s, projection_mode="full" if beta_basis == "full" else "b0_only")
    profile = beta_prior_profiles()[beta_prior_profile]
    projection = project_residual_and_jacobian(a, j_x, b_matrix, weights, profile.for_mode("full" if beta_basis == "full" else "b0_only"))
    j_red = projection.projected_jacobian
    normal_before = j_x.T @ (weights[:, None] * j_x)
    normal_after = j_red.T @ (weights[:, None] * j_red)
    trace_before = float(np.trace(normal_before))
    trace_after = float(np.trace(normal_after))
    retention_trace = trace_after / max(trace_before, 1e-30)
    retention_logdet = _logdet_ratio(normal_before, normal_after)
    rank_before = _matrix_rank(_weighted_matrix(j_x, weights))
    rank_after = _matrix_rank(_weighted_matrix(j_red, weights))
    coherence_max, coherence_mean = _subspace_coherence(b_matrix, j_x, weights)
    pos_before = float(np.sum(weights[:, None] * j_x[:, :3] * j_x[:, :3]))
    pos_after = float(np.sum(weights[:, None] * j_red[:, :3] * j_red[:, :3]))
    vel_before = float(np.sum(weights[:, None] * j_x[:, 3:6] * j_x[:, 3:6]))
    vel_after = float(np.sum(weights[:, None] * j_red[:, 3:6] * j_red[:, 3:6]))
    position_loss = 1.0 - pos_after / max(pos_before, 1e-30)
    velocity_loss = 1.0 - vel_after / max(vel_before, 1e-30)
    try:
        red_cond = float(np.linalg.cond(normal_after))
    except Exception:
        red_cond = np.inf
    if retention_trace < 0.5 or rank_before - rank_after >= 2:
        status = "projection_loss_high"
    elif retention_trace < 0.7 or coherence_max > 0.9:
        status = "projection_loss_medium"
    else:
        status = "ok"
    return ProjectionDiagnostics(
        beta_basis=beta_basis,
        retention_trace_ratio=float(retention_trace),
        retention_logdet_ratio=float(retention_logdet),
        rank_before=rank_before,
        rank_after=rank_after,
        rank_loss=int(rank_before - rank_after),
        subspace_coherence_max=float(coherence_max),
        subspace_coherence_mean=float(coherence_mean),
        velocity_column_loss_ratio=float(velocity_loss),
        position_column_loss_ratio=float(position_loss),
        beta_normal_condition=float(projection.beta_normal_condition),
        reduced_condition_number=red_cond,
        diagnostic_status=status,
    )


def diagnostics_to_dict(diag: ProjectionDiagnostics) -> dict[str, float | int | str]:
    return {
        "beta_basis": diag.beta_basis,
        "retention_trace_ratio": float(diag.retention_trace_ratio),
        "retention_logdet_ratio": float(diag.retention_logdet_ratio),
        "rank_before": int(diag.rank_before),
        "rank_after": int(diag.rank_after),
        "rank_loss": int(diag.rank_loss),
        "subspace_coherence_max": float(diag.subspace_coherence_max),
        "subspace_coherence_mean": float(diag.subspace_coherence_mean),
        "velocity_column_loss_ratio": float(diag.velocity_column_loss_ratio),
        "position_column_loss_ratio": float(diag.position_column_loss_ratio),
        "beta_normal_condition": float(diag.beta_normal_condition),
        "reduced_condition_number": float(diag.reduced_condition_number),
        "diagnostic_status": diag.diagnostic_status,
    }
