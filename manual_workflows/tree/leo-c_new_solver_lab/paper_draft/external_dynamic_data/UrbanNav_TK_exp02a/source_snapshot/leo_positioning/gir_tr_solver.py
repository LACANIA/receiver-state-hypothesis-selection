"""GIR-TR: geometry-preserving influence-robust trust-region solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry_weights import GeometryWeightDiagnostics, build_geometry_preserving_weights
from .influence_diagnostics import robust_sigma_mad
from .quality import physical_plausibility
from .trajectory_models import FULL_CTD_CONFIG, pack_state, residuals_and_jacobian_ctd, trajectory_positions, unpack_state


@dataclass
class GirTrResult:
    state: np.ndarray
    numerical_success: bool
    converged: bool
    iterations: int
    accepted_steps: int
    rejected_steps: int
    initial_objective: float
    final_objective: float
    residual: np.ndarray
    weights: np.ndarray
    robust_weights: np.ndarray
    geometry_floor: np.ndarray
    condition_number: float
    lambda_final: float
    trust_radius_final: float
    diagnostics: GeometryWeightDiagnostics
    failure_reason: str = ""


def _solve_linear(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _robust_cost(residual: np.ndarray, c_scale_mps: float, weights: np.ndarray | None = None) -> float:
    residual = np.asarray(residual, dtype=float)
    scaled = residual / max(float(c_scale_mps), 1e-9)
    per = 0.5 * c_scale_mps * c_scale_mps * np.log1p(scaled * scaled)
    if weights is not None:
        per = np.asarray(weights, dtype=float) * per
    return float(np.sum(per))


def _mad_scale(residual: np.ndarray, k: float = 2.5) -> float:
    sigma = robust_sigma_mad(residual)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return 2.0
    return float(np.clip(k * sigma, 0.5, 10.0))


def _physical_ok(theta: np.ndarray, dataset_type: str = "synthetic") -> bool:
    _p, v, b0, bdot = unpack_state(theta, FULL_CTD_CONFIG)
    ok, _reason = physical_plausibility(dataset_type, float(np.linalg.norm(v)), float(b0), float(bdot))
    return bool(ok)


def solve_gir_tr(
    initial_state: np.ndarray,
    time_s: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    t0_s: float,
    scale_mode: str = "fixed",
    dataset_type: str = "synthetic",
    max_iter_per_scale: int = 10,
    lambda0: float = 1e-3,
    trust_radius0: float = 100_000.0,
    floor_base: float = 0.02,
    floor_gain: float = 0.20,
    hard_outlier_threshold_mps: float = 10.0,
    hard_outlier_cap: float = 0.05,
) -> GirTrResult:
    theta = np.asarray(initial_state, dtype=float).reshape(-1).copy()
    if theta.size != 8:
        raise ValueError("GIR-TR expects the full CTD state dimension 8.")
    r0, j0, _ = residuals_and_jacobian_ctd(theta, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, FULL_CTD_CONFIG)
    base_scale = _mad_scale(r0) if scale_mode == "mad" else 2.0
    schedule = [10.0, 5.0, max(base_scale, 2.0), base_scale] if scale_mode == "mad" else [10.0, 5.0, 2.0, 1.0]
    lam = float(lambda0)
    trust_radius = float(trust_radius0)
    accepted = 0
    rejected = 0
    total_iterations = 0
    failure_reasons: list[str] = []
    weights, diag, robust_weights, floor = build_geometry_preserving_weights(
        r0, j0, schedule[0], floor_base, floor_gain, hard_outlier_threshold_mps, hard_outlier_cap
    )
    current_obj = _robust_cost(r0, schedule[0], weights)
    initial_obj = current_obj

    for scale in schedule:
        for _inner in range(max_iter_per_scale):
            total_iterations += 1
            residual, jac, _pred = residuals_and_jacobian_ctd(theta, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, FULL_CTD_CONFIG)
            weights, diag, robust_weights, floor = build_geometry_preserving_weights(
                residual, jac, scale, floor_base, floor_gain, hard_outlier_threshold_mps, hard_outlier_cap
            )
            normal = jac.T @ (weights[:, None] * jac)
            gradient = jac.T @ (weights * residual)
            diag_scale = np.diag(np.maximum(np.diag(normal), 1.0))
            try:
                eigvals, eigvecs = np.linalg.eigh(0.5 * (normal + normal.T))
                max_eval = max(float(np.max(np.abs(eigvals))), 1.0)
                weak = eigvals < max_eval * 1e-8
                geo_damping = eigvecs @ np.diag(np.where(weak, max_eval * 1e-3, 0.0)) @ eigvecs.T
            except Exception:
                geo_damping = np.zeros_like(normal)
            lhs = normal + lam * diag_scale + geo_damping
            rhs = -gradient
            try:
                delta = _solve_linear(lhs, rhs)
            except Exception as exc:  # noqa: BLE001
                failure_reasons.append(f"linear_solve_failed: {exc}")
                break
            if not np.all(np.isfinite(delta)):
                failure_reasons.append("non_finite_delta")
                break
            step_norm = float(np.linalg.norm(delta))
            if step_norm > trust_radius:
                delta = delta * (trust_radius / max(step_norm, 1e-12))
                step_norm = trust_radius
            candidate = theta + delta
            if not _physical_ok(candidate, dataset_type):
                rejected += 1
                lam = min(lam * 10.0, 1e12)
                trust_radius = max(trust_radius * 0.5, 1.0)
                continue
            cand_residual, cand_jac, _cand_pred = residuals_and_jacobian_ctd(
                candidate, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, FULL_CTD_CONFIG
            )
            cand_weights, _cand_diag, _cand_robust, _cand_floor = build_geometry_preserving_weights(
                cand_residual, cand_jac, scale, floor_base, floor_gain, hard_outlier_threshold_mps, hard_outlier_cap
            )
            cand_obj = _robust_cost(cand_residual, scale, cand_weights)
            predicted_reduction = float(-(gradient @ delta + 0.5 * delta @ normal @ delta))
            actual_reduction = float(current_obj - cand_obj)
            rho = actual_reduction / max(predicted_reduction, 1e-12)
            if np.isfinite(cand_obj) and actual_reduction > 0.0 and rho > 0.05:
                theta = candidate
                current_obj = cand_obj
                accepted += 1
                lam = max(lam / 3.0, 1e-12)
                if rho > 0.75:
                    trust_radius = min(max(trust_radius, 2.0 * step_norm), 10_000_000.0)
                if step_norm < 1e-5:
                    break
            else:
                rejected += 1
                lam = min(lam * 5.0, 1e12)
                trust_radius = max(trust_radius * 0.5, 1.0)

    final_residual, final_jac, _final_pred = residuals_and_jacobian_ctd(
        theta, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, FULL_CTD_CONFIG
    )
    weights, diag, robust_weights, floor = build_geometry_preserving_weights(
        final_residual, final_jac, schedule[-1], floor_base, floor_gain, hard_outlier_threshold_mps, hard_outlier_cap
    )
    try:
        condition = float(np.linalg.cond(final_jac.T @ (weights[:, None] * final_jac)))
    except Exception:
        condition = np.inf
    numerical_success = bool(np.all(np.isfinite(theta)) and np.all(np.isfinite(final_residual)) and accepted > 0)
    converged = bool(numerical_success and (accepted > 0) and (not failure_reasons))
    if not numerical_success and not failure_reasons:
        failure_reasons.append("no_accepted_step")
    return GirTrResult(
        state=theta,
        numerical_success=numerical_success,
        converged=converged,
        iterations=total_iterations,
        accepted_steps=accepted,
        rejected_steps=rejected,
        initial_objective=initial_obj,
        final_objective=_robust_cost(final_residual, schedule[-1], weights),
        residual=final_residual,
        weights=weights,
        robust_weights=robust_weights,
        geometry_floor=floor,
        condition_number=condition,
        lambda_final=float(lam),
        trust_radius_final=float(trust_radius),
        diagnostics=diag,
        failure_reason=";".join(dict.fromkeys(failure_reasons)),
    )


def gir_positions(result: GirTrResult, time_s: np.ndarray, t0_s: float) -> np.ndarray:
    return trajectory_positions(result.state, time_s, t0_s, FULL_CTD_CONFIG)


def initial_full_state(p0_m: np.ndarray, velocity_mps: np.ndarray) -> np.ndarray:
    return pack_state(p0_m, velocity_mps, 0.0, 0.0, FULL_CTD_CONFIG)
