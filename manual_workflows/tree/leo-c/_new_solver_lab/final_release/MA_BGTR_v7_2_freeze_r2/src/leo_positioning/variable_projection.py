"""Variable projection utilities for BE-GTR.

BE-GTR treats the common Doppler bias and bias drift as linear nuisance
variables:

    y = h(x) + B beta + error
    beta = [b0, bdot]
    B_i = [1, tau_i]

For a fixed geometric state x = [p0, v], beta is solved analytically and is not
updated as a joint LM state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BetaPriorProfile:
    name: str
    beta_prior: np.ndarray | None
    sigma: np.ndarray | None

    @property
    def omega(self) -> np.ndarray | None:
        if self.sigma is None:
            return None
        return np.diag(1.0 / (self.sigma * self.sigma))

    def for_mode(self, projection_mode: str) -> "BetaPriorProfile":
        if projection_mode == "full":
            return self
        if projection_mode == "b0_only":
            if self.beta_prior is None or self.sigma is None:
                return BetaPriorProfile(self.name, None, None)
            return BetaPriorProfile(self.name, self.beta_prior[:1].copy(), self.sigma[:1].copy())
        if projection_mode == "none":
            return BetaPriorProfile(self.name, None, None)
        raise ValueError(f"Unknown projection mode: {projection_mode}")


def beta_prior_profiles() -> dict[str, BetaPriorProfile]:
    return {
        "B0_none": BetaPriorProfile("B0_none", None, None),
        "B1_weak": BetaPriorProfile(
            "B1_weak",
            np.array([0.0, 0.0], dtype=float),
            np.array([10.0, 0.2], dtype=float),
        ),
        "B2_moderate": BetaPriorProfile(
            "B2_moderate",
            np.array([0.0, 0.0], dtype=float),
            np.array([5.0, 0.05], dtype=float),
        ),
        "B3_static": BetaPriorProfile(
            "B3_static",
            np.array([0.0, 0.0], dtype=float),
            np.array([2.0, 0.02], dtype=float),
        ),
    }


@dataclass
class ProjectionResult:
    beta: np.ndarray
    reduced_residual: np.ndarray
    projected_jacobian: np.ndarray
    projection_matrix: np.ndarray
    beta_normal_condition: float
    regularized: bool
    regularization_lambda: float
    weighted_orthogonality_error: float
    prior_cost: float
    reduced_cost: float


def bias_design_matrix(time_s: np.ndarray, t0_s: float, projection_mode: str = "full") -> np.ndarray:
    tau = np.asarray(time_s, dtype=float) - float(t0_s)
    if projection_mode == "full":
        return np.column_stack([np.ones_like(tau), tau])
    if projection_mode == "b0_only":
        return np.ones((len(tau), 1), dtype=float)
    if projection_mode == "none":
        return np.zeros((len(tau), 0), dtype=float)
    raise ValueError(f"Unknown projection mode: {projection_mode}")


def expand_beta(beta: np.ndarray, projection_mode: str) -> np.ndarray:
    beta = np.asarray(beta, dtype=float).reshape(-1)
    if projection_mode == "full":
        return beta
    if projection_mode == "b0_only":
        return np.array([float(beta[0]), 0.0], dtype=float)
    if projection_mode == "none":
        return np.array([0.0, 0.0], dtype=float)
    raise ValueError(f"Unknown projection mode: {projection_mode}")


def solve_beta(
    a: np.ndarray,
    b_matrix: np.ndarray,
    weights: np.ndarray,
    profile: BetaPriorProfile,
    condition_threshold: float = 1e10,
    initial_tikhonov: float = 1e-10,
) -> tuple[np.ndarray, float, bool, float]:
    a = np.asarray(a, dtype=float).reshape(-1)
    b_matrix = np.asarray(b_matrix, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weighted_b = weights[:, None] * b_matrix
    normal = b_matrix.T @ weighted_b
    rhs = b_matrix.T @ (weights * a)
    if b_matrix.shape[1] == 0:
        return np.zeros(0, dtype=float), np.nan, False, 0.0
    if profile.omega is not None and profile.beta_prior is not None:
        normal = normal + profile.omega
        rhs = rhs + profile.omega @ profile.beta_prior
    try:
        cond = float(np.linalg.cond(normal))
    except Exception:
        cond = np.inf
    reg = 0.0
    regularized = False
    normal_solve = normal
    if (not np.isfinite(cond)) or cond > condition_threshold:
        scale = float(np.trace(normal) / max(normal.shape[0], 1))
        reg = max(initial_tikhonov * max(scale, 1.0), 1e-12)
        normal_solve = normal + reg * np.eye(normal.shape[0])
        regularized = True
    try:
        beta = np.linalg.solve(normal_solve, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(normal_solve, rhs, rcond=None)[0]
        regularized = True
        if reg == 0.0:
            reg = initial_tikhonov
    return beta, cond, regularized, reg


def project_residual_and_jacobian(
    a: np.ndarray,
    j_x: np.ndarray,
    b_matrix: np.ndarray,
    weights: np.ndarray,
    profile: BetaPriorProfile,
) -> ProjectionResult:
    """Return reduced residual and projected Jacobian for a fixed x."""
    a = np.asarray(a, dtype=float).reshape(-1)
    j_x = np.asarray(j_x, dtype=float)
    b_matrix = np.asarray(b_matrix, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if b_matrix.shape[1] == 0:
        reduced_residual = a.copy()
        projected_jacobian = j_x.copy()
        reduced_cost = 0.5 * float(np.sum(weights * reduced_residual * reduced_residual))
        return ProjectionResult(
            beta=np.zeros(0, dtype=float),
            reduced_residual=reduced_residual,
            projected_jacobian=projected_jacobian,
            projection_matrix=np.eye(len(a), dtype=float),
            beta_normal_condition=np.nan,
            regularized=False,
            regularization_lambda=0.0,
            weighted_orthogonality_error=0.0,
            prior_cost=0.0,
            reduced_cost=reduced_cost,
        )
    beta, cond, regularized, reg = solve_beta(a, b_matrix, weights, profile)

    weighted_b = weights[:, None] * b_matrix
    normal = b_matrix.T @ weighted_b
    if profile.omega is not None:
        normal = normal + profile.omega
    if reg > 0.0:
        normal = normal + reg * np.eye(normal.shape[0])
    try:
        inv_normal_bt_w = np.linalg.solve(normal, b_matrix.T * weights[None, :])
    except np.linalg.LinAlgError:
        inv_normal_bt_w = np.linalg.lstsq(normal, b_matrix.T * weights[None, :], rcond=None)[0]
    projection = np.eye(len(a), dtype=float) - b_matrix @ inv_normal_bt_w
    # The residual itself must use beta_star. With beta priors, P a alone
    # omits the prior offset term and makes the trust-region objective
    # inconsistent with the beta solve.
    reduced_residual = a - b_matrix @ beta
    projected_jacobian = projection @ j_x
    # With an active beta prior, B^T W r is balanced by the prior residual.
    # The unregularized no-prior case should be close to zero.
    ortho = b_matrix.T @ (weights * reduced_residual)
    weighted_orthogonality_error = float(np.linalg.norm(ortho, ord=np.inf))
    prior_cost = 0.0
    if profile.omega is not None and profile.beta_prior is not None:
        beta_diff = beta - profile.beta_prior
        prior_cost = float(beta_diff @ (profile.omega @ beta_diff))
    reduced_cost = 0.5 * float(np.sum(weights * reduced_residual * reduced_residual) + prior_cost)
    return ProjectionResult(
        beta=beta,
        reduced_residual=reduced_residual,
        projected_jacobian=projected_jacobian,
        projection_matrix=projection,
        beta_normal_condition=cond,
        regularized=regularized,
        regularization_lambda=reg,
        weighted_orthogonality_error=weighted_orthogonality_error,
        prior_cost=prior_cost,
        reduced_cost=reduced_cost,
    )


def joint_linear_beta_solution(
    a: np.ndarray,
    b_matrix: np.ndarray,
    weights: np.ndarray,
    profile: BetaPriorProfile,
) -> np.ndarray:
    beta, _cond, _regularized, _reg = solve_beta(a, b_matrix, weights, profile)
    return beta


def projection_diagnostics_dict(result: ProjectionResult) -> dict[str, Any]:
    beta_full = result.beta if result.beta.size == 2 else expand_beta(result.beta, "b0_only")
    return {
        "beta0_estimated_mps": float(beta_full[0]) if beta_full.size else 0.0,
        "beta_dot_estimated_mps2": float(beta_full[1]) if beta_full.size > 1 else 0.0,
        "beta_normal_condition_number": float(result.beta_normal_condition),
        "projection_weighted_orthogonality_error": float(result.weighted_orthogonality_error),
        "beta_regularized": bool(result.regularized),
        "beta_regularization_lambda": float(result.regularization_lambda),
        "beta_prior_cost": float(result.prior_cost),
        "reduced_cost": float(result.reduced_cost),
    }
