"""Gated BE-GTR projection-mode selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .be_gtr_solver import BeGtrResult, solve_be_gtr
from .projection_diagnostics import ProjectionDiagnostics, compute_projection_diagnostics
from .trajectory_models import FULL_CTD_CONFIG, pack_state
from .trajectory_solvers import TrajectorySolverResult, solve_ctd_lm
from .variable_projection import BetaPriorProfile


@dataclass
class GatingDecision:
    chosen_projection_mode: str
    gating_reason: str
    full_diagnostics: ProjectionDiagnostics
    b0_diagnostics: ProjectionDiagnostics


@dataclass
class GatedBeGtrResult:
    decision: GatingDecision
    be_gtr_result: BeGtrResult | None
    ctd_result: TrajectorySolverResult | None


def choose_projection_mode(
    full_diag: ProjectionDiagnostics,
    b0_diag: ProjectionDiagnostics,
    full_retention_threshold: float = 0.50,
    full_rank_loss_threshold: int = 2,
    b0_retention_threshold: float = 0.70,
) -> tuple[str, str]:
    if full_diag.retention_trace_ratio < full_retention_threshold or full_diag.rank_loss >= full_rank_loss_threshold:
        if b0_diag.retention_trace_ratio < b0_retention_threshold:
            return (
                "none",
                "full_projection_forbidden_by_retention_or_rank_loss; b0_projection_forbidden_by_retention",
            )
        return "b0_only", "full_projection_forbidden_by_retention_or_rank_loss"
    return "full", "full_projection_allowed"


def solve_gated_be_gtr(
    initial_x: np.ndarray,
    time_s: np.ndarray,
    sat_pos_m: np.ndarray,
    sat_vel_mps: np.ndarray,
    meas_mps: np.ndarray,
    t0_s: float,
    beta_profile: BetaPriorProfile,
    robust: bool = False,
    max_iter_per_scale: int = 20,
    include_prior_cost: bool = True,
) -> GatedBeGtrResult:
    weights = np.ones(len(meas_mps), dtype=float)
    full_diag = compute_projection_diagnostics(
        initial_x, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, weights, "full", beta_profile.name
    )
    b0_diag = compute_projection_diagnostics(
        initial_x, time_s, sat_pos_m, sat_vel_mps, meas_mps, t0_s, weights, "b0_only", beta_profile.name
    )
    chosen, reason = choose_projection_mode(full_diag, b0_diag)
    decision = GatingDecision(chosen, reason, full_diag, b0_diag)
    if chosen == "none":
        theta0 = pack_state(initial_x[:3], initial_x[3:6], 0.0, 0.0, FULL_CTD_CONFIG)
        ctd = solve_ctd_lm(
            theta0,
            time_s,
            sat_pos_m,
            sat_vel_mps,
            meas_mps,
            t0_s,
            FULL_CTD_CONFIG,
            robust=robust,
            max_iter=80,
        )
        return GatedBeGtrResult(decision=decision, be_gtr_result=None, ctd_result=ctd)
    be = solve_be_gtr(
        initial_x,
        time_s,
        sat_pos_m,
        sat_vel_mps,
        meas_mps,
        t0_s,
        beta_profile,
        robust=robust,
        max_iter_per_scale=max_iter_per_scale,
        projection_mode=chosen,
        include_prior_cost=include_prior_cost,
    )
    return GatedBeGtrResult(decision=decision, be_gtr_result=be, ctd_result=None)
