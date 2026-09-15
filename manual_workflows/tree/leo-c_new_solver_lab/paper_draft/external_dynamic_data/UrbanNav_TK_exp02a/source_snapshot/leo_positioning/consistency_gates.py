"""State consistency diagnostics across initial guesses for MA-BGTR-v4."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _median_pairwise(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)] if arr.ndim == 2 else arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    distances: list[float] = []
    for i, j in combinations(range(len(arr)), 2):
        distances.append(float(np.linalg.norm(arr[i] - arr[j])) if arr.ndim == 2 else abs(float(arr[i] - arr[j])))
    return float(np.median(distances)) if distances else 0.0


def compute_consistency_diagnostics(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (
            str(row.get("dataset_type")),
            str(row.get("scenario")),
            str(row.get("candidate_model")),
            str(row.get("beta_prior_profile")),
        )
        groups.setdefault(key, []).append(row)

    out: list[dict] = []
    for (dataset_type, scenario, model, profile), group in groups.items():
        pos = np.array([[float(r.get("final_ecef_x_m", np.nan)), float(r.get("final_ecef_y_m", np.nan)), float(r.get("final_ecef_z_m", np.nan))] for r in group], dtype=float)
        vel = np.array([[float(r.get("estimated_vx_mps", np.nan)), float(r.get("estimated_vy_mps", np.nan)), float(r.get("estimated_vz_mps", np.nan))] for r in group], dtype=float)
        beta = np.array([float(r.get("beta0_estimated_mps", np.nan)) for r in group], dtype=float)
        bdot = np.array([float(r.get("beta_dot_estimated_mps2", np.nan)) for r in group], dtype=float)
        residual = np.array([float(r.get("full_residual_rmse_mps", r.get("residual_rmse_mps", np.nan))) for r in group], dtype=float)
        quality_fraction = float(np.mean([bool(r.get("quality_pass", False)) for r in group])) if group else 0.0
        pos_spread = _median_pairwise(pos)
        vel_spread = _median_pairwise(vel)
        beta_spread = _median_pairwise(beta)
        bdot_spread = _median_pairwise(bdot)
        residual_spread = _median_pairwise(residual)
        reasons: list[str] = []
        pos_limit = 500.0 if dataset_type == "real" else 1000.0
        vel_limit = 2.0 if dataset_type == "real" else 20.0
        if pos_spread >= pos_limit:
            reasons.append("position_spread_high")
        if vel_spread >= vel_limit:
            reasons.append("velocity_spread_high")
        if bdot_spread >= 0.05:
            reasons.append("bdot_spread_high")
        if quality_fraction < 0.8:
            reasons.append("quality_fraction_low")
        out.append(
            {
                "dataset_type": dataset_type,
                "scenario": scenario,
                "candidate_model": model,
                "beta_prior_profile": profile,
                "position_solution_spread_m": pos_spread,
                "velocity_solution_spread_mps": vel_spread,
                "bias_solution_spread_mps": beta_spread,
                "bdot_solution_spread_mps2": bdot_spread,
                "residual_spread_mps": residual_spread,
                "quality_pass_fraction": quality_fraction,
                "consistency_gate_pass": len(reasons) == 0,
                "consistency_reason": ";".join(reasons),
            }
        )
    return out


def apply_consistency_gates(rows: list[dict], diagnostics: list[dict]) -> list[dict]:
    index = {
        (d["dataset_type"], d["scenario"], d["candidate_model"], d["beta_prior_profile"]): d
        for d in diagnostics
    }
    out: list[dict] = []
    for row in rows:
        key = (
            str(row.get("dataset_type")),
            str(row.get("scenario")),
            str(row.get("candidate_model")),
            str(row.get("beta_prior_profile")),
        )
        diag = index.get(key, {})
        penalty = 0.0
        if not bool(diag.get("consistency_gate_pass", True)):
            penalty += 5.0
        penalty += min(float(diag.get("position_solution_spread_m", 0.0)) / (500.0 if key[0] == "real" else 1000.0), 10.0)
        penalty += min(float(diag.get("velocity_solution_spread_mps", 0.0)) / (2.0 if key[0] == "real" else 20.0), 10.0)
        updated = dict(row)
        updated.update(
            {
                "consistency_gate_pass": bool(diag.get("consistency_gate_pass", True)),
                "consistency_penalty": float(penalty),
                "consistency_reason": diag.get("consistency_reason", ""),
                "position_solution_spread_m": diag.get("position_solution_spread_m", np.nan),
                "velocity_solution_spread_mps": diag.get("velocity_solution_spread_mps", np.nan),
                "bias_solution_spread_mps": diag.get("bias_solution_spread_mps", np.nan),
                "bdot_solution_spread_mps2": diag.get("bdot_solution_spread_mps2", np.nan),
            }
        )
        out.append(updated)
    return out
