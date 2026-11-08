from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2


PROJECT = Path(r"C:\Users\32377\Desktop\leo-c_new_solver_lab")
OUT = Path(r"C:\Users\32377\Desktop\leo-c\tmp\pdfs\leo_doppler_theory_note")
OUT.mkdir(parents=True, exist_ok=True)

DATA04 = PROJECT / "paper_draft/geometry_development/PAPER_DATA04_RICH_GEOMETRY_DEV"
DATA07 = PROJECT / "paper_draft/independent_validation/PAPER_DATA07_SPATIAL_HOLDOUT_FREEZE"
MONITOR = PROJECT / "paper_draft/monitor_development/PAPER_MONITOR01_R3_DEVELOPMENT_EXPANSION/raw_candidate_results/PAPER_MONITOR01_R3_G120_FL120_RAW_RESULTS.csv"

ALPHA = 0.05
CHI2_3_95 = float(chi2.ppf(1.0 - ALPHA, 3))
SIGMA = 0.30
DISPLACEMENTS_M = np.array([25.0, 50.0, 100.0, 132.01, 523.20, 1000.0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_matrix(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def required_noncentrality(df: int, target_power: float, alpha: float = ALPHA) -> float:
    threshold = float(chi2.ppf(1.0 - alpha, df))

    def objective(noncentrality: float) -> float:
        return float(ncx2.sf(threshold, df, noncentrality) - target_power)

    upper = 1.0
    while objective(upper) < 0.0:
        upper *= 2.0
        if upper > 1.0e9:
            raise RuntimeError("failed to bracket noncentrality")
    return float(brentq(objective, 0.0, upper))


LAMBDA_REQ_LRT_80 = required_noncentrality(3, 0.80)
LAMBDA_REQ_LRT_90 = required_noncentrality(3, 0.90)


def residualize(nuisance: np.ndarray, values: np.ndarray) -> np.ndarray:
    if nuisance.size == 0:
        return values.copy()
    coefficients = np.linalg.lstsq(nuisance, values, rcond=1.0e-12)[0]
    return values - nuisance @ coefficients


def geometry_arrays(path: Path) -> dict[str, np.ndarray | float | int]:
    frame = pd.read_csv(path)
    sat_pos = frame[["satellite_ecef_x_m", "satellite_ecef_y_m", "satellite_ecef_z_m"]].to_numpy(float)
    sat_vel = frame[["satellite_ecef_vx_mps", "satellite_ecef_vy_mps", "satellite_ecef_vz_mps"]].to_numpy(float)
    receiver = frame[["receiver_anchor_ecef_x_m", "receiver_anchor_ecef_y_m", "receiver_anchor_ecef_z_m"]].iloc[0].to_numpy(float)
    relative_time = frame["relative_time_s"].to_numpy(float)
    duration_s = float(relative_time.max() - relative_time.min())
    tau = relative_time - 0.5 * (relative_time.min() + relative_time.max())
    relative = sat_pos - receiver[None, :]
    ranges = np.linalg.norm(relative, axis=1)
    unit = relative / ranges[:, None]
    projected_velocity = sat_vel - unit * np.sum(unit * sat_vel, axis=1)[:, None]
    jacobian_position = -projected_velocity / ranges[:, None]
    jacobian_velocity = -unit + tau[:, None] * jacobian_position
    jacobian_nuisance = np.column_stack([jacobian_velocity, np.ones(len(tau)), tau])
    return {
        "sat_pos": sat_pos,
        "sat_vel": sat_vel,
        "receiver": receiver,
        "ranges": ranges,
        "unit": unit,
        "tau": tau,
        "jp": jacobian_position,
        "jn": jacobian_nuisance,
        "rows": len(frame),
        "duration_s": duration_s,
    }


def schur_from_arrays(jp: np.ndarray, jn: np.ndarray, sigma: float = SIGMA) -> tuple[np.ndarray, np.ndarray, int]:
    position = jp / sigma
    nuisance = jn / sigma
    projected_position = residualize(nuisance, position)
    schur = projected_position.T @ projected_position
    schur = 0.5 * (schur + schur.T)
    return schur, projected_position, int(np.linalg.matrix_rank(nuisance))


def position_only_information(jp: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    weighted = jp / sigma
    information = weighted.T @ weighted
    return 0.5 * (information + information.T)


def normalized_information(information: np.ndarray) -> np.ndarray:
    maximum = float(np.linalg.eigvalsh(information)[-1])
    if maximum <= 0.0:
        return np.zeros_like(information)
    return information / maximum


def visible_plane_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    _, vectors_first = np.linalg.eigh(first)
    _, vectors_second = np.linalg.eigh(second)
    angles = subspace_angles(vectors_first[:, 1:], vectors_second[:, 1:])
    return float(np.degrees(np.max(angles)))


def nonlinear_metrics(arrays: dict[str, np.ndarray | float | int], schur: np.ndarray, displacement_m: float) -> dict[str, float]:
    sat_pos = np.asarray(arrays["sat_pos"])
    sat_vel = np.asarray(arrays["sat_vel"])
    receiver = np.asarray(arrays["receiver"])
    ranges = np.asarray(arrays["ranges"])
    unit = np.asarray(arrays["unit"])
    jp = np.asarray(arrays["jp"])
    jn = np.asarray(arrays["jn"])
    eigenvalues, eigenvectors = np.linalg.eigh(schur)
    weak = eigenvectors[:, 0]
    delta = displacement_m * weak
    shifted_relative = sat_pos - (receiver + delta)[None, :]
    shifted_ranges = np.linalg.norm(shifted_relative, axis=1)
    shifted_unit = shifted_relative / shifted_ranges[:, None]
    exact_change = np.sum(shifted_unit * sat_vel, axis=1) - np.sum(unit * sat_vel, axis=1)
    linear_change = jp @ delta
    nuisance = jn / SIGMA
    projected_exact = residualize(nuisance, exact_change / SIGMA)
    projected_linear = residualize(nuisance, linear_change / SIGMA)
    projected_remainder = projected_exact - projected_linear
    linear_norm = float(np.linalg.norm(projected_linear))
    exact_norm = float(np.linalg.norm(projected_exact))
    remainder_norm = float(np.linalg.norm(projected_remainder))
    safe_range = ranges - displacement_m
    if np.any(safe_range <= 0.0):
        analytic_bound = float("inf")
    else:
        speed = np.linalg.norm(sat_vel, axis=1)
        row_bound_mps = 1.5 * speed * displacement_m**2 / safe_range**2
        analytic_bound = float(np.linalg.norm(row_bound_mps / SIGMA))
    return {
        "linear_norm": linear_norm,
        "exact_norm": exact_norm,
        "remainder_norm": remainder_norm,
        "actual_remainder_ratio": remainder_norm / linear_norm if linear_norm > 0.0 else float("inf"),
        "analytic_bound_norm": analytic_bound,
        "analytic_bound_ratio": analytic_bound / linear_norm if linear_norm > 0.0 else float("inf"),
        "lambda_linear": linear_norm**2,
        "lambda_exact": exact_norm**2,
        "lambda_relative_error": (exact_norm**2 - linear_norm**2) / linear_norm**2 if linear_norm > 0.0 else float("nan"),
        "minimum_range_m": float(np.min(ranges)),
        "maximum_satellite_speed_mps": float(np.max(np.linalg.norm(sat_vel, axis=1))),
    }


def nonlinear_radius_for_ratio(arrays: dict[str, np.ndarray | float | int], schur: np.ndarray, target_ratio: float) -> float:
    minimum_range = float(np.min(np.asarray(arrays["ranges"])))
    upper = min(100_000.0, 0.20 * minimum_range)

    def objective(radius: float) -> float:
        return nonlinear_metrics(arrays, schur, radius)["analytic_bound_ratio"] - target_ratio

    if objective(upper) <= 0.0:
        return upper
    return float(brentq(objective, 1.0e-6, upper))


def calculate_window(record: dict[str, object], geometry_path: Path) -> dict[str, object]:
    arrays = geometry_arrays(geometry_path)
    schur, _, nuisance_rank = schur_from_arrays(np.asarray(arrays["jp"]), np.asarray(arrays["jn"]))
    position_only = position_only_information(np.asarray(arrays["jp"]))
    eigenvalues, eigenvectors = np.linalg.eigh(schur)
    normalized_full = normalized_information(schur)
    normalized_position_only = normalized_information(position_only)
    joint_normalized = normalized_full + normalized_position_only
    row: dict[str, object] = {
        **record,
        "geometry_path": str(geometry_path),
        "geometry_sha256": sha256(geometry_path),
        "observation_rows": int(arrays["rows"]),
        "duration_reconstructed_s": float(arrays["duration_s"]),
        "nuisance_rank": nuisance_rank,
        "df_global": int(arrays["rows"]) - nuisance_rank,
        "schur_json": json.dumps(schur.tolist(), separators=(",", ":")),
        "eig_min": float(eigenvalues[0]),
        "eig_mid": float(eigenvalues[1]),
        "eig_max": float(eigenvalues[2]),
        "weak_vector_json": json.dumps(eigenvectors[:, 0].tolist(), separators=(",", ":")),
        "information_condition": float(eigenvalues[2] / eigenvalues[0]),
        "position_only_full_visible_plane_angle_deg": visible_plane_angle_deg(position_only, schur),
        "normalized_floor_position_only": float(np.linalg.eigvalsh(normalized_position_only)[0]),
        "normalized_floor_full": float(np.linalg.eigvalsh(normalized_full)[0]),
        "normalized_joint_floor": float(np.linalg.eigvalsh(joint_normalized)[0]),
    }
    lambda_req_global_80 = required_noncentrality(int(row["df_global"]), 0.80)
    lambda_req_global_90 = required_noncentrality(int(row["df_global"]), 0.90)
    row["required_radius_lrt80_weak_m"] = math.sqrt(LAMBDA_REQ_LRT_80 / float(eigenvalues[0]))
    row["required_radius_lrt90_weak_m"] = math.sqrt(LAMBDA_REQ_LRT_90 / float(eigenvalues[0]))
    row["required_radius_global80_weak_m"] = math.sqrt(lambda_req_global_80 / float(eigenvalues[0]))
    row["required_radius_global90_weak_m"] = math.sqrt(lambda_req_global_90 / float(eigenvalues[0]))
    row["analytic_local_radius_10pct_m"] = nonlinear_radius_for_ratio(arrays, schur, 0.10)
    row["analytic_local_radius_50pct_m"] = nonlinear_radius_for_ratio(arrays, schur, 0.50)
    threshold_lrt = CHI2_3_95
    threshold_global = float(chi2.ppf(1.0 - ALPHA, int(row["df_global"])))
    for displacement in DISPLACEMENTS_M:
        label = f"{displacement:g}".replace(".", "p")
        for direction, eigenvalue in zip(("weak", "mid", "strong"), eigenvalues):
            noncentrality = float(displacement**2 * eigenvalue)
            row[f"lambda_{direction}_{label}m"] = noncentrality
            row[f"power_lrt_{direction}_{label}m"] = float(ncx2.sf(threshold_lrt, 3, noncentrality))
            row[f"power_global_{direction}_{label}m"] = float(ncx2.sf(threshold_global, int(row["df_global"]), noncentrality))
        nonlinear = nonlinear_metrics(arrays, schur, float(displacement))
        for key, value in nonlinear.items():
            row[f"nonlinear_{key}_{label}m"] = value
    return row


def selected_geometries() -> list[tuple[dict[str, object], Path]]:
    selected: list[tuple[dict[str, object], Path]] = []
    manifest04_path = DATA04 / "frozen_geometry/PAPER_DATA04_GEOMETRY_MANIFEST.csv"
    manifest04 = pd.read_csv(manifest04_path)
    for _, item in manifest04.iterrows():
        record = {
            "dataset": "DATA04",
            "sequence_id": str(item["sequence_id"]),
            "window_kind": f"D{int(item['duration_s'])}",
            "duration_s": int(item["duration_s"]),
            "satellite_tier": str(item["minimum_satellite_tier"]),
            "information_stratum": str(item["information_tier"]),
            "anchor_id": str(item["anchor_id"]),
            "center_utc": str(item["center_utc"]),
            "manifest_sha256": sha256(manifest04_path),
        }
        selected.append((record, Path(str(item["geometry_file"]))))
    manifest07_path = DATA07 / "frozen_geometry/PAPER_DATA07_GEOMETRY_MANIFEST.csv"
    manifest07 = pd.read_csv(manifest07_path)
    manifest07 = manifest07[manifest07["window_kind"] == "D120"].copy()
    for _, item in manifest07.iterrows():
        record = {
            "dataset": "DATA07",
            "sequence_id": str(item["sequence_id"]),
            "window_kind": "D120",
            "duration_s": 120,
            "satellite_tier": str(item["minimum_satellite_tier"]),
            "information_stratum": str(item["information_tercile"]),
            "anchor_id": str(item["anchor_id"]),
            "center_utc": str(item["center_utc"]),
            "manifest_sha256": sha256(manifest07_path),
        }
        selected.append((record, Path(str(item["geometry_file"]))))
    return selected


def crosscheck_information(windows: pd.DataFrame) -> dict[str, float | int]:
    checks: list[float] = []
    info04 = pd.read_csv(DATA04 / "outputs/PAPER_DATA04_INFORMATION_ENVELOPE.csv")
    manifest04 = pd.read_csv(DATA04 / "frozen_geometry/PAPER_DATA04_GEOMETRY_MANIFEST.csv")
    lookup04 = manifest04.set_index("sequence_id")
    for _, row in windows[windows["dataset"] == "DATA04"].iterrows():
        manifest = lookup04.loc[row["sequence_id"]]
        match = info04[
            (info04["tier_sequence_id"] == manifest["tier_sequence_id"])
            & (info04["state_model"] == "I3_full_ctd")
            & np.isclose(info04["noise_sigma_mps"].astype(float), SIGMA)
            & (info04["duration_s"].astype(int) == int(row["duration_s"]))
        ]
        if len(match) != 1:
            continue
        stored = parse_matrix(match.iloc[0]["schur_position_information_json"])
        calculated = parse_matrix(row["schur_json"])
        checks.append(float(np.max(np.abs(stored - calculated))))
    info07 = pd.read_csv(DATA07 / "outputs/PAPER_DATA07_INFORMATION_ENVELOPE.csv")
    manifest07 = pd.read_csv(DATA07 / "frozen_geometry/PAPER_DATA07_GEOMETRY_MANIFEST.csv")
    lookup07 = manifest07[manifest07["window_kind"] == "D120"].set_index("sequence_id")
    for _, row in windows[windows["dataset"] == "DATA07"].iterrows():
        manifest = lookup07.loc[row["sequence_id"]]
        match = info07[
            (info07["parent_sequence_id"] == manifest["parent_sequence_id"])
            & (info07["anchor_id"] == manifest["anchor_id"])
            & (info07["center_utc"] == manifest["center_utc"])
            & (info07["duration_s"].astype(int) == 120)
        ]
        if len(match) != 1:
            continue
        stored = parse_matrix(match.iloc[0]["schur_position_information_json"])
        calculated = parse_matrix(row["schur_json"])
        checks.append(float(np.max(np.abs(stored - calculated))))
    return {
        "matrices_checked": len(checks),
        "maximum_absolute_matrix_difference": float(max(checks)) if checks else float("nan"),
        "median_absolute_matrix_difference": float(np.median(checks)) if checks else float("nan"),
    }


def half_window_metrics(record: dict[str, object], path: Path) -> dict[str, object] | None:
    arrays = geometry_arrays(path)
    tau = np.asarray(arrays["tau"])
    if float(arrays["duration_s"]) < 119.0:
        return None
    left = tau < 0.0
    right = ~left
    jp = np.asarray(arrays["jp"])
    jn = np.asarray(arrays["jn"])
    left_s, _, _ = schur_from_arrays(jp[left], jn[left])
    right_s, _, _ = schur_from_arrays(jp[right], jn[right])
    full_s, _, _ = schur_from_arrays(jp, jn)
    left_eig = np.linalg.eigvalsh(left_s)
    right_eig = np.linalg.eigvalsh(right_s)
    sum_s = left_s + right_s
    sum_eig = np.linalg.eigvalsh(sum_s)
    full_eig = np.linalg.eigvalsh(full_s)
    angle = visible_plane_angle_deg(left_s, right_s)
    radius_left = math.sqrt(LAMBDA_REQ_LRT_80 / left_eig[0])
    radius_right = math.sqrt(LAMBDA_REQ_LRT_80 / right_eig[0])
    radius_sum = math.sqrt(LAMBDA_REQ_LRT_80 / sum_eig[0])
    radius_shared = math.sqrt(LAMBDA_REQ_LRT_80 / full_eig[0])
    return {
        "dataset": record["dataset"],
        "sequence_id": record["sequence_id"],
        "satellite_tier": record["satellite_tier"],
        "information_stratum": record["information_stratum"],
        "left_rows": int(np.sum(left)),
        "right_rows": int(np.sum(right)),
        "visible_plane_angle_deg": angle,
        "left_floor": float(left_eig[0]),
        "right_floor": float(right_eig[0]),
        "separate_nuisance_joint_floor": float(sum_eig[0]),
        "shared_nuisance_full_floor": float(full_eig[0]),
        "shared_over_separate_floor_ratio": float(full_eig[0] / sum_eig[0]),
        "required_radius_lrt80_left_m": radius_left,
        "required_radius_lrt80_right_m": radius_right,
        "required_radius_lrt80_separate_joint_m": radius_sum,
        "required_radius_lrt80_shared_full_m": radius_shared,
        "joint_radius_gain_vs_better_half": float(min(radius_left, radius_right) / radius_sum),
    }


def candidate_pair_metrics() -> pd.DataFrame:
    raw = pd.read_csv(MONITOR)
    raw = raw[(raw["success"].astype(str).str.lower() == "true") & raw["schur_position_covariance_m2_json"].notna()].copy()
    output: list[dict[str, object]] = []
    for observation_id, group in raw.groupby("observation_id", sort=True):
        if set(group["method"]) != {"G120-H-v0.1", "FL120-PCV-H-v0.1"} or len(group) != 2:
            continue
        g = group[group["method"] == "G120-H-v0.1"].iloc[0]
        f = group[group["method"] == "FL120-PCV-H-v0.1"].iloc[0]
        cov_g = parse_matrix(g["schur_position_covariance_m2_json"])
        cov_f = parse_matrix(f["schur_position_covariance_m2_json"])
        info_g = np.linalg.pinv(cov_g, rcond=1.0e-12)
        info_f = np.linalg.pinv(cov_f, rcond=1.0e-12)
        pos_g = np.asarray(json.loads(g["output_ecef_position_m_json"]), dtype=float)
        pos_f = np.asarray(json.loads(f["output_ecef_position_m_json"]), dtype=float)
        difference = pos_g - pos_f
        covariance_sum = cov_g + cov_f
        compatibility = float(difference @ np.linalg.pinv(covariance_sum, rcond=1.0e-12) @ difference)
        normalized_sum = normalized_information(info_g) + normalized_information(info_f)
        output.append({
            "observation_id": observation_id,
            "position_disagreement_m": float(np.linalg.norm(difference)),
            "independence_reference_chi2": compatibility,
            "independence_reference_pass_95": bool(compatibility <= CHI2_3_95),
            "visible_plane_angle_deg": visible_plane_angle_deg(info_g, info_f),
            "normalized_floor_g120": float(np.linalg.eigvalsh(normalized_information(info_g))[0]),
            "normalized_floor_fl120": float(np.linalg.eigvalsh(normalized_information(info_f))[0]),
            "normalized_joint_floor": float(np.linalg.eigvalsh(normalized_sum)[0]),
        })
    return pd.DataFrame(output)


def summarize_numeric(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        output[column] = {
            "n": int(len(values)),
            "min": float(np.min(values)),
            "p05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }
    return output


def make_plots(windows: pd.DataFrame, halves: pd.DataFrame, pairs: pd.DataFrame) -> None:
    displacement_grid = np.linspace(0.0, 1200.0, 241)
    groups = {
        "DATA04 frozen (48)": windows[windows["dataset"] == "DATA04"],
        "DATA04 D120 (12)": windows[(windows["dataset"] == "DATA04") & (windows["duration_s"] == 120)],
        "DATA07 D120 (36)": windows[windows["dataset"] == "DATA07"],
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for label, group in groups.items():
        floors = group["eig_min"].to_numpy(float)
        powers = np.array([[ncx2.sf(CHI2_3_95, 3, radius**2 * floor) for floor in floors] for radius in displacement_grid])
        ax.plot(displacement_grid, np.median(powers, axis=1), label=label)
        if label == "DATA07 D120 (36)":
            ax.fill_between(displacement_grid, np.quantile(powers, 0.10, axis=1), np.quantile(powers, 0.90, axis=1), alpha=0.16)
    ax.axhline(0.80, color="black", linestyle="--", linewidth=1.0, label="80% power")
    ax.set(xlabel="Position displacement along weakest Schur direction (m)", ylabel="3-dof LRT detection power", ylim=(0, 1.01), xlim=(0, 1200))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "power_curve.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    axes[0].scatter(windows["nonlinear_actual_remainder_ratio_1000m"] * 100.0, windows["nonlinear_analytic_bound_ratio_1000m"] * 100.0, c=(windows["dataset"] == "DATA07").astype(int), cmap="coolwarm", alpha=0.8)
    axes[0].set(xlabel="Exact projected remainder / linear signal (%)", ylabel="Analytic bound / linear signal (%)")
    axes[0].grid(alpha=0.25)
    axes[1].scatter(halves["visible_plane_angle_deg"], halves["joint_radius_gain_vs_better_half"], c=(halves["dataset"] == "DATA07").astype(int), cmap="coolwarm", alpha=0.8)
    axes[1].set(xlabel="Half-window visible-plane angle (deg)", ylabel="80% radius gain: better half / joint")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "nonlinear_joint.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.scatter(pairs["visible_plane_angle_deg"], pairs["position_disagreement_m"], alpha=0.55, s=14)
    ax.set(xlabel="G120-FL120 visible-plane angle (deg)", ylabel="Output-position disagreement (m)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "candidate_pair.png", dpi=220)
    plt.close(fig)


def main() -> None:
    records = selected_geometries()
    window_rows: list[dict[str, object]] = []
    half_rows: list[dict[str, object]] = []
    for record, path in records:
        window_rows.append(calculate_window(record, path))
        half = half_window_metrics(record, path)
        if half is not None:
            half_rows.append(half)
    windows = pd.DataFrame(window_rows)
    halves = pd.DataFrame(half_rows)
    pairs = candidate_pair_metrics()
    windows.to_csv(OUT / "window_numeric_metrics.csv", index=False)
    halves.to_csv(OUT / "half_window_joint_metrics.csv", index=False)
    pairs.to_csv(OUT / "g120_fl120_pair_metrics.csv", index=False)

    group_masks = {
        "DATA04_all": windows["dataset"] == "DATA04",
        "DATA04_D120": (windows["dataset"] == "DATA04") & (windows["duration_s"] == 120),
        "DATA07_D120": windows["dataset"] == "DATA07",
        "all_D120": windows["duration_s"] == 120,
    }
    summary_columns = [
        "eig_min", "eig_mid", "eig_max", "information_condition",
        "required_radius_lrt80_weak_m", "required_radius_lrt90_weak_m",
        "required_radius_global80_weak_m", "required_radius_global90_weak_m",
        "power_lrt_weak_132p01m", "power_lrt_weak_523p2m", "power_lrt_weak_1000m",
        "power_global_weak_132p01m", "power_global_weak_523p2m", "power_global_weak_1000m",
        "nonlinear_actual_remainder_ratio_523p2m", "nonlinear_actual_remainder_ratio_1000m",
        "nonlinear_analytic_bound_ratio_523p2m", "nonlinear_analytic_bound_ratio_1000m",
        "nonlinear_lambda_relative_error_1000m",
        "analytic_local_radius_10pct_m", "analytic_local_radius_50pct_m",
        "position_only_full_visible_plane_angle_deg", "normalized_joint_floor",
    ]
    group_summaries = {name: summarize_numeric(windows[mask], summary_columns) for name, mask in group_masks.items()}
    half_summary = summarize_numeric(halves, [
        "visible_plane_angle_deg", "left_floor", "right_floor", "separate_nuisance_joint_floor",
        "shared_nuisance_full_floor", "shared_over_separate_floor_ratio",
        "required_radius_lrt80_left_m", "required_radius_lrt80_right_m",
        "required_radius_lrt80_separate_joint_m", "required_radius_lrt80_shared_full_m",
        "joint_radius_gain_vs_better_half",
    ])
    pair_summary = summarize_numeric(pairs, [
        "position_disagreement_m", "independence_reference_chi2", "visible_plane_angle_deg",
        "normalized_floor_g120", "normalized_floor_fl120", "normalized_joint_floor",
    ])
    pair_summary["independence_reference_pass_95_fraction"] = {
        "n": int(len(pairs)),
        "value": float(pairs["independence_reference_pass_95"].mean()),
    }
    result = {
        "calculation_identity": {
            "project_root": str(PROJECT),
            "noise_sigma_mps": SIGMA,
            "alpha": ALPHA,
            "chi2_3_95": CHI2_3_95,
            "lambda_required_lrt_80": LAMBDA_REQ_LRT_80,
            "lambda_required_lrt_90": LAMBDA_REQ_LRT_90,
            "data04_manifest_sha256": sha256(DATA04 / "frozen_geometry/PAPER_DATA04_GEOMETRY_MANIFEST.csv"),
            "data07_manifest_sha256": sha256(DATA07 / "frozen_geometry/PAPER_DATA07_GEOMETRY_MANIFEST.csv"),
            "monitor_raw_sha256": sha256(MONITOR),
        },
        "counts": {
            "frozen_geometry_windows": int(len(windows)),
            "data04_windows": int((windows["dataset"] == "DATA04").sum()),
            "data04_d120_windows": int(((windows["dataset"] == "DATA04") & (windows["duration_s"] == 120)).sum()),
            "data07_d120_windows": int((windows["dataset"] == "DATA07").sum()),
            "independent_half_window_pairs": int(len(halves)),
            "g120_fl120_same_data_pairs": int(len(pairs)),
        },
        "stored_information_crosscheck": crosscheck_information(windows),
        "group_summaries": group_summaries,
        "half_window_summary": half_summary,
        "candidate_pair_summary": pair_summary,
    }
    (OUT / "numeric_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    make_plots(windows, halves, pairs)
    print(json.dumps({
        "status": "complete",
        "counts": result["counts"],
        "crosscheck": result["stored_information_crosscheck"],
        "output": str(OUT / "numeric_results.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
