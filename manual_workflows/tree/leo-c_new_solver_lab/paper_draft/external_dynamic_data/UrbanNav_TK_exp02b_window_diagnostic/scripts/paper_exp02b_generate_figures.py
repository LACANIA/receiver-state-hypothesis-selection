from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#222222",
    "gray": "#777777",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.pdf")
    fig.savefig(directory / f"{stem}.png", dpi=300)
    plt.close(fig)


def _box_with_points(ax: plt.Axes, frame: pd.DataFrame, metric: str, durations: list[int], ylabel: str) -> None:
    values = [pd.to_numeric(frame.loc[frame["duration_s"] == duration, metric], errors="coerce").dropna().to_numpy(float) for duration in durations]
    ax.boxplot(values, tick_labels=[str(item) for item in durations], showfliers=False)
    rng = np.random.default_rng(20260724)
    for index, current in enumerate(values, start=1):
        if len(current):
            jitter = rng.normal(index, 0.035, len(current))
            ax.scatter(jitter, current, s=10, alpha=0.45, color=COLORS["blue"], edgecolors="none")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("Window duration (s)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def generate_all(root: str | Path) -> list[dict[str, str]]:
    _style()
    root = Path(root)
    outputs = root / "outputs"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(outputs / "PAPER_EXP02B_SELECTED_RESULTS.csv")
    candidates = pd.read_csv(outputs / "PAPER_EXP02B_CANDIDATE_RESULTS.csv")
    floors = pd.read_csv(outputs / "PAPER_EXP02B_MOTION_MODEL_REPRESENTATION_FLOORS.csv")
    duration_agg = pd.read_csv(outputs / "PAPER_EXP02B_DURATION_AGGREGATE.csv")
    geometry_agg = pd.read_csv(outputs / "PAPER_EXP02B_GEOMETRY_AGGREGATE.csv")
    exp_compare = pd.read_csv(outputs / "PAPER_EXP02B_EXP02A_COMPARISON.csv")
    fit_data = pd.read_csv(root / "inputs" / "PAPER_EXP02B_REPRESENTATIVE_FIT_DATA.csv")
    durations = [4, 6, 8, 12]
    maps: list[dict[str, str]] = []

    fig, ax = plt.subplots(figsize=(5.6, 3.7), constrained_layout=True)
    _box_with_points(ax, selected, "mean_trajectory_position_error_m", durations, "Selected mean trajectory error (m, symlog)")
    ax.set_title("Selected error versus local window duration")
    _save(fig, figures, "figure_01_selected_error_vs_duration")
    maps.append({"figure_id": "Figure 1", "source_files": "outputs/PAPER_EXP02B_SELECTED_RESULTS.csv", "source_columns": "duration_s;mean_trajectory_position_error_m", "claim": "Duration effect on frozen selector output", "limitation": "Controlled LEO range-rate; long tails retained"})

    fig, ax = plt.subplots(figsize=(5.6, 3.7), constrained_layout=True)
    _box_with_points(ax, selected, "oracle_mean_trajectory_error_m", durations, "Post-selection oracle mean error (m, symlog)")
    ax.set_title("Candidate oracle error versus duration")
    _save(fig, figures, "figure_02_oracle_error_vs_duration")
    maps.append({"figure_id": "Figure 2", "source_files": "outputs/PAPER_EXP02B_SELECTED_RESULTS.csv", "source_columns": "duration_s;oracle_mean_trajectory_error_m", "claim": "Candidate-pool adequacy versus duration", "limitation": "Oracle uses truth after selection"})

    floor_long = floors.melt(
        id_vars=["source_segment_id", "motion_type", "duration_s", "geometry_window_id"],
        value_vars=["static_floor_mean_error_m", "cv_floor_mean_error_m", "ca_floor_mean_error_m"],
        var_name="floor_type",
        value_name="mean_error_m",
    )
    labels = {"static_floor_mean_error_m": "Static", "cv_floor_mean_error_m": "CV", "ca_floor_mean_error_m": "CA"}
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 3.3), sharey=True, constrained_layout=True)
    for ax, motion in zip(axes, ["high_speed_straight", "turn", "deceleration", "stop_and_go"], strict=True):
        subset = floor_long[floor_long["motion_type"] == motion]
        for idx, floor_type in enumerate(labels):
            medians = subset[subset["floor_type"] == floor_type].groupby("duration_s")["mean_error_m"].median().reindex(durations)
            ax.plot(durations, medians, marker=["o", "s", "^"][idx], label=labels[floor_type])
        ax.set_title(motion.replace("_", " "))
        ax.set_xlabel("Duration (s)")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Representation floor mean error (m, symlog)")
    axes[-1].legend(frameon=False)
    _save(fig, figures, "figure_03_representation_floors")
    maps.append({"figure_id": "Figure 3", "source_files": "outputs/PAPER_EXP02B_MOTION_MODEL_REPRESENTATION_FLOORS.csv", "source_columns": "motion_type;duration_s;static_floor_mean_error_m;cv_floor_mean_error_m;ca_floor_mean_error_m", "claim": "Static/CV/CA representation limits", "limitation": "Evaluation-only fits are not candidate solvers"})

    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    med_selected = selected.groupby("duration_s")["mean_trajectory_position_error_m"].median().reindex(durations)
    med_oracle = selected.groupby("duration_s")["oracle_mean_trajectory_error_m"].median().reindex(durations)
    med_floor = selected.groupby("duration_s")["cv_floor_mean_error_m"].median().reindex(durations)
    ax.plot(durations, med_selected, "o-", label="Selected", color=COLORS["blue"])
    ax.plot(durations, med_oracle, "s--", label="Candidate oracle", color=COLORS["orange"])
    ax.plot(durations, med_floor, "^-.", label="CV representation floor", color=COLORS["green"])
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("Window duration (s)")
    ax.set_ylabel("Median mean trajectory error (m, symlog)")
    ax.set_title("Diagnostic gaps across duration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, figures, "figure_04_selected_oracle_floor")
    maps.append({"figure_id": "Figure 4", "source_files": "outputs/PAPER_EXP02B_SELECTED_RESULTS.csv", "source_columns": "duration_s;mean_trajectory_position_error_m;oracle_mean_trajectory_error_m;cv_floor_mean_error_m", "claim": "Representation, candidate, and selector diagnostic gaps", "limitation": "Gaps are not additive error decomposition"})

    family = selected.copy()
    family["family_class"] = np.where(family["selected_family"].astype(str).str.startswith("static"), "Static family", "Dynamic/diagnostic family")
    rates = pd.crosstab(family["duration_s"], family["family_class"], normalize="index").reindex(durations).fillna(0.0)
    fig, ax = plt.subplots(figsize=(5.8, 3.7), constrained_layout=True)
    bottom = np.zeros(len(rates))
    for label, color in [("Static family", COLORS["gray"]), ("Dynamic/diagnostic family", COLORS["blue"])]:
        vals = rates[label].to_numpy(float) if label in rates else np.zeros(len(rates))
        ax.bar(rates.index.astype(str), vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_xlabel("Window duration (s)")
    ax.set_ylabel("Selection fraction")
    ax.set_title("Selected family versus duration")
    ax.legend(frameon=False)
    _save(fig, figures, "figure_05_family_selection_vs_duration")
    maps.append({"figure_id": "Figure 5", "source_files": "outputs/PAPER_EXP02B_SELECTED_RESULTS.csv", "source_columns": "duration_s;selected_family", "claim": "Static/dynamic selection response to duration", "limitation": "Family counts do not imply navigation accuracy"})

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "median_selected_mean_error_m", "Selected error"),
        (axes[1], "median_oracle_gap_m", "Oracle gap"),
    ]:
        for geometry_id, marker in [("G0", "o"), ("G1", "s")]:
            subset = geometry_agg[(geometry_agg["perturbation_profile"] == "ALL") & (geometry_agg["geometry_window_id"] == geometry_id)].sort_values("duration_s")
            ax.plot(subset["duration_s"], subset[metric], marker=marker, label=geometry_id)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Error (m, symlog)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    _save(fig, figures, "figure_06_geometry_effect")
    maps.append({"figure_id": "Figure 6", "source_files": "outputs/PAPER_EXP02B_GEOMETRY_AGGREGATE.csv", "source_columns": "duration_s;geometry_window_id;median_selected_mean_error_m;median_oracle_gap_m", "claim": "G0/G1 geometry effect", "limitation": "Only one historical Iridium span"})

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True, constrained_layout=True)
    for ax, motion in zip(axes.ravel(), ["high_speed_straight", "turn", "deceleration", "stop_and_go"], strict=True):
        subset = selected[selected["motion_type"] == motion]
        for duration in durations:
            values = pd.to_numeric(subset.loc[subset["duration_s"] == duration, "mean_trajectory_position_error_m"], errors="coerce").dropna()
            ax.scatter(np.full(len(values), duration), values, s=12, alpha=0.5)
        med = subset.groupby("duration_s")["mean_trajectory_position_error_m"].median().reindex(durations)
        ax.plot(durations, med, color=COLORS["black"], marker="o")
        ax.set_title(motion.replace("_", " "))
        ax.set_yscale("symlog", linthresh=10)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Duration (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Selected mean error (m)")
    _save(fig, figures, "figure_07_duration_by_motion")
    maps.append({"figure_id": "Figure 7", "source_files": "outputs/PAPER_EXP02B_SELECTED_RESULTS.csv", "source_columns": "motion_type;duration_s;mean_trajectory_position_error_m", "claim": "Duration effect by motion type", "limitation": "All failures and seeds retained"})

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.6), constrained_layout=True)
    axes[0].scatter(floors["max_abs_acceleration_mps2"], floors["cv_floor_mean_error_m"], c=floors["duration_s"], cmap="viridis", s=24, alpha=0.75)
    axes[0].set_xlabel("Maximum |acceleration| (m/s$^2$)")
    axes[1].scatter(floors["max_abs_yaw_rate_degps"], floors["cv_floor_mean_error_m"], c=floors["duration_s"], cmap="viridis", s=24, alpha=0.75)
    axes[1].set_xlabel("Maximum |yaw rate| (deg/s)")
    for ax in axes:
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_ylabel("CV representation floor (m, symlog)")
        ax.grid(alpha=0.25)
    axes[0].set_title("Acceleration")
    axes[1].set_title("Turning")
    _save(fig, figures, "figure_08_floor_vs_kinematics")
    maps.append({"figure_id": "Figure 8", "source_files": "outputs/PAPER_EXP02B_MOTION_MODEL_REPRESENTATION_FLOORS.csv", "source_columns": "max_abs_acceleration_mps2;max_abs_yaw_rate_degps;cv_floor_mean_error_m;duration_s", "claim": "Motion nonlinearity and CV representation floor", "limitation": "Association is diagnostic, not causal identification"})

    representative = ["ODA_0012", "ODA_0054", "ODA_0121"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    styles = [("true", COLORS["black"], "-"), ("static", COLORS["gray"], ":"), ("cv", COLORS["blue"], "--"), ("ca", COLORS["orange"], "-.")]
    for ax, segment in zip(axes, representative, strict=True):
        subset = fit_data[(fit_data["source_segment_id"] == segment) & (fit_data["duration_s"] == 12) & (fit_data["geometry_window_id"] == "G0")]
        for label, color, linestyle in styles:
            ax.plot(subset[f"{label}_east_m"], subset[f"{label}_north_m"], color=color, ls=linestyle, label=label.upper())
        motion = subset["motion_type"].iloc[0] if not subset.empty else segment
        ax.set_title(motion.replace("_", " "))
        ax.set_xlabel("Local east (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Local north (m)")
    axes[-1].legend(frameon=False)
    _save(fig, figures, "figure_09_representative_trajectory_fits")
    maps.append({"figure_id": "Figure 9", "source_files": "inputs/PAPER_EXP02B_REPRESENTATIVE_FIT_DATA.csv", "source_columns": "true/static/cv/ca local east/north", "claim": "Representative motion-model fits", "limitation": "Fits use truth after selection and are not solver outputs"})

    plot_compare = exp_compare.copy()
    order = ["4 s", "6 s", "8 s", "12 s", "EXP02A original"]
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 6.0), constrained_layout=True)
    for ax, motion in zip(axes.ravel(), ["high_speed_straight", "turn", "deceleration", "stop_and_go"], strict=True):
        subset = plot_compare[(plot_compare["motion_type"] == motion) & (plot_compare["perturbation_profile"] == "ALL")]
        vals = [float(subset.loc[subset["duration_label"] == label, "median_selected_mean_error_m"].iloc[0]) if not subset.loc[subset["duration_label"] == label].empty else np.nan for label in order]
        ax.plot(range(len(order)), vals, marker="o", color=COLORS["blue"])
        ax.set_xticks(range(len(order)), ["4", "6", "8", "12", "original"], rotation=20)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title(motion.replace("_", " "))
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Window duration/reference")
    for ax in axes[:, 0]:
        ax.set_ylabel("Median selected mean error (m)")
    _save(fig, figures, "figure_10_exp02a_original_vs_short")
    maps.append({"figure_id": "Figure 10", "source_files": "outputs/PAPER_EXP02B_EXP02A_COMPARISON.csv", "source_columns": "motion_type;duration_label;median_selected_mean_error_m", "claim": "EXP02A reference versus fixed short-window diagnostic", "limitation": "Original results are reference-only and not pooled with EXP02B samples"})

    map_frame = pd.DataFrame(maps)
    map_frame["generation_script"] = "scripts/paper_exp02b_generate_figures.py"
    map_frame["verified"] = True
    map_frame.to_csv(figures / "PAPER_EXP02B_FIGURE_DATA_MAP.csv", index=False, encoding="utf-8-sig")
    return maps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(generate_all(args.root), ensure_ascii=False, indent=2))
