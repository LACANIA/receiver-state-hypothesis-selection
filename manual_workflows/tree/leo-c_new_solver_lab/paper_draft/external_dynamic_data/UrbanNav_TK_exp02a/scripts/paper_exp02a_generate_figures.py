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
    "yellow": "#F0E442",
    "black": "#222222",
    "gray": "#777777",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def _finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    return values[np.isfinite(values)]


def generate_all(root: str | Path) -> list[dict[str, str]]:
    _style()
    root = Path(root)
    figures = root / "figures"
    outputs = root / "outputs"
    protocol = root / "protocol"
    inputs = root / "inputs"
    figures.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(outputs / "PAPER_EXP02A_SELECTED_RESULTS.csv")
    candidates = pd.read_csv(outputs / "PAPER_EXP02A_CANDIDATE_RESULTS.csv")
    window_protocol = pd.read_csv(protocol / "PAPER_EXP02A_WINDOW_PROTOCOL.csv")
    transplant = pd.read_csv(inputs / "PAPER_EXP02A_TRAJECTORY_TRANSPLANT_MAP.csv")
    mode_comparison = pd.read_csv(outputs / "PAPER_EXP02A_EVIDENCE_MODE_COMPARISON.csv")
    failures = pd.read_csv(outputs / "PAPER_EXP02A_FAILURE_CASES.csv")

    maps: list[dict[str, str]] = []

    # Figure 1: full UrbanNav routes and pre-frozen windows.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    for ax, route in zip(axes, ["Odaiba", "Shinjuku"], strict=True):
        source = pd.read_csv(inputs / f"UrbanNav_TK_{route}_trajectory_for_plot.csv")
        ax.plot(source["enu_e_m"], source["enu_n_m"], color=COLORS["gray"], lw=0.7, label="Full reference route")
        route_windows = window_protocol[window_protocol["route"] == route]
        for _, win in route_windows.iterrows():
            mask = (source["time_s"] >= float(win["window_start_time_s"])) & (source["time_s"] <= float(win["window_end_time_s"]))
            style = "-" if bool(win["available_for_solver_run"]) else "--"
            ax.plot(source.loc[mask, "enu_e_m"], source.loc[mask, "enu_n_m"], lw=2.2, ls=style, label=f"{win['experiment_window_id']} ({win['motion_type']})")
        ax.set_title(route)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.axis("equal")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", frameon=False, fontsize=6)
    _save(fig, figures, "figure_01_trajectory_window_overview")
    maps.append({"figure_id": "Figure 1", "source_files": "inputs/UrbanNav_TK_*_trajectory_for_plot.csv; protocol/PAPER_EXP02A_WINDOW_PROTOCOL.csv", "claim": "Fixed UrbanNav route windows and unavailable-window transparency"})

    # Figure 2: conceptual motion transplant, drawn from the audited map values.
    fig, ax = plt.subplots(figsize=(8.5, 4.1), constrained_layout=True)
    ax.axis("off")
    boxes = [(0.04, 0.55, "UrbanNav local ENU\nrecorded vehicle motion"), (0.38, 0.55, "Rotate relative displacement\nand velocity"), (0.72, 0.55, "Frozen Iridium anchor\nECEF trajectory")]
    for x, y, label in boxes:
        rect = plt.Rectangle((x, y), 0.24, 0.23, facecolor="#F5F5F5", edgecolor=COLORS["black"], lw=1.1)
        ax.add_patch(rect)
        ax.text(x + 0.12, y + 0.115, label, ha="center", va="center")
    for x0, x1 in [(0.28, 0.38), (0.62, 0.72)]:
        ax.annotate("", xy=(x1, 0.665), xytext=(x0, 0.665), arrowprops=dict(arrowstyle="->", lw=1.5, color=COLORS["blue"]))
    ax.text(0.50, 0.38, r"$p_r(t)=p_{anchor}+R_{ENU\rightarrow ECEF}\,\Delta p_{ENU}(t)$", ha="center", fontsize=11)
    ax.text(0.50, 0.25, r"$v_r(t)=R_{ENU\rightarrow ECEF}\,v_{ENU}(t)$", ha="center", fontsize=11)
    ax.text(0.50, 0.09, "Relative timing preserved; no Tokyo/LEO absolute-time synchronization and no native LEO RF data", ha="center", color=COLORS["red"], fontsize=8)
    _save(fig, figures, "figure_02_motion_transplant_geometry")
    maps.append({"figure_id": "Figure 2", "source_files": "inputs/PAPER_EXP02A_TRAJECTORY_TRANSPLANT_MAP.csv", "claim": "Audited real-trajectory motion-transplant rule"})

    # Figure 3: selected mean trajectory error by motion type.
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    motion_order = ["high_speed_straight", "turn", "deceleration", "stop_and_go"]
    available = selected[selected["run_available"].astype(bool)].copy()
    data = [_finite(available.loc[available["motion_type"] == motion, "mean_trajectory_position_error_m"]) for motion in motion_order]
    positions = np.arange(len(motion_order))
    valid = [(pos, values, motion) for pos, values, motion in zip(positions, data, motion_order, strict=True) if len(values)]
    if valid:
        ax.boxplot([item[1] for item in valid], positions=[item[0] for item in valid], widths=0.55, showfliers=True, whis=(0, 100))
        for pos, values, _ in valid:
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(pos + jitter, values, s=12, alpha=0.55, color=COLORS["blue"])
    for pos, motion in enumerate(motion_order):
        unavailable_count = int((~window_protocol.loc[window_protocol["motion_type"] == motion, "available_for_solver_run"].astype(bool)).sum())
        if unavailable_count:
            ax.text(pos, 0.98, f"{unavailable_count} unavailable", transform=ax.get_xaxis_transform(), ha="center", va="top", color=COLORS["red"], fontsize=7)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(positions, [m.replace("_", "\n") for m in motion_order])
    ax.set_ylabel("Mean trajectory position error (m, symlog)")
    ax.set_title("Selected solution error by pre-frozen motion type")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, figures, "figure_03_selected_error_by_motion_type")
    maps.append({"figure_id": "Figure 3", "source_files": "outputs/PAPER_EXP02A_SELECTED_RESULTS.csv; protocol/PAPER_EXP02A_WINDOW_PROTOCOL.csv", "claim": "Motion-type error including unavailable categories and full long tails"})

    # Figure 4: selected model distribution.
    counts = available.groupby(["motion_type", "selected_model"]).size().unstack(fill_value=0).reindex(motion_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    bottom = np.zeros(len(counts))
    palette = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"], COLORS["sky"], COLORS["red"], COLORS["gray"]]
    for idx, model in enumerate(counts.columns):
        values = counts[model].to_numpy(float)
        ax.bar(np.arange(len(counts)), values, bottom=bottom, label=model, color=palette[idx % len(palette)], edgecolor="white", linewidth=0.4)
        bottom += values
    ax.set_xticks(np.arange(len(counts)), [m.replace("_", "\n") for m in counts.index])
    ax.set_ylabel("Selection count")
    ax.set_title("Frozen-selector model distribution")
    ax.legend(ncol=2, frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, figures, "figure_04_selected_model_distribution")
    maps.append({"figure_id": "Figure 4", "source_files": "outputs/PAPER_EXP02A_SELECTED_RESULTS.csv", "claim": "Model-adaptive selection distribution without outcome filtering"})

    # Figure 5: selected versus fixed baselines.
    comparison = pd.read_csv(outputs / "PAPER_EXP02A_BASELINE_COMPARISON.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.3), constrained_layout=True)
    order = ["M0_static_position", "M2_ctd_full", "M3_ctd_no_drift", "M4_ctd_no_bias", "M7_robust_ctd_full", "M13_robust_ctd_full_plus_gir_refine"]
    medians, low, high = [], [], []
    for model in order:
        group = comparison[comparison["baseline_model"] == model]
        vals = _finite(group["baseline_minus_selected_mean_error_m"])
        medians.append(float(np.median(vals)) if len(vals) else np.nan)
        low.append(float(np.quantile(vals, 0.25)) if len(vals) else np.nan)
        high.append(float(np.quantile(vals, 0.75)) if len(vals) else np.nan)
    medians_arr = np.asarray(medians)
    ax.errorbar(np.arange(len(order)), medians_arr, yerr=[medians_arr - np.asarray(low), np.asarray(high) - medians_arr], fmt="o", color=COLORS["black"], ecolor=COLORS["blue"], capsize=4)
    ax.axhline(0, color=COLORS["red"], ls="--", lw=1)
    ax.set_xticks(np.arange(len(order)), [m.split("_")[0] for m in order])
    ax.set_ylabel("Baseline minus selected mean error (m)")
    ax.set_title("Paired selected-versus-fixed-candidate comparison")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, figures, "figure_05_selected_vs_fixed_baselines")
    maps.append({"figure_id": "Figure 5", "source_files": "outputs/PAPER_EXP02A_BASELINE_COMPARISON.csv", "claim": "All predeclared paired baseline comparisons"})

    # Figure 6: P2 Mode A versus Mode B.
    fig, ax = plt.subplots(figsize=(5.2, 4.7), constrained_layout=True)
    if not mode_comparison.empty:
        x = pd.to_numeric(mode_comparison["observable_only_mean_error_m"], errors="coerce")
        y = pd.to_numeric(mode_comparison["configuration_assisted_mean_error_m"], errors="coerce")
        finite = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[finite], y[finite], color=COLORS["purple"], alpha=0.75, s=24)
        limit = max(float(np.nanmax(x[finite])) if finite.any() else 1.0, float(np.nanmax(y[finite])) if finite.any() else 1.0, 1.0)
        ax.plot([0, limit], [0, limit], color=COLORS["black"], ls="--", lw=1)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("Observable-only mean error (m)")
    ax.set_ylabel("Configuration-assisted mean error (m)")
    ax.set_title("P2 evidence-mode ablation")
    ax.grid(alpha=0.25)
    _save(fig, figures, "figure_06_p2_evidence_mode_comparison")
    maps.append({"figure_id": "Figure 6", "source_files": "outputs/PAPER_EXP02A_EVIDENCE_MODE_COMPARISON.csv", "claim": "Controlled metadata-availability ablation under identical candidate fits"})

    # Figure 7: error versus kinematic severity.
    merged = available.merge(window_protocol[["experiment_window_id", "max_abs_yaw_rate_degps", "max_abs_acceleration_mps2"]], on="experiment_window_id", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    axes[0].scatter(merged["max_abs_yaw_rate_degps"], merged["mean_trajectory_position_error_m"], c=merged["seed"], cmap="viridis", s=20, alpha=0.7)
    axes[0].set_xlabel("Maximum |yaw rate| (deg/s)")
    axes[0].set_ylabel("Mean trajectory error (m)")
    axes[0].set_yscale("symlog", linthresh=10)
    axes[1].scatter(merged["max_abs_acceleration_mps2"], merged["mean_trajectory_position_error_m"], c=merged["seed"], cmap="viridis", s=20, alpha=0.7)
    axes[1].set_xlabel("Maximum |acceleration| (m/s²)")
    axes[1].set_ylabel("Mean trajectory error (m)")
    axes[1].set_yscale("symlog", linthresh=10)
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, figures, "figure_07_error_vs_kinematics")
    maps.append({"figure_id": "Figure 7", "source_files": "outputs/PAPER_EXP02A_SELECTED_RESULTS.csv; protocol/PAPER_EXP02A_WINDOW_PROTOCOL.csv", "claim": "Trajectory error versus audited turning/acceleration severity"})

    # Figure 8: residual-position alignment.
    fig, ax = plt.subplots(figsize=(6.4, 4.7), constrained_layout=True)
    finite = np.isfinite(pd.to_numeric(candidates["trimmed_validation_rmse_mps"], errors="coerce")) & np.isfinite(pd.to_numeric(candidates["mean_trajectory_position_error_m"], errors="coerce"))
    families = candidates.loc[finite, "model_family"].astype(str).unique()
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for idx, family in enumerate(families):
        subset = candidates.loc[finite & candidates["model_family"].eq(family)]
        ax.scatter(subset["trimmed_validation_rmse_mps"], subset["mean_trajectory_position_error_m"], marker=markers[idx % len(markers)], s=16, alpha=0.45, label=family)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("Trimmed validation RMSE (m/s)")
    ax.set_ylabel("Mean trajectory position error (m)")
    ax.set_title("Residual evidence versus navigation error")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    _save(fig, figures, "figure_08_residual_vs_trajectory_error")
    maps.append({"figure_id": "Figure 8", "source_files": "outputs/PAPER_EXP02A_CANDIDATE_RESULTS.csv", "claim": "Residual-position alignment and mismatch"})

    # Figure 9: all recorded failure/unavailable cases, without clipping long tails.
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2), constrained_layout=True)
    unavailable = window_protocol[~window_protocol["available_for_solver_run"].astype(bool)]
    threshold_rows = unavailable if not unavailable.empty else window_protocol
    threshold_color = COLORS["red"] if not unavailable.empty else COLORS["green"]
    axes[0].barh(threshold_rows["source_segment_id"], threshold_rows["geometry_unique_satellite_count"], color=threshold_color)
    axes[0].axvline(4, color=COLORS["black"], ls="--", lw=1, label="Minimum = 4")
    axes[0].set_xlabel("Unique satellites")
    axes[0].set_title("Geometry threshold audit" if unavailable.empty else "Protocol-unavailable windows")
    axes[0].legend(frameon=False)
    selected_failures = failures[failures["case_type"].isin(["large_selected_error", "large_oracle_gap", "selected_quality_failure"])].copy()
    selected_failures = selected_failures.sort_values("mean_trajectory_position_error_m", ascending=False).head(12)
    selected_failures["_mode"] = np.where(selected_failures["case_id"].astype(str).str.contains("configuration-assisted"), "B", "A")
    selected_failures["_seed"] = selected_failures["run_id"].astype(str).str.extract(r"(2026072[4-6])$")[0].fillna("seed?")
    selected_failures["_label"] = (
        selected_failures["source_segment_id"].astype(str)
        + " | " + selected_failures["selected_model"].astype(str).str.replace("_", " ", regex=False)
        + " | " + selected_failures["_seed"].astype(str)
        + " | Mode " + selected_failures["_mode"].astype(str)
        + " | " + selected_failures["case_type"].astype(str).str.replace("_", " ", regex=False)
    )
    axes[1].barh(selected_failures["_label"], selected_failures["mean_trajectory_position_error_m"], color=COLORS["orange"])
    axes[1].set_xscale("symlog", linthresh=10)
    axes[1].set_xlabel("Mean trajectory position error (m, symlog)")
    axes[1].set_title("Recorded high-error cases")
    axes[1].tick_params(axis="y", labelsize=6)
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    _save(fig, figures, "figure_09_failure_case_examples")
    maps.append({"figure_id": "Figure 9", "source_files": "outputs/PAPER_EXP02A_FAILURE_CASES.csv; protocol/PAPER_EXP02A_WINDOW_PROTOCOL.csv", "claim": "Unavailable and failed/high-error cases retained"})

    pd.DataFrame(maps).to_csv(figures / "PAPER_EXP02A_FIGURE_DATA_MAP.csv", index=False, encoding="utf-8-sig")
    return maps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(generate_all(args.root), ensure_ascii=False, indent=2))
