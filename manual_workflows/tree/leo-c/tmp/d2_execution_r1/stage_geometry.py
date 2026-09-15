from __future__ import annotations

import json
import math
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sgp4 import omm
from sgp4.api import Satrec
from skyfield.api import EarthSatellite, load
from skyfield.framelib import itrs

from d2_common import (
    EXEC_ROOT,
    EXPECTED_RAW_SHA,
    NORMALIZED_XML,
    PROTOCOL_ROOT,
    RAW_XML,
    ensure_no_formal_output,
    load_json,
    plain_path,
    protocol_contract_hashes,
    registered_asset_hash_audit,
    sha256_file,
    utc_now_z,
    write_json,
)


@dataclass(frozen=True)
class SatelliteEntry:
    norad: str
    name: str
    fields: dict[str, str]
    epoch: datetime
    skyfield: EarthSatellite
    satrec: Satrec


def parse_datetime(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_omm_xml(path: Path, ts: Any) -> list[SatelliteEntry]:
    root = ET.parse(path).getroot()
    entries: list[SatelliteEntry] = []
    for segment in root.findall(".//segment"):
        fields: dict[str, str] = {}
        for node in segment.iter():
            if len(list(node)) == 0 and node.text is not None:
                fields[node.tag.split("}")[-1]] = node.text.strip()
        required = {
            "OBJECT_NAME", "EPOCH", "MEAN_MOTION", "ECCENTRICITY", "INCLINATION",
            "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY", "EPHEMERIS_TYPE",
            "CLASSIFICATION_TYPE", "NORAD_CAT_ID", "ELEMENT_SET_NO", "REV_AT_EPOCH",
            "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
        }
        missing = sorted(required.difference(fields))
        if missing:
            raise RuntimeError(f"OMM XML missing fields for one segment: {missing}")
        model = Satrec()
        omm.initialize(model, fields)
        entries.append(
            SatelliteEntry(
                norad=str(fields["NORAD_CAT_ID"]),
                name=str(fields["OBJECT_NAME"]),
                fields=fields,
                epoch=parse_datetime(fields["EPOCH"]),
                skyfield=EarthSatellite.from_omm(ts, fields),
                satrec=model,
            )
        )
    entries.sort(key=lambda item: int(item.norad))
    ids = [entry.norad for entry in entries]
    if len(entries) != 80 or len(set(ids)) != 80:
        raise RuntimeError(f"canonical OMM identity error: records={len(entries)} unique={len(set(ids))}")
    return entries


def enu_up(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    return np.asarray([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)], dtype=float)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=np.int8)
    changes = np.diff(np.r_[0, values, 0])
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def moments(start: datetime, count: int, step_s: int) -> list[datetime]:
    return [start + timedelta(seconds=index * step_s) for index in range(count)]


def propagated_state(entry: SatelliteEntry, ts: Any, epochs: list[datetime]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = ts.from_datetimes(epochs)
    geocentric = entry.skyfield.at(times)
    position, velocity = geocentric.frame_xyz_and_velocity(itrs)
    pos = np.asarray(position.m, dtype=float).T
    vel = np.asarray(velocity.m_per_s, dtype=float).T
    messages = geocentric.message
    if messages is None:
        message_ok = np.ones(len(epochs), dtype=bool)
    elif isinstance(messages, list):
        message_ok = np.asarray([message is None for message in messages], dtype=bool)
    else:
        message_ok = np.asarray([messages is None] * len(epochs), dtype=bool)
    finite = np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1)
    return pos, vel, message_ok & finite


def visibility_matrix(
    positions: np.ndarray,
    valid: np.ndarray,
    receiver_ecef: np.ndarray,
    receiver_up: np.ndarray,
) -> np.ndarray:
    los = positions[:, None, :] - receiver_ecef[None, :, :]
    norms = np.linalg.norm(los, axis=2)
    up_component = np.einsum("trj,rj->tr", los, receiver_up)
    sin_elevation = np.divide(up_component, norms, out=np.full_like(up_component, -1.0), where=norms > 0.0)
    return (sin_elevation >= math.sin(math.radians(10.0))) & valid[:, None]


def propagate_interval_coarse(
    satellites: list[SatelliteEntry],
    ts: Any,
    start: datetime,
    end: datetime,
    receiver_ecef: np.ndarray,
    receiver_up: np.ndarray,
) -> np.ndarray:
    count = int(round((end - start).total_seconds() / 10.0)) + 1
    epochs = moments(start, count, 10)
    counts = np.zeros((len(receiver_ecef), count), dtype=np.uint8)
    for entry in satellites:
        positions, _velocities, ok = propagated_state(entry, ts, epochs)
        age_ok = np.asarray([abs((moment - entry.epoch).total_seconds()) <= 14.0 * 86400.0 for moment in epochs])
        visible = visibility_matrix(positions, ok & age_ok, receiver_ecef, receiver_up)
        counts += visible.T.astype(np.uint8)
    return counts


def search_receiver_arc(
    satellites: list[SatelliteEntry],
    ts: Any,
    interval_start: datetime,
    interval_end: datetime,
    receiver: pd.Series,
    coarse_counts: np.ndarray,
) -> dict[str, Any]:
    receiver_ecef = receiver[["ecef_x_m", "ecef_y_m", "ecef_z_m"]].to_numpy(float)
    receiver_up = enu_up(float(receiver["latitude_deg"]), float(receiver["longitude_deg"]))
    runs = contiguous_runs(coarse_counts >= 4)
    for coarse_start, coarse_end in runs:
        fine_start_index = max(0, (coarse_start - 1) * 10)
        fine_end_index = min(int((interval_end - interval_start).total_seconds()), (coarse_end + 1) * 10)
        if fine_end_index - fine_start_index < 120:
            continue
        # Resource-bounded chronological scan. It carries the exact count of
        # consecutive feasible seconds across chunks and therefore returns the
        # same first 121-sample run as a single full-window state cube.
        cursor = fine_start_index
        consecutive = 0
        selected_absolute_start: int | None = None
        while cursor <= fine_end_index and selected_absolute_start is None:
            chunk_end = min(fine_end_index, cursor + 3599)
            chunk_epochs = moments(interval_start + timedelta(seconds=cursor), chunk_end - cursor + 1, 1)
            chunk_counts = np.zeros(len(chunk_epochs), dtype=np.uint8)
            for entry in satellites:
                positions, _velocities, ok = propagated_state(entry, ts, chunk_epochs)
                age_ok = np.asarray([abs((moment - entry.epoch).total_seconds()) <= 14.0 * 86400.0 for moment in chunk_epochs])
                chunk_counts += visibility_matrix(
                    positions,
                    ok & age_ok,
                    receiver_ecef[None, :],
                    receiver_up[None, :],
                )[:, 0].astype(np.uint8)
            for offset, feasible in enumerate(chunk_counts >= 4):
                consecutive = consecutive + 1 if feasible else 0
                if consecutive >= 121:
                    selected_absolute_start = cursor + offset - 120
                    break
            cursor = chunk_end + 1
        if selected_absolute_start is None:
            continue

        arc_start = interval_start + timedelta(seconds=selected_absolute_start)
        arc_end = arc_start + timedelta(seconds=120)
        selected_epochs = moments(arc_start, 121, 1)
        selected_positions = np.empty((121, len(satellites), 3), dtype=np.float64)
        selected_velocities = np.empty_like(selected_positions)
        selected_visible = np.zeros((121, len(satellites)), dtype=bool)
        for sat_index, entry in enumerate(satellites):
            positions, velocities, ok = propagated_state(entry, ts, selected_epochs)
            age_ok = np.asarray([abs((moment - entry.epoch).total_seconds()) <= 14.0 * 86400.0 for moment in selected_epochs])
            selected_positions[:, sat_index, :] = positions
            selected_velocities[:, sat_index, :] = velocities
            selected_visible[:, sat_index] = visibility_matrix(
                positions,
                ok & age_ok,
                receiver_ecef[None, :],
                receiver_up[None, :],
            )[:, 0]
        selected_counts = selected_visible.sum(axis=1)
        if np.any(selected_counts < 4):
            raise RuntimeError("engineering_chunked_geometry_equivalence_failure")
        epoch_indices, sat_indices = np.where(selected_visible)
        flat_pos = selected_positions[epoch_indices, sat_indices, :]
        flat_vel = selected_velocities[epoch_indices, sat_indices, :]
        rel_time = epoch_indices.astype(float)
        sat_ids = np.asarray([int(satellites[index].norad) for index in sat_indices], dtype=np.int64)
        los = flat_pos - receiver_ecef[None, :]
        los_norm = np.linalg.norm(los, axis=1)
        range_rate = np.sum((los / los_norm[:, None]) * flat_vel, axis=1)
        if not np.isfinite(range_rate).all():
            continue
        intersection = np.flatnonzero(selected_visible.all(axis=0))
        union = np.flatnonzero(selected_visible.any(axis=0))
        return {
            "eligible": True,
            "arc_start": arc_start,
            "arc_end": arc_end,
            "time_s": rel_time,
            "sat_pos_m": flat_pos,
            "sat_vel_mps": flat_vel,
            "satellite_number": sat_ids,
            "row_index": np.arange(len(rel_time), dtype=np.int64),
            "minimum_satellite_count": int(selected_counts.min()),
            "median_satellite_count": float(np.median(selected_counts)),
            "maximum_satellite_count": int(selected_counts.max()),
            "satellite_union": ";".join(satellites[index].norad for index in union),
            "satellite_intersection": ";".join(satellites[index].norad for index in intersection),
            "observation_row_count": int(len(rel_time)),
            "coarse_run_count": len(runs),
        }
    return {"eligible": False, "coarse_run_count": len(runs)}


def main() -> None:
    started = time.perf_counter()
    ensure_no_formal_output()
    if sha256_file(RAW_XML) != EXPECTED_RAW_SHA:
        raise RuntimeError("d2_execution_blocked_protected_asset_change")
    registered = registered_asset_hash_audit()
    if not all(row["exists"] and row["sha_match"] and row["byte_match"] for row in registered):
        raise RuntimeError("d2_execution_blocked_protected_asset_change")
    receiver_contract = load_json(PROTOCOL_ROOT / "D2_RECEIVER_GENERATION_CONTRACT.json")
    interval_contract = load_json(PROTOCOL_ROOT / "D2_INTERVAL_AND_BLOCK_CONTRACT.json")
    geometry_contract = load_json(PROTOCOL_ROOT / "D2_GEOMETRY_EXECUTION_CONTRACT.json")
    if receiver_contract["status"] != "complete_and_locked" or interval_contract["target_block_count"] != 96:
        raise RuntimeError("d2_execution_blocked_receiver_block_identity")
    if geometry_contract["search"]["ranking"] != "chronologically_first_feasible":
        raise RuntimeError("d2_execution_invalidated_protocol_change_required")

    EXEC_ROOT.mkdir(parents=True, exist_ok=True)
    (EXEC_ROOT / "geometry").mkdir(exist_ok=True)
    for stale_geometry in (EXEC_ROOT / "geometry").glob("D2_B*_GEOMETRY.npz"):
        stale_geometry.unlink()
    receiver_source = PROTOCOL_ROOT / "D2_RECEIVER_REGISTRY.csv"
    receiver_target = EXEC_ROOT / "D2_RECEIVER_REGISTRY.csv"
    shutil.copyfile(receiver_source, receiver_target)
    if sha256_file(receiver_source) != sha256_file(receiver_target):
        raise RuntimeError("d2_execution_blocked_receiver_block_identity")
    receivers = pd.read_csv(receiver_target)
    expected_ids = [f"D2_R{index:02d}" for index in range(48)]
    if receivers["receiver_id"].astype(str).tolist() != expected_ids:
        raise RuntimeError("d2_execution_blocked_receiver_block_identity")
    if not receivers["receiver_registry_locked"].astype(bool).all() or not receivers["exclusion_pass"].astype(bool).all():
        raise RuntimeError("d2_execution_blocked_receiver_block_identity")

    intervals = [
        ("D2_A", parse_datetime(interval_contract["D2_A_start_utc"]), parse_datetime(interval_contract["D2_A_end_utc"])),
        ("D2_B", parse_datetime(interval_contract["D2_B_start_utc"]), parse_datetime(interval_contract["D2_B_end_utc"])),
    ]
    block_rows: list[dict[str, Any]] = []
    for receiver_index, receiver in receivers.iterrows():
        for interval_index, (interval_id, start, end) in enumerate(intervals):
            block_index = 2 * int(receiver_index) + interval_index
            block_rows.append(
                {
                    "block_id": f"D2_B{block_index:03d}",
                    "block_index": block_index,
                    "receiver_id": receiver["receiver_id"],
                    "receiver_index": int(receiver_index),
                    "interval_id": interval_id,
                    "interval_index": interval_index,
                    "interval_start_utc": iso_z(start),
                    "interval_end_utc": iso_z(end),
                    "independent_unit": "orbit-time-receiver block",
                    "geometry_status": "PENDING",
                }
            )
    blocks = pd.DataFrame(block_rows)
    blocks.to_csv(EXEC_ROOT / "D2_BLOCK_REGISTRY.csv", index=False)

    ts = load.timescale(builtin=True)
    satellites = parse_omm_xml(RAW_XML, ts)
    median_epoch = sorted(entry.epoch for entry in satellites)[len(satellites) // 2]
    receiver_ecef = receivers[["ecef_x_m", "ecef_y_m", "ecef_z_m"]].to_numpy(float)
    receiver_up = np.asarray([enu_up(float(row.latitude_deg), float(row.longitude_deg)) for row in receivers.itertuples(index=False)])
    geometry_rows: list[dict[str, Any]] = []
    interval_timings: dict[str, float] = {}
    for interval_index, (interval_id, interval_start, interval_end) in enumerate(intervals):
        interval_started = time.perf_counter()
        coarse = propagate_interval_coarse(satellites, ts, interval_start, interval_end, receiver_ecef, receiver_up)
        for receiver_index, receiver in receivers.iterrows():
            block_index = 2 * int(receiver_index) + interval_index
            block_id = f"D2_B{block_index:03d}"
            result = search_receiver_arc(
                satellites,
                ts,
                interval_start,
                interval_end,
                receiver,
                coarse[int(receiver_index)],
            )
            base = {
                "block_id": block_id,
                "block_index": block_index,
                "receiver_id": receiver["receiver_id"],
                "interval_id": interval_id,
                "interval_start_utc": iso_z(interval_start),
                "interval_end_utc": iso_z(interval_end),
                "search_order": "chronological_ascending",
                "coarse_grid_s": 10,
                "fine_grid_s": 1,
                "arc_duration_s": 120,
                "elevation_mask_deg": 10.0,
                "minimum_target_satellites": 4,
                "maximum_element_age_days": 14.0,
                "snapshot_median_element_epoch_utc": iso_z(median_epoch),
                "selection_rule": "chronologically_first_feasible",
                "coarse_run_count": int(result.get("coarse_run_count", 0)),
            }
            if result["eligible"]:
                geometry_path = EXEC_ROOT / "geometry" / f"{block_id}_GEOMETRY.npz"
                np.savez_compressed(
                    geometry_path,
                    block_id=np.asarray(block_id),
                    receiver_id=np.asarray(str(receiver["receiver_id"])),
                    interval_id=np.asarray(interval_id),
                    arc_start_utc=np.asarray(iso_z(result["arc_start"])),
                    arc_end_utc=np.asarray(iso_z(result["arc_end"])),
                    time_s=result["time_s"],
                    sat_pos_m=result["sat_pos_m"],
                    sat_vel_mps=result["sat_vel_mps"],
                    satellite_number=result["satellite_number"],
                    row_index=result["row_index"],
                    receiver_anchor_ecef_m=receiver[["ecef_x_m", "ecef_y_m", "ecef_z_m"]].to_numpy(float),
                    lat_deg=np.asarray(float(receiver["latitude_deg"])),
                    lon_deg=np.asarray(float(receiver["longitude_deg"])),
                    height_m=np.asarray(float(receiver["altitude_m"])),
                )
                base.update(
                    {
                        "geometry_status": "ELIGIBLE",
                        "selected_arc_start_utc": iso_z(result["arc_start"]),
                        "selected_arc_end_utc": iso_z(result["arc_end"]),
                        "minimum_satellite_count": result["minimum_satellite_count"],
                        "median_satellite_count": result["median_satellite_count"],
                        "maximum_satellite_count": result["maximum_satellite_count"],
                        "satellite_union_count": len(result["satellite_union"].split(";")) if result["satellite_union"] else 0,
                        "satellite_union": result["satellite_union"],
                        "satellite_intersection_count": len(result["satellite_intersection"].split(";")) if result["satellite_intersection"] else 0,
                        "satellite_intersection": result["satellite_intersection"],
                        "observation_row_count": result["observation_row_count"],
                        "element_age_status": "PASS",
                        "geometry_file": plain_path(geometry_path),
                        "geometry_sha256": sha256_file(geometry_path),
                    }
                )
            else:
                base.update(
                    {
                        "geometry_status": "PRE_METHOD_GEOMETRY_INELIGIBLE",
                        "selected_arc_start_utc": "",
                        "selected_arc_end_utc": "",
                        "minimum_satellite_count": 0,
                        "median_satellite_count": 0.0,
                        "maximum_satellite_count": 0,
                        "satellite_union_count": 0,
                        "satellite_union": "",
                        "satellite_intersection_count": 0,
                        "satellite_intersection": "",
                        "observation_row_count": 0,
                        "element_age_status": "NO_FEASIBLE_ARC",
                        "geometry_file": "",
                        "geometry_sha256": "",
                    }
                )
            geometry_rows.append(base)
        interval_timings[interval_id] = time.perf_counter() - interval_started
        print(json.dumps({"stage": "geometry", "interval": interval_id, "blocks_processed": 48, "elapsed_seconds": interval_timings[interval_id]}), flush=True)

    geometry = pd.DataFrame(geometry_rows).sort_values("block_index")
    if geometry["block_id"].tolist() != [f"D2_B{index:03d}" for index in range(96)]:
        raise RuntimeError("d2_execution_blocked_receiver_block_identity")
    geometry.to_csv(EXEC_ROOT / "D2_BLOCK_GEOMETRY_REGISTRY.csv", index=False)
    blocks = blocks.drop(columns=["geometry_status"]).merge(
        geometry[["block_id", "geometry_status", "selected_arc_start_utc", "selected_arc_end_utc", "minimum_satellite_count", "element_age_status"]],
        on="block_id",
        how="left",
        validate="one_to_one",
    )
    blocks.to_csv(EXEC_ROOT / "D2_BLOCK_REGISTRY.csv", index=False)
    eligible = int((geometry["geometry_status"] == "ELIGIBLE").sum())
    summary = {
        "stage": "geometry",
        "completed_utc": utc_now_z(),
        "snapshot_raw_sha256": sha256_file(RAW_XML),
        "snapshot_normalized_sha256": sha256_file(NORMALIZED_XML),
        "receiver_registry_sha256": sha256_file(receiver_target),
        "block_registry_sha256": sha256_file(EXEC_ROOT / "D2_BLOCK_REGISTRY.csv"),
        "geometry_registry_sha256": sha256_file(EXEC_ROOT / "D2_BLOCK_GEOMETRY_REGISTRY.csv"),
        "eligible_block_count": eligible,
        "ineligible_block_count": 96 - eligible,
        "minimum_eligible_blocks": 48,
        "headline_gate_pass": eligible >= 48,
        "interval_runtime_seconds": interval_timings,
        "total_runtime_seconds": time.perf_counter() - started,
        "geometry_used_information_ranking": False,
        "geometry_used_method_output": False,
        "geometry_used_truth": False,
        "registered_asset_hash_audit_pass": all(row["sha_match"] and row["byte_match"] for row in registered),
        "protocol_contract_hashes": protocol_contract_hashes(),
    }
    write_json(EXEC_ROOT / "D2_GEOMETRY_STAGE_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
