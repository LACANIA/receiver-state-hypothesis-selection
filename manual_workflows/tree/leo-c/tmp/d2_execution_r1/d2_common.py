from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAPER_F_ROOT = Path(
    'flagship'
)
SNAPSHOT_ROOT = PAPER_F_ROOT / "34_d2_snapshot_capture" / "PAPER_FLAGSHIP_D2_CANONICAL_OMM_SNAPSHOT_ELIGIBILITY_CAPTURE"
PROTOCOL_ROOT = PAPER_F_ROOT / "37_d2_minimum_protocol_completion" / "PAPER_FLAGSHIP_D2_MINIMUM_PROTOCOL_COMPLETION_AND_EXECUTION_BINDING_R1"
V057_ROOT = PAPER_F_ROOT / "03_working_manuscript" / "manuscript_v0_5_7_literature_novelty_minor_revision"
EXEC_ROOT = PAPER_F_ROOT / "38_d2_execution" / "PAPER_FLAGSHIP_D2_CROSS_EPOCH_GENERATION_AND_FROZEN_METHOD_EXECUTION"

RAW_XML = SNAPSHOT_ROOT / "canonical" / "D2_SNAPSHOT_CANDIDATE_001_IRIDIUM_NEXT_OMM.xml"
NORMALIZED_XML = SNAPSHOT_ROOT / "normalized" / "D2_SNAPSHOT_CANDIDATE_001_IRIDIUM_NEXT_OMM_NORMALIZED.xml"
EXPECTED_RAW_SHA = "d1b58796a12dff8d3a3020d58e93516d529850508fbb910e529f0af109059827"

EXP01_ROOT = Path('leo-c/_new_solver_lab/paper_draft/holdout_experiments')
EXP01_RUN = EXP01_ROOT / "paper_exp01_run_holdout.py"
EXP01_SCENARIOS = EXP01_ROOT / "PAPER_EXP01_HOLDOUT_SCENARIOS.csv"
RELEASE_ROOT = Path('leo-c/_new_solver_lab/final_release/MA_BGTR_v7_2_freeze_r2')
RELEASE_RUNTIME = RELEASE_ROOT / "scripts" / "release_runtime.py"
SELECTOR_REPLAY = PAPER_F_ROOT / "21_v052_identity_repair" / "PAPER_FLAGSHIP_V052_INFORMATION_CRITERION_SELECTOR_PUBLICATIONIZATION" / "selector_parity" / "selector_spec_replay_v052.py"
SELECTOR_SPEC = V057_ROOT / "supplementary_data" / "ma_bgtr_mode_a_selector_spec_v052.json"


def plain_path(path: Path) -> str:
    text = str(path)
    return text[4:] if text.startswith("\\\\?\\") else text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {plain_path(path)}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_no_formal_output() -> None:
    markers = [
        EXEC_ROOT / "D2_METHOD_OUTPUT_LOCK.json",
        EXEC_ROOT / "D2_EXECUTION_MAIN_DECISION.json",
        EXEC_ROOT / "PAPER_FLAGSHIP_D2_CROSS_EPOCH_GENERATION_AND_FROZEN_METHOD_EXECUTION_TOTAL_REPORT.md",
    ]
    if any(path.exists() for path in markers):
        raise RuntimeError("d2_execution_blocked_existing_output")


def registered_asset_hash_audit() -> list[dict[str, Any]]:
    registry = PROTOCOL_ROOT / "D2_EXECUTION_ASSET_HASHES.csv"
    rows = read_csv_rows(registry)
    out: list[dict[str, Any]] = []
    for row in rows:
        path = Path("\\\\?\\" + row["path"])
        exists = path.is_file()
        actual_sha = sha256_file(path) if exists else ""
        actual_bytes = path.stat().st_size if exists else -1
        out.append(
            {
                "asset_id": row["asset_id"],
                "path": row["path"],
                "registered_sha256": row["sha256"],
                "actual_sha256": actual_sha,
                "registered_bytes": int(row["bytes"]),
                "actual_bytes": int(actual_bytes),
                "exists": exists,
                "sha_match": actual_sha == row["sha256"],
                "byte_match": int(actual_bytes) == int(row["bytes"]),
            }
        )
    return out


def protocol_contract_hashes() -> list[dict[str, Any]]:
    names = [
        "D2_RECEIVER_GENERATION_CONTRACT.json",
        "D2_INTERVAL_AND_BLOCK_CONTRACT.json",
        "D2_GEOMETRY_EXECUTION_CONTRACT.json",
        "D2_SCENARIO_REUSE_BINDING.json",
        "D2_OBSERVATION_SCHEMA.json",
        "D2_TRUTH_AND_METHOD_LOCK_CONTRACT.json",
        "D2_RECEIVER_REGISTRY.csv",
        "D2_GENERIC_WRAPPER_PARITY_AUDIT.csv",
    ]
    return [
        {
            "asset_id": f"protocol_{Path(name).stem}",
            "path": plain_path(PROTOCOL_ROOT / name),
            "sha256": sha256_file(PROTOCOL_ROOT / name),
            "bytes": (PROTOCOL_ROOT / name).stat().st_size,
        }
        for name in names
    ]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
