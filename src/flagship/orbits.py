from __future__ import annotations
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from datetime import datetime,timedelta,timezone
import math
import xml.etree.ElementTree as ET
import numpy as np
from sgp4 import omm
from sgp4.api import Satrec
from skyfield.api import EarthSatellite,load
from skyfield.framelib import itrs

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
