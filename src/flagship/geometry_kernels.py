from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2,ncx2
SIGMA=0.30
ALPHA=0.05

def matrix(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=float)

def projector(columns: np.ndarray) -> np.ndarray:
    n = columns.shape[0]
    if columns.shape[1] == 0:
        return np.zeros((n, n), dtype=float)
    return columns @ np.linalg.pinv(columns, rcond=1.0e-12)

def schur(jp: np.ndarray, jn: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    ap = jp / SIGMA
    an = jn / SIGMA
    rn = np.eye(len(ap)) - projector(an)
    b = rn @ ap
    s = b.T @ b
    s = (s + s.T) / 2.0
    return s, b, int(np.linalg.matrix_rank(an)), int(np.linalg.matrix_rank(b))

def required_lambda(df: int, power: float) -> float:
    threshold = float(chi2.ppf(1.0 - ALPHA, df))
    f = lambda lam: float(ncx2.cdf(threshold, df, lam) - (1.0 - power))
    high = 1.0
    while f(high) > 0.0:
        high *= 2.0
    return float(brentq(f, 0.0, high, xtol=1.0e-14, rtol=1.0e-14))

@dataclass
class Geometry:
    sat_pos: np.ndarray
    sat_vel: np.ndarray
    receiver: np.ndarray
    ranges: np.ndarray
    unit: np.ndarray
    tau: np.ndarray
    jp: np.ndarray
    jn: np.ndarray

def load_geometry(path: Path) -> Geometry:
    df = pd.read_csv(path)
    ps = df[["satellite_ecef_x_m", "satellite_ecef_y_m", "satellite_ecef_z_m"]].to_numpy(float)
    vs = df[["satellite_ecef_vx_mps", "satellite_ecef_vy_mps", "satellite_ecef_vz_mps"]].to_numpy(float)
    pr = df[["receiver_anchor_ecef_x_m", "receiver_anchor_ecef_y_m", "receiver_anchor_ecef_z_m"]].iloc[0].to_numpy(float)
    rel = ps - pr
    r = np.sqrt(np.einsum("ij,ij->i", rel, rel))
    u = rel / r[:, None]
    transverse = vs - u * np.einsum("ij,ij->i", u, vs)[:, None]
    jp = -transverse / r[:, None]
    raw_t = df["relative_time_s"].to_numpy(float)
    tau = raw_t - (float(raw_t.min()) + float(raw_t.max())) / 2.0
    jv = -u + tau[:, None] * jp
    jn = np.concatenate((jv, np.ones((len(df), 1)), tau[:, None]), axis=1)
    return Geometry(ps, vs, pr, r, u, tau, jp, jn)

def visible_plane_angle(s1: np.ndarray, s2: np.ndarray) -> float:
    _, q1 = np.linalg.eigh(s1)
    _, q2 = np.linalg.eigh(s2)
    singular = np.linalg.svd(q1[:, 1:].T @ q2[:, 1:], compute_uv=False)
    singular = np.clip(singular, -1.0, 1.0)
    return float(np.degrees(np.max(np.arccos(singular))))

def nonlinear_values(g: Geometry, s: np.ndarray, rho: float, sign: float) -> dict[str, float]:
    vals, vecs = np.linalg.eigh(s)
    direction = sign * vecs[:, 0]
    dp = rho * direction
    shifted = g.sat_pos - (g.receiver + dp)[None, :]
    shifted_u = shifted / np.linalg.norm(shifted, axis=1)[:, None]
    exact = np.einsum("ij,ij->i", shifted_u, g.sat_vel) - np.einsum("ij,ij->i", g.unit, g.sat_vel)
    linear = g.jp @ dp
    rn = np.eye(len(g.jn)) - projector(g.jn / SIGMA)
    exact_p = rn @ (exact / SIGMA)
    linear_p = rn @ (linear / SIGMA)
    rem = exact_p - linear_p
    ln = float(np.linalg.norm(linear_p))
    en = float(np.linalg.norm(exact_p))
    rnrm = float(np.linalg.norm(rem))
    gap = g.ranges - rho
    if np.any(gap <= 0):
        bound = float("inf")
    else:
        row = 1.5 * np.linalg.norm(g.sat_vel, axis=1) * rho * rho / (gap * gap)
        bound = float(np.linalg.norm(row / SIGMA))
    return {
        "remainder_ratio": rnrm / ln,
        "bound_ratio": bound / ln,
        "lambda_linear": ln * ln,
        "lambda_exact": en * en,
        "lambda_relative_error": (en * en - ln * ln) / (ln * ln),
    }

def certification_radius(g: Geometry, s: np.ndarray, ratio: float) -> float:
    upper = min(100_000.0, 0.20 * float(g.ranges.min()))
    f = lambda r: nonlinear_values(g, s, r, 1.0)["bound_ratio"] - ratio
    if f(upper) <= 0.0:
        return upper
    return float(brentq(f, 1.0e-6, upper))
