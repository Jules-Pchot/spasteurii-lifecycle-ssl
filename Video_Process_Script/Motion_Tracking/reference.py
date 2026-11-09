"""Reference solutions of the motion / tracking exercises (one function per exercise).

The notebooks compare your version with these, and use them for the later steps while yours is not finished.
Try the exercise before opening this file.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


# --- 01 sampling ------------------------------------------------------------------------------------------------
def nearest_neighbor_distance(points: np.ndarray) -> np.ndarray:
    """(N, 2) positions -> (N,) distance of each point to its nearest other point."""
    distances, _ = cKDTree(points).query(points, k=2)  # k=1 is the point itself
    return distances[:, 1]


# --- 02 association ---------------------------------------------------------------------------------------------
def link_frames(previous: np.ndarray, current: np.ndarray, max_distance: float) -> np.ndarray:
    """Links between two frames: (P, 2) and (C, 2) positions -> (L, 2) index pairs (i in previous, j in current).

    Global assignment minimizing the total distance (Hungarian algorithm); pairs farther than max_distance are
    refused (gating).
    """
    if len(previous) == 0 or len(current) == 0:
        return np.empty((0, 2), dtype=int)
    cost = cdist(previous, current)
    gated = np.where(cost > max_distance, 1e9, cost)  # forbidden pairs: huge cost
    rows, cols = linear_sum_assignment(gated)
    keep = cost[rows, cols] <= max_distance
    return np.c_[rows[keep], cols[keep]]


def greedy_link(previous: np.ndarray, current: np.ndarray, max_distance: float) -> np.ndarray:
    """Greedy nearest neighbor: pairs taken one by one, shortest first, if both detections are still free."""
    if len(previous) == 0 or len(current) == 0:
        return np.empty((0, 2), dtype=int)
    cost = cdist(previous, current)
    used_i, used_j, pairs = set(), set(), []
    for flat in np.argsort(cost, axis=None):
        i, j = np.unravel_index(flat, cost.shape)
        if cost[i, j] > max_distance:
            break
        if i not in used_i and j not in used_j:
            pairs.append((i, j))
            used_i.add(i)
            used_j.add(j)
    return np.array(pairs, dtype=int).reshape(-1, 2)


# --- 03 Kalman --------------------------------------------------------------------------------------------------
def kalman_matrices(dt: float, process_noise: float, measurement_noise: float) -> dict:
    """Constant-velocity model, state (x, y, vx, vy), measurement (x, y)."""
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    q = process_noise ** 2  # random acceleration (px / s²)²
    Q = q * np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0], [0, dt ** 3 / 3, 0, dt ** 2 / 2],
                      [dt ** 2 / 2, 0, dt, 0], [0, dt ** 2 / 2, 0, dt]])
    R = measurement_noise ** 2 * np.eye(2)
    return {"F": F, "H": H, "Q": Q, "R": R}


def kalman_predict(x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prediction step: state and covariance one frame later."""
    return F @ x, F @ P @ F.T + Q


def kalman_update(x: np.ndarray, P: np.ndarray, z: np.ndarray, H: np.ndarray,
                  R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Correction step with the measurement z."""
    innovation = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain: how much the measurement is trusted
    return x + K @ innovation, (np.eye(len(x)) - K @ H) @ P


@dataclass
class KalmanModel:
    dt: float
    process_noise: float = 10.0     # px / s²
    measurement_noise: float = 0.5  # px
    initial_speed_std: float = 20.0  # px / s
    predict: Callable = kalman_predict
    update: Callable = kalman_update


# --- tracking (frame-by-frame linking) ---------------------------------------------------------------------------
def track(detections: pd.DataFrame, max_distance: float, link: Callable = link_frames, memory: int = 0,
          motion: KalmanModel | None = None) -> pd.DataFrame:
    """Adds a `track_id` column to the detections, frame by frame.

    memory: a track missing for up to `memory` frames can still be continued (gap closing).
    motion: None = a track is expected where it was last seen; KalmanModel = where its motion predicts it.
    """
    result = detections.sort_values("frame", kind="stable").copy()
    result["track_id"] = -1
    active = {}  # track_id -> {"frame", "x" (state), "P"}
    matrices = kalman_matrices(motion.dt, motion.process_noise, motion.measurement_noise) if motion else None
    next_id = 0
    for frame, group in result.groupby("frame", sort=True):
        points = group[["x", "y"]].to_numpy()
        ids = [i for i, a in active.items() if frame - a["frame"] <= memory + 1]
        predicted = {}
        for i in ids:
            state, cov = active[i]["x"], active[i]["P"]
            if motion:
                for _ in range(frame - active[i]["frame"]):
                    state, cov = motion.predict(state, cov, matrices["F"], matrices["Q"])
            predicted[i] = (state, cov)
        previous = np.array([predicted[i][0][:2] for i in ids]).reshape(-1, 2)
        pairs = link(previous, points, max_distance) if ids else np.empty((0, 2), dtype=int)
        assigned = np.full(len(points), -1)
        for i, j in np.asarray(pairs, dtype=int).reshape(-1, 2):
            assigned[j] = ids[i]
        for j in np.flatnonzero(assigned == -1):
            assigned[j], next_id = next_id, next_id + 1
        result.loc[group.index, "track_id"] = assigned
        for j, tid in enumerate(assigned):
            z = points[j]
            if tid in predicted and motion:
                state, cov = motion.update(*predicted[tid], z, matrices["H"], matrices["R"])
            elif motion:
                state = np.r_[z, 0.0, 0.0]
                cov = np.diag([motion.measurement_noise ** 2] * 2 + [motion.initial_speed_std ** 2] * 2)
            else:
                state, cov = z, None
            active[tid] = {"frame": frame, "x": state, "P": cov}
        active = {i: a for i, a in active.items() if frame - a["frame"] <= memory + 1}
    return result


# --- 04 trajectories ---------------------------------------------------------------------------------------------
def msd(xy: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean squared displacement of one track (consecutive frames), lags 1..max_lag (in frames) -> (max_lag,)."""
    return np.array([np.mean(np.sum((xy[lag:] - xy[:-lag]) ** 2, axis=1)) for lag in range(1, max_lag + 1)])


def fit_alpha(lag_seconds: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """MSD = K * lag^alpha, fitted as a line in log-log -> (alpha, K)."""
    alpha, log_k = np.polyfit(np.log(lag_seconds), np.log(values), 1)
    return alpha, float(np.exp(log_k))


# --- 05 reference motion and outliers ----------------------------------------------------------------------------
def velocities(tracks: pd.DataFrame, dt: float, id_column: str = "track_id") -> pd.DataFrame:
    """Velocity between consecutive detections of each track, stored at the earlier frame (px / s)."""
    t = tracks.sort_values([id_column, "frame"])
    nxt = t.groupby(id_column)[["frame", "x", "y"]].shift(-1)
    gap = (nxt.frame - t.frame) * dt
    out = t.assign(vx=(nxt.x - t.x) / gap, vy=(nxt.y - t.y) / gap)
    return out.dropna(subset=["vx"])


def local_reference_velocity(points: np.ndarray, velocity: np.ndarray, k: int) -> np.ndarray:
    """(N, 2) positions and velocities -> (N, 2) median velocity of the k nearest OTHER bacteria."""
    k = min(k, len(points) - 1)
    _, neighbors = cKDTree(points).query(points, k=k + 1)
    return np.median(velocity[neighbors[:, 1:]], axis=1)


def robust_z(values: np.ndarray) -> np.ndarray:
    """(value - median) / (1.4826 * MAD): a z-score that the outliers themselves do not distort."""
    median = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - median))
    return (values - median) / mad


# --- 06 evaluation -----------------------------------------------------------------------------------------------
def link_scores(tracked: pd.DataFrame) -> tuple[float, float]:
    """(precision, recall) of the frame-to-frame links, using the ground-truth `true_id`.

    precision: share of the links made by the tracker that join the same bacterium;
    recall: share of the true links (same bacterium, consecutive detections) that the tracker made.
    """
    def links(df, by):
        s = df.sort_values([by, "frame"])
        same = s[by].to_numpy()[1:] == s[by].to_numpy()[:-1]
        return set(zip(s.index.to_numpy()[:-1][same], s.index.to_numpy()[1:][same]))
    made, true = links(tracked, "track_id"), links(tracked, "true_id")
    return len(made & true) / max(len(made), 1), len(made & true) / max(len(true), 1)


def id_switches(tracked: pd.DataFrame) -> int:
    """Number of times a true bacterium changes track_id along its detections."""
    s = tracked.sort_values(["true_id", "frame"])
    same_bacterium = s.true_id.to_numpy()[1:] == s.true_id.to_numpy()[:-1]
    changed = s.track_id.to_numpy()[1:] != s.track_id.to_numpy()[:-1]
    return int(np.sum(same_bacterium & changed))
