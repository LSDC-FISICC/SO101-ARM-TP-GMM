"""Demonstrations and task frames for TP-GMM, built from the segmentation and the cube frames.

Variables (global, base frame)
------------------------------
``[t, x, y, z, h, g, pitch]``:

* ``t`` in [0, 1]: normalized time within the primitive (input of the regression).
* ``x, y, z``: TCP position [m] from FK.
* ``h``: wrist yaw as the heading of the **outward** jaw axis, h = wrap(psi + pi), with psi the
  heading of the TCP x axis used by the IK. The TCP x axis points back towards the robot, so
  psi sits near ±180° and 170/366 trajectories cross the ±180° cut; h stays around the
  azimuth of the cube and never does. Unwrapped along each trajectory.
* ``g``: gripper (RANGE_0_100).
* ``pitch``: approach pitch [rad] (optional output; resolves the IK redundancy).

Frames
------
* **cube**: b = cube centre (vision by default, FK at contact as alternative), A = Rz(phi')
  on the position block and an offset phi' on h. The cube looks the same every 90°: among
  phi + k·90° the training picks the one closest to the gripper heading at contact, so the
  cube frame's x axis points (roughly) outward from the robot in every demonstration and
  the approach is consistent across demos. At execution the grasp heading is unknown; k is
  chosen with the heading predicted from the cube azimuth plus the median offset learned
  from the demonstrations (``choose_cube_heading``).
* **box**: b = box position (configured, or the mean release point), A = Rz(yaw_box).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from so101_tpgmm.kinematics.fk import SO101Kinematics
from so101_tpgmm.kinematics.ik import approach_pitch, tool_yaw
from so101_tpgmm.tpgmm.tpgmm import TaskFrame

VARS_ALL = ("t", "x", "y", "z", "h", "g", "pitch")
POS = slice(1, 4)
H_IDX = 4


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def heading_from_tool_yaw(psi: np.ndarray) -> np.ndarray:
    """Outward jaw heading h = wrap(psi + pi)."""
    return wrap(np.asarray(psi) + np.pi)


def tool_yaw_from_heading(h: np.ndarray) -> np.ndarray:
    """Inverse of :func:`heading_from_tool_yaw` (target for the IK)."""
    return wrap(np.asarray(h) - np.pi)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def make_frame(position: np.ndarray, yaw: float, dim: int) -> TaskFrame:
    """Task frame acting on [t, x, y, z, h, g, (pitch)]: rotation + offset on position, offset on h."""
    A = np.eye(dim)
    A[POS, POS] = rot_z(yaw)
    b = np.zeros(dim)
    b[POS] = position
    b[H_IDX] = yaw
    return TaskFrame(A, b)


def canonical_cube_yaw(phi: float, heading: float) -> tuple[float, int]:
    """Among phi + k·pi/2, the one closest to ``heading``; returns (phi', k)."""
    cands = phi + np.arange(4) * np.pi / 2
    k = int(np.argmin(np.abs(wrap(heading - cands))))
    return float(wrap(cands[k])), k


def choose_cube_heading(phi: float, cube_xy: np.ndarray, pan_xy: np.ndarray, delta: float) -> tuple[float, int]:
    """Execution-time choice of phi': closest to (cube azimuth + learned heading offset)."""
    az = float(np.arctan2(cube_xy[1] - pan_xy[1], cube_xy[0] - pan_xy[0]))
    return canonical_cube_yaw(phi, az + delta)


@dataclass
class Demo:
    """One primitive of one cycle, resampled in normalized time."""

    episode: int
    primitive: str
    X: np.ndarray  # (N, D) global variables
    cube: np.ndarray  # (4,) x, y, z, phi' (canonical)
    cube_k: int
    phi_raw: float
    box: np.ndarray  # (4,) x, y, z, yaw
    duration: float  # [s]

    def frames(self) -> list[TaskFrame]:
        d = self.X.shape[1]
        return [make_frame(self.cube[:3], self.cube[3], d), make_frame(self.box[:3], self.box[3], d)]

    def local(self) -> np.ndarray:
        """(P, N, D) data seen from each frame."""
        return np.stack([F.to_local(self.X) for F in self.frames()])


def trajectory_variables(state: np.ndarray, kin: SO101Kinematics, start: int, end: int, n: int,
                         with_pitch: bool = True) -> np.ndarray:
    """[t, x, y, z, h, g, (pitch)] of frames start..end (inclusive), resampled to ``n`` points."""
    T = kin.fk_from_dataset(state[start:end + 1], "degrees")
    pax = kin.chain[0].origin[:2, 3]
    h = np.unwrap(heading_from_tool_yaw(tool_yaw(T[:, :3, :3])))
    h -= 2 * np.pi * np.round(h[len(h) // 2] / (2 * np.pi))  # anchor the middle in (-pi, pi]
    cols = [T[:, 0, 3], T[:, 1, 3], T[:, 2, 3], h, state[start:end + 1, 5]]
    if with_pitch:
        cols.append(approach_pitch(T, pax))
    Y = np.stack(cols, axis=1)
    s = np.linspace(0, 1, len(Y))
    tq = np.linspace(0, 1, n)
    out = np.stack([np.interp(tq, s, Y[:, j]) for j in range(Y.shape[1])], axis=1)
    return np.c_[tq, out]


def primitive_window(row, primitive: str) -> tuple[int, int]:
    if primitive == "approach_grasp":
        return int(row.approach_start), int(row.close_idx)
    if primitive == "transport_release":
        return int(row.close_idx), int(row.release_end)
    if primitive == "cycle":
        return int(row.approach_start), int(row.release_end)
    raise KeyError(primitive)


def training_table(seg: pd.DataFrame, frames: pd.DataFrame, exclude_aborted: bool = True) -> pd.DataFrame:
    """Cycles usable for training: vision-usable and (optionally) without aborted attempts."""
    df = seg.merge(frames.drop(columns=[c for c in ("cycle",) if c in frames.columns]), on="episode")
    df = df[df.usable.astype(bool)]
    if exclude_aborted:
        df = df[~df.aborted_before.astype(bool)]
    return df.reset_index(drop=True)


def box_pose_from_config(cfg: dict, releases: np.ndarray) -> np.ndarray:
    """Box frame (x, y, z, yaw): fixed from the YAML, or the mean release TCP position."""
    b = cfg["tpgmm"]["frames"]["box"]
    if b.get("mode", "estimate") == "fixed":
        return np.r_[np.asarray(b["position"], dtype=float), np.deg2rad(b.get("yaw_deg", 0.0))]
    return np.r_[np.mean(releases, axis=0), np.deg2rad(b.get("yaw_deg", 0.0))]


def build_demos(df: pd.DataFrame, ds, kin: SO101Kinematics, cfg: dict, primitives: list[str],
                box: np.ndarray | None = None) -> tuple[dict[str, list[Demo]], dict]:
    """Demonstrations per primitive and the context needed at execution (box, heading offset)."""
    tcfg = cfg["tpgmm"]
    n = int(tcfg["samples_per_primitive"])
    src = tcfg["frames"]["cube_source"]
    with_pitch = "pitch" in tcfg["outputs"]
    pax = kin.chain[0].origin[:2, 3]
    states = {int(e): ds.episode(int(e)).state for e in df.episode}
    if box is None:
        rel = np.array([kin.fk_from_dataset(states[int(r.episode)][int(r.open_idx)], "degrees")[:3, 3]
                        for r in df.itertuples()])
        box = box_pose_from_config(cfg, rel)
    demos: dict[str, list[Demo]] = {p: [] for p in primitives}
    deltas = []
    for r in df.itertuples():
        st = states[int(r.episode)]
        Tg = kin.fk_from_dataset(st[int(r.close_idx)], "degrees")
        h_grasp = float(heading_from_tool_yaw(tool_yaw(Tg[:3, :3])))
        cx, cy, cz, phi = ((r.vision_x, r.vision_y, r.vision_z, r.vision_yaw) if src == "vision"
                           else (r.fk_x, r.fk_y, r.fk_z, r.fk_yaw))
        phi_c, k = canonical_cube_yaw(float(phi), h_grasp)
        az = np.arctan2(cy - pax[1], cx - pax[0])
        deltas.append(wrap(h_grasp - az))
        for p in primitives:
            s, e = primitive_window(r, p)
            X = trajectory_variables(st, kin, s, e, n, with_pitch)
            demos[p].append(Demo(int(r.episode), p, X, np.array([cx, cy, cz, phi_c]), k, float(phi), box.copy(),
                                 (e - s) / ds.fps))
    delta = float(np.angle(np.mean(np.exp(1j * np.array(deltas)))))  # circular mean
    return demos, {"box": box.tolist(), "heading_offset": delta, "pan_xy": pax.tolist(),
                   "var_names": list(VARS_ALL[: 6 + int(with_pitch)])}
