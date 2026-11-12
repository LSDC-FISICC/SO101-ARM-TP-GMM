"""Offline metrics comparing a generated rollout with a held-out demonstration.

Time alignment: the policy is run with the demonstration's own primitive durations, so
the comparison measures *where* the policy goes, not how fast the operator was. The
monolithic model has no primitive boundary: its grasp instant is where its predicted
gripper closes (and, for reference, the demonstration's contact time — "oracle timing").
"""

from __future__ import annotations

import numpy as np

from so101_tpgmm.kinematics.ik import SO101IK
from so101_tpgmm.tpgmm.demos import Demo, tool_yaw_from_heading
from so101_tpgmm.tpgmm.policy import Rollout


def resample(t: np.ndarray, Y: np.ndarray, tq: np.ndarray, angular: tuple[int, ...] = ()) -> np.ndarray:
    Y = np.array(Y, dtype=float)
    for a in angular:
        Y[:, a] = np.unwrap(Y[:, a])
    return np.stack([np.interp(tq, t, Y[:, j]) for j in range(Y.shape[1])], axis=1)


def demo_cycle(dA: Demo, dT: Demo) -> tuple[np.ndarray, np.ndarray]:
    """Time [s] and outputs of a demonstrated cycle (approach then transport)."""
    t = np.r_[dA.X[:, 0] * dA.duration, dA.duration + dT.X[1:, 0] * dT.duration]
    return t, np.vstack([dA.X[:, 1:], dT.X[1:, 1:]])


def closing_index(g: np.ndarray) -> int:
    """Predicted grasp instant of a trajectory without primitive boundaries: after the gripper's
    opening peak, the first sample below the midpoint between that peak and the following minimum."""
    i_max = int(np.argmax(g[: max(2, int(0.8 * len(g)))]))
    after = g[i_max:]
    lo = after[: max(2, int(0.6 * len(after)))].min()
    mid = 0.5 * (g[i_max] + lo)
    below = np.flatnonzero(after <= mid)
    return i_max + int(below[0]) if below.size else i_max


def cycle_metrics(ro: Rollout, dA: Demo, dT: Demo, monolithic: bool) -> dict[str, float]:
    """Trajectory RMSE, grasp/release errors and heading error for one held-out cycle."""
    t_d, Y_d = demo_cycle(dA, dT)
    tq = np.linspace(0, t_d[-1], 200)
    P = resample(ro.t, ro.Y, tq, (3,))
    D = resample(t_d, Y_d, tq, (3,))
    dh = np.angle(np.exp(1j * (P[:, 3] - D[:, 3])))
    out = {
        "rmse_pos_mm": float(np.sqrt(np.mean(np.sum((P[:, :3] - D[:, :3]) ** 2, axis=1))) * 1e3),
        "rmse_heading_deg": float(np.degrees(np.sqrt(np.mean(dh**2)))),
        "rmse_gripper": float(np.sqrt(np.mean((P[:, 4] - D[:, 4]) ** 2))),
        "release_err_mm": float(np.linalg.norm(ro.Y[-1, :3] - dT.X[-1, 1:4]) * 1e3),
    }
    grasp_true = dA.X[-1, 1:4]
    i_oracle = int(np.argmin(np.abs(ro.t - dA.duration)))
    if monolithic:
        ig = closing_index(ro.Y[:, 4])
    else:
        ig = ro.boundaries[0]
    out["grasp_err_mm"] = float(np.linalg.norm(ro.Y[ig, :3] - grasp_true) * 1e3)
    out["grasp_err_oracle_timing_mm"] = float(np.linalg.norm(ro.Y[i_oracle, :3] - grasp_true) * 1e3)
    out["grasp_time_err_s"] = float(ro.t[ig] - dA.duration)
    out["grasp_heading_err_deg"] = float(np.degrees(abs(np.angle(np.exp(1j * (ro.Y[ig, 3] - dA.X[-1, 4]))))))
    # per-primitive RMSE on normalized time (monolithic: split at the demo's contact time)
    for name, sl_pred, d in (("approach", slice(0, i_oracle + 1), dA), ("transport", slice(i_oracle, None), dT)):
        seg = ro.Y[sl_pred]
        if len(seg) < 2:
            out[f"rmse_{name}_mm"] = float("nan")
            continue
        s = np.linspace(0, 1, len(seg))
        Pn = resample(s, seg, np.linspace(0, 1, 50), (3,))
        out[f"rmse_{name}_mm"] = float(np.sqrt(np.mean(np.sum((Pn[:, :3] - d.X[:, 1:4]) ** 2, axis=1))) * 1e3)
    return out


def ik_feasibility(ro: Rollout, ik: SO101IK, q0_rad: np.ndarray, stride: int = 2) -> dict[str, float]:
    """Fraction of rollout points with an IK solution inside the joint limits."""
    idx = np.arange(0, len(ro.Y), stride)
    names = ro.names
    pitch = ro.Y[idx, names.index("pitch")] if "pitch" in names else None
    res = ik.solve_trajectory(ro.Y[idx, :3], tool_yaw_from_heading(ro.Y[idx, names.index("h")]), q0_rad,
                              max_restarts=3, pitch=pitch)
    return {"ik_feasible": res.feasible_fraction, "ik_max_step_deg": float(np.degrees(res.max_joint_step[1:].max()))}


def _resample_arclength(P: np.ndarray, n: int = 200) -> np.ndarray:
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if s[-1] < 1e-9:
        return np.repeat(P[:1], n, axis=0)
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, P[:, j]) for j in range(P.shape[1])], axis=1)


def path_deviation(P_pred: np.ndarray, P_demo: np.ndarray) -> float:
    """Time-free path distance [m]: symmetric mean of closest-point distances between the two
    paths, both resampled uniformly in arc length."""
    A, B = _resample_arclength(P_pred), _resample_arclength(P_demo)
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return float(0.5 * (D.min(axis=1).mean() + D.min(axis=0).mean()))


def retime(ro: Rollout, durations: list[float]) -> Rollout:
    """Rescale each primitive of a rollout to the given durations (for a policy without a clock, e.g.
    a dynamical system, before time-aligned metrics)."""
    t = np.empty_like(ro.t)
    t0, start = 0.0, 0
    for i, end in enumerate(ro.boundaries):
        seg = ro.t[start:end + 1]
        span = seg[-1] - seg[0] if len(seg) > 1 else 1.0
        t[start:end + 1] = t0 + (seg - seg[0]) / max(span, 1e-9) * durations[i]
        t0, start = t[end], end + 1
    t[start:] = t0
    return Rollout(t, ro.Y, ro.cov, ro.primitive, ro.names, ro.boundaries, ro.cube_heading, ro.cube_k)
