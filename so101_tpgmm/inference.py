"""From a fixed-camera image to a joint trajectory for the SO-101.

Pipeline: detect cubes (HSV) → validate (partial blobs are skipped) → cube pose in the base
frame (calibrated homography, yaw folded to ±45°) → order by the configured colour list →
for every cube: Cartesian trajectory from the policy (segmented TP-GMM by default, or the
monolithic TP-GMM / IK planner) starting at the robot's current pose → joint trajectory
by IK (pitch output as soft target) → feasibility check (a cube with any unreachable point
is skipped and reported, never clipped) → joint-space return to home → next cube.

Joint outputs are in the dataset/LeRobot units: degrees (``DEGREES`` mode) and the gripper
in 0-100, ready for ``SO101Follower.send_action``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from so101_tpgmm.kinematics.ik import SO101IK, approach_pitch, tool_yaw
from so101_tpgmm.planner.ik_planner import min_jerk
from so101_tpgmm.tpgmm.cube_symmetry import ambiguous_candidates, heading_target
from so101_tpgmm.tpgmm.demos import heading_from_tool_yaw, tool_yaw_from_heading
from so101_tpgmm.vision.detection import CubeDetection, DetectorParams, detect_cubes
from so101_tpgmm.vision.geometry import PixelToBase
from so101_tpgmm.vision.scene import SceneParams, validate_detections

JOINT_KEYS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


@dataclass
class CubePlan:
    """Plan (or reason for skipping) of one detected cube."""

    color: str
    position: np.ndarray  # base frame [m]
    yaw: float  # [rad], folded
    order: int
    status: str = "planned"  # planned | skipped: <reason>
    ik_feasible: float = float("nan")
    n_samples: int = 0
    grasp_yaw: float = float("nan")  # chosen equivalent cube yaw (jaw heading frame) [rad]
    ambiguous: bool = False  # two equivalent yaws were close: chosen by IK feasibility / joint motion


@dataclass
class Plan:
    """Concatenated joint trajectory for all cubes."""

    t: np.ndarray  # (N,) [s]
    joints_deg: np.ndarray  # (N, 5)
    gripper: np.ndarray  # (N,)
    cartesian: np.ndarray  # (N, 6) [x, y, z, h, g, pitch] (NaN during joint-space moves)
    phase: list[str]  # per sample
    cube_index: np.ndarray  # (N,) -1 = return home
    cubes: list[CubePlan] = field(default_factory=list)
    detections: list[CubeDetection] = field(default_factory=list)


def cartesian_state(ik: SO101IK, q_deg: np.ndarray, gripper: float) -> np.ndarray:
    """[x, y, z, h, g, pitch] of the robot at joints ``q_deg``."""
    T = ik.kin.fk(np.deg2rad(q_deg[:5]))
    return np.r_[T[:3, 3], heading_from_tool_yaw(tool_yaw(T[:3, :3])), gripper, approach_pitch(T, ik.pan_axis_xy)]


def joint_move(q_from: np.ndarray, q_to: np.ndarray, g_from: float, g_to: float, duration: float, dt: float):
    n = max(2, int(round(duration / dt)) + 1)
    s = min_jerk(np.linspace(0, 1, n))[:, None]
    return q_from + (q_to - q_from) * s, g_from + (g_to - g_from) * s[:, 0]


def order_cubes(dets: list[CubeDetection], xy: np.ndarray, color_order: list[str]) -> list[int]:
    """Indices sorted by colour priority, then by distance to the robot."""
    rank = {c: i for i, c in enumerate(color_order)}
    return sorted(range(len(dets)), key=lambda i: (rank.get(dets[i].color, len(rank)), float(np.hypot(*xy[i]))))


def plan_cube(policy, ik: SO101IK, cp: CubePlan, q_deg: np.ndarray, g: float, cfg: dict, heading: float | None = None,
              box: np.ndarray | None = None):
    """Cartesian rollout and IK for one cube from joints ``q_deg`` (fills ``cp``'s status fields).

    With ``heading`` None and an ambiguous cube (two equivalent yaws within the margin), both are
    planned and the IK-feasible one with the least joint motion is kept. ``box``: box frame (x, y, z, yaw)
    for the transport primitive (default: the training box). Returns (rollout, IK result),
    or (None, None) when the cube is skipped (``cp.status`` says why).
    """
    icfg = cfg["inference"]
    dt = float(icfg["dt"])
    start = cartesian_state(ik, q_deg, g)
    if heading is None:
        target = heading_target(cp.position[:2], np.asarray(policy.meta["pan_xy"]), policy.meta["heading_offset"])
        options = ambiguous_candidates(cp.yaw, target, np.deg2rad(icfg.get("ambiguity_margin_deg", 10.0)))
    else:
        options = [heading]
    cp.ambiguous = len(options) > 1
    best = None
    for h in options:  # with an ambiguous cube, keep the feasible option with least joint motion
        ro_c = policy.rollout(cp.position, cp.yaw, start=start, dt=dt, heading=h, box=box)
        res_c = ik.solve_trajectory(ro_c.Y[:, :3], tool_yaw_from_heading(ro_c.Y[:, ro_c.names.index("h")]),
                                    np.deg2rad(q_deg), max_restarts=3,
                                    pitch=ro_c.Y[:, ro_c.names.index("pitch")] if "pitch" in ro_c.names else None)
        score = (res_c.success.mean(), -float(np.abs(np.diff(res_c.q, axis=0)).sum()))
        if best is None or score > best[0]:
            best = (score, ro_c, res_c, h)
    _, ro, res, cp.grasp_yaw = best
    cp.ik_feasible, cp.n_samples = res.feasible_fraction, len(ro.Y)
    if icfg.get("require_full_ik", True) and not res.success.all():
        bad = np.flatnonzero(~res.success)
        cp.status = (f"skipped: {len(bad)} unreachable points (first at t={ro.t[bad[0]]:.2f} s, "
                     f"pos err {res.pos_err[bad[0]] * 1e3:.1f} mm)")
        return None, None
    return ro, res


def plan_from_image(rgb: np.ndarray, policy, ik: SO101IK, p2b: PixelToBase, cube_z: float, cfg: dict,
                    start_joints_deg: np.ndarray | None = None, start_gripper: float | None = None,
                    colors: list[str] | None = None) -> Plan:
    """Full pipeline (see module docstring)."""
    icfg, vcfg = cfg["inference"], cfg["vision"]
    dp = DetectorParams.from_config(vcfg)
    sp = SceneParams.from_config(vcfg["scene"], cfg["scene"]["cube_side"])
    dets = detect_cubes(rgb, dp)
    validate_detections(dets, p2b, sp)
    xy = p2b.to_base(np.array([d.center_px for d in dets])) if dets else np.zeros((0, 2))
    order = order_cubes(dets, xy, colors or icfg["color_order"])
    home = np.asarray(icfg["home_joints_deg"], dtype=float)
    q = np.asarray(home if start_joints_deg is None else start_joints_deg, dtype=float)
    g = float(icfg["home_gripper"] if start_gripper is None else start_gripper)
    dt = float(icfg["dt"])
    T, Q, G, C, PH, CI = [], [], [], [], [], []
    t0 = 0.0

    def append(qs, gs, cart, phase, ci):
        nonlocal t0
        n = len(qs)
        tt = t0 + dt * np.arange(n)
        T.append(tt), Q.append(qs), G.append(gs), C.append(cart), PH.extend([phase] * n), CI.append(np.full(n, ci))
        t0 = tt[-1] + dt

    cubes = []
    for rank_i, i in enumerate(order):
        d = dets[i]
        cp = CubePlan(d.color, np.r_[xy[i], cube_z], p2b.yaw_to_base(d.center_px, d.yaw_img), rank_i)
        cubes.append(cp)
        if "partial_blob" in d.flags:
            cp.status = "skipped: partial blob (cube partly hidden or lost in the mask)"
            continue
        ro, res = plan_cube(policy, ik, cp, q, g, cfg)
        if ro is None:
            continue
        qs = np.rad2deg(res.q)
        gs = np.clip(ro.Y[:, ro.names.index("g")], 0, 100)
        for k, name in enumerate(policy.primitives):
            m = ro.primitive == k
            append(qs[m], gs[m], ro.Y[m], name, rank_i)
        qh, gh = joint_move(qs[-1], home, gs[-1], float(icfg["home_gripper"]), float(icfg["return_home_s"]), dt)
        append(qh[1:], gh[1:], np.full((len(qh) - 1, ro.Y.shape[1]), np.nan), "return_home", -1)
        q, g = home.copy(), float(icfg["home_gripper"])
    if not T:
        return Plan(np.zeros(0), np.zeros((0, 5)), np.zeros(0), np.zeros((0, 6)), [], np.zeros(0, int), cubes, dets)
    return Plan(np.concatenate(T), np.vstack(Q), np.concatenate(G), np.vstack(C), PH, np.concatenate(CI), cubes, dets)


def save_plan(plan: Plan, path_stem, meta: dict) -> None:
    """Write ``<stem>.npz`` and ``<stem>.csv`` (LeRobot action keys as CSV columns)."""
    import json
    from pathlib import Path

    import pandas as pd

    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    cubes = [{"color": c.color, "position": c.position.tolist(), "yaw_deg": float(np.degrees(c.yaw)),
              "grasp_yaw_deg": float(np.degrees(c.grasp_yaw)), "ambiguous": c.ambiguous,
              "order": c.order, "status": c.status, "ik_feasible": c.ik_feasible} for c in plan.cubes]
    np.savez(stem.with_suffix(".npz"), t=plan.t, joints_deg=plan.joints_deg, gripper=plan.gripper,
             cartesian=plan.cartesian, phase=np.array(plan.phase), cube_index=plan.cube_index,
             action_keys=np.array(JOINT_KEYS), meta=json.dumps({**meta, "cubes": cubes}))
    df = pd.DataFrame(np.c_[plan.joints_deg, plan.gripper], columns=JOINT_KEYS)
    df.insert(0, "t", plan.t)
    df["phase"] = plan.phase
    df["cube_index"] = plan.cube_index
    df.round(5).to_csv(stem.with_suffix(".csv"), index=False)
