"""One simulated task episode: random cubes, a learned policy, MuJoCo physics.

Per cube (in the configured colour order), exactly the deployment pipeline minus the camera:
current measured joints → policy rollout from that state → IK (pitch as soft target, a cube with
any unreachable point is skipped) → joint trajectory sent to the simulated position servos →
joint-space return home → next cube.

**"Known position" convention.** The policies were trained with cube frames from the fixed camera,
whose cube position is biased by ~2 cm with respect to the fingers of the FK model (cube top face
seen in perspective + joint-calibration error; ``docs/mujoco_sim.md``). Giving them the physical
cube centre would reproduce that calibration bias, not test the policy. So the position handed to
the policy is the one in *its* convention that makes its nominal grasp centre the fingers on the
cube, with the finger tips ``tip_height`` above the table: found by a few fixed-point iterations
on the policy's own rollout (rollout → IK of the grasp sample → finger geometry → correction).
This is what a perfectly calibrated perception would deliver. ``noise_mm`` adds Gaussian
perception noise on top of it (the vision error of the real pipeline is ~8 mm median).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from so101_tpgmm.evaluation.metrics import closing_index
from so101_tpgmm.inference import CubePlan, cartesian_state, joint_move, plan_cube
from so101_tpgmm.kinematics.ik import SO101IK
from so101_tpgmm.sim.runner import Simulator, gripper_to_rad
from so101_tpgmm.sim.scene import SceneSpec, SimCube, build_model
from so101_tpgmm.tpgmm.cube_symmetry import ambiguous_candidates, heading_target
from so101_tpgmm.tpgmm.demos import tool_yaw_from_heading


class FingerGeometry:
    """Finger-tip midpoint and lowest finger point for a joint configuration (kinematics only)."""

    def __init__(self, spec: SceneSpec, open_cmd: float = 32.0) -> None:
        import mujoco

        self.mj = mujoco
        self.model = build_model(SceneSpec(spec.robot_xml, table_plane=spec.table_plane, finger_friction=spec.finger_friction, cubes=[]))
        self.data = mujoco.MjData(self.model)
        names = [self.model.geom(g).name for g in range(self.model.ngeom)]
        self.fixed = [g for g, n in enumerate(names) if n.startswith("gripper_pad")]
        self.jaw = [g for g, n in enumerate(names) if n.startswith("moving_jaw_so101_v1_pad")]
        self.jaw_rad = float(gripper_to_rad(open_cmd))
        self._signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])

    def __call__(self, q_deg: np.ndarray) -> tuple[np.ndarray, float]:
        d, m = self.data, self.model
        d.qpos[:5] = np.deg2rad(q_deg[:5])
        d.qpos[5] = self.jaw_rad
        self.mj.mj_kinematics(m, d)

        def tip(G):  # mean of the two lowest slices of a finger
            return np.mean(sorted((d.geom_xpos[g].copy() for g in G), key=lambda v: v[2])[:2], axis=0)

        low = min(((self._signs * m.geom_size[g]) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()
                  for g in self.fixed + self.jaw)
        return (tip(self.fixed) + tip(self.jaw)) / 2, float(low)


def grasp_index(ro) -> int:
    return int(ro.boundaries[0]) if len(ro.primitive) and ro.primitive.max() > 0 else closing_index(ro.Y[:, 4])


def policy_input(policy, ik: SO101IK, geom: FingerGeometry, spec: SceneSpec, cube: SimCube, heading: float,
                 q_deg: np.ndarray, g: float, cfg: dict, tip_height: float = 0.01, iters: int = 4,
                 tol: float = 1e-3) -> tuple[np.ndarray, float]:
    """Cube position in the policy's convention (see module docstring) and the residual [m]."""
    dt = float(cfg["inference"]["dt"])
    start = cartesian_state(ik, q_deg, g)
    p = np.array([cube.xyz[0], cube.xyz[1], float(cfg["vision"].get("plane_z", 0.0064))])
    err = np.inf
    for _ in range(iters):
        ro = policy.rollout(p, cube.yaw, start=start, dt=dt, heading=heading)
        i = grasp_index(ro)
        y = ro.Y[i]
        pitch = y[ro.names.index("pitch")] if "pitch" in ro.names else None
        r = ik.solve(y[:3], float(tool_yaw_from_heading(y[3])), np.deg2rad(q_deg), max_restarts=3, pitch=pitch)
        mid, low = geom(np.rad2deg(r.q))
        e = np.array([cube.xyz[0] - mid[0], cube.xyz[1] - mid[1],
                      spec.table_height(cube.xyz[0], cube.xyz[1]) + tip_height - low])
        p = p + e
        err = float(np.linalg.norm(e))
        if err < tol:
            break
    return p, err


def sample_cubes(rng: np.random.Generator, demo_xy: np.ndarray, cfg: dict) -> list[SimCube]:
    """Cubes at demonstrated positions + a uniform disc (stays in the trained workspace), random yaw,
    separated from each other and from the box."""
    s = cfg["sim"]
    colors, sep, rad = list(s["colors"])[: int(s["n_cubes"])], float(s["min_separation"]), float(s["sample_radius"])
    bc = np.asarray(s["box"]["center_xy"], float)
    keep_out = np.asarray(s["box"]["inner"], float) / 2 + s["box"]["wall_t"] + float(s["box_clearance"]) + 0.02
    out: list[SimCube] = []
    for _ in range(10000):
        if len(out) == len(colors):
            break
        a, r = rng.uniform(0, 2 * np.pi), rad * np.sqrt(rng.uniform())
        xy = demo_xy[rng.integers(len(demo_xy))] + r * np.array([np.cos(a), np.sin(a)])
        if np.all(np.abs(xy - bc) < keep_out) or any(np.linalg.norm(xy - c.xyz[:2]) < sep for c in out):
            continue
        out.append(SimCube(colors[len(out)], np.r_[xy, 0.0], float(rng.uniform(-np.pi / 4, np.pi / 4))))
    if len(out) < len(colors):
        raise RuntimeError("could not place the cubes")
    return out


@dataclass
class CubeOutcome:
    color: str
    xyz0: list
    yaw0_deg: float
    status: str = "planned"  # planned | skipped: ...
    ambiguous: bool = False
    input_residual_mm: float = float("nan")
    lifted: bool = False  # rose > 3 cm above its start while the gripper was closed
    in_box: bool = False  # at the end of the episode (inside the walls, possibly resting on another cube)
    final_xyz: list = field(default_factory=list)
    displaced_mm: float = 0.0  # how far the cube moved before its own turn (pushed by earlier cycles)
    held_at_release: bool = False  # still in the gripper (> 3 cm above its start) when the gripper opened
    released_over_box: bool = False  # cube inside the box footprint when the gripper opened
    cycle_s: float = float("nan")


def release_index(closed: np.ndarray) -> int | None:
    """Last sample before the release: the gripper starts closed (rest), opens for the approach, closes on
    the cube (grasp) and opens again (release)."""
    opened = np.flatnonzero(~closed)
    if not len(opened):
        return None
    grasp = np.flatnonzero(closed[opened[0]:])
    if not len(grasp):
        return None
    g0 = opened[0] + grasp[0]
    rel = np.flatnonzero(~closed[g0:])
    return int(g0 + rel[0] - 1) if len(rel) else None


def in_box(spec: SceneSpec, p: np.ndarray, footprint: bool = False) -> bool:
    """Inside the walls (possibly resting on another cube); ``footprint``: inside in xy at any height."""
    R = spec.table_rotation()
    c = np.array([*spec.box_center_xy, spec.table_height(*spec.box_center_xy)])
    loc = R.T @ (p - c)
    ix, iy = np.asarray(spec.box_inner) / 2
    inside = abs(loc[0]) < ix and abs(loc[1]) < iy
    return bool(inside if footprint else inside and loc[2] < spec.box_wall_h + spec.cube_side)  # stacked counts


def run_episode(policy, ik: SO101IK, cfg: dict, spec: SceneSpec, geom: FingerGeometry, rng: np.random.Generator,
                noise_mm: float = 0.0, grasp_mode: str = "contact", render: tuple[int, int] | None = None,
                camera=None, render_every: int = 2, frame_callback=None,
                box_offsets: list | None = None) -> tuple[list[CubeOutcome], list]:
    """Pick and place every cube of ``spec`` with ``policy``; returns per-cube outcomes and video frames
    (``frame_callback(sim)``, if given, is called instead of the built-in renderer).

    ``box_offsets``: optional [dx, dy] per picked cube (in order) added to the policy's box frame, so the
    k-th cube is released in its own slot of the box (a task parameter the policies were never trained
    to vary: the box did not move in the demonstrations)."""
    icfg = cfg["inference"]
    dt = float(icfg["dt"])
    home = np.asarray(icfg["home_joints_deg"], float)
    g_home = float(icfg["home_gripper"])
    sim = Simulator(spec, render=render, camera=camera, grasp_mode=grasp_mode)
    sim.frame_callback = frame_callback
    sim.reset(home, g_home)
    rank = {c: i for i, c in enumerate(icfg["color_order"])}
    order = sorted(range(len(spec.cubes)), key=lambda i: (rank.get(spec.cubes[i].color, 99),
                                                         float(np.hypot(*spec.cubes[i].xyz[:2]))))
    start_pos = sim.cube_positions()
    outcomes = [CubeOutcome(c.color, np.round(start_pos[i], 4).tolist(), float(np.degrees(c.yaw)))
                for i, c in enumerate(spec.cubes)]
    frames = []
    for n_done, i in enumerate(order):
        box = None
        if box_offsets:
            off = box_offsets[n_done % len(box_offsets)]
            box = np.asarray(policy.meta["box"], float) + np.r_[off[0], off[1], 0.0, 0.0]
        cube, oc = spec.cubes[i], outcomes[i]
        now = sim.cube_positions()[i]
        oc.displaced_mm = float(np.linalg.norm(now[:2] - start_pos[i, :2]) * 1e3)
        cube_now = SimCube(cube.color, now.copy(), cube.yaw)  # the position is known (current, not initial)
        q_deg, g = sim.joints_deg_gripper()
        target = heading_target(now[:2], np.asarray(policy.meta["pan_xy"]), policy.meta["heading_offset"])
        options = ambiguous_candidates(cube.yaw, target, np.deg2rad(icfg.get("ambiguity_margin_deg", 10.0)))
        best = None
        for h in options:
            p, res_err = policy_input(policy, ik, geom, spec, cube_now, h, q_deg, g, cfg)
            if noise_mm > 0:
                p[:2] += rng.normal(0, noise_mm * 1e-3, 2)
            cp = CubePlan(cube.color, p, cube.yaw, len(frames))
            ro, res = plan_cube(policy, ik, cp, q_deg, g, cfg, heading=h, box=box)
            score = (ro is not None, -float(np.abs(np.diff(res.q, axis=0)).sum()) if ro is not None else 0.0)
            if best is None or score > best[0]:
                best = (score, cp, ro, res, res_err)
        _, cp, ro, res, res_err = best
        oc.ambiguous, oc.input_residual_mm = len(options) > 1, res_err * 1e3
        if ro is None:
            oc.status = cp.status
            continue
        qs, gs = np.rad2deg(res.q), np.clip(ro.Y[:, ro.names.index("g")], 0, 100)
        qh, gh = joint_move(qs[-1], home, gs[-1], g_home, float(icfg["return_home_s"]), dt)
        out = sim.execute(np.vstack([qs, qh[1:]]), np.r_[gs, gh[1:]], dt, render_every=render_every)
        z = out.cubes_xyz[:, i, 2]
        g_all = np.r_[gs, gh[1:]]
        closed = g_all < sim.open_threshold
        oc.lifted = bool(np.any((z - now[2] > 0.03) & closed))
        k_rel = release_index(closed)
        if k_rel is not None:
            oc.held_at_release = bool(z[k_rel] - now[2] > 0.03)
            oc.released_over_box = bool(oc.held_at_release and in_box(spec, out.cubes_xyz[k_rel, i], footprint=True))
        oc.cycle_s = float(len(qs) * dt)
        frames += out.frames
    end = sim.cube_positions()
    for i, oc in enumerate(outcomes):
        oc.final_xyz = np.round(end[i], 4).tolist()
        oc.in_box = in_box(spec, end[i])
    return outcomes, frames
