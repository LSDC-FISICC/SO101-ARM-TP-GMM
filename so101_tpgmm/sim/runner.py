"""Execute joint trajectories (LeRobot units) in the MuJoCo scene and measure the outcome.

Commands go through the model's position actuators at the control rate of the plan (30 Hz,
zero-order hold), exactly as ``SO101Follower.send_action`` would receive them: joints in degrees
(DEGREES mode) and the gripper in 0-100 (RANGE_0_100). The gripper command is mapped linearly to
the jaw joint range of the model (0 → closed limit, 100 → open limit); see ``gripper_to_rad``.
With this map the finger tips touch at 0 and are 4 cm apart at ~26; the demonstrated holding
command (~20) squeezes a 4 cm cube, as on the real arm.

Grasp modes:

* ``contact``: pure physics (friction between finger pads and cube).
* ``assist``: when the gripper is closing and **both** fingers touch a cube, that cube is welded to the
  gripper at its current relative pose until the command opens again. The approach still has to put
  the cube between the fingers (a cube knocked over or missed is not grasped), but holding no longer
  depends on friction, cube mass and pad compliance, which were not measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from so101_tpgmm.sim.scene import JOINTS, SceneSpec, build_model

GRIPPER_RANGE = (-0.17453297762778586, 1.7453291995659765)  # jaw joint limits in the MJCF [rad]


def gripper_to_rad(g: np.ndarray | float, gripper_range=GRIPPER_RANGE) -> np.ndarray:
    lo, hi = gripper_range
    return lo + np.clip(np.asarray(g, float), 0, 100) / 100.0 * (hi - lo)


@dataclass
class SimResult:
    t: np.ndarray
    q: np.ndarray  # (N, 6) measured joints [rad] at the control rate
    cubes_xyz: np.ndarray  # (N, n_cubes, 3)
    tracking_err_deg: np.ndarray  # (N, 5) command − measured
    frames: list = field(default_factory=list)


class Simulator:
    def __init__(self, spec: SceneSpec, render: tuple[int, int] | None = None, camera=None,
                 grasp_mode: str = "contact", open_threshold: float = 30.0) -> None:
        import mujoco

        self.mj = mujoco
        self.spec = spec
        self.model = build_model(spec)
        self.data = mujoco.MjData(self.model)
        self.act = [self.model.actuator(j).id for j in JOINTS]
        self.qadr = [self.model.jnt_qposadr[self.model.joint(j).id] for j in JOINTS]
        self.cube_bodies = [self.model.body(f"cube_{i}").id for i in range(len(spec.cubes))]
        self.renderer = mujoco.Renderer(self.model, *render) if render else None
        self.camera = camera
        self.gripper_range = tuple(self.model.jnt_range[self.model.joint("gripper").id])
        self.grasp_mode, self.open_threshold = grasp_mode, open_threshold
        self.welds = [self.model.equality(f"grasp_{i}").id for i in range(len(spec.cubes))]
        self.attached: int | None = None
        geom_name = lambda g: self.model.geom(g).name  # noqa: E731
        self.fixed_pads = {g for g in range(self.model.ngeom) if geom_name(g).startswith("gripper_pad")}
        self.jaw_pads = {g for g in range(self.model.ngeom) if geom_name(g).startswith("moving_jaw_so101_v1_pad")}
        self.cube_geoms = {self.model.geom(f"cube_{i}").id: i for i in range(len(spec.cubes))}
        self.gripper_body = self.model.body("gripper").id
        self.frame_callback = None  # optional f(sim) called every ``render_every`` control steps (video)

    def reset(self, q_deg: np.ndarray, gripper: float) -> None:
        q = np.r_[np.deg2rad(q_deg[:5]), gripper_to_rad(gripper, self.gripper_range)]
        self.mj.mj_resetData(self.model, self.data)
        self.attached = None
        for a, v in zip(self.qadr, q):
            self.data.qpos[a] = v
        self.data.ctrl[self.act] = q
        self.mj.mj_forward(self.model, self.data)
        self.settle(0.5)

    def settle(self, seconds: float) -> None:
        for _ in range(int(seconds / self.model.opt.timestep)):
            self.mj.mj_step(self.model, self.data)

    def joints(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self.qadr])

    def joints_deg_gripper(self) -> tuple[np.ndarray, float]:
        q = self.joints()
        lo, hi = self.gripper_range
        return np.rad2deg(q[:5]), float((q[5] - lo) / (hi - lo) * 100)

    def pinched_cube(self) -> int | None:
        """Index of a cube touched by both fingers, if any."""
        touch: dict[int, set[str]] = {}
        for k in range(self.data.ncon):
            a, b = self.data.contact[k].geom1, self.data.contact[k].geom2
            for x, y in ((a, b), (b, a)):
                if x in self.cube_geoms:
                    side = "fixed" if y in self.fixed_pads else "jaw" if y in self.jaw_pads else None
                    if side:
                        touch.setdefault(self.cube_geoms[x], set()).add(side)
        both = [i for i, s in touch.items() if s == {"fixed", "jaw"}]
        return both[0] if both else None

    def _update_assist(self, g_cmd: float) -> None:
        mj, m, d = self.mj, self.model, self.data
        if self.attached is not None and g_cmd > self.open_threshold:
            d.eq_active[self.welds[self.attached]] = 0
            self.attached = None
        elif self.attached is None and g_cmd < self.open_threshold:
            i = self.pinched_cube()
            if i is not None:
                b1, b2 = self.gripper_body, self.cube_bodies[i]
                R1 = d.xmat[b1].reshape(3, 3)
                rel = R1.T @ (d.xpos[b2] - d.xpos[b1])
                q1inv, q = np.zeros(4), np.zeros(4)
                mj.mju_negQuat(q1inv, d.xquat[b1])
                mj.mju_mulQuat(q, q1inv, d.xquat[b2])
                e = self.welds[i]
                m.eq_data[e, 0:3] = 0.0
                m.eq_data[e, 3:6] = rel
                m.eq_data[e, 6:10] = q
                m.eq_data[e, 10] = 1.0
                d.eq_active[e] = 1
                self.attached = i

    def cube_positions(self) -> np.ndarray:
        return np.array([self.data.xpos[b].copy() for b in self.cube_bodies]).reshape(-1, 3)

    def execute(self, joints_deg: np.ndarray, gripper: np.ndarray, dt: float, render_every: int = 1) -> SimResult:
        """Send the plan sample by sample (zero-order hold for ``dt``) and record the state."""
        n_sub = max(1, int(round(dt / self.model.opt.timestep)))
        T, Q, C, E, F = [], [], [], [], []
        for i in range(len(joints_deg)):
            cmd = np.r_[np.deg2rad(joints_deg[i, :5]), gripper_to_rad(gripper[i], self.gripper_range)]
            self.data.ctrl[self.act] = cmd
            if self.grasp_mode == "assist":
                self._update_assist(float(gripper[i]))
            for _ in range(n_sub):
                self.mj.mj_step(self.model, self.data)
            q = self.joints()
            T.append(self.data.time), Q.append(q), C.append(self.cube_positions())
            E.append(np.rad2deg(cmd[:5] - q[:5]))
            if i % render_every == 0:
                if self.frame_callback is not None:
                    self.frame_callback(self)
                elif self.renderer is not None:
                    self.renderer.update_scene(self.data, camera=self.camera if self.camera is not None else -1)
                    F.append(self.renderer.render().copy())
        return SimResult(np.array(T), np.array(Q), np.array(C), np.array(E), F)
