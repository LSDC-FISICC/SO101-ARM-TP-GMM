"""Gymnasium environment: SO-101 picks one cube and drops it in the box, in MuJoCo.

Observation and action: :class:`so101_tpgmm.rl.task.PickPlaceTask` (identical on the real robot).
The simulator is the one of ``scripts/simulate_mujoco.py`` (official SO-101 model, finger boxes,
table plane in FK coordinates, contact physics), built once and reset in place.

**Reward** (staged, as in robosuite's pick-and-place; uses the true simulated state, which the
policy does not observe):

* reach: 0.1 · (1 − tanh(10 d)), d = finger-tip midpoint ↔ cube centre;
* grasp: 0.35 while both fingers touch the cube;
* lift / carry: 0.35 + 0.15 · (1 − tanh(5 d_h)) while the cube is pinched and off the table,
  d_h = cube ↔ a point above the box;
* success: the cube rests inside the box, released → reward 1 for this step **and** for every
  remaining step (the episode ends early; equivalent to staying at the maximum, so finishing sooner
  is never worse than hovering);
* −0.05 when the IK cannot reach the commanded target; residual mode: −``rl.residual_action_penalty`` · mean(a²), so the
  policy departs from the TP-GMM only where it pays off.

The reward of a step is the maximum of the reached stages. **Domain randomization** (per episode,
``rl.domain_randomization``): table height, cube mass and friction, servo stiffness; the observed
cube pose is the true one plus noise (``rl.cube_obs_noise_*``).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from so101_tpgmm.config import resolve_path
from so101_tpgmm.rl.task import PickPlaceTask, TaskConfig
from so101_tpgmm.sim.episode import FingerGeometry, in_box, policy_input, sample_cubes
from so101_tpgmm.sim.runner import Simulator, gripper_to_rad
from so101_tpgmm.sim.scene import SceneSpec, SimCube
from so101_tpgmm.tpgmm.cube_symmetry import ambiguous_candidates, heading_target


def release_point(cfg: dict, spec: SceneSpec) -> np.ndarray:
    """Point above the middle of the box (observation target for the transport)."""
    bc = np.asarray(cfg["sim"]["box"]["center_xy"], float)
    return np.r_[bc, spec.table_height(*bc) + cfg["sim"]["box"]["wall_h"] + 0.04]


class SO101PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, cfg: dict, mode: str = "scratch", randomize: bool = True, seed: int | None = None,
                 base_policy: str = "segmented", render_size: tuple[int, int] | None = None) -> None:
        import pandas as pd

        from so101_tpgmm.kinematics.ik import ik_from_config

        self.cfg, self.mode, self.randomize = cfg, mode, randomize
        self.rcfg = cfg["rl"]
        xml = resolve_path(cfg["sim"]["robot_dir"]) / "so101_new_calib.xml"
        self.scene = SceneSpec.from_config(cfg, xml, [SimCube("red", np.array([0.25, 0.0, 0.0]), 0.0)])
        self.sim = Simulator(self.scene, render=render_size)
        self.ik = ik_from_config(cfg)
        f = pd.read_csv(resolve_path(cfg["vision"]["frames_csv"]))
        self.demo_xy = f[["fk_x", "fk_y"]].dropna().to_numpy()
        self.policy = None
        self.geom = None
        if mode == "residual":
            from so101_tpgmm.policies import load_policy

            self.policy = load_policy(cfg, base_policy)
            self.geom = FingerGeometry(self.scene)
        self.release = release_point(cfg, self.scene)
        self.task = PickPlaceTask(TaskConfig.from_config(cfg, mode), self.ik, self.release,
                                  table_height=self.scene.table_height, base_policy=self.policy)
        n = PickPlaceTask.obs_size(mode)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (PickPlaceTask.act_dim,), np.float32)
        m = self.sim.model
        self._cube_qadr = m.jnt_qposadr[m.joint("cube_0").id]
        self._cube_vadr = m.jnt_dofadr[m.joint("cube_0").id]
        self._cube_geom = m.geom("cube_0").id
        self._cube_body = m.body("cube_0").id
        self._table_geom = m.geom("table").id
        self._box_body = m.body("box").id
        self._nominal = dict(friction=m.geom_friction[self._cube_geom].copy(), mass=float(m.body_mass[self._cube_body]),
                             inertia=m.body_inertia[self._cube_body].copy(), gain=m.actuator_gainprm[:, 0].copy(),
                             bias=m.actuator_biasprm[:, 1].copy(), table=m.geom_pos[self._table_geom].copy(),
                             box=m.body_pos[self._box_body].copy())
        self.table_offset = 0.0
        self._seed = seed
        self.home = np.asarray(cfg["inference"]["home_joints_deg"], float)
        self.g_home = float(cfg["inference"]["home_gripper"])

    # ------------------------------------------------------------------ helpers
    def table_height(self, x: float, y: float) -> float:
        return self.scene.table_height(x, y) + self.table_offset

    def _randomize(self) -> None:
        m, nom, dr = self.sim.model, self._nominal, self.rcfg["domain_randomization"]
        rng = self.np_random
        on = self.randomize
        self.table_offset = rng.uniform(-1, 1) * dr["table_offset_mm"] * 1e-3 if on else 0.0
        m.geom_pos[self._table_geom] = nom["table"] + [0, 0, self.table_offset]
        m.body_pos[self._box_body] = nom["box"] + [0, 0, self.table_offset]
        mass = rng.uniform(*dr["cube_mass_kg"]) if on else nom["mass"]
        m.body_mass[self._cube_body] = mass
        m.body_inertia[self._cube_body] = nom["inertia"] * mass / nom["mass"]
        m.geom_friction[self._cube_geom, 0] = rng.uniform(*dr["cube_friction"]) if on else nom["friction"][0]
        kp = rng.uniform(*dr["servo_kp_scale"]) if on else 1.0
        m.actuator_gainprm[:, 0] = nom["gain"] * kp
        m.actuator_biasprm[:, 1] = nom["bias"] * kp

    def _place(self, cube: SimCube) -> None:
        mj, m, d = self.sim.mj, self.sim.model, self.sim.data
        mj.mj_resetData(m, d)
        self.sim.attached = None
        q = np.r_[np.deg2rad(self.home), gripper_to_rad(self.g_home, self.sim.gripper_range)]
        for a, v in zip(self.sim.qadr, q):
            d.qpos[a] = v
        d.ctrl[self.sim.act] = q
        pos, R = self.scene.cube_on_table(cube.xyz[0], cube.xyz[1], cube.yaw)
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, R.ravel())
        d.qpos[self._cube_qadr:self._cube_qadr + 3] = pos + [0, 0, self.table_offset]
        d.qpos[self._cube_qadr + 3:self._cube_qadr + 7] = quat
        mj.mj_forward(m, d)
        self.sim.settle(0.3)

    def finger_midpoint(self) -> np.ndarray:
        d = self.sim.data

        def tip(G):
            return np.mean(sorted((d.geom_xpos[g].copy() for g in G), key=lambda v: v[2])[:2], axis=0)

        return (tip(self.sim.fixed_pads) + tip(self.sim.jaw_pads)) / 2

    def cube_pose(self) -> tuple[np.ndarray, float]:
        d = self.sim.data
        R = d.xmat[self._cube_body].reshape(3, 3)
        return d.xpos[self._cube_body].copy(), float(np.arctan2(R[1, 0], R[0, 0]))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is None and self._seed is not None:  # first reset of an environment built with a seed
            seed, self._seed = self._seed, None
        super().reset(seed=seed)
        cfg1 = {**self.cfg, "sim": {**self.cfg["sim"], "n_cubes": 1}}
        cube = (options or {}).get("cube") or sample_cubes(self.np_random, self.demo_xy, cfg1)[0]
        self._randomize()
        self._place(cube)
        true_xyz, true_yaw = self.cube_pose()
        nz = self.rcfg["cube_obs_noise_mm"] * 1e-3 if self.randomize else 0.0
        ny = np.deg2rad(self.rcfg["cube_obs_noise_yaw_deg"]) if self.randomize else 0.0
        obs_xyz = true_xyz + np.r_[self.np_random.normal(0, nz, 2), 0.0]
        obs_yaw = true_yaw + self.np_random.normal(0, ny)
        base_input = None
        if self.mode == "residual":  # the TP-GMM gets the observed pose in its own convention
            target = heading_target(obs_xyz[:2], np.asarray(self.policy.meta["pan_xy"]), self.policy.meta["heading_offset"])
            h = ambiguous_candidates(obs_yaw, target, 0.0)[0]
            p, _ = policy_input(self.policy, self.ik, self.geom, self.scene, SimCube(cube.color, obs_xyz, obs_yaw), h,
                                self.home, self.g_home, self.cfg)
            base_input = (p, obs_yaw, h)
        q, g = self.sim.joints_deg_gripper()
        self.task.reset(q, g, obs_xyz, obs_yaw, base_input=base_input)
        self.cube0 = true_xyz
        self.grasped_once = False
        return self.task.observe(q, g), {"cube": true_xyz}

    def stage_reward(self) -> tuple[float, dict]:
        cube, _ = self.cube_pose()
        th = self.table_height(cube[0], cube[1])
        side = self.scene.cube_side
        pinched = self.sim.pinched_cube() == 0
        lifted = pinched and cube[2] - th > side / 2 + 0.01
        released_in_box = (not pinched) and in_box(self.scene, cube - [0, 0, self.table_offset])
        d = float(np.linalg.norm(self.finger_midpoint() - cube))
        r = 0.1 * (1 - np.tanh(10 * d))
        if pinched:
            r = max(r, 0.35)
        if lifted:
            r = max(r, 0.35 + 0.15 * (1 - np.tanh(5 * np.linalg.norm(cube - self.release))))
        self.grasped_once |= bool(lifted)
        success = bool(released_in_box and self.grasped_once)
        return float(r), dict(pinched=pinched, lifted=lifted, success=success, reach_dist=d,
                              cube_off_table=bool(cube[2] < th - 0.05))

    def step(self, action):
        q_seq, g_seq, ok = self.task.command(action)
        self.sim.execute(q_seq, g_seq, 1 / 30, render_every=10**9)
        q, g = self.sim.joints_deg_gripper()
        if not (np.isfinite(self.sim.data.qpos).all() and np.abs(self.cube_pose()[0]).max() < 2.0):
            # numerical blow-up: end the episode without learning from it
            info = dict(pinched=False, lifted=False, success=False, is_success=False, reach_dist=np.nan,
                        cube_off_table=False, sim_unstable=True, ik_ok=ok)
            return self.task.observe(np.nan_to_num(q), float(np.nan_to_num(g))), 0.0, False, True, info
        r, info = self.stage_reward()
        if not ok:
            r -= 0.05
        if self.mode == "residual":  # stay close to the TP-GMM unless a correction pays off
            r -= float(self.rcfg.get("residual_action_penalty", 0.02)) * float(np.mean(np.square(action)))
        terminated = info["success"] or info["cube_off_table"]
        if info["success"]:
            r = 1.0 + (self.task.n_steps - self.task.k)  # stay-at-maximum equivalent
        truncated = self.task.done_time and not terminated
        info.update(ik_ok=ok, is_success=info["success"], sim_unstable=False)
        return self.task.observe(q, g), float(r), bool(terminated), bool(truncated), info

    def render(self):
        if self.sim.renderer is None:
            return None
        self.sim.renderer.update_scene(self.sim.data)
        return self.sim.renderer.render()
