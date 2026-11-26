"""Gymnasium environment on the real SO-101 (LeRobot ``SO101Follower``), same interface as in MuJoCo.

Used to **deploy** a policy trained in simulation (``scripts/run_sac_real.py --episodes N``) and to
**fine-tune** it on the robot (``--finetune STEPS``: SAC keeps learning from real episodes).

Per episode:

1. The arm returns home (min-jerk, joint space) and the operator places a cube (Enter).
2. The cube pose comes from the fixed camera (``CameraCubeSource``: the vision pipeline of
   ``scripts/infer.py``, including the taped-grid correction of a moved camera view), once — as in simulation, the policy does not track the cube while the
   arm occludes it. The observation uses the cube centre *between the fingers* (vision position +
   ``rl.vision_to_finger_heading_mm`` rotated to the grasp heading); the TP-GMM of the residual
   mode receives the vision position, the convention it was trained with.
3. Each agent step: action → IK → joint commands streamed at 30 Hz; joints read back.
4. Reward, from what the robot can measure: 0.35 while the gripper is commanded closed but stays more
   open than commanded (``rl.real.holding_margin``, ``holding_gripper``: a cube between the jaws; an empty
   gripper reaches its command), and at the end, if the
   operator confirms the cube is in the box, the same terminal bonus as in simulation.

Safety: the task's workspace box and table floor, the joint rate limit, LeRobot's
``max_relative_target`` per command, and an abort when the measured joints lag the commands by more
than ``rl.real.max_tracking_err_deg`` (collision or stall). Ctrl+C holds the arm where it is.
``dry_run=True`` uses the simulated follower of ``so101_tpgmm.robot.so101_runner`` (tests).
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np

from so101_tpgmm.inference import joint_move
from so101_tpgmm.rl.task import PickPlaceTask, TaskConfig
from so101_tpgmm.robot.so101_runner import SO101Runner
from so101_tpgmm.tpgmm.cube_symmetry import ambiguous_candidates, heading_target
from so101_tpgmm.vision.camera import TopCamera


def finger_offset_xy(vision_xy: np.ndarray, heading: float, offset_heading_m: np.ndarray) -> np.ndarray:
    """Cube centre between the fingers from the vision position (offset given in the grasp heading frame)."""
    c, s = np.cos(heading), np.sin(heading)
    return np.asarray(vision_xy, float) + np.array([[c, -s], [s, c]]) @ np.asarray(offset_heading_m, float)


class CameraCubeSource(TopCamera):
    """Pose of the first cube (colour order) from one image of the fixed camera."""

    def __call__(self) -> tuple[np.ndarray, float]:
        from so101_tpgmm.inference import order_cubes
        from so101_tpgmm.vision.detection import DetectorParams, detect_cubes
        from so101_tpgmm.vision.scene import SceneParams, validate_detections

        rgb = self.grab()
        p2b, cfg = self.view(rgb)
        vcfg = cfg["vision"]
        dets = detect_cubes(rgb, DetectorParams.from_config(vcfg))
        validate_detections(dets, p2b, SceneParams.from_config(vcfg["scene"], cfg["scene"]["cube_side"]))
        dets = [d for d in dets if "partial_blob" not in d.flags]
        if not dets:
            raise RuntimeError("no cube detected")
        xy = p2b.to_base(np.array([d.center_px for d in dets]))
        i = order_cubes(dets, xy, cfg["inference"]["color_order"])[0]
        return np.r_[xy[i], self.plane_z], float(p2b.yaw_to_base(dets[i].center_px, dets[i].yaw_img))


class RealSO101Env(gym.Env):
    def __init__(self, cfg: dict, mode: str = "scratch", runner: SO101Runner | None = None, cube_source=None,
                 ask=input, dry_run: bool = False, realtime: bool = True, base_policy: str = "segmented") -> None:
        from so101_tpgmm.kinematics.ik import ik_from_config

        self.cfg, self.mode, self.ask, self.realtime = cfg, mode, ask, realtime
        self.rcfg = cfg["rl"]
        self.runner = runner or SO101Runner(dry_run=dry_run)
        self.cube_source = cube_source or CameraCubeSource(cfg)
        from so101_tpgmm.policies import load_policy

        self.ik = ik_from_config(cfg)
        tpgmm = load_policy(cfg, base_policy)  # its cube-yaw rule is used in both modes
        self.rule = (np.asarray(tpgmm.meta["pan_xy"]), float(tpgmm.meta["heading_offset"]))
        self.policy = tpgmm if mode == "residual" else None
        a, b, c = cfg["sim"]["table_plane"]
        table = lambda x, y: a + b * x + c * y  # noqa: E731  (the table in FK coordinates, see docs/mujoco_sim.md)
        bc = np.asarray(cfg["sim"]["box"]["center_xy"], float)
        release = np.r_[bc, table(*bc) + cfg["sim"]["box"]["wall_h"] + 0.04]
        self.task = PickPlaceTask(TaskConfig.from_config(cfg, mode), self.ik, release, table_height=table,
                                  base_policy=self.policy)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (PickPlaceTask.obs_size(mode),), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (PickPlaceTask.act_dim,), np.float32)
        self.home = np.asarray(cfg["inference"]["home_joints_deg"], float)
        self.g_home = float(cfg["inference"]["home_gripper"])

    def _stream(self, q_deg: np.ndarray, g: np.ndarray) -> None:
        for qi, gi in zip(q_deg, g):
            t = time.perf_counter()
            self.runner._send(np.r_[qi, gi])
            if self.realtime:
                time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - t)))

    def _go_home(self, seconds: float = 3.0) -> None:
        cur = self.runner.read_state()
        qs, gs = joint_move(cur[:5], self.home, cur[5], self.g_home, seconds, 1 / 30)
        self._stream(qs, gs)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._go_home()
        self.ask("Place a cube in the workspace and press Enter… ")
        vis_xyz, yaw = self.cube_source()
        h = ambiguous_candidates(yaw, heading_target(vis_xyz[:2], *self.rule), 0.0)[0]
        off = np.asarray(self.rcfg["vision_to_finger_heading_mm"], float) * 1e-3
        cube = vis_xyz.copy()
        cube[:2] = finger_offset_xy(vis_xyz[:2], h, off)
        a, b, c = self.cfg["sim"]["table_plane"]
        cube[2] = a + b * cube[0] + c * cube[1] + self.cfg["scene"]["cube_side"] / 2
        q = self.runner.read_state()
        self.task.reset(q[:5], q[5], cube, yaw, base_input=(vis_xyz, yaw, h))
        self.closed_steps = 0
        return self.task.observe(q[:5], q[5]), {"cube_vision": vis_xyz, "cube_obs": cube}

    def step(self, action):
        q_seq, g_seq, ok = self.task.command(action)
        aborted = False
        try:
            self._stream(q_seq, g_seq)
        except KeyboardInterrupt:
            self.runner._send(self.runner.read_state())
            aborted = True
        q = self.runner.read_state()
        err = np.abs(q[:5] - q_seq[-1]).max()
        if err > self.rcfg["real"]["max_tracking_err_deg"]:
            self.runner._send(q)  # hold
            aborted = True
        lo, hi = self.rcfg["real"]["holding_gripper"]
        # holding: commanded closed, but the jaws are stopped open by an object (an empty gripper reaches the command)
        holding = g_seq[-1] < 30 and q[5] - g_seq[-1] > self.rcfg["real"]["holding_margin"] and lo <= q[5] <= hi
        r = 0.35 if holding else 0.0
        if not ok:
            r -= 0.05
        done = self.task.done_time or aborted
        info = dict(ik_ok=ok, holding=holding, aborted=aborted, tracking_err_deg=float(err), is_success=False)
        if done and not aborted:
            ans = self.ask("Is the cube in the box? [y/N] ").strip().lower()
            info["is_success"] = ans.startswith("y")
            if info["is_success"]:
                r = 1.0
        return self.task.observe(q[:5], q[5]), float(r), bool(done and info["is_success"]), bool(done and not info["is_success"]), info
