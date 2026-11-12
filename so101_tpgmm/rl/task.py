"""Backend-agnostic pick-and-place task for reinforcement learning (MuJoCo and the real SO-101).

Everything here uses only what the real robot also has: measured joints (degrees, gripper 0-100),
the TCP from our FK, a cube pose from the camera taken at the start of the episode (no tracking
of the cube while the arm occludes it), the release point of the box and the elapsed time.

**Action** (5-D, in [-1, 1]), turned into joint commands by the same IK as the other policies:

* ``scratch``: Δx, Δy, Δz of a commanded TCP target (× ``max_step_m`` per agent step, cumulative),
  Δ heading (× ``max_yaw_step_deg``), gripper command (absolute, 0 … ``gripper_max``).
* ``residual``: offsets added to the TP-GMM plan at the current time — position
  (× ``residual_m``), heading (× ``residual_yaw_deg``), gripper (× ``residual_gripper``). With a zero
  action the robot executes the TP-GMM exactly, so learning starts from a working policy.

The target is clipped to a workspace box (z above the table), the IK is warm-started from the last
command, and an unreachable target keeps the previous joint command (and is reported, so the
environment can penalize it). Joint commands are interpolated from the previous command to the
new one at 30 Hz within the agent step, as the follower expects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from so101_tpgmm.kinematics.ik import SO101IK, approach_pitch, tool_yaw
from so101_tpgmm.tpgmm.demos import heading_from_tool_yaw, tool_yaw_from_heading, wrap

POS_SCALE = 0.2  # [m] observation scale for positions


@dataclass
class TaskConfig:
    mode: str = "scratch"  # scratch | residual
    agent_hz: float = 10.0
    max_step_m: float = 0.02
    max_yaw_step: float = np.deg2rad(15)
    gripper_max: float = 60.0
    residual_m: float = 0.02
    residual_yaw: float = np.deg2rad(15)
    residual_gripper: float = 30.0
    ws_min: tuple = (0.05, -0.25, 0.008)
    ws_max: tuple = (0.45, 0.30, 0.25)
    horizon_s: float = 15.0
    residual_extra_s: float = 2.0

    @classmethod
    def from_config(cls, cfg: dict, mode: str) -> "TaskConfig":
        r = cfg["rl"]
        return cls(mode=mode, agent_hz=float(r["agent_hz"]), max_step_m=float(r["max_step_m"]),
                   max_yaw_step=np.deg2rad(r["max_yaw_step_deg"]), gripper_max=float(r["gripper_max"]),
                   residual_m=float(r["residual_m"]), residual_yaw=np.deg2rad(r["residual_yaw_deg"]),
                   residual_gripper=float(r["residual_gripper"]), ws_min=tuple(r["workspace_min"]),
                   ws_max=tuple(r["workspace_max"]), horizon_s=float(r["horizon_s"]["scratch"]),
                   residual_extra_s=float(r["horizon_s"]["residual_extra"]))


class PickPlaceTask:
    """Observation and action logic of one pick-and-place episode (one cube)."""

    @staticmethod
    def obs_size(mode: str) -> int:
        return 25 if mode == "scratch" else 31

    act_dim = 5
    max_joint_step = np.deg2rad(20.0)  # per agent step (the follower also clamps each 30 Hz command)
    pitch_rate = np.deg2rad(3.0)  # scratch: the pitch target drifts from the start pitch to the default
    gripper_rate = 150.0  # [units/s] gripper command rate limit (an STS3215 turns at ~5 rad/s unloaded;
    #                       faster jaw commands are not executable and make the simulated contacts explode)

    def __init__(self, tc: TaskConfig, ik: SO101IK, release_xyz: np.ndarray, table_height=lambda x, y: 0.0,
                 base_policy=None) -> None:
        if tc.mode not in ("scratch", "residual"):
            raise ValueError(tc.mode)
        if tc.mode == "residual" and base_policy is None:
            raise ValueError("residual mode needs a base policy (TP-GMM)")
        self.tc, self.ik, self.base_policy = tc, ik, base_policy
        self.release_xyz = np.asarray(release_xyz, float)
        self.table_height = table_height
        self.dt_agent = 1.0 / tc.agent_hz

    # ------------------------------------------------------------------ episode
    def reset(self, q_deg: np.ndarray, gripper: float, cube_xyz: np.ndarray, cube_yaw: float,
              base_input: tuple | None = None) -> None:
        """Start an episode from the measured state and the observed cube pose.

        ``base_input`` (residual mode): (cube position in the TP-GMM's convention, yaw, heading), see
        ``so101_tpgmm.sim.episode.policy_input``; default: the observed pose.
        """
        self.q_cmd = np.deg2rad(np.asarray(q_deg, float)[:5])
        self.g_cmd = float(gripper)
        T = self.ik.kin.fk(self.q_cmd)
        self.target = T[:3, 3].copy()
        self.heading = float(heading_from_tool_yaw(tool_yaw(T[:3, :3])))
        self.pitch = float(approach_pitch(T, self.ik.pan_axis_xy))
        self.cube_xyz, self.cube_yaw = np.asarray(cube_xyz, float).copy(), float(cube_yaw)
        self.k = 0
        self.base = None
        horizon = self.tc.horizon_s
        if self.tc.mode == "residual":
            from so101_tpgmm.inference import cartesian_state

            p, yaw, h = base_input if base_input is not None else (self.cube_xyz, self.cube_yaw, None)
            start = cartesian_state(self.ik, np.rad2deg(self.q_cmd), self.g_cmd)
            ro = self.base_policy.rollout(np.asarray(p, float), float(yaw), start=start, dt=1 / 30, heading=h)
            names = ro.names
            self.base = dict(t=ro.t, xyz=ro.Y[:, :3], h=np.unwrap(ro.Y[:, names.index("h")]),
                             g=ro.Y[:, names.index("g")],
                             pitch=ro.Y[:, names.index("pitch")] if "pitch" in names else None)
            horizon = float(ro.t[-1]) + self.tc.residual_extra_s
        self.n_steps = int(round(horizon * self.tc.agent_hz))

    def base_at(self, t: float) -> tuple[np.ndarray, float, float, float | None]:
        b = self.base
        i = int(np.clip(np.searchsorted(b["t"], t), 0, len(b["t"]) - 1))
        return b["xyz"][i], float(b["h"][i]), float(b["g"][i]), (float(b["pitch"][i]) if b["pitch"] is not None else None)

    # ------------------------------------------------------------------ action → commands
    def command(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Joint commands for the next agent step: (joints_deg (n, 5), gripper (n,), ik_ok), n = 30 Hz samples."""
        a = np.clip(np.asarray(action, float), -1, 1)
        tc = self.tc
        pitch = None
        if tc.mode == "scratch":
            target = self.target + a[:3] * tc.max_step_m
            heading = self.heading + a[3] * tc.max_yaw_step
            g = (a[4] + 1) / 2 * tc.gripper_max
            p_def = self.ik.default_pitch if self.ik.default_pitch is not None else self.pitch
            self.pitch += float(np.clip(p_def - self.pitch, -self.pitch_rate, self.pitch_rate))
            pitch = self.pitch
        else:
            bx, bh, bg, pitch = self.base_at((self.k + 1) * self.dt_agent)
            target = bx + a[:3] * tc.residual_m
            heading = bh + a[3] * tc.residual_yaw
            g = float(np.clip(bg + a[4] * tc.residual_gripper, 0, 100))
        lo, hi = np.array(tc.ws_min, float), np.array(tc.ws_max, float)
        target = np.asarray(target, float).copy()
        target[:2] = np.clip(target[:2], lo[:2], hi[:2])
        target[2] = np.clip(target[2], self.table_height(target[0], target[1]) + lo[2], hi[2])  # z floor: above the table
        r = self.ik.solve(target, float(tool_yaw_from_heading(heading)), self.q_cmd, max_restarts=0, pitch=pitch)
        ok = bool(r.success)
        q_new = r.q if ok else self.q_cmd
        step = np.abs(q_new - self.q_cmd).max()
        if step > self.max_joint_step:  # rate limit (safety); the target is still reached over the next steps
            q_new = self.q_cmd + (q_new - self.q_cmd) * self.max_joint_step / step
        if ok:
            self.target, self.heading = target, float(wrap(heading))
        g = float(np.clip(g, self.g_cmd - self.gripper_rate * self.dt_agent, self.g_cmd + self.gripper_rate * self.dt_agent))
        n = max(1, int(round(30 * self.dt_agent)))
        s = np.linspace(1 / n, 1, n)[:, None]
        q_seq = np.rad2deg(self.q_cmd + (q_new - self.q_cmd) * s)
        g_seq = self.g_cmd + (g - self.g_cmd) * s[:, 0]
        self.q_cmd, self.g_cmd = q_new, float(g)
        self.k += 1
        return q_seq, g_seq, ok

    @property
    def done_time(self) -> bool:
        return self.k >= self.n_steps

    # ------------------------------------------------------------------ observation
    def observe(self, q_deg: np.ndarray, gripper: float) -> np.ndarray:
        """Observation from the measured joints (degrees, gripper 0-100) and the episode's cube snapshot."""
        q = np.deg2rad(np.asarray(q_deg, float)[:5])
        T = self.ik.kin.fk(q)
        tcp = T[:3, 3]
        h = float(heading_from_tool_yaw(tool_yaw(T[:3, :3])))
        d_yaw = 4 * (self.cube_yaw - h)  # the cube looks the same every 90°
        pitch = float(approach_pitch(T, self.ik.pan_axis_xy))
        t = self.k * self.dt_agent
        o = [q / np.pi, [gripper / 100.0], tcp / POS_SCALE, [np.cos(h), np.sin(h), pitch],
             (self.cube_xyz - tcp) / POS_SCALE, [np.cos(d_yaw), np.sin(d_yaw)],
             (self.release_xyz - tcp) / POS_SCALE, (self.target - tcp) / POS_SCALE,
             [self.g_cmd / 100.0, min(1.0, self.k / max(1, self.n_steps))]]
        if self.tc.mode == "residual":
            bx, bh, bg, _ = self.base_at(t)
            o += [(bx - tcp) / POS_SCALE, [np.cos(bh - h), np.sin(bh - h)], [bg / 100.0]]
        return np.concatenate([np.ravel(x) for x in o]).astype(np.float32)
