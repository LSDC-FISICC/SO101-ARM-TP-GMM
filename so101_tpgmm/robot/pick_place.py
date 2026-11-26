"""Closed-loop pick-and-place on the real SO-101: verify every grasp and replan a missed cube.

``so101_runner`` plays a precomputed plan open loop: when a grasp misses (the camera calibration or the
cube pose is off by more than the ~12 mm the open jaws leave on each side), the arm carries nothing to
the box and goes on. Here, one cube at a time:

1. Go home (the arm out of the camera's view), take a photo with the fixed camera (view corrected on the
   taped grid), detect the cube of the current colour and plan it from the measured joints.
2. Execute up to the grasp check: ``grasp_check.settle_s`` after the gripper command reaches its hold value
   (the arm is already lifting).
3. Verify (``so101_tpgmm.grasp``): measured − commanded gripper (the cube stops the jaws), optionally
   confirmed by the wrist camera (cube between the jaws).
4. Held: finish the transport, release and return home. Missed: open, lift straight up, return home and
   go back to 1 (a new photo: the cube may have been pushed), up to ``grasp_check.max_retries`` times.

A cube that is no longer detected (dropped into the box, out of the image) ends its attempts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from so101_tpgmm.grasp import GraspCheckParams, check_index, verify_grasp, wrist_blob_fraction
from so101_tpgmm.inference import plan_from_image, recovery_trajectory
from so101_tpgmm.vision.detection import DetectorParams


@dataclass
class CubeResult:
    color: str
    status: str = "pending"  # released | not found | skipped: ... | missed after N attempts | aborted
    attempts: int = 0
    verdicts: list[str] = field(default_factory=list)


class PickPlaceExecutor:
    def __init__(self, cfg: dict, runner, camera, policy, ik, wrist_camera=None, check: bool = True,
                 realtime: bool = True, log=print) -> None:
        self.cfg, self.runner, self.camera, self.policy, self.ik = cfg, runner, camera, policy, ik
        self.wrist, self.check, self.realtime, self.log = wrist_camera, check, realtime, log
        self.p = GraspCheckParams.from_config(cfg)
        icfg = cfg["inference"]
        self.dt = float(icfg["dt"])
        self.home, self.g_home = np.asarray(icfg["home_joints_deg"], float), float(icfg["home_gripper"])
        self.detector = DetectorParams.from_config(cfg["vision"])

    def _stream(self, q, g) -> bool:
        return self.runner.stream(q, g, self.dt, realtime=self.realtime)

    def run(self, colors: list[str]) -> list[CubeResult]:
        results = []
        for color in colors:
            r = self.pick(color)
            results.append(r)
            self.log(f"[{color}] {r.status} after {r.attempts} attempt(s)")
            if r.status == "aborted":
                break
        return results

    def pick(self, color: str) -> CubeResult:
        r = CubeResult(color)
        tries = 1 + (self.p.max_retries if self.check else 0)
        while r.attempts < tries:
            if not self.runner.move_to(self.home, self.g_home, 2.0, realtime=self.realtime):
                r.status = "aborted"
                return r
            rgb = self.camera.grab()
            p2b, cfg = self.camera.view(rgb)
            q_now = self.runner.read_state()
            plan = plan_from_image(rgb, self.policy, self.ik, p2b, self.camera.plane_z, cfg, q_now[:5], float(q_now[5]),
                                   colors=[color], only_colors=True, max_cubes=1)
            if not plan.cubes:
                r.status = "not found" if r.attempts == 0 else f"not found after {r.attempts} attempt(s) (dropped?)"
                return r
            cp = plan.cubes[0]
            if cp.status != "planned":
                r.status = cp.status
                return r
            r.attempts += 1
            m = np.flatnonzero(plan.cube_index == 0)
            q, g = plan.joints_deg[m], plan.gripper[m]
            rest_q, rest_g = plan.joints_deg[m[-1] + 1:], plan.gripper[m[-1] + 1:]  # return home
            self.log(f"[{color}] attempt {r.attempts}: cube at ({cp.position[0] * 100:.1f}, {cp.position[1] * 100:.1f}) cm")
            if not self.check:
                ok = self._stream(np.vstack([q, rest_q]), np.r_[g, rest_g])
                r.status = "released (not verified)" if ok else "aborted"
                return r
            kc = check_index(g, cp.k_contact - int(m[0]), self.dt, self.p.settle_s)
            if not self._stream(q[:kc + 1], g[:kc + 1]):
                r.status = "aborted"
                return r
            state = self.runner.read_state()
            blob = wrist_blob_fraction(self.wrist.grab(), color, self.detector, self.p.wrist_roi) if self.wrist else None
            v = verify_grasp(state[5], g[kc], self.p, blob)
            r.verdicts.append(str(v))
            self.log(f"[{color}]   {v}")
            if v.held:
                ok = self._stream(np.vstack([q[kc + 1:], rest_q]), np.r_[g[kc + 1:], rest_g])
                r.status = "released" if ok else "aborted"
                return r
            rq, rg = recovery_trajectory(self.ik, state[:5], float(g[kc]), self.cfg, self.p.open_gripper,
                                         self.p.retreat_height_m)
            if not self._stream(rq, rg):
                r.status = "aborted"
                return r
        r.status = f"missed after {r.attempts} attempts"
        return r
