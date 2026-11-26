"""Stub interface to execute a planned trajectory on the real SO-101 follower (LeRobot).

Runs **locally** on the machine connected to the arm (``pip install -e ".[robot]"``).
It streams the joint trajectory written by ``scripts/infer.py`` to ``SO101Follower``:

1. Connect with ``use_degrees=True``. The dataset and every plan are in LeRobot's
   ``DEGREES`` mode; a follower configured in ``RANGE_M100_100`` would interpret the
   same numbers as a different posture. The runner refuses to run in that case.
2. Move slowly (min-jerk, ``approach_s``) from the current state to the first sample.
3. Stream the samples at their timestamps, with LeRobot's ``max_relative_target`` as an extra
   per-step safety clamp, and stop (holding the last command) on Ctrl+C.

``--dry-run`` replaces the robot by a simulated one (no LeRobot needed), which is what the tests
use. It plays a plan open loop; grasp verification and replanning of a missed cube are in
``so101_tpgmm/robot/pick_place.py`` (``scripts/run_pick_place.py``). Collision checks and a proper
emergency stop are out of scope.

    python -m so101_tpgmm.robot.so101_runner --npz outputs/inference/plan.npz --port /dev/ttyACM0
    python -m so101_tpgmm.robot.so101_runner --npz outputs/inference/plan.npz --dry-run
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

JOINT_KEYS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def _import_lerobot():
    """SO101Follower and its config across LeRobot versions."""
    try:  # LeRobot >= 0.4 (SO-100/101 merged into so_follower)
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:  # older layout
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    return SO101Follower, SO101FollowerConfig


@dataclass
class SimulatedFollower:
    """Stand-in for ``SO101Follower`` in dry runs: the state follows the commands instantly."""

    state: np.ndarray = field(default_factory=lambda: np.array([-5.7, -104.3, 91.3, 76.3, 2.8, 1.5]))
    sent: list[dict] = field(default_factory=list)
    use_degrees: bool = True

    def connect(self) -> None:
        pass

    def get_observation(self) -> dict:
        return {k: float(v) for k, v in zip(JOINT_KEYS, self.state)}

    def send_action(self, action: dict) -> dict:
        self.sent.append(dict(action))
        self.state = np.array([action[k] for k in JOINT_KEYS])
        return action

    def disconnect(self) -> None:
        pass


class SO101Runner:
    """Play a joint trajectory (degrees + gripper 0-100) on the SO-101 follower."""

    def __init__(self, port: str | None = None, robot_id: str = "so101_follower", max_step_deg: float = 8.0,
                 dry_run: bool = False, robot=None) -> None:
        self.dry_run = dry_run
        if robot is not None:
            self.robot = robot
        elif dry_run:
            self.robot = SimulatedFollower()
        else:
            Follower, Config = _import_lerobot()
            cfg = Config(port=port, id=robot_id, use_degrees=True, max_relative_target=max_step_deg)
            self.robot = Follower(cfg)
        self.check_units()

    def check_units(self) -> None:
        cfg = getattr(self.robot, "config", self.robot)
        if not getattr(cfg, "use_degrees", True):
            raise RuntimeError("The follower is not in DEGREES mode (use_degrees=False): the plan's joint values "
                               "would be read as RANGE_M100_100 and the arm would go to a different posture.")

    def connect(self) -> None:
        self.robot.connect()

    def disconnect(self) -> None:
        self.robot.disconnect()

    def read_state(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([obs[k] for k in JOINT_KEYS], dtype=float)

    def _send(self, q: np.ndarray) -> None:
        self.robot.send_action({k: float(v) for k, v in zip(JOINT_KEYS, q)})

    def stream(self, joints_deg: np.ndarray, gripper: np.ndarray, dt: float = 1 / 30, realtime: bool = True) -> bool:
        """Send the samples at ``dt``; on Ctrl+C hold where the arm is and return False."""
        try:
            t_start = time.perf_counter()
            for k, q in enumerate(np.c_[joints_deg, gripper]):
                if realtime:
                    delay = k * dt - (time.perf_counter() - t_start)
                    if delay > 0:
                        time.sleep(delay)
                self._send(q)
        except KeyboardInterrupt:
            self._send(self.read_state())
            return False
        return True

    def move_to(self, q_deg: np.ndarray, gripper: float, seconds: float = 2.0, realtime: bool = True) -> bool:
        """Min-jerk joint move from the measured state."""
        cur = self.read_state()
        n = max(2, int(seconds * 30))
        s = np.linspace(0, 1, n)
        s = (10 * s**3 - 15 * s**4 + 6 * s**5)[:, None]
        target = np.r_[np.asarray(q_deg, float)[:5], gripper]
        path = cur + (target - cur) * s
        return self.stream(path[:, :5], path[:, 5], realtime=realtime)

    def play(self, t: np.ndarray, joints_deg: np.ndarray, gripper: np.ndarray, approach_s: float = 2.0,
             realtime: bool = True) -> int:
        """Stream the trajectory; returns the number of samples sent."""
        target = np.c_[joints_deg, gripper]
        cur = self.read_state()
        n0 = max(2, int(approach_s * 30))
        s = np.linspace(0, 1, n0)
        s = 10 * s**3 - 15 * s**4 + 6 * s**5
        lead = cur + (target[0] - cur) * s[:, None]
        sent = 0
        try:
            t_start = time.perf_counter()
            for q in lead:
                self._send(q)
                sent += 1
                if realtime:
                    time.sleep(1 / 30)
            t_start = time.perf_counter()
            for ti, q in zip(t - t[0], target):
                if realtime:
                    delay = ti - (time.perf_counter() - t_start)
                    if delay > 0:
                        time.sleep(delay)
                self._send(q)
                sent += 1
        except KeyboardInterrupt:
            self._send(self.read_state())  # hold where it is
        return sent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--port", default=None)
    ap.add_argument("--id", default="so101_follower")
    ap.add_argument("--max-step-deg", type=float, default=8.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    data = np.load(args.npz)
    runner = SO101Runner(args.port, args.id, args.max_step_deg, dry_run=args.dry_run)
    runner.connect()
    try:
        n = runner.play(data["t"], data["joints_deg"], data["gripper"], realtime=not args.dry_run)
        print(f"sent {n} commands ({'dry run' if args.dry_run else args.port})")
    finally:
        runner.disconnect()


if __name__ == "__main__":
    main()
