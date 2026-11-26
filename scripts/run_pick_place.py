#!/usr/bin/env python
"""Pick and place every cube on the real SO-101 with grasp verification and replanning.

Runs on the machine connected to the arm and the cameras. For each colour: photo from the fixed camera →
plan (TP-GMM by default) → execute up to the grasp → verify the grasp (measured gripper, optionally the wrist
camera) → finish, or open, retreat and replan from a new photo (``so101_tpgmm/robot/pick_place.py``).

    python scripts/run_pick_place.py --port /dev/ttyACM0 --camera 3 --colors red yellow green
    python scripts/run_pick_place.py --port /dev/ttyACM0 --camera 3 --wrist-camera 1    # + wrist confirmation
    python scripts/run_pick_place.py --camera 3 --dry-run       # simulated follower (always "missed")
    python scripts/run_pick_place.py --port /dev/ttyACM0 --camera 3 --no-check          # open loop, as before
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_tpgmm.config import load_config  # noqa: E402
from so101_tpgmm.policies import POLICIES, load_policy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--policy", choices=list(POLICIES), default=None)
    ap.add_argument("--colors", nargs="*", default=None, help="default: inference.color_order")
    ap.add_argument("--port", default=None)
    ap.add_argument("--id", default="so101_follower")
    ap.add_argument("--max-step-deg", type=float, default=8.0)
    ap.add_argument("--camera", default="3", help="fixed top camera (OpenCV index or /dev/videoN)")
    ap.add_argument("--wrist-camera", default=None, help="wrist camera to confirm the grasp (optional)")
    ap.add_argument("--register", action=argparse.BooleanOptionalAction, default=True,
                    help="correct a moved camera view on the taped grid")
    ap.add_argument("--check", action=argparse.BooleanOptionalAction, default=True,
                    help="verify the grasp and replan a missed cube (--no-check: open loop)")
    ap.add_argument("--dry-run", action="store_true", help="simulated follower (no robot)")
    args = ap.parse_args()

    from so101_tpgmm.kinematics.ik import ik_from_config
    from so101_tpgmm.robot.pick_place import PickPlaceExecutor
    from so101_tpgmm.robot.so101_runner import SO101Runner
    from so101_tpgmm.vision.camera import TopCamera, WristCamera

    cfg = load_config(args.config, args.set)
    cam = lambda c: int(c) if c.isdigit() else c  # noqa: E731
    camera = TopCamera(cfg, cam(args.camera), register=args.register)
    wrist = WristCamera(cam(args.wrist_camera)) if args.wrist_camera else None
    runner = SO101Runner(args.port, args.id, args.max_step_deg, dry_run=args.dry_run)
    policy = load_policy(cfg, args.policy or cfg["inference"]["policy"])
    ex = PickPlaceExecutor(cfg, runner, camera, policy, ik_from_config(cfg), wrist_camera=wrist, check=args.check,
                           realtime=not args.dry_run)
    runner.connect()
    try:
        results = ex.run(args.colors or list(cfg["inference"]["color_order"]))
    finally:
        runner.disconnect()
    print("\nsummary:")
    for r in results:
        print(f"  {r.color:6s} {r.status} ({r.attempts} attempt(s))")
        for v in r.verdicts:
            print(f"           {v}")


if __name__ == "__main__":
    main()
