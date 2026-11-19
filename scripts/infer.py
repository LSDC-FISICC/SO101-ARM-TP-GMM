#!/usr/bin/env python
"""Phase 5: fixed-camera image → cube detections → TP-GMM trajectories → joints by IK → .npz/.csv.

Examples::

    python scripts/infer.py --image snapshot.png                      # robot assumed at home
    python scripts/infer.py --image snapshot.png --start-joints 0 -100 90 75 0 --start-gripper 2
    python scripts/infer.py --episode 12                              # dataset reference frame (testing)
    python scripts/infer.py --image snapshot.png --policy ik_planner --colors green red

Outputs in ``inference.output_dir``: <name>.npz, <name>.csv (columns = LeRobot action keys,
degrees / gripper 0-100) and <name>_detections.jpg. Execute with
``python -m so101_tpgmm.robot.so101_runner --npz <name>.npz --port /dev/ttyACM0``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402
from so101_tpgmm.inference import plan_from_image, save_plan  # noqa: E402
from so101_tpgmm.kinematics.ik import ik_from_config  # noqa: E402
from so101_tpgmm.policies import POLICIES, load_policy  # noqa: E402
from so101_tpgmm.vision.calibration import VisionCalibration  # noqa: E402
from so101_tpgmm.vision.draw import draw_detections, draw_plan_overlay  # noqa: E402
from so101_tpgmm.vision.geometry import PixelToBase  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str)
    src.add_argument("--episode", type=int)
    ap.add_argument("--policy", choices=list(POLICIES), default=None)
    ap.add_argument("--colors", nargs="*", default=None, help="colour order (default: inference.color_order)")
    ap.add_argument("--start-joints", type=float, nargs=5, default=None, help="current joints [deg]")
    ap.add_argument("--start-gripper", type=float, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--show", action=argparse.BooleanOptionalAction, default=True,
                    help="open a window with the detections and the planned grasps (--no-show: only save it)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    icfg, vcfg = cfg["inference"], cfg["vision"]

    if args.image:
        rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        name = args.name or Path(args.image).stem
        start, gstart = args.start_joints, args.start_gripper
    else:
        import pandas as pd

        from so101_tpgmm.data.lerobot_dataset import LeRobotV3Dataset
        from so101_tpgmm.vision.scene import SceneParams, reference_frame_index

        ds = LeRobotV3Dataset(resolve_path(cfg["dataset"]["local_dir"]), cfg["dataset"]["state_key"])
        seg = pd.read_csv(resolve_path(cfg["segmentation"]["output_csv"])).set_index("episode")
        i = reference_frame_index(int(seg.loc[args.episode, "approach_start"]),
                                  SceneParams.from_config(vcfg["scene"], cfg["scene"]["cube_side"]))
        ep = ds.episode(args.episode)
        rgb = ds.read_frames(cfg["dataset"]["fixed_camera_key"], args.episode, ep.timestamps[[i]])[0][0]
        name = args.name or f"episode_{args.episode:03d}"
        start = args.start_joints if args.start_joints is not None else ep.state[i, :5]
        gstart = args.start_gripper if args.start_gripper is not None else float(ep.state[i, 5])

    H, tcp_offset, plane_z = VisionCalibration.load_h(resolve_path(vcfg["calibration"]["output"]))
    intr = vcfg.get("intrinsics")
    p2b = PixelToBase(H, *(np.array(intr["K"]), np.array(intr["dist"])) if intr else (None, None))
    policy_name = args.policy or icfg["policy"]
    policy = load_policy(cfg, policy_name)
    ik = ik_from_config(cfg)
    plan = plan_from_image(rgb, policy, ik, p2b, plane_z, cfg, None if start is None else np.asarray(start),
                           gstart, args.colors)

    out = resolve_path(icfg["output_dir"]) / name
    save_plan(plan, out, {"policy": policy_name, "source": args.image or f"episode {args.episode}"})
    labels = []
    for d in plan.detections:
        c = next(c for c in plan.cubes if np.allclose(c.position[:2], p2b.to_base(d.center_px[None])[0]))
        labels.append(f"#{c.order + 1} {d.color} {'OK' if c.status == 'planned' else 'SKIP'}")
    img = draw_detections(rgb, plan.detections, labels or None, header=f"{name}: {policy_name}")
    cv2.imwrite(str(out) + "_detections.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    overlay = draw_plan_overlay(img, plan, p2b, ik.kin, tcp_offset)
    overlay_path = str(out) + "_plan.jpg"
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"{len(plan.cubes)} cube(s) detected:")
    for c in plan.cubes:
        print(f"  #{c.order + 1} {c.color:6s} at ({c.position[0] * 100:.1f}, {c.position[1] * 100:.1f}) cm, "
              f"yaw {np.degrees(c.yaw):+.0f}°: {c.status}")
    print(f"trajectory: {len(plan.t)} samples, {plan.t[-1] if len(plan.t) else 0:.1f} s → {out}.npz / .csv")
    print(f"scene: {overlay_path}")
    if args.show:
        show(overlay, f"{name}: detections and planned grasps (any key closes)", overlay_path)


def show(rgb: np.ndarray, title: str, path: str) -> None:
    """Display the overlay: OpenCV window, else matplotlib, else the system image viewer."""
    big = cv2.resize(rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    try:
        cv2.imshow(title, cv2.cvtColor(big, cv2.COLOR_RGB2BGR))  # not available in opencv-python-headless
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    except cv2.error:
        pass
    try:
        import matplotlib

        if matplotlib.get_backend().lower() == "agg":
            raise RuntimeError("no interactive matplotlib backend")
        import matplotlib.pyplot as plt

        plt.figure(title, figsize=(big.shape[1] / 100, big.shape[0] / 100))
        plt.imshow(rgb)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        return
    except Exception:  # noqa: BLE001 - any GUI failure falls through to the image viewer
        pass
    import shutil
    import subprocess

    viewer = shutil.which("xdg-open") or shutil.which("open")
    if viewer:
        subprocess.Popen([viewer, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print(f"(no display available: open {path})")


if __name__ == "__main__":
    main()
