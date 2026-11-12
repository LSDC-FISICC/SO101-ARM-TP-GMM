#!/usr/bin/env python
"""Save images with the cube detections drawn (contour, minAreaRect, centre, yaw, flags).

Examples::

    python scripts/debug_detections.py --episodes 0 59 170          # reference frames
    python scripts/debug_detections.py --episodes 59 --frame first  # first frame instead
    python scripts/debug_detections.py --all --out outputs/debug_detections
    python scripts/debug_detections.py --image path/to/photo.png     # any image (e.g. live camera)

Labels show the colour, the yaw in the image and, when a calibration exists
(outputs/vision_calibration.yaml), the cube position and yaw in the base frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402
from so101_tpgmm.vision.calibration import VisionCalibration  # noqa: E402
from so101_tpgmm.vision.detection import DetectorParams, detect_cubes  # noqa: E402
from so101_tpgmm.vision.draw import draw_detections  # noqa: E402
from so101_tpgmm.vision.geometry import PixelToBase  # noqa: E402
from so101_tpgmm.vision.scene import SceneParams, reference_frame_index, validate_detections  # noqa: E402


def annotate(rgb: np.ndarray, cfg: dict, header: str, p2b: PixelToBase | None) -> np.ndarray:
    vcfg = cfg["vision"]
    dp = DetectorParams.from_config(vcfg)
    sp = SceneParams.from_config(vcfg["scene"], cfg["scene"]["cube_side"])
    dets = detect_cubes(rgb, dp)
    labels = []
    if p2b is not None:
        validate_detections(dets, p2b, sp)
        for d in dets:
            x, y = p2b.to_base(d.center_px[None])[0]
            labels.append(f"{d.color} ({x * 100:.1f},{y * 100:.1f})cm {np.degrees(p2b.yaw_to_base(d.center_px, d.yaw_img)):+.0f}deg")
    img = draw_detections(rgb, dets, labels or None, dp.exclude_boxes, sp.arm_home_roi_px,
                          header=f"{header} - {len(dets)} cube(s)")
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--episodes", type=int, nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--frame", choices=["reference", "first"], default="reference")
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--out", type=str, default="outputs/debug_detections")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cal_path = resolve_path(cfg["vision"]["calibration"]["output"])
    p2b = PixelToBase(VisionCalibration.load_h(cal_path)[0]) if cal_path.exists() else None

    if args.image:
        rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        img = annotate(rgb, cfg, Path(args.image).name, p2b)
        dst = out / f"{Path(args.image).stem}_detections.jpg"
        cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(dst)
        return

    from so101_tpgmm.data.lerobot_dataset import LeRobotV3Dataset

    ds = LeRobotV3Dataset(resolve_path(cfg["dataset"]["local_dir"]), cfg["dataset"]["state_key"])
    cam_key = cfg["dataset"]["fixed_camera_key"]
    seg = pd.read_csv(resolve_path(cfg["segmentation"]["output_csv"])).set_index("episode")
    sp = SceneParams.from_config(cfg["vision"]["scene"], cfg["scene"]["cube_side"])
    episodes = range(ds.num_episodes) if args.all else args.episodes
    for e in episodes:
        ep = ds.episode(e)
        i = 0 if args.frame == "first" or e not in seg.index else reference_frame_index(int(seg.loc[e, "approach_start"]), sp)
        frames, _ = ds.read_frames(cam_key, e, ep.timestamps[[i]])
        img = annotate(frames[0], cfg, f"ep {e} frame {i} ({args.frame})", p2b)
        cv2.imwrite(str(out / f"ep{e:03d}_{args.frame}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"wrote {len(list(episodes))} image(s) to {out}")


if __name__ == "__main__":
    main()
