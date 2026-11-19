#!/usr/bin/env python
"""Capture one frame of the fixed (top) camera at the calibration resolution, ready for infer.py.

    python scripts/capture_top.py --camera 3 --out foto.png
    python scripts/capture_top.py --camera /dev/video3 --out foto.png --preview

Opens the camera with the same mode as the dataset3 recording (640x480, MJPG, 30 fps), drops the
first frames (auto-exposure settling) and refuses to save if the camera delivers another size,
because a frame captured at another resolution and resized can have another field of view. Then
checks the view against the calibration view (taped grid) and prints the grasp error it would cause
uncorrected (infer.py corrects it by default with --register on).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402
from so101_tpgmm.vision.calibration import VisionCalibration  # noqa: E402
from so101_tpgmm.vision.geometry import PixelToBase  # noqa: E402
from so101_tpgmm.vision.register import register_view  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="3", help="OpenCV index or device path (top camera; ACT uses index 3)")
    ap.add_argument("--out", default="foto.png")
    ap.add_argument("--warmup", type=int, default=30, help="frames dropped before saving")
    ap.add_argument("--preview", action="store_true", help="show the live image first; space saves, q quits")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, [])
    w, h = cfg["camera"]["image_size"]

    cam = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {args.camera}")
    frame = None
    for _ in range(max(1, args.warmup)):
        ok, frame = cap.read()
    if not ok or frame is None:
        sys.exit("camera read failed")
    if args.preview:
        while True:
            ok, frame = cap.read()
            cv2.imshow("top camera: space = save, q = quit", frame)
            k = cv2.waitKey(30) & 0xFF
            if k == ord(" "):
                break
            if k == ord("q"):
                sys.exit("not saved")
        cv2.destroyAllWindows()
    cap.release()
    if frame.shape[:2] != (h, w):
        sys.exit(f"the camera delivered {frame.shape[1]}x{frame.shape[0]}, not {w}x{h}: wrong camera or mode, not saved")
    cv2.imwrite(args.out, frame)
    print(f"saved {args.out} ({w}x{h})")

    vcfg = cfg["vision"]
    H, _, _ = VisionCalibration.load_h(resolve_path(vcfg["calibration"]["output"]))
    ref = cv2.imread(str(resolve_path(vcfg["reference_image"])))
    try:
        reg = register_view(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cv2.cvtColor(ref, cv2.COLOR_BGR2RGB), PixelToBase(H))
    except RuntimeError as e:
        print(f"WARNING: {e}")
        return
    print(reg.summary())
    if not reg.ok:
        print("WARNING: the taped grid does not align with the calibration view: is this the top camera, "
              "is the whole grid visible?")
    elif np.max(reg.shift_mm) > 5:
        print("the camera moved since the calibration: infer.py corrects it (--register on, the default)")
    blend = cv2.addWeighted(frame, 0.5, cv2.warpPerspective(ref, reg.W, (w, h)), 0.5, 0)
    out = str(Path(args.out).with_suffix("")) + "_vs_reference.jpg"
    cv2.imwrite(out, blend)
    print(f"reference warped onto the photo (tape should overlap): {out}")


if __name__ == "__main__":
    main()
