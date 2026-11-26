"""OpenCV cameras of the SO-101 setup: the fixed top camera (calibrated) and the wrist camera."""

from __future__ import annotations

import numpy as np


class TopCamera:
    """The fixed (top) camera in the recording mode, with the taped-grid view correction of ``infer.py``."""

    def __init__(self, cfg: dict, camera: int | str = 0, register: bool = True) -> None:
        import cv2

        from so101_tpgmm.config import resolve_path
        from so101_tpgmm.vision.calibration import VisionCalibration
        from so101_tpgmm.vision.geometry import PixelToBase

        self.cv2, self.cfg, self.register = cv2, cfg, register
        H, _, self.plane_z = VisionCalibration.load_h(resolve_path(cfg["vision"]["calibration"]["output"]))
        intr = cfg["vision"].get("intrinsics")
        self.p2b = PixelToBase(H, *(np.array(intr["K"]), np.array(intr["dist"])) if intr else (None, None))
        self.cap = cv2.VideoCapture(camera)
        w, h = self.size = tuple(cfg["camera"]["image_size"])  # the recording / calibration mode
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.last_good = None  # (p2b, cfg) of the last successful view registration

    def grab(self) -> np.ndarray:
        """Current RGB frame (the driver buffers a few frames: drop them, they predate the cube)."""
        for _ in range(5):
            ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        if bgr.shape[1::-1] != self.size:
            raise RuntimeError(f"camera delivered {bgr.shape[1]}x{bgr.shape[0]}, the calibration is for "
                               f"{self.size[0]}x{self.size[1]}")
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)

    def view(self, rgb: np.ndarray):
        """Pixel -> base and config for this image's view (taped-grid registration, as in infer.py)."""
        if not self.register:
            return self.p2b, self.cfg
        from so101_tpgmm.vision.register import correct_view

        p2b, cfg, reg = correct_view(rgb, self.p2b, self.cfg)
        if reg is not None and reg.ok:
            self.last_good = (p2b, cfg)
        elif self.last_good is not None:  # e.g. a hand over the tape: keep the previous correction
            print("  using the previous view correction")
            return self.last_good
        return p2b, cfg



class WristCamera:
    """The wrist camera (used only to confirm a grasp; ``so101_tpgmm.grasp.wrist_blob_fraction``)."""

    def __init__(self, camera: int | str, size: tuple[int, int] = (640, 480)) -> None:
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

    def grab(self) -> np.ndarray:
        for _ in range(5):  # drop buffered frames
            ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError("wrist camera read failed")
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
