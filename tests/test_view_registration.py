"""View registration on the taped grid: a moved/zoomed camera is detected and corrected."""

import cv2
import numpy as np

from so101_tpgmm.config import resolve_path
from so101_tpgmm.vision.geometry import PixelToBase, apply_homography
from so101_tpgmm.vision.register import register_view, registered_p2b, warp_box

# pixel -> base of roughly the dataset3 calibration (only its general shape matters here)
H = np.array([[-1.1176e-04, 9.6872e-04, 4.3968e-02], [8.8654e-04, 9.5677e-05, -2.8609e-01],
              [-2.0675e-04, -6.7702e-05, 1.0]])


def test_moved_camera_is_corrected():
    ref = cv2.cvtColor(cv2.imread(str(resolve_path("configs/reference_top.jpg"))), cv2.COLOR_BGR2RGB)
    p2b = PixelToBase(H)
    same = register_view(ref, ref, p2b)
    assert same.ok and same.shift_mm.max() < 1.0
    a, s = np.deg2rad(3.0), 0.9
    Wt = np.array([[s * np.cos(a), -s * np.sin(a), 45.0], [s * np.sin(a), s * np.cos(a), 20.0], [0, 0, 1]])
    moved = cv2.warpPerspective(ref, Wt, (640, 480), borderValue=(200, 200, 200))
    reg = register_view(moved, ref, p2b)
    assert reg.ok and reg.shift_mm.max() > 20  # uncorrected: grasps off by centimetres
    xy = np.c_[np.linspace(0.15, 0.33, 8), np.linspace(-0.12, 0.2, 8)]
    px_now = apply_homography(Wt, p2b.to_pixel(xy))
    err = np.linalg.norm(registered_p2b(p2b, reg).to_base(px_now) - xy, axis=1)
    assert err.max() < 0.003


def test_warp_box():
    W = np.array([[1.0, 0, 10], [0, 1.0, -5], [0, 0, 1]])
    assert warp_box([0, 10, 20, 30], W) == [10, 5, 30, 25]
