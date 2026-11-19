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


def test_sac_camera_source_corrects_moved_view(monkeypatch, tmp_path):
    """SAC on the robot (scratch and residual) gets the cube through CameraCubeSource: same correction."""
    import yaml

    from so101_tpgmm.config import load_config
    from so101_tpgmm.rl import real_env

    ref = cv2.imread(str(resolve_path("configs/reference_top.jpg")))
    scene = ref.copy()
    rect = ((200.0, 250.0), (38.0, 38.0), 20.0)
    cv2.fillPoly(scene, [cv2.boxPoints(rect).astype(np.int32)], (40, 160, 40))  # BGR green cube
    s = 0.9
    Wt = np.array([[s, 0, 45.0], [0, s, 35.0], [0, 0, 1]])
    moved = cv2.warpPerspective(scene, Wt, (640, 480), borderValue=(200, 200, 200))
    calib = tmp_path / "cal.yaml"
    calib.write_text(yaml.safe_dump({"H_pixel_to_base_xy": H.tolist(), "tcp_offset": [0, 0, 0], "plane_z": 0.0064}))
    cfg = load_config(None, [f"vision.calibration.output={calib}"])

    class FakeCap:
        def __init__(self, frame):
            self.frame = frame

        def set(self, *a):
            return True

        def read(self):
            return True, self.frame.copy()

    xy = {}
    for name, frame, register in (("ref", scene, False), ("moved", moved, True), ("raw", moved, False)):
        monkeypatch.setattr(cv2, "VideoCapture", lambda cam, f=frame: FakeCap(f))
        xy[name] = real_env.CameraCubeSource(cfg, 0, register=register)()[0][:2]
    assert np.linalg.norm(xy["moved"] - xy["ref"]) < 0.003
    assert np.linalg.norm(xy["raw"] - xy["ref"]) > 0.01
