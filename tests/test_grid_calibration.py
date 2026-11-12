"""Grid calibration on a synthetic rendering of the taped lanes."""

import cv2
import numpy as np
import pytest

from so101_tpgmm.vision.camera_pose import apply_homography, decompose_homography, focal_from_height
from so101_tpgmm.vision.grid_calibration import GridSpec, extract_inner_edges, fit_grid, line_value

TAPE = 0.016
SPEC = GridSpec(4, 0.10, 0.40, 0.02)


def render(H_grid_to_px: np.ndarray, size=(640, 480)) -> np.ndarray:
    """Draw the grid at 1 px/mm in grid coordinates, then warp it into the camera image."""
    mm = 1000
    off = 40  # mm margin
    W = int((4 * 0.10 + 5 * TAPE) * mm) + 2 * off
    Hh = int((0.40 + 2 * TAPE) * mm) + 2 * off
    top = np.full((Hh, W, 3), 235, np.uint8)
    blue = (40, 45, 150)
    p = SPEC.lane_width + TAPE

    def to_canvas(X, Y):  # grid metres -> canvas pixels (Y up)
        return int(round((X + TAPE) * mm)) + off, Hh - (int(round((Y + TAPE) * mm)) + off)

    # outer rectangle of tape then white lanes
    x0, y0 = to_canvas(-TAPE, -TAPE)
    x1, y1 = to_canvas(4 * p - TAPE + TAPE, 0.40 + TAPE)
    cv2.rectangle(top, (x0, y1), (x1, y0), blue, -1)
    for i in range(4):
        a = to_canvas(i * p, 0.0)
        b = to_canvas(i * p + SPEC.lane_width, 0.40)
        cv2.rectangle(top, (a[0], b[1]), (b[0] - 1, a[1] - 1), (235, 235, 235), -1)
    # canvas pixel -> grid metres
    C = np.array([[1 / mm, 0, -off / mm - TAPE], [0, -1 / mm, (Hh - off) / mm - TAPE], [0, 0, 1]])
    img = cv2.warpPerspective(top, H_grid_to_px @ C, size, flags=cv2.INTER_AREA, borderValue=(235, 235, 235))
    return cv2.GaussianBlur(img, (3, 3), 0)


@pytest.fixture(scope="module")
def synthetic():
    # camera ~55 cm above, slightly tilted: build grid->pixel homography from K [r1 r2 t]
    f, tilt = 500.0, np.radians(8)
    K = np.array([[f, 0, 319.5], [0, f, 239.5], [0, 0, 1]])
    z = np.array([0, np.sin(tilt), -np.cos(tilt)])  # looks towards +Y (the robot), like the real camera
    x = np.array([1.0, 0, 0])
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    C = np.array([0.22, 0.12, 0.55])
    t = -R @ C
    H = K @ np.c_[R[:, 0], R[:, 1], t]
    H /= H[2, 2]
    return render(H), H, f, C


def test_recovers_tape_width_and_mapping(synthetic):
    img, H_true, _, _ = synthetic
    edges = extract_inner_edges(img, SPEC)
    assert len(edges.uv) > 1000
    cal = fit_grid(edges, SPEC, fit_distortion=False)
    assert cal.tape_width == pytest.approx(TAPE, abs=1.5e-3)
    corners = np.array([[0, 0], [0.1, 0.4], [3 * (0.1 + TAPE) + 0.1, 0.0], [0.2, 0.2]])
    px = apply_homography(H_true, corners)
    np.testing.assert_allclose(cal.pixel_to_grid(px), corners, atol=2.5e-3)


def test_pose_from_height(synthetic):
    img, _, f_true, C_true = synthetic
    cal = fit_grid(extract_inner_edges(img, SPEC), SPEC, fit_distortion=False)
    f = focal_from_height(cal.H, C_true[2], (640, 480))
    assert f == pytest.approx(f_true, rel=0.02)
    pose = decompose_homography(cal.H, f, (640, 480))
    np.testing.assert_allclose(pose.center[:2], C_true[:2], atol=0.015)
    assert pose.tilt_deg == pytest.approx(8, abs=1.0)


def test_line_values():
    v = line_value(np.array([0, 1, 2, 7, 100, 101]), SPEC, TAPE)
    np.testing.assert_allclose(v, [0, 0.1, 0.1 + TAPE, 3 * (0.1 + TAPE) + 0.1, 0, 0.4])
