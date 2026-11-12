"""Vision: colour detection, 90° yaw folding, red hue wrap, homography plane, RANSAC, scene checks."""

import cv2
import numpy as np
import pytest

from so101_tpgmm.vision.calibration import calibrate_homography
from so101_tpgmm.vision.detection import ColorSpec, DetectorParams, detect_cubes, fold_yaw_90, hue_mask
from so101_tpgmm.vision.geometry import PixelToBase
from so101_tpgmm.vision.scene import SceneParams, associate, changed_fraction, validate_detections

SPECS = [ColorSpec("red", [(0, 10), (170, 179)]), ColorSpec("yellow", [(18, 35)]), ColorSpec("green", [(40, 90)]),
         ColorSpec("blue", [(100, 130)], min_s=150, min_v=130)]
PARAMS = DetectorParams(colors=SPECS)


def hsv_color(h, s=220, v=200):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0, 0])


def draw_square(img, center, side, angle_deg, rgb):
    box = cv2.boxPoints((center, (side, side), angle_deg)).astype(np.int32)
    cv2.fillPoly(img, [box], rgb)


def scene():
    img = np.full((480, 640, 3), 225, np.uint8)
    draw_square(img, (150, 150), 38, 30, hsv_color(2))  # red, low hue
    draw_square(img, (300, 150), 38, -20, hsv_color(176))  # red, high hue (wraps)
    draw_square(img, (450, 150), 38, 75, hsv_color(27))  # yellow, 75 deg == -15 deg
    draw_square(img, (150, 330), 38, 10, hsv_color(60))  # green
    cv2.rectangle(img, (250, 300), (560, 314), hsv_color(118, 110, 96), -1)  # blue tape strip
    draw_square(img, (500, 400), 12, 0, hsv_color(60))  # too small
    return img


def test_fold_yaw_90():
    deg = np.array([10, 100, -80, 55, 45, -45, 190])
    np.testing.assert_allclose(np.degrees(fold_yaw_90(np.radians(deg))), [10, 10, 10, -35, -45, -45, 10], atol=1e-9)


def test_red_hue_wraps():
    img = scene()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m_two = hue_mask(hsv, SPECS[0])
    m_wrap = hue_mask(hsv, ColorSpec("red", [(170, 10)]))  # one wrapped interval
    assert m_two[150, 150] and m_two[150, 300] and m_wrap[150, 150] and m_wrap[150, 300]
    assert not hue_mask(hsv, ColorSpec("red", [(0, 10)]))[150, 300]  # a single low interval misses it


def test_detect_cubes_colors_positions_and_folded_yaw():
    dets = detect_cubes(scene(), PARAMS)
    found = sorted((d.color, round(d.center_px[0]), round(d.center_px[1]), round(np.degrees(d.yaw_img))) for d in dets)
    assert [f[0] for f in found] == ["green", "red", "red", "yellow"]  # tape and small blob rejected
    expected = {(150, 150): 30, (300, 150): -20, (450, 150): -15, (150, 330): 10}
    for _, x, y, yaw in found:
        exp = expected[min(expected, key=lambda c: (c[0] - x) ** 2 + (c[1] - y) ** 2)]
        assert abs(((yaw - exp) + 45) % 90 - 45) <= 2


def test_yaw_ambiguity_of_90_degrees():
    a, b = np.full((200, 200, 3), 225, np.uint8), np.full((200, 200, 3), 225, np.uint8)
    draw_square(a, (100, 100), 40, 20, hsv_color(60))
    draw_square(b, (100, 100), 40, 110, hsv_color(60))
    ya, yb = detect_cubes(a, PARAMS)[0].yaw_img, detect_cubes(b, PARAMS)[0].yaw_img
    assert abs(ya - yb) < np.radians(2)


def test_yaw_to_base_handles_mirroring():
    """The camera image is mirrored w.r.t. the table: an image angle of +20 deg is -20 deg on the table."""
    mirror = PixelToBase(np.diag([0.001, -0.001, 1.0]))
    assert np.degrees(mirror.yaw_to_base(np.array([100.0, 100.0]), np.radians(20))) == pytest.approx(-20, abs=1e-6)
    rot = PixelToBase(np.array([[0, -0.001, 0], [0.001, 0, 0], [0, 0, 1.0]]))  # 90 deg rotation
    assert np.degrees(rot.yaw_to_base(np.array([50.0, 50.0]), np.radians(20))) == pytest.approx(20, abs=1e-6)


def synthetic_camera():
    f, C = 500.0, np.array([0.25, 0.0, 0.55])
    K = np.array([[f, 0, 319.5], [0, f, 239.5], [0, 0, 1]])
    R = np.array([[0, -1.0, 0], [-1.0, 0, 0], [0, 0, -1.0]])  # looking down, mirrored like the real camera
    return K, R, C


def project(K, R, C, P):
    pc = (P - C) @ R.T
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:]


def test_homography_on_the_wrong_plane_is_biased():
    K, R, C = synthetic_camera()
    rng = np.random.default_rng(0)
    xy = np.c_[rng.uniform(0.08, 0.42, 200), rng.uniform(-0.2, 0.25, 200)]
    table, top = np.c_[xy, np.zeros(200)], np.c_[xy, np.full(200, 0.04)]
    H_table, _ = cv2.findHomography(project(K, R, C, table), xy)
    H_top, _ = cv2.findHomography(project(K, R, C, top), xy)
    uv_top = project(K, R, C, top)  # what the detector sees: the top of the cube
    err_wrong = np.linalg.norm(PixelToBase(H_table).to_base(uv_top) - xy, axis=1)
    err_right = np.linalg.norm(PixelToBase(H_top).to_base(uv_top) - xy, axis=1)
    assert np.max(err_right) < 1e-6
    assert np.median(err_wrong) > 0.005  # ~h/H x distance from the nadir


def test_ransac_calibration_rejects_pushed_cubes():
    K, R, C = synthetic_camera()
    rng = np.random.default_rng(1)
    n = 150
    offset = np.array([-0.0075, -0.017, -0.017])
    T = np.tile(np.eye(4), (n, 1, 1))
    cube = np.c_[rng.uniform(0.08, 0.42, n), rng.uniform(-0.2, 0.25, n), np.full(n, 0.01)]
    for i in range(n):
        yaw = rng.uniform(-np.pi, np.pi)
        T[i, :3, :3] = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]) @ np.diag([1, -1, -1])
    T[:, :3, 3] = cube - np.einsum("nij,j->ni", T[:, :3, :3], offset)
    observed = cube.copy()
    pushed = rng.choice(n, 25, replace=False)
    observed[pushed, :2] += rng.normal(0, 0.04, (25, 2)) + 0.03  # cube moved before the grasp
    uv = project(K, R, C, observed)
    cal = calibrate_homography(uv, T, 0.015)
    good = np.setdiff1d(np.arange(n), pushed)
    assert cal.inliers[good].mean() > 0.97 and cal.inliers[pushed].mean() < 0.2
    assert np.median(cal.errors_m[good]) < 1e-3


def test_association_and_max_distance():
    a = associate(np.array([0.2, 0.1]), np.array([[0.5, 0.1], [0.21, 0.1]]), 0.03)
    assert a.detection_index == 1 and a.distance_m == pytest.approx(0.01)
    assert associate(np.array([0.2, 0.1]), np.array([[0.5, 0.1]]), 0.03).detection_index is None
    assert associate(np.array([0.2, 0.1]), np.zeros((0, 2)), 0.03).detection_index is None


def test_scene_validation_flags():
    img = np.full((480, 640, 3), 225, np.uint8)
    draw_square(img, (150, 300), 38, 0, hsv_color(60))  # full cube
    draw_square(img, (450, 300), 38, 0, hsv_color(60))
    img[270:330, 459:475] = 225  # a strip missing -> ~28x38 px blob, below 0.7 x the expected 40x40
    dets = sorted(detect_cubes(img, DetectorParams(colors=SPECS, max_aspect=2.5, min_fill=0.5)), key=lambda d: d.center_px[0])
    p2b = PixelToBase(np.diag([0.001, 0.001, 1.0]))  # 1 mm/px -> a 4 cm cube is 40 px
    flags = validate_detections(dets, p2b, SceneParams(), tcp_ref=np.array([0.2, 0, 0.1]), tcp_home=np.zeros(3),
                                static_fraction=0.01)
    assert set(flags) == {"arm_not_home", "scene_not_static"}
    assert "partial_blob" not in dets[0].flags and "partial_blob" in dets[1].flags
    a, b = np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10, 3), np.uint8)
    b[0, 0] = 200
    assert changed_fraction(a, b, 40, []) == pytest.approx(0.01)
    assert changed_fraction(a, b, 40, [(0, 0, 2, 2)]) == 0.0


def test_reference_frame_fixes_hand_in_first_frame(real_dataset):
    """Episodes 168/170: the cube is still in the operator's hand in the first frame (no detection);
    the reference frame (just before motion onset) shows it on the table. In 59 the hand is in
    the first frame too, but its skin blob is rejected by the area filter."""
    import pandas as pd

    from so101_tpgmm.config import load_config, resolve_path
    from so101_tpgmm.vision.scene import reference_frame_index

    cfg = load_config()
    seg_path = resolve_path(cfg["segmentation"]["output_csv"])
    if not seg_path.exists():
        pytest.skip("run scripts/segment_dataset.py first")
    seg = pd.read_csv(seg_path).set_index("episode")
    dp = DetectorParams.from_config(cfg["vision"])
    sp = SceneParams.from_config(cfg["vision"]["scene"], 0.04)
    for e in (59, 168, 170):
        ep = real_dataset.episode(e)
        i = reference_frame_index(int(seg.loc[e, "approach_start"]), sp)
        first, ref = real_dataset.read_frames("observation.images.top", e, ep.timestamps[[0, i]])[0]
        if e in (168, 170):
            assert len(detect_cubes(first, dp)) == 0
        assert len(detect_cubes(ref, dp)) == 1


def test_optional_undistortion():
    from so101_tpgmm.vision.geometry import undistort_points

    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    uv = np.array([[100.0, 50.0], [320.0, 240.0], [600.0, 400.0]])
    np.testing.assert_allclose(undistort_points(uv, K, np.zeros(5)), uv, atol=1e-6)
    dist = np.array([-0.2, 0.05, 0, 0, 0])  # barrel
    und = undistort_points(uv, K, dist)
    assert np.allclose(und[1], uv[1], atol=1e-6)  # principal point unchanged
    assert np.linalg.norm(und[0] - K[:2, 2]) > np.linalg.norm(uv[0] - K[:2, 2])  # pushed outwards
    p2b = PixelToBase(np.eye(3), K, dist)
    np.testing.assert_allclose(p2b.to_base(uv), und, atol=1e-9)
