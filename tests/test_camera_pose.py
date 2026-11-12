"""Homography fitting / decomposition on a synthetic camera."""

import cv2
import numpy as np
import pytest

from so101_tpgmm.vision.camera_pose import (
    apply_homography,
    decompose_homography,
    fit_homography_with_tcp_offset,
    focal_from_height,
    homography_is_orientation_reversing,
)

SIZE = (640, 480)


def look_down_camera(f=500.0, center=(0.25, -0.02, 0.5), tilt_deg=5.0):
    K = np.array([[f, 0, 319.5], [0, f, 239.5], [0, 0, 1]])
    # camera z axis points down (slightly tilted about world y), camera y points toward -x (robot)
    t = np.radians(tilt_deg)
    z = np.array([np.sin(t), 0, -np.cos(t)])
    y = np.array([-1.0, 0, 0])
    y = y - z * (y @ z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.stack([x, y, z])  # rows = camera axes in world -> world->camera rotation
    C = np.array(center)
    return K, R, C


def project(K, R, C, P):
    pc = (P - C) @ R.T
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:]


def synthetic_grasps(n=300, offset=np.array([-0.007, -0.017, -0.017]), seed=0):
    rng = np.random.default_rng(seed)
    T = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        yaw = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        T[i, :3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.diag([1, -1, -1])  # tool pointing down
    cube = np.c_[rng.uniform(0.08, 0.40, n), rng.uniform(-0.15, 0.25, n), np.zeros(n)]
    T[:, :3, 3] = cube - np.einsum("nij,j->ni", T[:, :3, :3], offset)
    return T, cube


def test_recovers_offset_and_mapping():
    K, R, C = look_down_camera()
    T, cube = synthetic_grasps()
    uv = project(K, R, C, cube) + np.random.default_rng(1).normal(0, 0.3, (len(cube), 2))
    fit = fit_homography_with_tcp_offset(uv, T)
    assert fit.median_error_mm < 1.0
    np.testing.assert_allclose(fit.tcp_offset[:2], [-0.007, -0.017], atol=1.5e-3)
    no_off = fit_homography_with_tcp_offset(uv, T, fit_offset=False)
    assert no_off.median_error_mm > 5 * fit.median_error_mm


def test_decomposition_with_known_height_recovers_camera():
    K, R, C = look_down_camera(f=500, center=(0.25, -0.02, 0.5), tilt_deg=5)
    P = np.c_[np.random.default_rng(2).uniform(0.05, 0.45, (200, 2)) * [1, 1] - [0, 0.2], np.zeros(200)]
    H, _ = cv2.findHomography(project(K, R, C, P), P[:, :2])
    f = focal_from_height(H, 0.5, SIZE)
    assert f == pytest.approx(500, rel=1e-3)
    pose = decompose_homography(H, f, SIZE)
    np.testing.assert_allclose(pose.center, C, atol=1e-3)
    assert pose.tilt_deg == pytest.approx(5, abs=0.1)
    assert homography_is_orientation_reversing(H)


def test_mirrored_world_is_detected():
    """A mirrored joint convention (e.g. pan sign flipped) mirrors base y: the map stops reversing orientation."""
    K, R, C = look_down_camera()
    P = np.c_[np.random.default_rng(3).uniform(0.05, 0.4, (50, 2)), np.zeros(50)]
    uv = project(K, R, C, P)
    H, _ = cv2.findHomography(uv, P[:, :2])
    Hm, _ = cv2.findHomography(uv, P[:, :2] * [1, -1])
    assert homography_is_orientation_reversing(H) and not homography_is_orientation_reversing(Hm)
    np.testing.assert_allclose(apply_homography(H, uv), P[:, :2], atol=1e-6)
