"""Camera-to-robot geometry from the dataset itself.

Two steps:

1. :func:`fit_homography_with_tcp_offset` -- a pixel -> base-XY homography fitted on
   (cube pixel centre in the first frame, FK pose at the grasp instant) pairs,
   jointly with a constant offset ``o`` of the cube centre in the TCP frame::

       H(u) ~= (R_tcp o + p_tcp)_xy

   The offset is needed because ``gripper_frame_link`` is not the centre of a
   grasped cube (the SO-101 has one fixed jaw); ignoring it rotates the error
   with the wrist and biases the homography.

2. :func:`decompose_homography` -- with square pixels and the principal point at
   the image centre, a plane homography leaves a one-parameter family of
   cameras indexed by the focal length ``f``. For a near-overhead camera ``f`` and
   the height are confounded (only ``f / Z`` is observable), so one metric
   measurement fixes it: :func:`focal_from_height` or :func:`focal_from_footprint`.

Both measured constraints were found consistent in Phase 0 (docs/dataset_report.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import brentq, least_squares


def apply_homography(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Map ``(N, 2)`` pixels through a 3x3 homography."""
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    q = np.c_[uv, np.ones(len(uv))] @ np.asarray(H).T
    return q[:, :2] / q[:, 2:3]


@dataclass
class HomographyFit:
    """Result of :func:`fit_homography_with_tcp_offset`."""

    H: np.ndarray  # 3x3, pixel -> base XY [m]
    tcp_offset: np.ndarray  # (3,) cube centre in TCP frame [m]
    errors_m: np.ndarray  # (N,) planar reprojection error
    plane_z: float  # median base-frame z of the fitted cube centres

    @property
    def median_error_mm(self) -> float:
        return float(np.median(self.errors_m) * 1e3)


def fit_homography_with_tcp_offset(
    uv: np.ndarray,
    T_tcp: np.ndarray,
    fit_offset: bool = True,
    loss_scale_m: float = 0.005,
) -> HomographyFit:
    """Robustly fit ``H`` (and optionally the TCP->cube offset).

    Args:
        uv: ``(N, 2)`` cube pixel centres.
        T_tcp: ``(N, 4, 4)`` FK TCP poses at the grasp instant (base frame).
        fit_offset: Also estimate the constant offset in the TCP frame.
        loss_scale_m: Soft-L1 scale; residuals above it are down-weighted
            (cubes pushed during teleoperation are outliers).
    """
    uv = np.asarray(uv, dtype=np.float64)
    T = np.asarray(T_tcp, dtype=np.float64)
    H0, _ = cv2.findHomography(uv, T[:, :2, 3], cv2.RANSAC, 0.01)
    if H0 is None:
        raise RuntimeError("Initial RANSAC homography failed.")
    H0 = H0 / H0[2, 2]
    h0 = H0.ravel()[:8]

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H = np.append(x[:8], 1.0).reshape(3, 3)
        o = x[8:] if fit_offset else np.zeros(3)
        return H, o

    def residuals(x: np.ndarray) -> np.ndarray:
        H, o = unpack(x)
        cube = T[:, :3, :3] @ o + T[:, :3, 3]
        return (apply_homography(H, uv) - cube[:, :2]).ravel()

    x0 = np.r_[h0, np.zeros(3)] if fit_offset else h0
    xs = np.r_[np.abs(h0) + 1e-6, [0.01] * 3] if fit_offset else np.abs(h0) + 1e-6
    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=loss_scale_m, x_scale=xs)
    H, o = unpack(sol.x)
    cube = T[:, :3, :3] @ o + T[:, :3, 3]
    err = np.linalg.norm(apply_homography(H, uv) - cube[:, :2], axis=1)
    return HomographyFit(H=H, tcp_offset=np.array(o), errors_m=err, plane_z=float(np.median(cube[:, 2])))


@dataclass
class CameraPose:
    """Pinhole camera w.r.t. the homography plane (base XY axes, z up, z=0 on the plane)."""

    K: np.ndarray  # 3x3 intrinsics
    R: np.ndarray  # world(plane) -> camera rotation
    t: np.ndarray  # world(plane) -> camera translation
    center: np.ndarray  # camera centre in plane coordinates [m]

    @property
    def focal_px(self) -> float:
        return float(self.K[0, 0])

    @property
    def tilt_deg(self) -> float:
        """Angle between the optical axis and the vertical."""
        return float(np.degrees(np.arccos(abs(self.R[2, 2]))))

    def hfov_deg(self, width: int) -> float:
        return float(2 * np.degrees(np.arctan(width / 2 / self.focal_px)))


def decompose_homography(H_pix_to_plane: np.ndarray, f: float, image_size: tuple[int, int]) -> CameraPose:
    """Camera pose for a given focal length (principal point at the image centre)."""
    w, h = image_size
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    G = np.linalg.inv(H_pix_to_plane)  # plane (X, Y, 1) -> pixel
    M = np.linalg.inv(K) @ G
    M = M * 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    if M[2, 2] < 0:  # the plane must be in front of the camera
        M = -M
    R = np.c_[M[:, 0], M[:, 1], np.cross(M[:, 0], M[:, 1])]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    t = M[:, 2]
    return CameraPose(K=K, R=R, t=t, center=-R.T @ t)


def homography_is_orientation_reversing(H_pix_to_plane: np.ndarray, at_px: tuple[float, float] = (320.0, 240.0)) -> bool:
    """True if pixel->plane flips orientation (Jacobian determinant < 0 at ``at_px``).

    It must, for a camera looking *down* on a z-up plane with image axes x-right,
    y-down. A non-flipping map would put the camera under the table, which
    reveals a mirrored joint convention (e.g. a wrong shoulder_pan sign).
    """
    u0 = np.array(at_px, dtype=np.float64)
    e = 0.5
    p = apply_homography(H_pix_to_plane, np.array([u0, u0 + [e, 0], u0 + [0, e]]))
    J = np.c_[(p[1] - p[0]) / e, (p[2] - p[0]) / e]
    return bool(np.linalg.det(J) < 0)


def focal_from_height(H: np.ndarray, height_above_plane: float, image_size: tuple[int, int]) -> float:
    """Focal length that places the camera ``height_above_plane`` metres above the plane."""
    return float(brentq(lambda f: decompose_homography(H, f, image_size).center[2] - height_above_plane, 50, 5000))


def focal_from_footprint(H: np.ndarray, footprint_x: float, image_size: tuple[int, int]) -> float:
    """Focal length that puts the camera centre above base x = ``footprint_x``."""
    return float(brentq(lambda f: decompose_homography(H, f, image_size).center[0] - footprint_x, 50, 5000))
