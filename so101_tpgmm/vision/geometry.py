"""Pixel -> robot-base geometry for the fixed camera.

:class:`PixelToBase` maps pixels to XY in the base frame on **one** horizontal plane.
Which plane matters: a homography is exact only for points on the plane it was fitted
on. Cube detections are blob centroids of a cube seen from above (essentially its top
face, 4 cm above the table); a homography of the *table* plane applied to them is
off by about ``h / H_cam`` times the distance from the camera nadir (~7 %, up to
1.5 cm here). The homography used downstream is therefore fitted on the detections
themselves (``vision/calibration.py``), so it maps *detected cube centroids* to *cube
centres* and the plane choice is consistent by construction.

Optional lens undistortion is applied to pixels before the homography when intrinsics
are configured.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from so101_tpgmm.vision.camera_pose import apply_homography
from so101_tpgmm.vision.detection import fold_yaw_90


def undistort_points(uv: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Undistort pixel coordinates (returns pixels in the ideal pinhole image)."""
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)


@dataclass
class PixelToBase:
    """Pixel -> base-frame XY on the calibration plane, with optional undistortion."""

    H: np.ndarray
    K: np.ndarray | None = None
    dist: np.ndarray | None = None

    def _und(self, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        if self.K is not None and self.dist is not None:
            return undistort_points(uv, self.K, self.dist)
        return uv

    def to_base(self, uv: np.ndarray) -> np.ndarray:
        """(N, 2) pixels -> (N, 2) base XY [m]."""
        return apply_homography(self.H, self._und(uv))

    def to_pixel(self, xy: np.ndarray) -> np.ndarray:
        """(N, 2) base XY [m] on the calibration plane -> (N, 2) pixels (inverse of :meth:`to_base`)."""
        und = apply_homography(np.linalg.inv(self.H), np.atleast_2d(np.asarray(xy, dtype=np.float64)))
        if self.K is None or self.dist is None:
            return und
        import cv2

        n = np.c_[(und - self.K[:2, 2]) / np.diag(self.K)[:2], np.ones(len(und))]  # normalized, z = 1
        px, _ = cv2.projectPoints(n, np.zeros(3), np.zeros(3), self.K, self.dist)
        return px.reshape(-1, 2)

    def yaw_to_base(self, center_px: np.ndarray, yaw_img: float, length_px: float = 10.0) -> float:
        """Image-plane orientation -> base-frame yaw [rad], folded to [-pi/4, pi/4).

        The direction is pushed through the homography (two points on the edge
        direction), so mirroring and perspective are handled exactly.
        """
        c = np.asarray(center_px, dtype=np.float64)
        d = np.array([np.cos(yaw_img), np.sin(yaw_img)]) * length_px / 2
        a, b = self.to_base(np.array([c - d, c + d]))
        return float(fold_yaw_90(np.arctan2(b[1] - a[1], b[0] - a[0])))

    def local_scale(self, center_px: np.ndarray) -> float:
        """Metres per pixel around a pixel (mean of the two image directions)."""
        c = np.asarray(center_px, dtype=np.float64)
        p = self.to_base(np.array([c, c + [1, 0], c + [0, 1]]))
        return float(np.mean(np.linalg.norm(p[1:] - p[0], axis=1)))
