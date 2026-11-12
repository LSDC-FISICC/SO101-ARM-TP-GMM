"""Pixel -> base homography calibrated from the dataset itself.

Correspondences: the centre of the grasped cube detected in the reference frame of
each episode, and the cube centre at the contact instant from FK,
``p_tcp + R_tcp · offset`` (the offset is estimated jointly; ``gripper_frame_link``
is not the centre of a grasped cube, see docs/dataset_report.md §6).

Robust fit: (1) RANSAC homography on TCP positions to find gross outliers (wrong
cube, cube pushed away before closing), (2) soft-L1 least squares of H + offset on the
RANSAC inliers, (3) re-classification of all points with the refined model and a
final refit. Errors are reported in millimetres on the plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from so101_tpgmm.vision.camera_pose import apply_homography, fit_homography_with_tcp_offset


@dataclass
class VisionCalibration:
    """Result of :func:`calibrate_homography`."""

    H: np.ndarray  # pixel -> base XY [m] (cube centre)
    tcp_offset: np.ndarray  # cube centre in the TCP frame [m]
    inliers: np.ndarray  # (N,) bool
    errors_m: np.ndarray  # (N,) planar error of every correspondence
    plane_z: float  # base z of the cube centres used [m]
    ransac_threshold: float

    @property
    def inlier_median_mm(self) -> float:
        return float(np.median(self.errors_m[self.inliers]) * 1e3)

    @property
    def inlier_p90_mm(self) -> float:
        return float(np.percentile(self.errors_m[self.inliers], 90) * 1e3)

    def save(self, path: str | Path) -> None:
        data = {
            "H_pixel_to_base_xy": np.asarray(self.H).tolist(),
            "tcp_offset": np.asarray(self.tcp_offset).tolist(),
            "plane_z": self.plane_z,
            "ransac_threshold": self.ransac_threshold,
            "n_correspondences": int(len(self.errors_m)),
            "n_inliers": int(self.inliers.sum()),
            "inlier_median_error_mm": round(self.inlier_median_mm, 2),
            "inlier_p90_error_mm": round(self.inlier_p90_mm, 2),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @staticmethod
    def load_h(path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
        """(H, tcp_offset, plane_z) from a saved calibration."""
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return np.array(d["H_pixel_to_base_xy"]), np.array(d["tcp_offset"]), float(d["plane_z"])


def calibrate_homography(
    uv: np.ndarray,
    T_grasp: np.ndarray,
    ransac_threshold: float = 0.015,
    fit_offset: bool = True,
) -> VisionCalibration:
    """Robust pixel -> base calibration (see module docstring).

    Args:
        uv: (N, 2) detected cube centres [px] (undistorted if intrinsics are known).
        T_grasp: (N, 4, 4) FK TCP poses at the contact instant.
        ransac_threshold: inlier distance on the plane [m].
        fit_offset: estimate the cube-centre offset in the TCP frame (recommended).
    """
    uv = np.asarray(uv, dtype=np.float64)
    T = np.asarray(T_grasp, dtype=np.float64)
    if len(uv) < 8:
        raise ValueError("Need at least 8 correspondences.")
    _, mask = cv2.findHomography(uv, T[:, :2, 3], cv2.RANSAC, ransac_threshold)
    inl = mask.ravel().astype(bool)
    fit = fit_homography_with_tcp_offset(uv[inl], T[inl], fit_offset=fit_offset)

    def errors(H, o):
        cube = T[:, :3, :3] @ o + T[:, :3, 3]
        return np.linalg.norm(apply_homography(H, uv) - cube[:, :2], axis=1), cube

    err, _ = errors(fit.H, fit.tcp_offset)
    inl = err <= ransac_threshold
    fit = fit_homography_with_tcp_offset(uv[inl], T[inl], fit_offset=fit_offset)
    err, cube = errors(fit.H, fit.tcp_offset)
    inl = err <= ransac_threshold
    return VisionCalibration(fit.H, fit.tcp_offset, inl, err, float(np.median(cube[inl, 2])), ransac_threshold)
