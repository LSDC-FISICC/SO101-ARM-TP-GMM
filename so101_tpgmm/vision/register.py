"""Register the fixed camera's current view to the calibration view, using the taped grid.

The pixel -> base homography (``outputs/vision_calibration.yaml``) was fitted on the dataset3
recordings. If the camera is moved, tilted, re-zoomed, or the photo is taken at another resolution
and resized, the same table point lands on another pixel and every detected cube is planned at the
wrong place. The error grows with the distance from the point where both views agree. On the
2026-11 robot test, a view scaled by ~0.9 shifted the grasps by 12-26 mm, enough for a fingertip to
land on top of a 4 cm cube.

The blue tape grid and the robot are fixed to the table, so the tape is a calibration target that is
always in the picture. :func:`register_view` aligns the tape mask of the current image to that of
a reference frame of the recording (``configs/reference_top.jpg``, median background of the dataset's
first frames) with a homography ``W`` (reference pixel -> current pixel). Since the tape lies on the
table plane, ``H_now = H_ref · W⁻¹`` is the pixel -> base homography of the current view.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from so101_tpgmm.vision.geometry import PixelToBase, apply_homography

# Dark-blue tape in OpenCV HSV (H 0-180). The box rim is a similar colour: masked in the reference.
TAPE_HSV_LO = (100, 80, 30)
TAPE_HSV_HI = (135, 255, 200)


@dataclass
class Registration:
    W: np.ndarray            # reference pixel -> current pixel (3x3)
    ecc: float               # ECC correlation of the tape masks after alignment (1 = perfect)
    shift_mm: np.ndarray     # per workspace point: base-XY error if the view change were ignored
    scale: float             # mean image scale current / reference

    @property
    def ok(self) -> bool:
        return self.ecc >= 0.7

    def summary(self) -> str:
        return (f"view vs calibration: scale {self.scale:.3f}, alignment ECC {self.ecc:.2f}, "
                f"uncorrected grasp error median {np.median(self.shift_mm):.1f} mm, max {self.shift_mm.max():.1f} mm")


def tape_mask(rgb: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, TAPE_HSV_LO, TAPE_HSV_HI).astype(np.float32) / 255.0
    return cv2.GaussianBlur(m, (0, 0), sigma)


def _moment_guess(ref: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Similarity (scale + shift) matching the centroid and spread of the two tape masks."""
    def stats(m):
        ys, xs = np.nonzero(m > 0.5)
        c = np.array([xs.mean(), ys.mean()])
        return c, np.sqrt(((xs - c[0]) ** 2 + (ys - c[1]) ** 2).mean())
    (c0, s0), (c1, s1) = stats(ref), stats(cur)
    k = s1 / s0
    return np.array([[k, 0, c1[0] - k * c0[0]], [0, k, c1[1] - k * c0[1]], [0, 0, 1]], np.float32)


def register_view(rgb: np.ndarray, ref_rgb: np.ndarray, p2b: PixelToBase,
                  workspace_xy: np.ndarray | None = None) -> Registration:
    """Align the tape of ``rgb`` to that of ``ref_rgb`` (both RGB). Images of other sizes are resized."""
    h, w = ref_rgb.shape[:2]
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    ref, cur = tape_mask(ref_rgb), tape_mask(rgb)
    if (ref > 0.5).sum() < 500 or (cur > 0.5).sum() < 500:
        raise RuntimeError("tape grid not visible enough to register the view")
    tmpl_mask = np.ones((h, w), np.uint8)
    # the box rim (top right, tape-coloured) can move between sessions: register on the tape only
    tmpl_mask[: int(0.21 * h), int(0.69 * w):] = 0
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)
    best = (-1.0, None)
    for W0 in (_moment_guess(ref, cur), np.eye(3, dtype=np.float32)):
        W = W0.copy()
        try:
            for s in (8, 4, 2):
                a = cv2.GaussianBlur(ref, (0, 0), s)
                b = cv2.GaussianBlur(cur, (0, 0), s)
                cc, W = cv2.findTransformECC(a, b, W, cv2.MOTION_HOMOGRAPHY, crit, tmpl_mask, 5)
        except cv2.error:
            continue
        if cc > best[0]:
            best = (cc, W)
    if best[1] is None:
        raise RuntimeError("view registration did not converge")
    cc, W = best[0], best[1].astype(np.float64)
    if workspace_xy is None:  # table area of the demonstrations (base frame, metres)
        gx, gy = np.meshgrid(np.linspace(0.15, 0.33, 7), np.linspace(-0.12, 0.20, 9))
        workspace_xy = np.c_[gx.ravel(), gy.ravel()]
    ref_px = p2b.to_pixel(workspace_xy)
    naive = p2b.to_base(apply_homography(W, ref_px))  # what the old calibration would compute now
    shift = np.linalg.norm(naive - workspace_xy, axis=1) * 1e3
    scale = float(np.sqrt(abs(np.linalg.det(W[:2, :2]))))
    return Registration(W, float(cc), shift, scale)


def registered_p2b(p2b: PixelToBase, reg: Registration) -> PixelToBase:
    """Pixel -> base for the current view. With lens intrinsics, the warp is applied to the
    undistorted pixels (an approximation, exact without distortion; ``vision.intrinsics`` is null)."""
    return PixelToBase(p2b.H @ np.linalg.inv(reg.W), p2b.K, p2b.dist)


def warp_box(box, W: np.ndarray) -> list[int]:
    """Axis-aligned pixel box [x0, y0, x1, y1] of the reference view -> bounding box in the current view."""
    x0, y0, x1, y1 = box
    c = apply_homography(W, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))
    return [int(np.floor(c[:, 0].min())), int(np.floor(c[:, 1].min())), int(np.ceil(c[:, 0].max())),
            int(np.ceil(c[:, 1].max()))]


def correct_view(rgb: np.ndarray, p2b: PixelToBase, cfg: dict, apply: bool = True, log=print
                 ) -> tuple[PixelToBase, dict, Registration | None]:
    """Register ``rgb`` (RGB, calibration size) to ``vision.reference_image`` and correct the calibration.

    Returns the pixel -> base for this view, a copy of ``cfg`` whose pixel regions defined on the
    calibration view (``vision.exclude_boxes``, ``vision.scene.arm_home_roi_px``) follow the camera, and
    the registration (None if it failed). Used by ``scripts/infer.py`` (TP-GMM, LPV-DS, IK planner) and
    by ``rl/real_env.py`` (SAC on the robot, scratch and residual), so every policy sees the same correction.
    """
    import copy

    from so101_tpgmm.config import resolve_path

    ref = cv2.imread(str(resolve_path(cfg["vision"]["reference_image"])))
    try:
        reg = register_view(rgb, cv2.cvtColor(ref, cv2.COLOR_BGR2RGB), p2b)
    except RuntimeError as e:
        log(f"WARNING: view not registered ({e}); using the calibration as is")
        return p2b, cfg, None
    log(reg.summary())
    if not reg.ok:
        log("WARNING: the taped grid could not be aligned reliably; using the calibration as is. "
            "Check that the whole grid is visible and the camera points at the table.")
        return p2b, cfg, reg
    if np.max(reg.shift_mm) > 5:
        log(f"  the camera view differs from the calibration view "
            f"({'corrected' if apply else 'NOT corrected: --register on'})")
    if not apply:
        return p2b, cfg, reg
    cfg = copy.deepcopy(cfg)
    vcfg = cfg["vision"]
    vcfg["exclude_boxes"] = [warp_box(b, reg.W) for b in vcfg.get("exclude_boxes", [])]
    vcfg["scene"]["arm_home_roi_px"] = warp_box(vcfg["scene"]["arm_home_roi_px"], reg.W)
    return registered_p2b(p2b, reg), cfg, reg
