"""Scene snapshot for an episode: reference frame, validation and cycle↔cube association.

Reference frame
---------------
The first frame is **not** a reliable snapshot: in 3 episodes (59, 168, 170) the
operator's hand is still placing the cube (skin is even detected as "red"). We use the
last frame before the arm starts moving (motion onset from the segmentation, minus a
small margin): the operator is then on the leader arm and the follower is at home.

Validation (flags, never silent)
--------------------------------
* ``arm_not_home``: the TCP at the reference frame is farther than ``home_tolerance_m``
  from the median home TCP. (Joint angles are not used: the rest posture varies by
  up to ~9 deg per joint between episodes while the TCP stays within ~3 cm.)
* ``scene_not_static``: the table changed between the reference frame and a frame
  ``static_window_s`` earlier (hand or object moving).
* ``near_arm``: the cube's bounding box touches the image region of the arm at home.
* ``partial_blob``: the blob is smaller than ``expected_area_ratio[0]`` × the expected
  footprint of a cube at that position: partly hidden by the arm, or partly lost in
  the mask (a dark red cube lying on the dark tape). The centre may be biased.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from so101_tpgmm.vision.detection import CubeDetection
from so101_tpgmm.vision.geometry import PixelToBase


@dataclass
class SceneParams:
    """Thresholds for the scene snapshot (``vision.scene`` in the YAML)."""

    reference_margin_frames: int = 2
    home_tolerance_m: float = 0.05
    static_window_s: float = 0.5
    static_diff_threshold: int = 40
    static_max_fraction: float = 0.002
    arm_home_roi_px: tuple[int, int, int, int] = (265, 0, 375, 135)
    expected_area_ratio: tuple[float, float] = (0.7, 2.0)
    cube_side: float = 0.04

    @classmethod
    def from_config(cls, cfg: dict, cube_side: float) -> "SceneParams":
        keys = cls.__dataclass_fields__.keys()
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (cfg or {}).items() if k in keys}
        return cls(**kw, cube_side=cube_side)


def reference_frame_index(approach_start: int, params: SceneParams) -> int:
    """Frame used to observe the scene: just before the arm starts moving."""
    return max(0, int(approach_start) - params.reference_margin_frames)


def changed_fraction(a: np.ndarray, b: np.ndarray, threshold: int, ignore: list[tuple[int, int, int, int]]) -> float:
    """Fraction of pixels whose max channel difference exceeds ``threshold``."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) > threshold
    for x0, y0, x1, y1 in ignore:
        d[y0:y1, x0:x1] = False
    return float(d.mean())


def validate_detections(
    dets: list[CubeDetection],
    p2b: PixelToBase,
    params: SceneParams,
    tcp_ref: np.ndarray | None = None,
    tcp_home: np.ndarray | None = None,
    static_fraction: float | None = None,
) -> list[str]:
    """Annotate ``dets[i].flags`` in place; returns scene-level flags."""
    scene_flags: list[str] = []
    if tcp_ref is not None and tcp_home is not None:
        if np.linalg.norm(np.asarray(tcp_ref) - np.asarray(tcp_home)) > params.home_tolerance_m:
            scene_flags.append("arm_not_home")
    if static_fraction is not None and static_fraction > params.static_max_fraction:
        scene_flags.append("scene_not_static")
    ax0, ay0, ax1, ay1 = params.arm_home_roi_px
    for d in dets:
        x0, y0, x1, y1 = d.bbox
        if x0 <= ax1 and x1 >= ax0 and y0 <= ay1 and y1 >= ay0:
            d.flags.append("near_arm")
        side_px = params.cube_side / p2b.local_scale(d.center_px)
        ratio = d.area_px / side_px**2
        if ratio < params.expected_area_ratio[0]:
            d.flags.append("partial_blob")
        elif ratio > params.expected_area_ratio[1]:
            d.flags.append("too_large")
    return scene_flags


@dataclass
class Association:
    """Detected cube assigned to a grasp cycle."""

    detection_index: int | None
    distance_m: float
    candidates: list[float] = field(default_factory=list)


def associate(grasp_xy: np.ndarray, det_xy: np.ndarray, max_distance: float) -> Association:
    """Nearest detected cube (base XY) to the grasp position; None beyond ``max_distance``."""
    if len(det_xy) == 0:
        return Association(None, float("inf"))
    d = np.linalg.norm(np.asarray(det_xy) - np.asarray(grasp_xy)[None, :2], axis=1)
    i = int(np.argmin(d))
    return Association(i if d[i] <= max_distance else None, float(d[i]), sorted(d.tolist()))
