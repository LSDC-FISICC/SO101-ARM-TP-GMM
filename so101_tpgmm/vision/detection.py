"""Colour-based cube detection in the fixed camera (HSV thresholds, configurable in YAML).

Pipeline per colour: hue/saturation/value mask (red uses two hue intervals because
OpenCV hue wraps at 180) -> exclusion boxes (the box with a yellow object in it,
the camera stand) -> morphological opening/closing -> connected components ->
area, aspect-ratio and fill filters (reject tape segments and fragments) ->
``cv2.minAreaRect`` for centre and orientation. The orientation of a cube is
only defined modulo 90 deg, so it is folded into [-45, 45) deg.

The yaw returned here is in **image** coordinates; use
:class:`so101_tpgmm.vision.geometry.PixelToBase` to express it in the robot base frame
(the image is mirrored with respect to the table, so the sign flips).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def fold_yaw_90(angle: np.ndarray | float) -> np.ndarray | float:
    """Fold an angle [rad] into [-pi/4, pi/4): a cube looks the same every 90 deg."""
    return (np.asarray(angle) + np.pi / 4) % (np.pi / 2) - np.pi / 4


@dataclass
class ColorSpec:
    """HSV thresholds of one cube colour (OpenCV ranges: H 0-179, S/V 0-255)."""

    name: str
    hue: list[tuple[int, int]]  # inclusive intervals; red needs two
    min_s: int = 120
    min_v: int = 50


@dataclass
class DetectorParams:
    """Thresholds of :func:`detect_cubes` (see ``vision`` in configs/default.yaml)."""

    colors: list[ColorSpec]
    area_px: tuple[int, int] = (900, 2600)
    max_aspect: float = 1.6
    min_fill: float = 0.6
    open_kernel: int = 3
    close_kernel: int = 5
    exclude_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict) -> "DetectorParams":
        colors = [ColorSpec(name, [tuple(r) for r in c["hue"]], c.get("min_s", 120), c.get("min_v", 50))
                  for name, c in cfg["colors"].items()]
        return cls(
            colors=colors,
            area_px=tuple(cfg.get("area_px", (900, 2600))),
            max_aspect=cfg.get("max_aspect", 1.6),
            min_fill=cfg.get("min_fill", 0.6),
            open_kernel=cfg.get("open_kernel", 3),
            close_kernel=cfg.get("close_kernel", 5),
            exclude_boxes=[tuple(b) for b in cfg.get("exclude_boxes", [])],
        )


@dataclass
class CubeDetection:
    """One detected cube (pixel coordinates)."""

    color: str
    center_px: np.ndarray  # (2,) sub-pixel centre of the minAreaRect
    yaw_img: float  # [rad] in [-pi/4, pi/4), image axes (x right, y down)
    size_px: tuple[float, float]
    area_px: int
    fill: float
    contour: np.ndarray = field(repr=False)
    flags: list[str] = field(default_factory=list)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        x, y, w, h = cv2.boundingRect(self.contour)
        return x, y, x + w, y + h


def hue_mask(hsv: np.ndarray, spec: ColorSpec) -> np.ndarray:
    """Boolean mask for one colour; each hue interval may also wrap (lo > hi)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = np.zeros(h.shape, dtype=bool)
    for lo, hi in spec.hue:
        m |= ((h >= lo) & (h <= hi)) if lo <= hi else ((h >= lo) | (h <= hi))
    return m & (s >= spec.min_s) & (v >= spec.min_v)


def detect_cubes(rgb: np.ndarray, params: DetectorParams) -> list[CubeDetection]:
    """Detect cubes of every configured colour in an RGB image."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    k_open = np.ones((params.open_kernel,) * 2, np.uint8)
    k_close = np.ones((params.close_kernel,) * 2, np.uint8)
    out: list[CubeDetection] = []
    for spec in params.colors:
        m = hue_mask(hsv, spec).astype(np.uint8)
        for x0, y0, x1, y1 in params.exclude_boxes:
            m[y0:y1, x0:x1] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not params.area_px[0] <= area <= params.area_px[1]:
                continue
            contours, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = max(contours, key=cv2.contourArea)
            (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
            if min(w, h) < 1:
                continue
            fill = area / (w * h)
            if max(w, h) / min(w, h) > params.max_aspect or fill < params.min_fill:
                continue
            out.append(CubeDetection(
                color=spec.name, center_px=np.array([cx, cy]), yaw_img=float(fold_yaw_90(np.deg2rad(ang))),
                size_px=(float(w), float(h)), area_px=area, fill=float(fill), contour=cnt,
            ))
    return out
