"""Metric camera calibration from the blue floor-tape grid.

The table carries ``n_lanes`` taped rectangles side by side. Their **inner**
dimensions are known (``lane_width`` x ``lane_length``, measured on the real
setup); the tape width is not, and is estimated.

Grid frame ``G`` (right-handed, z up, metres): origin at the inner corner of lane 0
at the end *far from the robot* (bottom-left in the image), X across the lanes
(image right), Y along the lanes towards the robot (image up). Lane ``i`` spans
``X in [i*p, i*p + lane_width]``, ``Y in [0, lane_length]``, ``p = lane_width + tape_width``.

Instead of corners (several are hidden by the robot and the box) we use every
visible **inner tape edge pixel**, located with sub-pixel precision on the
saturation profile, and fit the homography so that each edge point lands on
its known grid line (point-to-line residuals). An optional one-coefficient
radial distortion is fitted jointly to test for lens distortion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from so101_tpgmm.vision.camera_pose import apply_homography


@dataclass
class GridSpec:
    """Physical description of the taped grid."""

    n_lanes: int = 4
    lane_width: float = 0.10
    lane_length: float = 0.40
    tape_width_guess: float = 0.02


@dataclass
class EdgePoints:
    """Sub-pixel edge samples with the grid line they belong to."""

    uv: np.ndarray  # (N, 2) pixels
    axis: np.ndarray  # (N,) 0 -> constrains grid X, 1 -> constrains grid Y
    line: np.ndarray  # (N,) line id (see GridCalibration.line_value)
    source: list[str] = field(default_factory=list)


@dataclass
class GridCalibration:
    """Result of :func:`fit_grid`."""

    H: np.ndarray  # undistorted pixel -> grid XY [m]
    tape_width: float
    k1: float  # radial distortion coefficient (normalised by norm_f)
    norm_f: float
    center: tuple[float, float]
    spec: GridSpec
    residuals_m: np.ndarray
    edges: EdgePoints

    def undistort(self, uv: np.ndarray) -> np.ndarray:
        return undistort_radial(uv, self.k1, self.norm_f, self.center)

    def pixel_to_grid(self, uv: np.ndarray) -> np.ndarray:
        """Pixels -> grid-plane XY (table surface) [m]."""
        return apply_homography(self.H, self.undistort(np.atleast_2d(uv)))

    @property
    def rms_mm(self) -> float:
        return float(np.sqrt(np.mean(self.residuals_m**2)) * 1e3)


def undistort_radial(uv: np.ndarray, k1: float, f: float, c: tuple[float, float]) -> np.ndarray:
    """First-order inverse radial model ``p_u = c + (p - c)(1 + k1 r^2)``, ``r`` normalised by ``f``."""
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    d = uv - np.asarray(c)
    r2 = np.sum(d**2, axis=1, keepdims=True) / f**2
    return np.asarray(c) + d * (1.0 + k1 * r2)


def tape_mask(rgb: np.ndarray, hue_range: tuple[int, int] = (100, 140), min_sat: int = 50) -> np.ndarray:
    """Boolean mask of the blue tape (OpenCV hue in [0, 180))."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 0] >= hue_range[0]) & (hsv[..., 0] <= hue_range[1]) & (hsv[..., 1] > min_sat)


def _subpixel_crossing(profile: np.ndarray, i0: int, i1: int, level: float) -> float | None:
    """First crossing of ``level`` by ``profile`` between indices i0 -> i1 (either direction)."""
    step = 1 if i1 >= i0 else -1
    for i in range(i0, i1, step):
        a, b = profile[i], profile[i + step]
        if (a - level) * (b - level) <= 0 and a != b:
            return i + step * (level - a) / (b - a)
    return None


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def extract_inner_edges(
    rgb: np.ndarray,
    spec: GridSpec,
    exclude: tuple[tuple[int, int, int, int], ...] = (),
    tape_px_range: tuple[int, int] = (4, 30),
) -> EdgePoints:
    """Sample the inner edges of the tape grid with sub-pixel precision.

    Args:
        rgb: Clean background image (e.g. median of first frames).
        spec: Grid description (only ``n_lanes`` is used here).
        exclude: Pixel boxes ``(x0, y0, x1, y1)`` to ignore (box, robot, ...).
        tape_px_range: Accepted tape width in pixels along a scanline.
    """
    mask = tape_mask(rgb)
    for x0, y0, x1, y1 in exclude:
        mask[y0:y1, x0:x1] = False
    sat = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float64), (3, 3), 0)
    h, w = mask.shape
    n_strips = spec.n_lanes + 1

    # Rough strip positions from projections.
    xs, _ = find_peaks(mask[h // 4 : 3 * h // 4].sum(0), distance=40)
    xs = np.sort(xs[np.argsort(mask[h // 4 : 3 * h // 4].sum(0)[xs])[-n_strips:]])
    proj_y = mask[:, xs[0] + 10 : xs[-1] - 10].sum(1)
    ys, _ = find_peaks(proj_y, distance=100)
    ys = np.sort(ys[np.argsort(proj_y[ys])[-2:]])
    if len(xs) != n_strips or len(ys) != 2:
        raise RuntimeError(f"Expected {n_strips} vertical and 2 horizontal tape strips, got {xs}, {ys}")
    y_top, y_bot = int(ys[0]), int(ys[1])

    uv, axis, line, src = [], [], [], []
    lo, hi = tape_px_range

    def edge_pair(profile: np.ndarray, a: int, b: int) -> tuple[float | None, float | None]:
        if a - 5 < 0 or b + 5 >= profile.size:
            return None, None
        peak = profile[a : b + 1].max()
        left_bg, right_bg = profile[a - 5 : a - 2].mean(), profile[b + 3 : b + 6].mean()
        if peak - max(left_bg, right_bg) < 25:
            return None, None
        e0 = _subpixel_crossing(profile, a - 4, a + (b - a) // 2, 0.5 * (peak + left_bg))
        e1 = _subpixel_crossing(profile, b + 4, a + (b - a) // 2, 0.5 * (peak + right_bg))
        return e0, e1

    # Vertical strips: scan rows between the horizontal strips.
    # Line ids 0..2*n_lanes-1: lane i left edge = 2i, right edge = 2i+1 (axis X).
    for y in range(y_top + 15, y_bot - 15):
        for a, b in _runs(mask[y]):
            if not lo <= b - a + 1 <= hi:
                continue
            j = int(np.argmin(np.abs(xs - (a + b) / 2)))
            if abs(xs[j] - (a + b) / 2) > 35:
                continue
            e_left, e_right = edge_pair(sat[y], a, b)
            if j > 0 and e_left is not None:  # left edge of strip j = right inner edge of lane j-1
                uv.append((e_left, y)); axis.append(0); line.append(2 * (j - 1) + 1); src.append("v")
            if j < spec.n_lanes and e_right is not None:  # right edge of strip j = left inner edge of lane j
                uv.append((e_right, y)); axis.append(0); line.append(2 * j); src.append("v")
    # Horizontal strips: scan columns inside each lane.
    # Line id 100 = far (bottom in image) inner edge, Y = 0; 101 = near (top) inner edge, Y = L.
    for x in range(int(xs[0]) + 15, int(xs[-1]) - 15):
        if np.min(np.abs(xs - x)) < 18:
            continue
        col = mask[:, x]
        for a, b in _runs(col):
            if not lo <= b - a + 1 <= hi:
                continue
            mid = (a + b) / 2
            e_up, e_down = edge_pair(sat[:, x], a, b)
            if abs(mid - y_bot) < 30 and e_up is not None:
                uv.append((x, e_up)); axis.append(1); line.append(100); src.append("h")
            elif abs(mid - y_top) < 30 and e_down is not None:
                uv.append((x, e_down)); axis.append(1); line.append(101); src.append("h")
    return EdgePoints(np.array(uv, float), np.array(axis), np.array(line), src)


def line_value(line_id: np.ndarray, spec: GridSpec, tape_width: float) -> np.ndarray:
    """Grid coordinate (X or Y) of each edge line id."""
    line_id = np.asarray(line_id)
    p = spec.lane_width + tape_width
    lane = line_id // 2
    xval = lane * p + (line_id % 2) * spec.lane_width
    yval = np.where(line_id == 101, spec.lane_length, 0.0)
    return np.where(line_id >= 100, yval, xval)


def fit_grid(
    edges: EdgePoints,
    spec: GridSpec,
    image_size: tuple[int, int] = (640, 480),
    fit_distortion: bool = True,
    norm_f: float = 500.0,
    loss_scale_m: float = 0.001,
) -> GridCalibration:
    """Fit pixel->grid homography (+ tape width, + optional radial k1) to edge points."""
    c = ((image_size[0] - 1) / 2, (image_size[1] - 1) / 2)
    uv, ax = edges.uv, edges.axis

    # Initial homography from the rough corners implied by the edge samples.
    def rough_corners() -> tuple[np.ndarray, np.ndarray]:
        img, world = [], []
        for ln_x, X in ((0, 0.0), (2 * spec.n_lanes - 1, None)):
            for ln_y in (100, 101):
                sel_x = edges.line == ln_x
                sel_y = edges.line == ln_y
                if sel_x.sum() < 10 or sel_y.sum() < 10:
                    continue
                fx = np.polyfit(uv[sel_x, 1], uv[sel_x, 0], 1)  # u = a v + b
                fy = np.polyfit(uv[sel_y, 0], uv[sel_y, 1], 1)  # v = c u + d
                v = (fy[0] * fx[1] + fy[1]) / (1 - fy[0] * fx[0])
                img.append((fx[0] * v + fx[1], v))
                Xw = line_value(np.array([ln_x]), spec, spec.tape_width_guess)[0]
                world.append((Xw, line_value(np.array([ln_y]), spec, 0)[0]))
        return np.array(img), np.array(world)

    ci, cw = rough_corners()
    if len(ci) < 4:
        raise RuntimeError("Not enough edges to initialise the grid homography.")
    H0 = cv2.getPerspectiveTransform(ci[:4].astype(np.float32), cw[:4].astype(np.float32))
    H0 = H0 / H0[2, 2]

    def unpack(x: np.ndarray) -> tuple[np.ndarray, float, float]:
        return np.append(x[:8], 1).reshape(3, 3), x[8], (x[9] if fit_distortion else 0.0)

    def residuals(x: np.ndarray) -> np.ndarray:
        H, tw, k1 = unpack(x)
        g = apply_homography(H, undistort_radial(uv, k1, norm_f, c))
        target = line_value(edges.line, spec, tw)
        return g[np.arange(len(g)), ax] - target

    x0 = np.r_[H0.ravel()[:8], spec.tape_width_guess] + 0.0
    xs = np.r_[np.abs(H0.ravel()[:8]) + 1e-7, 0.005]
    if fit_distortion:
        x0, xs = np.r_[x0, 0.0], np.r_[xs, 0.05]
    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=loss_scale_m, x_scale=xs)
    H, tw, k1 = unpack(sol.x)
    return GridCalibration(H, float(tw), float(k1), norm_f, c, spec, residuals(sol.x), edges)
