"""Gripper event detection with hysteresis (grasp cycles).

The SO-101 gripper signal (LeRobot RANGE_0_100, 0 = closed) shows three regimes
in this dataset (Phase 0, ``docs/dataset_report.md``):

* closed-empty  ~0-12  (rest pose, or a grasp that missed the cube)
* holding       ~17-23 (jaws stopped by the cube: the plateau level encodes the cube width)
* open          >=25   (approach / release)

A naive single threshold either counts the holding plateau as "open" or
creates phantom cycles when the signal hovers near the threshold. We therefore:

1. median-filter the signal,
2. binarize with **hysteresis** (OPEN above ``open_threshold``, CLOSED below
   ``close_threshold``, otherwise keep the previous state),
3. drop OPEN runs shorter than ``min_open_s`` (debounce),
4. classify each CLOSED run between two OPEN runs by its **plateau level**:
   ``[hold_min, close_threshold]`` = holding a cube; below ``hold_min`` = empty
   closure (missed grasp); plus a minimum duration ``min_hold_s``.

A *grasp cycle* is OPEN -> HOLD -> OPEN. The contact instant (``close_idx``)
is the first frame within ``plateau_tol`` of the hold plateau, and the release
instant (``open_idx``) the last such frame before the gripper reopens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter


@dataclass
class GripperParams:
    """Thresholds of the detector (gripper units, RANGE_0_100)."""

    open_threshold: float = 25.5
    close_threshold: float = 23.5
    hold_min: float = 13.0
    min_hold_s: float = 0.4
    min_open_s: float = 0.15
    smooth_window: int = 5
    plateau_tol: float = 1.5

    @classmethod
    def from_config(cls, cfg: dict) -> "GripperParams":
        """Build from the ``gripper`` section of the YAML config."""
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in cfg.items() if k in keys})

    def validate(self) -> None:
        """Reject parameter sets without a real hysteresis band."""
        if not self.open_threshold > self.close_threshold:
            raise ValueError("open_threshold must be > close_threshold (hysteresis band).")
        if not self.hold_min < self.close_threshold:
            raise ValueError("hold_min must be < close_threshold.")
        if self.smooth_window < 1 or self.smooth_window % 2 == 0:
            raise ValueError("smooth_window must be a positive odd integer.")


@dataclass
class GraspCycle:
    """One OPEN -> HOLD -> OPEN cycle. Indices are episode frame indices."""

    open_start: int  # gripper starts the OPEN run preceding the grasp
    close_idx: int  # jaws reach the cube (start of the hold plateau)
    open_idx: int  # last frame of the hold plateau (release starts after it)
    release_end: int  # end of the OPEN run after the hold (gripper closes again)
    hold_level: float


@dataclass
class GripperEvents:
    """Detector output for one episode."""

    cycles: list[GraspCycle] = field(default_factory=list)
    empty_closures: list[tuple[int, int, float]] = field(default_factory=list)  # (start, end, level)
    unreleased_holds: list[tuple[int, int, float]] = field(default_factory=list)
    open_runs: list[tuple[int, int]] = field(default_factory=list)


def hysteresis_binarize(g: np.ndarray, open_thr: float, close_thr: float) -> np.ndarray:
    """Boolean OPEN mask with hysteresis. Values in the band keep the previous state."""
    is_open = np.zeros(g.shape[0], dtype=bool)
    state = bool(g[0] >= 0.5 * (open_thr + close_thr)) if g.size else False
    for i, v in enumerate(g):
        if state and v < close_thr:
            state = False
        elif not state and v > open_thr:
            state = True
        is_open[i] = state
    return is_open


def runs(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    """Run-length encoding: list of (value, start, end_exclusive)."""
    if mask.size == 0:
        return []
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, mask.size]
    return [(bool(mask[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def detect_gripper_events(g: np.ndarray, fps: float, params: GripperParams | None = None) -> GripperEvents:
    """Detect grasp cycles in a gripper trajectory.

    Args:
        g: ``(T,)`` gripper values (RANGE_0_100).
        fps: Sampling rate.
        params: Detector thresholds.
    """
    p = params or GripperParams()
    p.validate()
    g = np.asarray(g, dtype=np.float64)
    gs = median_filter(g, size=p.smooth_window, mode="nearest") if p.smooth_window > 1 else g
    is_open = hysteresis_binarize(gs, p.open_threshold, p.close_threshold)

    # Debounce: short OPEN runs are glitches -> CLOSED.
    min_open = int(round(p.min_open_s * fps))
    for val, s, e in runs(is_open):
        if val and (e - s) < min_open:
            is_open[s:e] = False
    rl = runs(is_open)

    ev = GripperEvents(open_runs=[(s, e) for v, s, e in rl if v])
    min_hold = int(round(p.min_hold_s * fps))
    for k, (val, s, e) in enumerate(rl):
        if val:
            continue
        prev_open = k > 0 and rl[k - 1][0]
        next_open = k + 1 < len(rl) and rl[k + 1][0]
        if not prev_open:
            continue  # rest period before the first opening
        # Plateau level: median of the first second after the closing transient.
        # (Not the whole run: a held cube can later slip and the jaws close further.)
        seg = gs[s:e]
        settle = min(len(seg) // 3, int(0.3 * fps))
        level = float(np.median(seg[settle : settle + int(fps)]))
        holding = p.hold_min <= level <= p.close_threshold and (e - s) >= min_hold
        if not holding:
            if next_open or level < p.hold_min:
                ev.empty_closures.append((s, e, level))
            continue
        if not next_open:
            ev.unreleased_holds.append((s, e, level))
            continue
        near = np.flatnonzero(np.abs(gs[s:e] - level) <= p.plateau_tol)
        close_idx = s + int(near[0]) if near.size else s
        open_idx = s + int(near[-1]) if near.size else e - 1
        ev.cycles.append(
            GraspCycle(
                open_start=rl[k - 1][1],
                close_idx=close_idx,
                open_idx=open_idx,
                release_end=rl[k + 1][2],
                hold_level=level,
            )
        )
    return ev
