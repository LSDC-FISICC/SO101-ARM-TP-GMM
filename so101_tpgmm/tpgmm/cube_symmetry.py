"""Choosing among the 4 equivalent yaws of a cube (90° symmetry).

A cube detected with yaw phi can be grasped with the jaws closing along any of
phi + k·90°. The demonstrations show which one the operator chose: the grasp heading is
close to the cube's azimuth from the robot plus a learned offset (``heading_offset``),
with a spread of ~21–27° (Phase 5 analysis). The execution rule picks the candidate
closest to that target; its **margin** is how far the target is from the 45° decision
boundary.

Leave-one-out on the 366 training cycles: ~92 % agreement with the operator. Position-based
alternatives (optimized offset, offset linear in the azimuth, k-NN on the cube position)
did not do better; the disagreements concentrate where the margin is small (< 10°: ~30 %
disagreement vs ~3 % otherwise), i.e. where two grasps are equally plausible and the
operator's choice is not predictable from the geometry. There, :func:`ambiguous_candidates`
returns both options so the caller can decide with a robot criterion (IK feasibility,
joint motion).
"""

from __future__ import annotations

import numpy as np

from so101_tpgmm.tpgmm.demos import wrap


def candidates(phi: float) -> np.ndarray:
    """The 4 equivalent cube yaws, wrapped to (-pi, pi]."""
    return wrap(phi + np.arange(4) * np.pi / 2)


def heading_target(cube_xy: np.ndarray, pan_xy: np.ndarray, offset: float) -> float:
    """Expected grasp heading: cube azimuth from the pan axis + learned offset."""
    return float(np.arctan2(cube_xy[1] - pan_xy[1], cube_xy[0] - pan_xy[0]) + offset)


def rank_candidates(phi: float, target: float) -> tuple[np.ndarray, np.ndarray]:
    """Candidates sorted by angular distance to ``target``, and those distances [rad]."""
    c = candidates(phi)
    d = np.abs(wrap(target - c))
    order = np.argsort(d)
    return c[order], d[order]


def decision_margin(phi: float, target: float) -> float:
    """Distance [rad] from the target to the boundary between the two best candidates (0 = ambiguous)."""
    _, d = rank_candidates(phi, target)
    return float(np.pi / 4 - d[0])


def ambiguous_candidates(phi: float, target: float, margin_threshold: float) -> list[float]:
    """The best candidate, plus the runner-up when the decision margin is below the threshold."""
    c, _ = rank_candidates(phi, target)
    if decision_margin(phi, target) < margin_threshold:
        return [float(c[0]), float(c[1])]
    return [float(c[0])]
