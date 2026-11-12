"""Joint-unit inference and conversion for LeRobot SO-101 recordings.

LeRobot stores Feetech STS3215 positions after a per-motor normalization
(``lerobot/motors/motors_bus.py::_normalize``)::

    RANGE_M100_100: v = clamp(raw, min, max) -> (raw - min) / (max - min) * 200 - 100
    RANGE_0_100:    v = clamp(raw, min, max) -> (raw - min) / (max - min) * 100
    DEGREES:        v = (raw - mid) * 360 / (4096 - 1),     mid = (min + max) / 2   (NOT clamped)

``raw`` is an integer tick, and ``min``/``max`` are integer calibration bounds.
That leaves two fingerprints in the stored floats that tell us the mode
without guessing:

1. **Quantization.** In DEGREES mode ``v * 4095 / 360 = raw - mid`` is an integer
   or half-integer (constant per joint, depending on the parity of min+max),
   with the *same* step 360/4095 = 8/91 deg for every joint. In RANGE_M100_100
   the step is 200/(max-min), which differs per joint.
2. **Range.** RANGE_* modes clamp, so ``|v| > 100`` is impossible in
   RANGE_M100_100. DEGREES is not clamped.

The gripper always uses RANGE_0_100 in ``SOFollower``. We recover its tick span
(max - min) from the smallest gap between distinct values and verify it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

#: STS3215 encoder resolution (ticks per revolution).
STS3215_RESOLUTION: int = 4096
#: Degrees per tick as used by LeRobot's DEGREES mode (it divides by resolution - 1).
LEROBOT_DEG_PER_TICK: float = 360.0 / (STS3215_RESOLUTION - 1)


class JointUnits(str, Enum):
    """Normalization mode of the five arm joints."""

    DEGREES = "degrees"
    RANGE_M100_100 = "range_m100_100"


@dataclass
class JointQuantization:
    """Quantization diagnostics for a single joint under the DEGREES hypothesis."""

    name: str
    max_residual_ticks: float
    offset_is_half_tick: bool
    offset_consistent: bool
    min_value: float
    max_value: float


@dataclass
class UnitsReport:
    """Result of :func:`infer_joint_units`."""

    joint_units: JointUnits
    joints: list[JointQuantization]
    any_abs_above_100: bool
    gripper_span_ticks: int | None
    gripper_max_residual: float | None
    evidence: list[str] = field(default_factory=list)


def _degrees_quantization(values: np.ndarray, name: str) -> JointQuantization:
    q = np.asarray(values, dtype=np.float64) / LEROBOT_DEG_PER_TICK
    # raw - mid is an integer or a half integer -> 2q must be an integer.
    resid = np.abs(2.0 * q - np.round(2.0 * q)) / 2.0
    half = np.abs(q - np.round(q)) > 0.25
    return JointQuantization(
        name=name,
        max_residual_ticks=float(resid.max()),
        offset_is_half_tick=bool(half.mean() > 0.5),
        offset_consistent=bool(half.all() or (~half).all()),
        min_value=float(np.min(values)),
        max_value=float(np.max(values)),
    )


def estimate_range_span_ticks(values: np.ndarray, full_scale: float = 100.0) -> tuple[int, float]:
    """Estimate ``max - min`` (ticks) of a RANGE_* normalized signal.

    The smallest positive gap between distinct values is one tick, i.e.
    ``full_scale / span``. Returns the span and the max residual (in ticks)
    of ``v * span / full_scale`` from the nearest integer.
    """
    v = np.asarray(values, dtype=np.float64)
    u = np.unique(np.round(v, 6))
    gaps = np.diff(u)
    gaps = gaps[gaps > 1e-6]
    if gaps.size == 0:
        raise ValueError("Signal is constant; cannot estimate its quantization step.")
    step = float(np.min(gaps))
    span = int(round(full_scale / step))
    ticks = (v - v.min()) * span / full_scale
    resid = float(np.max(np.abs(ticks - np.round(ticks))))
    return span, resid


def infer_joint_units(
    states: np.ndarray,
    joint_names: list[str] | None = None,
    tol_ticks: float = 0.01,
) -> UnitsReport:
    """Infer the normalization mode of an ``(N, 6)`` SO-101 state array.

    Columns 0..4 are the arm joints and column 5 the gripper.

    Raises:
        ValueError: if the evidence is contradictory (neither hypothesis fits).
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError(f"Expected (N, 6) states, got {states.shape}")
    names = joint_names or [f"joint_{i}" for i in range(5)]
    joints = [_degrees_quantization(states[:, i], names[i]) for i in range(5)]
    deg_fits = all(j.max_residual_ticks <= tol_ticks and j.offset_consistent for j in joints)
    above_100 = bool(np.any(np.abs(states[:, :5]) > 100.0 + 1e-4))

    evidence: list[str] = []
    if deg_fits:
        evidence.append(
            "All 5 arm joints are exact multiples of 360/4095 deg (max residual "
            f"{max(j.max_residual_ticks for j in joints):.2e} ticks), with a constant "
            "integer/half-integer offset per joint: the LeRobot DEGREES fingerprint."
        )
    if above_100:
        evidence.append(
            "Some arm values have |v| > 100, impossible in RANGE_M100_100 (LeRobot clamps)."
        )

    try:
        g_span, g_resid = estimate_range_span_ticks(states[:, 5], 100.0)
        if g_resid <= 0.05:
            evidence.append(
                f"Gripper is quantized in steps of 100/{g_span}: RANGE_0_100 with a "
                f"calibrated span of {g_span} ticks ({g_span * LEROBOT_DEG_PER_TICK:.1f} deg)."
            )
    except ValueError:
        g_span, g_resid = None, None

    if deg_fits:
        units = JointUnits.DEGREES
    elif not above_100:
        units = JointUnits.RANGE_M100_100
        evidence.append("Quantization does not match DEGREES and all |v| <= 100.")
    else:
        raise ValueError(
            "Joint units are ambiguous: values exceed +-100 (so not RANGE_M100_100) but "
            "they do not follow the 360/4095 deg quantization of DEGREES mode."
        )
    return UnitsReport(units, joints, above_100, g_span, g_resid, evidence)


def joints_to_radians(values: np.ndarray, units: JointUnits | str) -> np.ndarray:
    """Convert stored arm-joint values (..., 5) to radians in the URDF convention.

    For DEGREES mode the conversion is a pure ``deg2rad``. LeRobot's DEGREES mode
    measures from the middle of the calibrated range, which is the zero of
    ``so101_new_calib.urdf`` (LeRobot's own ``RobotKinematics`` uses the same
    direct mapping).

    RANGE_M100_100 cannot be converted without the per-motor calibration file
    (range_min/range_max ticks), so it raises instead of silently guessing.
    """
    units = JointUnits(units)
    if units is JointUnits.DEGREES:
        return np.deg2rad(np.asarray(values, dtype=np.float64))
    raise NotImplementedError(
        "RANGE_M100_100 -> radians needs the motor calibration (range_min/max); "
        "use range_m100_100_to_radians() with the calibration values."
    )


def range_m100_100_to_radians(values: np.ndarray, range_min: np.ndarray, range_max: np.ndarray) -> np.ndarray:
    """Convert RANGE_M100_100 values to radians given calibration ticks per joint."""
    v = np.asarray(values, dtype=np.float64)
    lo = np.asarray(range_min, dtype=np.float64)
    hi = np.asarray(range_max, dtype=np.float64)
    raw = (v + 100.0) / 200.0 * (hi - lo) + lo
    mid = (lo + hi) / 2.0
    return np.deg2rad((raw - mid) * LEROBOT_DEG_PER_TICK)


def radians_to_joints(q_rad: np.ndarray, units: JointUnits | str = JointUnits.DEGREES) -> np.ndarray:
    """Inverse of :func:`joints_to_radians` (DEGREES only)."""
    if JointUnits(units) is not JointUnits.DEGREES:
        raise NotImplementedError("Only DEGREES is supported for commands.")
    return np.rad2deg(np.asarray(q_rad, dtype=np.float64))
