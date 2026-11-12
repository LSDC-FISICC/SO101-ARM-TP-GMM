"""Silent failure: joint units misinterpreted (degrees vs normalized [-100, 100])."""

import numpy as np
import pytest

from so101_tpgmm.data.units import (
    LEROBOT_DEG_PER_TICK,
    JointUnits,
    estimate_range_span_ticks,
    infer_joint_units,
    joints_to_radians,
    range_m100_100_to_radians,
)

RNG = np.random.default_rng(0)
# Plausible calibration ranges (ticks) per joint + gripper.
MINS = np.array([800, 900, 850, 1000, 100, 2000])
MAXS = np.array([3300, 3150, 3100, 3200, 3900, 3451])  # gripper span 1451


def _raw(n: int = 3000) -> np.ndarray:
    return np.stack([RNG.integers(lo, hi + 1, n) for lo, hi in zip(MINS, MAXS)], axis=1)


def _gripper(raw: np.ndarray) -> np.ndarray:
    return (raw[:, 5] - MINS[5]) / (MAXS[5] - MINS[5]) * 100


def lerobot_degrees(raw: np.ndarray) -> np.ndarray:
    mid = (MINS[:5] + MAXS[:5]) / 2
    return np.c_[(raw[:, :5] - mid) * 360 / 4095, _gripper(raw)].astype(np.float32)


def lerobot_m100(raw: np.ndarray) -> np.ndarray:
    v = (raw[:, :5] - MINS[:5]) / (MAXS[:5] - MINS[:5]) * 200 - 100
    return np.c_[v, _gripper(raw)].astype(np.float32)


def test_detects_degrees_even_through_float32():
    rep = infer_joint_units(lerobot_degrees(_raw()))
    assert rep.joint_units is JointUnits.DEGREES
    assert rep.gripper_span_ticks == 1451
    # half-integer offset for joints whose min+max is odd
    assert [j.offset_is_half_tick for j in rep.joints] == list((MINS[:5] + MAXS[:5]) % 2 == 1)


def test_detects_range_m100_100():
    rep = infer_joint_units(lerobot_m100(_raw()))
    assert rep.joint_units is JointUnits.RANGE_M100_100
    assert not rep.any_abs_above_100


def test_contradictory_evidence_raises():
    bad = lerobot_m100(_raw()).astype(np.float64)
    bad[0, 1] = 130.0  # |v| > 100 but no degree quantization
    with pytest.raises(ValueError, match="ambiguous"):
        infer_joint_units(bad)


def test_degrees_to_radians_is_explicit():
    np.testing.assert_allclose(joints_to_radians(np.array([90.0, -45.0, 0, 180, 30]), "degrees"),
                               [np.pi / 2, -np.pi / 4, 0, np.pi, np.pi / 6])
    with pytest.raises(NotImplementedError):
        joints_to_radians(np.zeros(5), JointUnits.RANGE_M100_100)


def test_m100_to_radians_matches_degrees_mode():
    raw = _raw(50)
    rad_from_m100 = range_m100_100_to_radians(lerobot_m100(raw)[:, :5].astype(np.float64), MINS[:5], MAXS[:5])
    rad_from_deg = np.deg2rad(lerobot_degrees(raw)[:, :5].astype(np.float64))
    np.testing.assert_allclose(rad_from_m100, rad_from_deg, atol=1e-5)


def test_gripper_span_estimate():
    span, resid = estimate_range_span_ticks(np.arange(0, 1452) / 1451 * 100)
    assert span == 1451 and resid < 1e-6


def test_deg_per_tick_constant():
    assert LEROBOT_DEG_PER_TICK == pytest.approx(8 / 91)


def test_real_dataset_is_degrees(real_dataset):
    rep = infer_joint_units(real_dataset.all_states())
    assert rep.joint_units is JointUnits.DEGREES
    assert rep.any_abs_above_100  # shoulder_lift rest pose at ~-104 deg
