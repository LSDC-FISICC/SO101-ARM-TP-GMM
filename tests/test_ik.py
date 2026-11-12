"""Inverse kinematics: FK -> IK round trips, limits, wrap-around and unreachable targets."""

import numpy as np
import pytest

from so101_tpgmm.config import load_config
from so101_tpgmm.kinematics.fk import SO101Kinematics
from so101_tpgmm.kinematics.ik import (
    SO101IK,
    approach_pitch,
    joint_limits_from_config,
    tool_yaw,
    wrap_angle,
    yaw_observability,
)


@pytest.fixture(scope="module")
def kin():
    return SO101Kinematics()


@pytest.fixture(scope="module")
def ik(kin):
    lo, hi = joint_limits_from_config(kin, load_config()["kinematics"]["joint_limits"])
    return SO101IK(kin, lo, hi)


def task_relevant_samples(kin, ik, n, seed=0):
    """Random joint configurations whose TCP is in front of the robot, near the table, pointing down."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        q = rng.uniform(ik.lower, ik.upper)
        T = kin.fk(q)
        if (T[0, 3] > 0.03 and -0.01 < T[2, 3] < 0.25 and yaw_observability(T[:3, :3]) > 0.3
                and abs(approach_pitch(T, ik.pan_axis_xy)) < np.deg2rad(60)):
            out.append((q, T))
    return out


def test_wrap_angle():
    np.testing.assert_allclose(wrap_angle(np.array([0, np.pi + 0.1, -np.pi - 0.1, 4 * np.pi])),
                               [0, -np.pi + 0.1, np.pi - 0.1, 0], atol=1e-12)


def test_limits_union_includes_rest_pose(kin):
    lo, hi = joint_limits_from_config(kin, load_config()["kinematics"]["joint_limits"])
    rest = np.deg2rad([-0.8, -104.4, 96.6, 78.1, 6.0])  # episode 0, frame 0
    assert not kin.within_limits(rest)[0]  # URDF alone rejects the recorded rest pose
    assert np.all((rest >= lo) & (rest <= hi))
    lo_u, hi_u = joint_limits_from_config(kin, {"source": "urdf"})
    np.testing.assert_allclose(lo_u, kin.lower)


def test_jacobian_matches_finite_differences(ik):
    q = np.deg2rad([10, -20, 30, 60, 15])
    p_t, z_t = np.array([0.2, 0.05, 0.02]), np.array([0.2, 0.0, -0.98])
    r, J = ik._residual_and_jac(q, p_t, 0.3, q, True, z_t)
    eps = 1e-7
    Jn = np.stack([(ik._residual_and_jac(q + eps * e, p_t, 0.3, q, True, z_t)[0] - r) / eps for e in np.eye(5)], 1)
    np.testing.assert_allclose(J, Jn, atol=1e-5)


@pytest.mark.parametrize("use_true_pitch", [True, False])
def test_fk_ik_round_trip(kin, ik, use_true_pitch):
    samples = task_relevant_samples(kin, ik, 60)
    ok = 0
    for q, T in samples:
        pitch = approach_pitch(T, ik.pan_axis_xy) if use_true_pitch else None
        r = ik.solve(T[:3, 3], tool_yaw(T[:3, :3]), pitch=pitch)
        if r.success:
            ok += 1
            Ts = kin.fk(r.q)
            assert np.linalg.norm(Ts[:3, 3] - T[:3, 3]) <= ik.pos_tol
            assert abs(wrap_angle(tool_yaw(Ts[:3, :3]) - tool_yaw(T[:3, :3]))) <= ik.yaw_tol
            assert np.all(r.q >= ik.lower - 1e-9) and np.all(r.q <= ik.upper + 1e-9)
    assert ok / len(samples) >= 0.95


def test_true_pitch_recovers_joints(kin, ik):
    """Position + yaw + pitch pin down all 5 joints (up to the elbow branch): with a
    seed on the right branch the recorded joints are recovered."""
    q = np.deg2rad([20, -10, 40, 70, -30])
    T = kin.fk(q)
    r = ik.solve(T[:3, 3], tool_yaw(T[:3, :3]), q_seed=q + 0.1, pitch=approach_pitch(T, ik.pan_axis_xy))
    assert r.success
    np.testing.assert_allclose(r.q, q, atol=np.deg2rad(0.5))


def test_yaw_wraparound(kin, ik):
    q = np.deg2rad([0, 0, 30, 70, 0])
    p = kin.fk_position(q)
    for yaw in (np.pi - 0.01, -np.pi + 0.01):
        r = ik.solve(p, yaw)
        assert r.success, (yaw, r)


def test_unreachable_target_is_flagged_not_clipped(ik):
    r = ik.solve(np.array([0.9, 0.0, 0.1]), 0.0)
    assert not r.success
    assert r.pos_err > 0.3  # reports the real error
    assert r.within_limits


def test_position_only(ik):
    r = ik.solve(np.array([0.25, 0.05, 0.03]), None)
    assert r.success and r.yaw_err == 0.0


def test_trajectory_on_real_episode(real_dataset, kin, ik):
    ep = real_dataset.episode(100)
    Q = np.deg2rad(ep.state[::6, :5])
    T = kin.fk(Q)
    res = ik.solve_trajectory(T[:, :3, 3], tool_yaw(T[:, :3, :3]), Q[0], pitch=approach_pitch(T, ik.pan_axis_xy))
    assert res.feasible_fraction == 1.0
    assert np.max(np.abs(res.q - Q)) < np.deg2rad(2.0)  # recorded joints recovered
