"""Forward kinematics from the vendored URDF."""

import numpy as np
import pytest

from so101_tpgmm.kinematics.fk import ARM_JOINTS, SO101Kinematics
from so101_tpgmm.kinematics.urdf import rpy_to_matrix


@pytest.fixture(scope="module")
def kin():
    return SO101Kinematics()


def test_chain_matches_dataset_joint_order(kin):
    assert kin.joint_names == ARM_JOINTS
    assert kin.chain[-1].child == "gripper_frame_link"


def test_zero_pose_equals_product_of_origins(kin):
    T = np.eye(4)
    for j in kin.chain:
        T = T @ j.origin
    np.testing.assert_allclose(kin.fk(np.zeros(5)), T, atol=1e-12)


def test_batch_equals_single_and_rotations_are_proper(kin):
    q = np.random.default_rng(1).uniform(kin.lower, kin.upper, size=(20, 5))
    Tb = kin.fk(q)
    for i in range(20):
        np.testing.assert_allclose(Tb[i], kin.fk(q[i]), atol=1e-12)
    R = Tb[:, :3, :3]
    np.testing.assert_allclose(R @ R.transpose(0, 2, 1), np.broadcast_to(np.eye(3), R.shape), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_pan_rotates_about_vertical(kin):
    """shoulder_pan must move the TCP on a horizontal circle (constant z and radius)."""
    q = np.zeros((3, 5))
    q[:, 0] = [-0.5, 0.0, 0.5]
    p = kin.fk_position(q)
    # The URDF writes pi as 3.14159 in several rpy fields, tilting the axis by ~3e-6 rad (< 1 um here).
    np.testing.assert_allclose(p[:, 2], p[0, 2], atol=1e-5)
    pan_axis = kin.chain[0].origin[:2, 3]
    r = np.linalg.norm(p[:, :2] - pan_axis, axis=1)
    np.testing.assert_allclose(r, r[0], atol=1e-5)


def test_rpy_convention():
    # URDF: R = Rz(yaw) Ry(pitch) Rx(roll)
    R = rpy_to_matrix(0, 0, np.pi / 2)
    np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_fk_from_dataset_converts_units(kin):
    deg = np.array([10.0, -20.0, 30.0, 40.0, -50.0, 99.0])  # gripper column ignored
    np.testing.assert_allclose(kin.fk_from_dataset(deg, "degrees"), kin.fk(np.deg2rad(deg[:5])))


def test_real_home_pose_is_low_and_in_front(real_dataset, kin):
    """Sanity of units/signs on real data: at rest the gripper lies near the table, in front of the base."""
    ep = real_dataset.episode(0)
    p = kin.fk_from_dataset(ep.state[0], "degrees")[:3, 3]
    assert 0.05 < p[0] < 0.25 and abs(p[1]) < 0.05 and -0.05 < p[2] < 0.05
