"""Closed-form IK (corrected Rivas 2026 decoupling) against the URDF FK."""

import numpy as np
import pytest

from so101_tpgmm.kinematics.analytic import AnalyticIK, rivas_eq1_position
from so101_tpgmm.kinematics.fk import SO101Kinematics

# Table V of the paper: joints [deg] (LeRobot convention) and X, Y, Z [m] in the pan-axis frame.
PAPER_TABLE_V = {
    "cero": ([0, 0, 0, 0, 0], (0.3525, 0.0000, 0.2344)),
    "pose_A": ([20, -30, 45, -15, 0], (0.2718, -0.0989, 0.1982)),
    "pose_B": ([-35, -50, 60, 10, 30], (0.2011, 0.1408, 0.1377)),
    "pose_C": ([45, -20, 25, -35, -45], (0.2057, -0.2057, 0.3050)),
    "pose_D": ([-15, 10, 35, -25, 60], (0.3151, 0.0844, 0.0764)),
}


@pytest.fixture(scope="module")
def kin():
    return SO101Kinematics()


@pytest.fixture(scope="module")
def aik(kin):
    return AnalyticIK(kin)


def wrapdiff(a, b):
    d = np.asarray(a) - np.asarray(b)
    return np.arctan2(np.sin(d), np.cos(d))


def test_geometry_extracted_from_urdf(aik):
    g = aik.g
    assert g.L1 == pytest.approx(0.1160, abs=2e-4)
    assert np.degrees(g.phi) == pytest.approx(76.03, abs=0.05)
    assert g.l6 == pytest.approx(0.1349, abs=2e-4) and g.l6b == pytest.approx(0.0052, abs=2e-4)
    assert g.a1 == pytest.approx(0.0304, abs=2e-4) and g.d1 == pytest.approx(0.1166, abs=2e-4)
    assert g.l78 == pytest.approx(0.1592, abs=3e-4)
    assert np.linalg.norm(g.tcp_offset) == pytest.approx(0.0079, abs=3e-4)


def test_round_trip_recovers_joints(kin, aik):
    rng = np.random.default_rng(0)
    for _ in range(300):
        q = rng.uniform(kin.lower, kin.upper)
        T = kin.fk(q)
        sols = aik.solve_pose(T)
        best = min(sols, key=lambda s: np.max(np.abs(wrapdiff(s, q))))
        assert np.max(np.abs(wrapdiff(best, q))) < np.deg2rad(0.05)
        # Residual comes only from the rounded constants of the URDF (pi written as 3.14159).
        assert np.max(np.abs(kin.fk(best)[:3, 3] - T[:3, 3])) < 5e-5


def test_pan_near_90_deg_is_not_singular(kin, aik):
    """The paper's beta = atan2(-r33, r13 / c1) divides by cos(q1) = 0 here."""
    for pan in (np.pi / 2, -np.pi / 2, np.deg2rad(-98.6)):
        q = np.array([pan, -0.3, 0.6, 1.0, 0.2])
        sols = aik.solve_pose(kin.fk(q))
        assert min(np.max(np.abs(wrapdiff(s, q))) for s in sols) < 1e-3


def test_paper_table_v_is_the_roll_axis_point_in_the_pan_frame(kin, aik):
    g = aik.g
    for name, (qd, xyz) in PAPER_TABLE_V.items():
        T = kin.fk(np.deg2rad(qd))
        p_roll = T[:3, 3] - T[:3, :3] @ g.tcp_offset
        p = p_roll - np.r_[g.pan_xy, 0.0]
        np.testing.assert_allclose(p, xyz, atol=3e-4, err_msg=name)


def test_paper_eq1_needs_sign_flip_and_forearm_offset(kin, aik):
    g = aik.g
    q = np.deg2rad([20, -30, 45, -15, 0])
    T = kin.fk(q)
    target = T[:3, 3] - T[:3, :3] @ g.tcp_offset - np.r_[g.pan_xy, 0.0]
    q_paper = q.copy()
    q_paper[0] = -q[0]  # q1 = -pan
    np.testing.assert_allclose(rivas_eq1_position(q_paper, g), target, atol=3e-4)
    # As printed (no forearm offset): ~5 mm error; with LeRobot's pan sign: y mirrored.
    assert np.linalg.norm(rivas_eq1_position(q_paper, g, forearm_offset=False) - target) > 4e-3
    assert np.sign(rivas_eq1_position(q, g)[1]) != np.sign(target[1])


def test_two_elbow_branches(kin, aik):
    q = np.deg2rad([10, -20, 40, 50, 0])
    sols = aik.solve_pose(kin.fk(q))
    front = [s for s in sols if abs(wrapdiff(s[0], q[0])) < 1e-3]
    assert len(front) == 2
    assert abs(front[0][2] - front[1][2]) > 0.1  # elbow up / down differ


def test_closed_form_fk_matches_urdf_chain(kin, aik):
    from so101_tpgmm.kinematics.analytic import closed_form_fk
    from so101_tpgmm.kinematics.ik import approach_pitch

    rng = np.random.default_rng(3)
    for _ in range(300):
        q = rng.uniform(kin.lower, kin.upper)
        T_ref, T_cf = kin.fk(q), closed_form_fk(q, aik.g)
        np.testing.assert_allclose(T_cf, T_ref, atol=5e-5)
        if abs(np.sin(q[0])) < 0.99 and np.hypot(*(T_ref[:2, 3] - aik.g.pan_xy)) > 0.05:
            beta = q[1] + q[2] + q[3]
            pitch = approach_pitch(T_ref, aik.g.pan_xy)
            assert abs(np.arctan2(np.sin(pitch - (np.pi / 2 - beta)), np.cos(pitch - (np.pi / 2 - beta)))) < 0.05 \
                or abs(np.arctan2(np.sin(pitch + (np.pi / 2 - beta)), np.cos(pitch + (np.pi / 2 - beta)))) < 0.05


def test_equations_section_is_generated_and_verified():
    from so101_tpgmm.config import load_config
    from so101_tpgmm.kinematics.equations import EQ_END, EQ_START, equations_markdown, verification_errors
    from so101_tpgmm.kinematics.ik import ik_from_config

    ik = ik_from_config(load_config())
    err = verification_errors(ik, n=200)
    assert err["fk"] < 1e-4 and err["ik_deg"] < 0.5 and err["q5_deg"] < 0.05
    md = equations_markdown(ik)
    assert md.startswith(EQ_START) and md.endswith(EQ_END) and "q_5 = -\\mathrm{atan2}(M_{32}, M_{33})" in md
