"""Stable LPV-DS: stability by construction, attractor convergence, policy rollout."""

import numpy as np
import pytest

from so101_tpgmm.lpvds.lpvds import ScalarDS, _unpack, fit_lpvds


def test_any_parameters_are_lyapunov_stable():
    rng = np.random.default_rng(0)
    for _ in range(20):
        K, d = 4, 3
        theta = rng.normal(size=6 + K * 9) * 2
        A, P = _unpack(theta, K, d)
        assert np.all(np.linalg.eigvalsh(P) > 0)
        for k in range(K):
            assert np.linalg.eigvalsh(A[k].T @ P + P @ A[k]).max() < 0


def test_fit_converges_to_target_from_curved_demos():
    rng = np.random.default_rng(1)
    target = np.array([0.0, 0.0, 0.01])
    P, V = [], []
    for _ in range(30):  # demos: an arc rising then descending onto the target
        x0 = np.array([-0.2, rng.uniform(-0.05, 0.05), 0.0])
        t = np.linspace(0, 1, 60)
        s = 10 * t**3 - 15 * t**4 + 6 * t**5
        path = x0 + (target - x0) * s[:, None]
        path[:, 2] += 0.08 * np.sin(np.pi * s)
        P.append(path), V.append(np.gradient(path, 3.0 / 59, axis=0))
    m, info = fit_lpvds(np.vstack(P), np.vstack(V), target, k_range=(3,))
    x = np.array([-0.25, 0.08, 0.05])  # outside the demonstrations
    for _ in range(3000):
        x = x + 0.01 * m.velocity(x)[0]
    assert np.linalg.norm(x - target) < 1e-3
    assert info["velocity_rmse"] < np.sqrt(np.mean(np.sum(np.vstack(V) ** 2, axis=1)))


def test_scalar_ds_is_stable_and_fits_gain():
    g = np.ones((200, 1))
    z = np.linspace(1.0, 0.0, 200)
    sds = ScalarDS.fit(g, z, 2.0 * (0.0 - z), 0.0)
    assert np.exp(sds.log_gain[0]) == pytest.approx(2.0, rel=0.05)
    assert sds.rate(g[:1], 0.5)[0] < 0


def test_lpvds_policy_rollout_reaches_cube_and_box():
    import sys
    sys.path.insert(0, "tests")
    from test_tpgmm import _fit_policies  # synthetic demos shared with the TP-GMM tests

    from so101_tpgmm.lpvds.policy import LPVDSPolicy

    _, _, approach = _fit_policies()
    # rebuild the transport demos exactly as the TP-GMM test does
    from test_tpgmm import Demo, min_jerk

    tr = []
    for d in approach:
        t = d.X[:, 0]
        pos = d.X[-1, 1:4] + (d.box[:3] - d.X[-1, 1:4]) * min_jerk(t)[:, None]
        pos[:, 2] += 0.08 * np.sin(np.pi * t)
        X = np.c_[t, pos, d.X[-1, 4] * (1 - t), 20 + 25 * (t > 0.8), 0.1 * np.ones_like(t)]
        tr.append(Demo(d.episode, "transport_release", X, d.cube, 0, d.phi_raw, d.box, 2.0))
    ctx = {"box": [0.06, 0.16, 0.08, 0.0], "heading_offset": 0.0, "pan_xy": [0.0, 0.0],
           "var_names": ["t", "x", "y", "z", "h", "g", "pitch"]}
    pol = LPVDSPolicy.fit({"approach_grasp": approach, "transport_release": tr}, ctx,
                          k_fixed={"approach_grasp": 3, "transport_release": 3})
    d = approach[0]
    ro = pol.rollout(d.cube[:3], d.cube[3], start=d.X[0, 1:], heading=d.cube[3])
    assert np.linalg.norm(ro.Y[ro.boundaries[0], :3] - d.X[-1, 1:4]) < 0.01  # this demo's cube (goal frame)
    assert np.linalg.norm(ro.Y[-1, :3] - np.array([0.06, 0.16, 0.08])) < 0.01
    assert len(ro.boundaries) == 2 and np.all(np.diff(ro.t) > 0)
