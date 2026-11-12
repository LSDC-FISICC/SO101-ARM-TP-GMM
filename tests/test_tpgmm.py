"""TP-GMM core (EM, product, GMR), frames, policies and the IK planner."""

import numpy as np
import pytest

from so101_tpgmm.planner.ik_planner import IKPlanner, min_jerk
from so101_tpgmm.tpgmm.demos import Demo, canonical_cube_yaw, choose_cube_heading, heading_from_tool_yaw, make_frame, tool_yaw_from_heading
from so101_tpgmm.tpgmm.gmm import GMM, gaussian_logpdf, gaussian_product, gmr
from so101_tpgmm.tpgmm.policy import TPGMMPolicy, blend_start
from so101_tpgmm.tpgmm.tpgmm import TPGMM, TaskFrame, load_models, save_models, select_by_bic

D = 7  # t, x, y, z, h, g, pitch
REG = np.array([1e-3, 4e-6, 4e-6, 4e-6, 1e-3, 1.0, 1e-3])


def test_gaussian_logpdf_matches_scipy():
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    S = A @ A.T + np.eye(3)
    X = rng.normal(size=(5, 3))
    np.testing.assert_allclose(gaussian_logpdf(X, np.ones(3), S), multivariate_normal(np.ones(3), S).logpdf(X))


def test_gaussian_product_closed_form_1d():
    mu, S = gaussian_product([np.array([0.0]), np.array([2.0])], [np.array([[1.0]]), np.array([[1.0]])])
    assert mu[0] == pytest.approx(1.0) and S[0, 0] == pytest.approx(0.5)


def test_gmr_single_gaussian_is_linear_regression():
    S = np.array([[1.0, 0.8], [0.8, 1.0]])
    g = GMM(np.array([1.0]), np.array([[0.0, 1.0]]), S[None])
    mean, cov = gmr(g, np.array([[0.0], [1.0]]), [0], [1])
    np.testing.assert_allclose(mean[:, 0], [1.0, 1.8])
    np.testing.assert_allclose(cov[:, 0, 0], 1 - 0.64)


def synthetic_demos(n=40, seed=0):
    """Reach a target point given in a rotated, translated frame; frame 2 is fixed and irrelevant."""
    rng = np.random.default_rng(seed)
    demos = []
    t = np.linspace(0, 1, 30)
    for _ in range(n):
        target = np.r_[rng.uniform(0.1, 0.3), rng.uniform(-0.2, 0.2), 0.02]
        yaw = rng.uniform(-0.7, 0.7)
        approach_local = np.array([-0.05, 0.0, 0.05])  # come from "behind" the target, from above
        start = np.array([0.05, 0.0, 0.1])
        via = target + make_frame(np.zeros(3), yaw, D).A[1:4, 1:4] @ approach_local
        s = min_jerk(t)[:, None]
        pos = np.where(t[:, None] < 0.5, start + (via - start) * min_jerk(2 * t)[:, None],
                       via + (target - via) * min_jerk(2 * t - 1)[:, None])
        X = np.c_[t, pos + rng.normal(0, 1e-3, pos.shape), yaw * s, 50 * s, 0.1 * np.ones_like(t)]
        demos.append(Demo(0, "p", X, np.r_[target, yaw], 0, yaw, np.array([0.06, 0.16, 0.08, 0.0]), 3.0))
    return demos


def test_tpgmm_generalizes_to_new_frames():
    demos = synthetic_demos()
    train, test = demos[:35], demos[35:]
    X = np.concatenate([d.local() for d in train], axis=1)
    m = TPGMM(6, 2, D, REG, (4,)).fit(X)
    assert np.isfinite(m.log_likelihood) and m.n_iter >= 2
    for d in test:
        mu, cov = m.reproduce(d.frames(), d.X[:, 0])
        assert np.linalg.norm(mu[-1, :3] - d.X[-1, 1:4]) < 0.006  # reaches the new target
        # heading (output 3 = variable 4) ends at the new frame's yaw; mid-course error is GMR approximation
        assert abs(mu[-1, 3] - d.X[-1, 4]) < 0.02
        assert np.max(np.abs(mu[:, 3] - d.X[:, 4])) < 0.15
        assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_em_log_likelihood_does_not_decrease():
    X = np.concatenate([d.local() for d in synthetic_demos(20)], axis=1)
    lls = [TPGMM(4, 2, D, REG, (4,)).fit(X, max_iter=it).log_likelihood for it in (1, 3, 10)]
    assert lls[0] <= lls[1] + 1e-6 <= lls[2] + 2e-6


def test_bic_selection_and_io(tmp_path):
    X = np.concatenate([d.local() for d in synthetic_demos(20)], axis=1)
    m, scores = select_by_bic(X, [2, 4], REG, (4,))
    assert set(scores) == {2, 4} and m.bic() == min(scores.values())
    save_models(tmp_path / "m.json", {"p": m}, {"x": 1})
    models, meta = load_models(tmp_path / "m.json")
    np.testing.assert_allclose(models["p"].means, m.means)
    assert meta == {"x": 1}


def test_angles_aligned_modulo_2pi_in_product():
    m = TPGMM(1, 2, D, REG, (4,))
    m.priors = np.array([1.0])
    m.means = np.zeros((1, 2, D))
    m.covs = np.tile(np.eye(D) * 0.01, (1, 2, 1, 1))
    f1, f2 = make_frame(np.zeros(3), 3.0, D), make_frame(np.zeros(3), 3.0 - 2 * np.pi, D)
    g = m.product([f1, f2])
    assert g.means[0, 4] == pytest.approx(3.0, abs=1e-9)  # not the average 3 - pi


def test_task_frame_round_trip():
    F = make_frame(np.array([0.2, -0.1, 0.03]), 0.7, D)
    x = np.random.default_rng(0).normal(size=(4, D))
    np.testing.assert_allclose(F.A @ F.to_local(x).T + F.b[:, None], x.T, atol=1e-12)
    assert isinstance(F, TaskFrame)


def test_heading_and_cube_symmetry():
    np.testing.assert_allclose(tool_yaw_from_heading(heading_from_tool_yaw(np.array([3.0, -3.0, 0.2]))), [3.0, -3.0, 0.2])
    phi_c, k = canonical_cube_yaw(np.radians(30), np.radians(125))
    assert np.degrees(phi_c) == pytest.approx(120) and k == 1
    # execution choice: azimuth of the cube + learned offset
    phi_c, _ = choose_cube_heading(np.radians(10), np.array([0.0, 0.3]), np.zeros(2), 0.0)
    assert np.degrees(phi_c) == pytest.approx(100)


def test_blend_start_is_continuous_and_fades():
    Y = np.zeros((50, 3))
    out = blend_start(Y, np.array([1.0, 2.0, 0.0]), 0.3)
    np.testing.assert_allclose(out[0], [1.0, 2.0, 0.0])
    np.testing.assert_allclose(out[20:], 0.0)
    assert np.all(np.diff(out[:16, 0]) <= 1e-12)


def _fit_policies():
    demos = synthetic_demos()
    a = [Demo(d.episode, "approach_grasp", d.X, d.cube, 0, d.phi_raw, d.box, 3.0) for d in demos]
    tr = []
    for d in demos:  # transport: from the target to the box, gripper closed then opening
        t = d.X[:, 0]
        pos = d.X[-1, 1:4] + (d.box[:3] - d.X[-1, 1:4]) * min_jerk(t)[:, None]
        pos[:, 2] += 0.08 * np.sin(np.pi * t)
        X = np.c_[t, pos, d.X[-1, 4] * (1 - t), 20 + 25 * (t > 0.8), 0.1 * np.ones_like(t)]
        tr.append(Demo(d.episode, "transport_release", X, d.cube, 0, d.phi_raw, d.box, 2.0))
    ctx = {"box": [0.06, 0.16, 0.08, 0.0], "heading_offset": 0.0, "pan_xy": [0.0, 0.0],
           "var_names": ["t", "x", "y", "z", "h", "g", "pitch"]}
    models = {p: TPGMM(6, 2, D, REG, (4,)).fit(np.concatenate([d.local() for d in ds_], axis=1))
              for p, ds_ in (("approach_grasp", a), ("transport_release", tr))}
    durations = {"approach_grasp": 3.0, "transport_release": 2.0, "cycle": 5.0}
    pol = TPGMMPolicy(models, {**ctx, "durations": durations, "boundary_blend": 0.3,
                                "primitives": ["approach_grasp", "transport_release"]})
    plan = IKPlanner().fit({"approach_grasp": a, "transport_release": tr}, ctx, durations)
    return pol, plan, a


def test_segmented_policy_and_planner_rollouts():
    pol, plan, demos = _fit_policies()
    d = demos[0]
    start = d.X[0, 1:]
    for p in (pol, plan):
        ro = p.rollout(d.cube[:3], d.cube[3], start=start, heading=d.cube[3])
        np.testing.assert_allclose(ro.Y[0], start, atol=1e-9)  # starts at the robot's pose
        assert ro.t[-1] == pytest.approx(5.0, abs=0.05) and len(ro.boundaries) == 2
        grasp = ro.Y[ro.boundaries[0], :3]
        assert np.linalg.norm(grasp - d.cube[:3]) < 0.01
        assert np.linalg.norm(ro.Y[-1, :3] - np.array([0.06, 0.16, 0.08])) < 0.01
        steps = np.linalg.norm(np.diff(ro.Y[:, :3], axis=0), axis=1)
        b = ro.boundaries[0]
        assert np.max(steps[b - 2:b + 2]) < 0.01  # no jump where the primitives meet
        assert np.max(steps) < 0.04


def test_planner_save_load(tmp_path):
    _, plan, demos = _fit_policies()
    plan.save(tmp_path / "p.json")
    p2 = IKPlanner.load(tmp_path / "p.json")
    d = demos[1]
    a = plan.rollout(d.cube[:3], d.cube[3], start=d.X[0, 1:])
    b = p2.rollout(d.cube[:3], d.cube[3], start=d.X[0, 1:])
    np.testing.assert_allclose(a.Y, b.Y)


def test_cube_symmetry_margin_and_ambiguity():
    from so101_tpgmm.tpgmm.cube_symmetry import ambiguous_candidates, candidates, decision_margin, heading_target

    assert np.allclose(np.sort(np.degrees(candidates(np.radians(10)))), [-170, -80, 10, 100])
    assert np.degrees(decision_margin(np.radians(10), np.radians(10))) == pytest.approx(45)  # unambiguous
    assert np.degrees(decision_margin(np.radians(10), np.radians(55))) == pytest.approx(0, abs=1e-9)  # boundary
    assert len(ambiguous_candidates(np.radians(10), np.radians(52), np.radians(10))) == 2
    assert len(ambiguous_candidates(np.radians(10), np.radians(20), np.radians(10))) == 1
    assert heading_target(np.array([0.0, 0.2]), np.zeros(2), 0.1) == pytest.approx(np.pi / 2 + 0.1)


def test_gmr_ds_recovers_a_linear_attractor():
    from so101_tpgmm.tpgmm.ds import divergence, field, fit_ds, integrate

    rng = np.random.default_rng(0)
    A = np.array([[-1.0, 0.3], [-0.3, -1.0]])  # stable spiral to the origin, divergence -2
    P = rng.uniform(-0.2, 0.2, (3000, 2))
    V = P @ A.T + rng.normal(0, 1e-3, P.shape)
    g, _ = fit_ds(P, V, k_range=(1, 2))
    x = integrate(g, np.array([0.15, -0.1]), dt=0.02, T=8.0)
    assert np.linalg.norm(x[-1]) < 0.01
    xs = ys = np.linspace(-0.1, 0.1, 21)
    XX, YY = np.meshgrid(xs, ys)
    F = field(g, np.c_[XX.ravel(), YY.ravel()])
    div = divergence(F[:, 0].reshape(XX.shape), F[:, 1].reshape(XX.shape), xs, ys)
    assert np.median(div) == pytest.approx(-2.0, abs=0.1)
