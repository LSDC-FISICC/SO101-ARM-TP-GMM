"""Evaluation metrics, inference pipeline and the robot runner stub."""

import numpy as np
import pytest

from so101_tpgmm.config import load_config, resolve_path
from so101_tpgmm.evaluation.metrics import closing_index, cycle_metrics
from so101_tpgmm.inference import JOINT_KEYS, joint_move, order_cubes
from so101_tpgmm.robot.so101_runner import SimulatedFollower, SO101Runner
from so101_tpgmm.tpgmm.demos import Demo
from so101_tpgmm.tpgmm.policy import Rollout
from so101_tpgmm.vision.detection import CubeDetection

BOX = np.array([0.06, 0.16, 0.08, 0.0])


def make_demo_pair():
    t = np.linspace(0, 1, 50)
    A = np.c_[t, 0.1 + 0.1 * t, 0.0 * t, 0.05 * (1 - t), 0.1 * t, 50 - 30 * (t > 0.9), 0.1 + 0 * t]
    T = np.c_[t, 0.2 - 0.14 * t, 0.16 * t, 0.08 * t, 0.1 + t, 20 + 25 * (t > 0.85), 0.1 + 0 * t]
    dA = Demo(0, "approach_grasp", A, np.r_[0.2, 0.0, 0.0, 0.0], 0, 0.0, BOX, 3.0)
    dT = Demo(0, "transport_release", T, np.r_[0.2, 0.0, 0.0, 0.0], 0, 0.0, BOX, 2.0)
    return dA, dT


def test_perfect_rollout_has_zero_error():
    dA, dT = make_demo_pair()
    Y = np.vstack([dA.X[:, 1:], dT.X[1:, 1:]])
    t = np.r_[dA.X[:, 0] * 3.0, 3.0 + dT.X[1:, 0] * 2.0]
    ro = Rollout(t, Y, np.zeros((len(Y), 6, 6)), np.r_[np.zeros(50), np.ones(49)].astype(int),
                 ["x", "y", "z", "h", "g", "pitch"], [49, 98])
    m = cycle_metrics(ro, dA, dT, monolithic=False)
    for k in ("rmse_pos_mm", "grasp_err_mm", "release_err_mm", "rmse_gripper", "rmse_approach_mm", "rmse_transport_mm"):
        assert m[k] == pytest.approx(0.0, abs=1e-6), k


def test_closing_index_finds_the_grasp():
    g = np.r_[np.full(10, 2.0), np.linspace(2, 50, 10), np.full(20, 50.0), np.linspace(50, 20, 10), np.full(20, 20.0),
              np.linspace(20, 45, 5)]
    i = closing_index(g)
    assert 40 <= i <= 50 and g[i] <= 35.0


def det(color, x):
    return CubeDetection(color, np.array([x, 100.0]), 0.0, (40, 40), 1600, 0.9, np.zeros((4, 1, 2), np.int32))


def test_order_cubes_by_color_then_distance():
    dets = [det("green", 0), det("red", 1), det("red", 2), det("purple", 3)]
    xy = np.array([[0.1, 0], [0.3, 0], [0.2, 0], [0.1, 0.1]])
    assert order_cubes(dets, xy, ["red", "yellow", "green"]) == [2, 1, 0, 3]


def test_joint_move_is_smooth_and_exact():
    q, g = joint_move(np.zeros(5), np.ones(5) * 10, 0.0, 50.0, 1.0, 1 / 30)
    np.testing.assert_allclose(q[0], 0) and np.testing.assert_allclose(q[-1], 10)
    assert g[-1] == pytest.approx(50.0) and np.all(np.diff(q[:, 0]) >= -1e-12)


def test_runner_dry_run_sends_lerobot_actions():
    r = SO101Runner(dry_run=True)
    r.connect()
    t = np.linspace(0, 1, 31)
    q = np.tile([0.0, -90.0, 80.0, 70.0, 0.0], (31, 1))
    n = r.play(t, q, np.full(31, 30.0), approach_s=0.5, realtime=False)
    assert n == 15 + 31
    assert set(r.robot.sent[-1]) == set(JOINT_KEYS) and r.robot.sent[-1]["gripper.pos"] == 30.0
    np.testing.assert_allclose(r.read_state()[:5], q[-1])


def test_runner_refuses_non_degree_follower():
    class Cfg:
        use_degrees = False

    class Robot(SimulatedFollower):
        config = Cfg()

    with pytest.raises(RuntimeError, match="DEGREES"):
        SO101Runner(robot=Robot())


@pytest.fixture(scope="module")
def trained():
    cfg = load_config()
    mdir = resolve_path(cfg["tpgmm"]["models_dir"])
    cal = resolve_path(cfg["vision"]["calibration"]["output"])
    if not (mdir / "segmented.json").exists() or not cal.exists():
        pytest.skip("run scripts/vision_pipeline.py and scripts/train.py first")
    return cfg


def test_multi_cube_scene_end_to_end(real_dataset, trained):
    """Paste two cubes from other episodes into a reference frame: 3 cubes, planned in colour order."""
    import pandas as pd

    from so101_tpgmm.inference import plan_from_image
    from so101_tpgmm.kinematics.ik import ik_from_config
    from so101_tpgmm.tpgmm.policy import TPGMMPolicy
    from so101_tpgmm.vision.calibration import VisionCalibration
    from so101_tpgmm.vision.geometry import PixelToBase

    cfg = trained
    frames = pd.read_csv(resolve_path(cfg["vision"]["frames_csv"])).set_index("episode")
    cam = cfg["dataset"]["fixed_camera_key"]

    def ref(e):
        ep = real_dataset.episode(e)
        return real_dataset.read_frames(cam, e, ep.timestamps[[int(frames.loc[e, "ref_frame"])]])[0][0]

    base = ref(0).copy()  # green cube
    for e, (dx, dy) in ((100, (200, 60)), (40, (330, 230))):  # a red and a yellow cube, moved
        img = ref(e)
        u, v = int(frames.loc[e, "u"]), int(frames.loc[e, "v"])
        patch = img[v - 28:v + 28, u - 28:u + 28]
        base[dy - 28:dy + 28, dx - 28:dx + 28] = patch
    H, _, z = VisionCalibration.load_h(resolve_path(cfg["vision"]["calibration"]["output"]))
    pol = TPGMMPolicy.load(resolve_path(cfg["tpgmm"]["models_dir"]) / "segmented.json")
    plan = plan_from_image(base, pol, ik_from_config(cfg), PixelToBase(H), z, cfg,
                           colors=["red", "yellow", "green"])
    assert [c.color for c in sorted(plan.cubes, key=lambda c: c.order)] == ["red", "yellow", "green"]
    planned = [c for c in plan.cubes if c.status == "planned"]
    assert len(planned) >= 2
    assert np.all(np.isfinite(plan.joints_deg)) and np.all(np.diff(plan.t) > 0)
    steps = np.abs(np.diff(plan.joints_deg, axis=0)).max()
    assert steps < 15.0  # degrees per 33 ms sample: continuous across cubes and returns home
