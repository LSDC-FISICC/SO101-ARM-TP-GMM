"""Silent failure: gripper threshold without hysteresis (phantom cycles)."""

import numpy as np
import pytest

from so101_tpgmm.segmentation.gripper import GripperParams, detect_gripper_events, hysteresis_binarize

FPS = 30.0


def ramp(a, b, n):
    return np.linspace(a, b, n)


def synthetic_cycle(release_open: float = 45.0) -> np.ndarray:
    """rest(1) -> open(50) -> hold(20) -> open(release) -> rest(1)."""
    return np.r_[np.full(60, 1.0), ramp(1, 50, 10), np.full(60, 50.0), ramp(50, 20, 10), np.full(90, 20.0),
                 ramp(20, release_open, 6), np.full(20, release_open), ramp(release_open, 1, 10), np.full(60, 1.0)]


def test_single_cycle_contact_and_release_instants():
    g = synthetic_cycle()
    ev = detect_gripper_events(g, FPS)
    assert len(ev.cycles) == 1
    c = ev.cycles[0]
    assert c.hold_level == pytest.approx(20.0, abs=0.5)
    assert 138 <= c.close_idx <= 141  # end of the 50->20 ramp (frames 130..139)
    assert 228 <= c.open_idx <= 231  # last frame on the hold plateau
    assert not ev.empty_closures and not ev.unreleased_holds


def test_noise_around_threshold_creates_no_phantom_cycles():
    rng = np.random.default_rng(0)
    g = synthetic_cycle()
    g[150:220] = 24.5 + rng.normal(0, 0.8, 70)  # hold plateau hovering at a naive threshold
    naive = (g > 24.5).astype(int)
    assert np.sum(np.diff(naive) == 1) > 5  # a single threshold would chatter
    ev = detect_gripper_events(g, FPS, GripperParams(open_threshold=27.0, close_threshold=23.5, hold_min=13))
    assert len(ev.cycles) <= 1


def test_final_rest_closure_is_not_a_missed_grasp():
    ev = detect_gripper_events(synthetic_cycle(), FPS)
    assert ev.empty_closures == []


def test_missed_grasp_is_reported():
    g = np.r_[np.full(30, 1.0), ramp(1, 50, 10), np.full(30, 50.0), ramp(50, 5, 10), np.full(30, 5.0),
              ramp(5, 50, 10), synthetic_cycle()[70:]]
    ev = detect_gripper_events(g, FPS)
    assert len(ev.empty_closures) == 1 and len(ev.cycles) == 1


def test_short_hold_is_rejected():
    g = synthetic_cycle()
    g[140:230] = 50.0
    g[180:185] = 20.0  # 5 frames ~ 0.17 s < min_hold_s
    assert detect_gripper_events(g, FPS).cycles == []


def test_hysteresis_keeps_state_inside_band():
    m = hysteresis_binarize(np.array([1, 30, 25, 24, 30, 22, 25]), open_thr=25.5, close_thr=23.5)
    assert m.tolist() == [False, True, True, True, True, False, False]


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        detect_gripper_events(synthetic_cycle(), FPS, GripperParams(open_threshold=20, close_threshold=23))


def test_real_dataset_mostly_one_cycle(real_dataset):
    n = [len(detect_gripper_events(real_dataset.episode(k).state[:, 5], real_dataset.fps).cycles)
         for k in range(0, real_dataset.num_episodes, 10)]
    assert np.mean(np.array(n) == 1) > 0.9
