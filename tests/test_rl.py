"""RL task core and environments (MuJoCo env skipped without MuJoCo / the robot model)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402
from so101_tpgmm.kinematics.ik import ik_from_config  # noqa: E402
from so101_tpgmm.rl.task import PickPlaceTask, TaskConfig  # noqa: E402

CFG = load_config(None, [])
HOME = np.asarray(CFG["inference"]["home_joints_deg"], float)


def _task(mode="scratch"):
    from so101_tpgmm.policies import load_policy

    base = load_policy(CFG, "segmented") if mode == "residual" else None
    if mode == "residual" and not (resolve_path(CFG["tpgmm"]["models_dir"]) / "segmented.json").exists():
        pytest.skip("no trained TP-GMM")
    return PickPlaceTask(TaskConfig.from_config(CFG, mode), ik_from_config(CFG), np.array([0.05, 0.19, 0.04]),
                         table_height=lambda x, y: -0.03, base_policy=base)


def test_scratch_actions_are_rate_limited_and_stay_above_the_table():
    t = _task()
    t.reset(HOME, 1.5, np.array([0.25, 0.0, -0.01]), 0.0)
    assert t.observe(HOME, 1.5).shape == (PickPlaceTask.obs_size("scratch"),)
    q_prev, g_prev = HOME, 1.5
    for _ in range(30):
        q, g, ok = t.command(np.array([0.0, 0.0, -1.0, 0.0, 1.0]))  # push down, open fully
        assert np.abs(q[-1] - q_prev).max() <= np.degrees(t.max_joint_step) + 1e-6
        assert np.all(np.abs(np.diff(np.r_[g_prev, g])) <= t.gripper_rate / 30 + 1e-6)
        q_prev, g_prev = q[-1], g[-1]
    assert t.target[2] >= -0.03 + TaskConfig().ws_min[2] - 1e-9  # table floor (relative to the table)
    assert t.g_cmd == pytest.approx(TaskConfig().gripper_max)


def test_residual_zero_action_follows_the_tp_gmm():
    t = _task("residual")
    t.reset(HOME, 1.5, np.array([0.25, 0.05, 0.0064]), 0.2)
    for _ in range(20):
        t.command(np.zeros(5))
    bx, _, _, _ = t.base_at(t.k * t.dt_agent)
    assert np.linalg.norm(t.target - bx) < 1e-9 or t.target[2] > bx[2]  # equal unless lifted by the table floor
    assert t.observe(HOME, 1.5).shape == (PickPlaceTask.obs_size("residual"),)


def test_real_env_dry_run():
    pytest.importorskip("mujoco")  # not needed, but keeps this test with the RL extras
    from so101_tpgmm.rl.real_env import RealSO101Env, finger_offset_xy

    answers = iter(["", "y"])
    env = RealSO101Env(CFG, "scratch", dry_run=True, realtime=False,
                       cube_source=lambda: (np.array([0.25, 0.05, 0.0064]), 0.2), ask=lambda _: next(answers))
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    # the observed cube is the vision position moved to the finger midpoint (heading frame offset)
    h = 0.0
    assert np.allclose(finger_offset_xy([0.0, 0.0], h, [0.01, 0.02]), [0.01, 0.02])
    sent0 = len(env.runner.robot.sent)
    env.task.n_steps = 3
    for _ in range(3):
        obs, r, te, tu, step_info = env.step(np.zeros(5))
    assert len(env.runner.robot.sent) - sent0 == 9  # 3 agent steps × 3 commands at 30 Hz
    assert te and step_info["is_success"]


@pytest.mark.skipif(not (resolve_path(CFG["sim"]["robot_dir"]) / "so101_new_calib.xml").exists(),
                    reason="run scripts/fetch_so101_mujoco.sh")
def test_sim_env_api_and_zero_residual():
    pytest.importorskip("mujoco")
    from so101_tpgmm.rl.sim_env import SO101PickPlaceEnv

    env = SO101PickPlaceEnv(CFG, "residual", randomize=False, seed=0)
    obs, _ = env.reset(seed=3)
    assert env.observation_space.contains(obs)
    done, n = False, 0
    while not done:
        obs, r, te, tu, info = env.step(np.zeros(5, np.float32))
        done, n = te or tu, n + 1
    assert info["success"]  # zero correction = the TP-GMM, which solves this layout
    assert n < env.task.n_steps  # success ends the episode early
