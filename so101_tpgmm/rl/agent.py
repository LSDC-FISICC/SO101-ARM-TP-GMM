"""A trained SAC policy as a closed-loop controller for the multi-cube simulation.

``scripts/simulate_mujoco.py`` runs the planning policies (TP-GMM, IK planner, LPV-DS) as open-loop
plans; a SAC policy instead acts at 10 Hz on the measured state through
:class:`so101_tpgmm.rl.task.PickPlaceTask`, one cube at a time, in the same simulator, with the same
cube pose information and the same return home between cubes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from so101_tpgmm.config import resolve_path
from so101_tpgmm.rl.task import PickPlaceTask, TaskConfig
from so101_tpgmm.sim.episode import in_box, policy_input
from so101_tpgmm.sim.scene import SimCube


class SACAgent:
    def __init__(self, cfg: dict, mode: str, model_path: str | Path | None = None, base: str = "segmented") -> None:
        import torch
        from stable_baselines3 import SAC

        from so101_tpgmm.policies import load_policy

        torch.set_num_threads(1)
        self.cfg, self.mode = cfg, mode
        path = Path(model_path) if model_path else resolve_path(cfg["rl"]["models_dir"]) / f"{mode}_s0" / "best_model.zip"
        if not path.exists():
            raise FileNotFoundError(f"{path}: train it with scripts/train_sac.py --mode {mode}")
        self.model = SAC.load(path, device="cpu")
        self.tpgmm = load_policy(cfg, base)
        self.meta = self.tpgmm.meta  # cube-yaw rule (pan axis, heading offset), as for the other policies
        self.primitives = ["approach_grasp", "transport_release"]

    def run_cube(self, sim, spec, i: int, obs_xyz: np.ndarray, yaw: float, heading: float, ik, geom,
                 render_every: int = 2):
        """Pick cube ``i`` (observed at ``obs_xyz``, ``yaw``) until it rests in the box or the horizon ends.

        Returns (joint commands [deg] (N, 5), gripper commands (N,), cube positions (N, n_cubes, 3), frames,
        policy time [s]).
        """
        from so101_tpgmm.rl.sim_env import release_point

        cfg = self.cfg
        task = PickPlaceTask(TaskConfig.from_config(cfg, self.mode), ik, release_point(cfg, spec),
                             table_height=spec.table_height,
                             base_policy=self.tpgmm if self.mode == "residual" else None)
        q, g = sim.joints_deg_gripper()
        base_input = None
        if self.mode == "residual":
            p, _ = policy_input(self.tpgmm, ik, geom, spec, SimCube("", obs_xyz, yaw), heading, q, g, cfg)
            base_input = (p, yaw, heading)
        task.reset(q, g, obs_xyz, yaw, base_input=base_input)
        obs = task.observe(q, g)
        Q, G, C, F = [], [], [], []
        lifted = False
        z0 = sim.cube_positions()[i, 2]
        while not task.done_time:
            a, _ = self.model.predict(obs, deterministic=True)
            qs, gs, _ = task.command(a)
            out = sim.execute(qs, gs, 1 / 30, render_every=render_every)
            Q.append(qs), G.append(gs), C.append(out.cubes_xyz)
            F += out.frames
            q, g = sim.joints_deg_gripper()
            obs = task.observe(q, g)
            c = sim.cube_positions()[i]
            pinched = sim.pinched_cube() == i
            lifted |= bool(pinched and c[2] - z0 > 0.01)
            if lifted and not pinched and in_box(spec, c):  # done, as in training
                break
        return np.vstack(Q), np.concatenate(G), np.concatenate(C), F, task.k * task.dt_agent
