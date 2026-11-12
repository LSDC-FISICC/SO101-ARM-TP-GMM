"""Load a trained policy by name (all share ``rollout(cube_xyz, cube_yaw, start, dt=, heading=)``)."""

from __future__ import annotations

from so101_tpgmm.config import resolve_path

POLICIES = ("segmented", "monolithic", "ik_planner", "lpv_ds")
RL_POLICIES = ("sac_scratch", "sac_residual")  # closed loop, so101_tpgmm.rl.agent.SACAgent


def load_policy(cfg: dict, name: str):
    from so101_tpgmm.lpvds.policy import LPVDSPolicy
    from so101_tpgmm.planner.ik_planner import IKPlanner
    from so101_tpgmm.tpgmm.policy import TPGMMPolicy

    if name in RL_POLICIES:
        from so101_tpgmm.rl.agent import SACAgent

        return SACAgent(cfg, name.split("_", 1)[1])
    mdir = resolve_path(cfg["tpgmm"]["models_dir"])
    if name == "ik_planner":
        pol = IKPlanner.load(mdir / "ik_planner.json")
        pol.primitives = ["approach_grasp", "transport_release"]
        return pol
    if name == "lpv_ds":
        return LPVDSPolicy.load(mdir / "lpv_ds.json")
    if name in ("segmented", "monolithic"):
        return TPGMMPolicy.load(mdir / f"{name}.json")
    raise ValueError(f"unknown policy {name!r}; expected one of {POLICIES + RL_POLICIES}")
