"""Load a trained policy by name (all share ``rollout(cube_xyz, cube_yaw, start, dt=, heading=)``)."""

from __future__ import annotations

from so101_tpgmm.config import resolve_path

POLICIES = ("segmented", "monolithic", "ik_planner", "lpv_ds")


def load_policy(cfg: dict, name: str):
    from so101_tpgmm.lpvds.policy import LPVDSPolicy
    from so101_tpgmm.planner.ik_planner import IKPlanner
    from so101_tpgmm.tpgmm.policy import TPGMMPolicy

    mdir = resolve_path(cfg["tpgmm"]["models_dir"])
    if name == "ik_planner":
        pol = IKPlanner.load(mdir / "ik_planner.json")
        pol.primitives = ["approach_grasp", "transport_release"]
        return pol
    if name == "lpv_ds":
        return LPVDSPolicy.load(mdir / "lpv_ds.json")
    if name in ("segmented", "monolithic"):
        return TPGMMPolicy.load(mdir / f"{name}.json")
    raise ValueError(f"unknown policy {name!r}; expected one of {POLICIES}")
