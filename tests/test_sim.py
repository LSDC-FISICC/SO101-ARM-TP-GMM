"""MuJoCo simulation: model consistency, finger geometry, table plane and a physical grasp.

Skipped when MuJoCo or the robot model (scripts/fetch_so101_mujoco.sh) is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402

CFG = load_config(None, [])
XML = resolve_path(CFG["sim"]["robot_dir"]) / "so101_new_calib.xml"
pytestmark = pytest.mark.skipif(not XML.exists(), reason="run scripts/fetch_so101_mujoco.sh")


def _spec(**kw):
    from so101_tpgmm.sim.scene import SceneSpec

    return SceneSpec(XML, **kw)


def test_mujoco_tcp_matches_project_fk():
    from so101_tpgmm.kinematics.fk import SO101Kinematics

    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    kin = SO101Kinematics(resolve_path(CFG["kinematics"]["urdf"]))
    sid = m.site("gripperframe").id
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(-1.2, 1.2, 5)
        d.qpos[:5] = q
        mujoco.mj_kinematics(m, d)
        assert np.linalg.norm(d.site_xpos[sid] - kin.fk(q)[:3, 3]) < 1e-5


def test_finger_gap_follows_gripper_command():
    """Tips touch at 0 and are ~4 cm apart near the demonstrated holding range (the cube fits)."""
    from so101_tpgmm.sim.runner import gripper_to_rad
    from so101_tpgmm.sim.scene import build_model

    m = build_model(_spec())
    d = mujoco.MjData(m)
    F = [g for g in range(m.ngeom) if m.geom(g).name.startswith("gripper_pad")]
    J = [g for g in range(m.ngeom) if m.geom(g).name.startswith("moving_jaw_so101_v1_pad")]
    assert len(F) >= 5 and len(J) >= 5
    qa = m.jnt_qposadr[m.joint("gripper").id]

    def tip_gap(g):
        d.qpos[:] = 0
        d.qpos[qa] = gripper_to_rad(g)
        mujoco.mj_kinematics(m, d)
        return mujoco.mj_geomDistance(m, d, F[0], J[0], 0.2, np.zeros(6))

    assert tip_gap(0) < 0.003
    assert tip_gap(20) < 0.04 < tip_gap(32)  # holding (~20) squeezes a 4 cm cube; the open approach clears it


def test_cube_rests_on_tilted_table():
    spec = _spec(table_plane=(-0.01, -0.06, 0.004))
    pos, R = spec.cube_on_table(0.3, -0.1, 0.4)
    n = spec.table_rotation()[:, 2]
    assert np.isclose(n @ R[:, 2], 1.0)
    bottom = pos - n * spec.cube_side / 2
    assert abs(bottom[2] - spec.table_height(bottom[0], bottom[1])) < 1e-3


def test_sampled_cubes_are_separated_and_clear_of_the_box():
    from so101_tpgmm.sim.episode import sample_cubes

    demo_xy = np.random.default_rng(1).uniform([0.1, -0.15], [0.38, 0.22], (300, 2))
    s = CFG["sim"]
    for seed in range(20):
        cubes = sample_cubes(np.random.default_rng(seed), demo_xy, CFG)
        xy = np.array([c.xyz[:2] for c in cubes])
        assert len(cubes) == s["n_cubes"]
        assert min(np.linalg.norm(a - b) for i, a in enumerate(xy) for b in xy[i + 1:]) >= s["min_separation"]
        half = np.asarray(s["box"]["inner"]) / 2 + s["box"]["wall_t"] + 0.02
        assert not np.any(np.all(np.abs(xy - np.asarray(s["box"]["center_xy"])) < half, axis=1))


@pytest.mark.parametrize("mode", ["contact", "assist"])
def test_physical_grasp_lifts_the_cube(mode):
    """Cube between the open fingers, close, lift: it must come up with the gripper."""
    from so101_tpgmm.kinematics.ik import ik_from_config
    from so101_tpgmm.sim.episode import FingerGeometry
    from so101_tpgmm.sim.runner import Simulator
    from so101_tpgmm.sim.scene import SimCube

    ik = ik_from_config(CFG)
    r = ik.solve(np.array([0.25, 0.02, 0.0]), 0.0, np.deg2rad([0, -40, 60, 70, 0]), pitch=np.deg2rad(15))
    q = np.rad2deg(r.q)
    geom = FingerGeometry(_spec())
    mid, low = geom(q)
    # table 1 cm below the finger tips, cube centred between them, aligned with the closing direction
    d = geom.data
    tip = lambda G: np.mean(sorted((d.geom_xpos[g] for g in G), key=lambda v: v[2])[:2], axis=0)  # noqa: E731
    v = tip(geom.jaw) - tip(geom.fixed)
    spec = _spec(table_plane=(low - 0.01, 0.0, 0.0),
                 cubes=[SimCube("red", np.r_[mid[:2], 0.0], float(np.arctan2(v[1], v[0])))], box_center_xy=(-0.3, 0.3))
    sim = Simulator(spec, grasp_mode=mode)
    sim.reset(q, 40.0)
    z0 = sim.cube_positions()[0, 2]
    n = 30
    sim.execute(np.tile(q, (n, 1)), np.linspace(40, 20, n), 1 / 30)  # close
    lift = q + np.array([0, -15, 0, 0, 0])
    sim.execute(np.linspace(q, lift, n), np.full(n, 20.0), 1 / 30)  # lift
    sim.execute(np.tile(lift, (n, 1)), np.full(n, 20.0), 1 / 30)  # hold
    assert sim.cube_positions()[0, 2] - z0 > 0.03


def test_release_index_skips_the_initial_closed_gripper():
    from so101_tpgmm.sim.episode import release_index

    g = np.r_[np.full(5, 2.0), np.full(5, 50.0), np.full(5, 20.0), np.full(5, 50.0)]  # rest, open, grasp, release
    assert release_index(g < 30) == 14
    assert release_index(np.full(10, True)) is None
