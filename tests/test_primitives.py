"""Cycle / primitive segmentation with physical gates (synthetic + real data)."""

import numpy as np
import pandas as pd
import pytest

from so101_tpgmm.segmentation.primitives import (
    SegmentationParams,
    load_cycles,
    motion_onset,
    segment_dataset,
    segment_episode,
)

FPS = 30.0


class FakeKin:
    """FK stub: the TCP height is read from the first joint column (in metres*100)."""

    def fk_from_dataset(self, state, units):
        T = np.tile(np.eye(4), (len(state), 1, 1))
        T[:, 2, 3] = state[:, 0] / 100.0
        return T


def build(segments):
    """segments: list of (n_frames, gripper_from, gripper_to, z_from_cm, z_to_cm)."""
    g, z = [], []
    for n, g0, g1, z0, z1 in segments:
        g.append(np.linspace(g0, g1, n))
        z.append(np.linspace(z0, z1, n))
    g, z = np.concatenate(g), np.concatenate(z)
    state = np.zeros((len(g), 6))
    state[:, 0] = z  # read by FakeKin
    state[:, 1] = np.cumsum(np.abs(np.gradient(z))) * 5  # joint motion where z moves (onset)
    state[:, 5] = g
    return state


REST = (40, 1, 1, 0, 0)
APPROACH = [(30, 1, 1, 0, 12), (10, 1, 50, 12, 12), (45, 50, 50, 12, 0), (10, 50, 20, 0, 0)]


def normal_cycle(release_z=8):
    return build([REST, *APPROACH, (20, 20, 20, 0, 0), (40, 20, 20, 0, release_z),
                  (8, 20, 45, release_z, release_z), (15, 45, 45, release_z, release_z),
                  (10, 45, 1, release_z, release_z), (40, 1, 1, release_z, 0), REST])


def test_normal_cycle_primitives_and_keypoints():
    st = normal_cycle()
    r = segment_episode(st, FPS, FakeKin())
    assert len(r.cycles) == 1 and not r.rejected
    c = r.cycles[0]
    assert 38 <= c.approach_start <= 45  # motion onset after the rest period
    assert 133 <= c.close_idx <= 137  # end of the 50 -> 20 closing
    assert c.open_idx < c.release_end and st[c.release_end, 5] > 40  # P2 ends when the opening completes
    assert st[c.pregrasp_idx, 0] >= 3.0 and c.pregrasp_idx < c.close_idx  # 3 cm above contact
    assert st[c.lift_idx, 0] >= 3.0 and c.lift_idx > c.close_idx
    assert c.primitive("approach_grasp") == (c.approach_start, c.close_idx)
    assert c.primitive("transport_release") == (c.close_idx, c.release_end)


def test_contact_in_air_is_rejected():
    st = build([REST, (30, 1, 1, 0, 15), (10, 1, 50, 15, 15), (20, 50, 50, 15, 15), (10, 50, 20, 15, 15),
                (40, 20, 20, 15, 10), (8, 20, 45, 10, 10), (15, 45, 45, 10, 10), (10, 45, 1, 10, 10), REST])
    r = segment_episode(st, FPS, FakeKin())
    assert r.cycles == [] and r.rejected[0][1] == "contact_in_air"


def test_release_at_table_marks_next_cycle_as_aborted():
    aborted = [(10, 50, 20, 0, 0), (30, 20, 20, 0, 0), (8, 20, 45, 0, 0), (20, 45, 45, 0, 0)]
    st = build([REST, *APPROACH[:3], *aborted, (10, 45, 20, 0, 0), (20, 20, 20, 0, 0), (40, 20, 20, 0, 8),
                (8, 20, 45, 8, 8), (15, 45, 45, 8, 8), (10, 45, 1, 8, 8), REST])
    r = segment_episode(st, FPS, FakeKin())
    assert [why for _, why in r.rejected] == ["release_at_table"]
    assert len(r.cycles) == 1 and r.cycles[0].aborted_before


def test_quiet_release_above_table_is_recovered():
    st = build([REST, *APPROACH, (20, 20, 20, 0, 0), (40, 20, 20, 0, 8), (5, 20, 24, 8, 8),
                (8, 24, 1, 8, 8), (40, 1, 1, 8, 0), REST])
    r = segment_episode(st, FPS, FakeKin())
    assert len(r.cycles) == 1 and r.cycles[0].quiet_release


def test_hold_until_episode_end_is_not_a_cycle():
    st = build([REST, *APPROACH, (60, 20, 20, 0, 0)])
    assert segment_episode(st, FPS, FakeKin()).cycles == []


def test_motion_onset_requires_sustained_motion():
    q = np.zeros((100, 6))
    q[20, 0] = 3.0  # a single glitch
    q[50:, 0] = np.arange(50) * 1.0  # 30 deg/s from frame 50
    assert motion_onset(q, FPS, 0, SegmentationParams()) == 50


def test_load_cycles_round_trip(tmp_path):
    st = normal_cycle()
    c = segment_episode(st, FPS, FakeKin(), episode=7).cycles[0]
    path = tmp_path / "cycles.csv"
    pd.DataFrame([c.__dict__]).to_csv(path, index=False)
    assert load_cycles(path)[0] == c


def test_real_dataset_one_valid_cycle_per_episode(real_dataset):
    from so101_tpgmm.config import load_config
    from so101_tpgmm.kinematics.fk import SO101Kinematics

    res, df = segment_dataset(real_dataset, SO101Kinematics(), load_config())
    assert len(df) == real_dataset.num_episodes
    assert all(len(r.cycles) == 1 for r in res)
    assert set(df[df.quiet_release].episode) == {106, 143, 168, 221, 234}
    assert {370, 374} <= {r.episode for r in res for _, why in r.rejected if why == "contact_in_air"}
