"""Gripper command sent to the follower, and base→pixel projection used by the scene overlay."""

from __future__ import annotations

import numpy as np

from so101_tpgmm.inference import gripper_command
from so101_tpgmm.vision.geometry import PixelToBase

GC = {"contact_level": 20.0, "release_level": 26.0, "hold_command": 2.0}


def test_holding_commands_a_squeeze_and_the_rest_is_unchanged():
    g = np.array([50.0, 30.0, 26.0, 23.0, 20.0, 18.0, 1.5])
    out = gripper_command(g, GC)
    assert np.allclose(out[:3], g[:3])                       # open / opening: as planned
    assert np.allclose(out[4:6], 2.0)                        # holding level (jaws on the cube): squeeze
    assert 2.0 < out[3] < 26.0                               # contact → release: continuous ramp
    assert np.all(np.diff(gripper_command(np.linspace(30, 15, 50), GC)) <= 1e-9)  # monotone, no jump back
    assert np.allclose(gripper_command(g, None), g)          # disabled: plan's value


def test_to_pixel_inverts_to_base():
    H = np.array([[0.0005, 0.00002, -0.12], [0.00001, -0.0006, 0.25], [0.00001, 0.00002, 1.0]])
    p2b = PixelToBase(H)
    uv = np.array([[100.0, 80.0], [320.0, 240.0], [600.0, 400.0]])
    assert np.allclose(p2b.to_pixel(p2b.to_base(uv)), uv, atol=1e-6)
