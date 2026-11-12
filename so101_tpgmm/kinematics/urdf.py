"""Tiny URDF parser for serial chains (no external robotics dependencies)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class UrdfJoint:
    """A URDF joint with its fixed origin transform and motion axis."""

    name: str
    type: str  # revolute | continuous | prismatic | fixed
    parent: str
    child: str
    origin: np.ndarray  # 4x4 homogeneous transform parent -> joint frame
    axis: np.ndarray  # unit 3-vector in the joint frame
    lower: float
    upper: float


def rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw -> rotation matrix (Rz(y) @ Ry(p) @ Rx(r))."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _parse_floats(s: str | None, n: int, default: float = 0.0) -> np.ndarray:
    if s is None:
        return np.full(n, default)
    vals = np.array([float(x) for x in s.split()])
    if vals.size != n:
        raise ValueError(f"Expected {n} floats, got {s!r}")
    return vals


def parse_urdf_joints(path: str | Path) -> dict[str, UrdfJoint]:
    """Parse all joints of a URDF file, keyed by name."""
    root = ET.parse(str(path)).getroot()
    joints: dict[str, UrdfJoint] = {}
    for j in root.findall("joint"):
        origin_el = j.find("origin")
        xyz = _parse_floats(origin_el.get("xyz") if origin_el is not None else None, 3)
        rpy = _parse_floats(origin_el.get("rpy") if origin_el is not None else None, 3)
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(*rpy)
        T[:3, 3] = xyz
        axis_el = j.find("axis")
        axis = _parse_floats(axis_el.get("xyz") if axis_el is not None else None, 3)
        if j.get("type") != "fixed":
            if axis_el is None:
                axis = np.array([1.0, 0.0, 0.0])  # URDF default
            axis = axis / np.linalg.norm(axis)
        lim = j.find("limit")
        lower = float(lim.get("lower", "-inf")) if lim is not None else -np.inf
        upper = float(lim.get("upper", "inf")) if lim is not None else np.inf
        joints[j.get("name")] = UrdfJoint(
            name=j.get("name"),
            type=j.get("type"),
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            origin=T,
            axis=axis,
            lower=lower,
            upper=upper,
        )
    return joints


def chain_between(joints: dict[str, UrdfJoint], base_link: str, tip_link: str) -> list[UrdfJoint]:
    """Ordered list of joints from ``base_link`` to ``tip_link``."""
    by_child = {j.child: j for j in joints.values()}
    chain: list[UrdfJoint] = []
    link = tip_link
    while link != base_link:
        if link not in by_child:
            raise ValueError(f"No path from {base_link!r} to {tip_link!r} (stuck at {link!r}).")
        j = by_child[link]
        chain.append(j)
        link = j.parent
    return chain[::-1]
