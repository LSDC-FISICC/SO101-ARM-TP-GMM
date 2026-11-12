"""Forward kinematics of the SO-101 from the official URDF (``so101_new_calib.urdf``).

Frames
------
* **Base frame** = URDF ``base_link``: origin at the bottom of the robot base,
  z up. All Cartesian quantities in this package are expressed here (metres).
* **TCP** = URDF ``gripper_frame_link``: a fixed frame between the jaw tips.

Units
-----
FK takes **radians**. Dataset values are converted explicitly with
:func:`so101_tpgmm.data.units.joints_to_radians`; :meth:`SO101Kinematics.fk_from_dataset`
does that conversion, so callers cannot pass degrees to FK by accident.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from so101_tpgmm.data.units import JointUnits, joints_to_radians
from so101_tpgmm.kinematics.urdf import UrdfJoint, chain_between, parse_urdf_joints

DEFAULT_URDF = Path(__file__).resolve().parent / "urdf" / "so101_new_calib.urdf"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _axis_angle_batch(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotation matrices (N, 3, 3) about a fixed unit ``axis`` by angles ``q`` (N,)."""
    x, y, z = axis
    c, s = np.cos(q), np.sin(q)
    C = 1.0 - c
    R = np.empty(q.shape + (3, 3))
    R[..., 0, 0] = c + x * x * C
    R[..., 0, 1] = x * y * C - z * s
    R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s
    R[..., 1, 1] = c + y * y * C
    R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s
    R[..., 2, 1] = z * y * C + x * s
    R[..., 2, 2] = c + z * z * C
    return R


class SO101Kinematics:
    """Batched FK along the URDF chain ``base_link -> tcp_link``.

    Args:
        urdf_path: URDF file (defaults to the vendored ``so101_new_calib.urdf``).
        base_link: Root link of the chain.
        tcp_link: Tool-center-point link.
        joint_names: Actuated joints in dataset order.
    """

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_URDF,
        base_link: str = "base_link",
        tcp_link: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ) -> None:
        self.urdf_path = Path(urdf_path)
        self.joint_names = list(joint_names or ARM_JOINTS)
        joints = parse_urdf_joints(self.urdf_path)
        self.chain: list[UrdfJoint] = chain_between(joints, base_link, tcp_link)
        actuated = [j.name for j in self.chain if j.type in ("revolute", "continuous")]
        if actuated != self.joint_names:
            raise ValueError(f"URDF chain actuated joints {actuated} != configured {self.joint_names}")
        self.lower = np.array([joints[n].lower for n in self.joint_names])
        self.upper = np.array([joints[n].upper for n in self.joint_names])

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Homogeneous TCP poses for joint angles in radians.

        Args:
            q: ``(5,)`` or ``(N, 5)`` radians.

        Returns:
            ``(4, 4)`` or ``(N, 4, 4)`` transforms base -> TCP.
        """
        q = np.asarray(q, dtype=np.float64)
        single = q.ndim == 1
        Q = np.atleast_2d(q)
        if Q.shape[1] != self.n_joints:
            raise ValueError(f"Expected {self.n_joints} joints, got {Q.shape}")
        n = Q.shape[0]
        T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        k = 0
        for j in self.chain:
            T = T @ j.origin
            if j.type in ("revolute", "continuous"):
                M = np.zeros((n, 4, 4))
                M[:, :3, :3] = _axis_angle_batch(j.axis, Q[:, k])
                M[:, 3, 3] = 1.0
                T = T @ M
                k += 1
            elif j.type != "fixed":
                raise NotImplementedError(f"Joint type {j.type!r} not supported")
        return T[0] if single else T

    def fk_position(self, q: np.ndarray) -> np.ndarray:
        """TCP position(s) in metres, ``(3,)`` or ``(N, 3)``."""
        T = self.fk(q)
        return T[..., :3, 3]

    def fk_from_dataset(self, values: np.ndarray, units: JointUnits | str) -> np.ndarray:
        """FK from stored dataset values ``(..., >=5)`` with explicit unit conversion."""
        v = np.asarray(values, dtype=np.float64)[..., : self.n_joints]
        return self.fk(joints_to_radians(v, units))

    def within_limits(self, q: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of configurations inside the URDF limits (radians)."""
        q = np.atleast_2d(q)
        return np.all((q >= self.lower - margin) & (q <= self.upper + margin), axis=-1)
