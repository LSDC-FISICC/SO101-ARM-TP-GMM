"""Closed-form IK of the SO-101 by kinematic decoupling (corrected Rivas 2026 model).

Follows the structure of the closed-form solution in S. Rivas, *Modelado,
Simulación, Teleoperación y Aprendizaje por Imitación con ACT del Manipulador
SO-101* (Universidad Galileo, LSDC, 2026): q1 from the wrist position, the
"pitch" beta = q2 + q3 + q4, a two-link planar arm for (q2, q3) with an L-shaped
first link, q4 = beta - q2 - q3, and q5 from the orientation. The corrections
found when checking it against ``so101_new_calib.urdf`` (docs/kinematics_paper_review.md):

1. **Joint signs.** LeRobot/URDF ``shoulder_pan`` and ``wrist_roll`` are the
   opposite of the paper's q1 and q5: q1 = -pan, q5 = pi - roll (eq. 1 of the paper
   with LeRobot angles mirrors y).
2. **Forearm offset.** The elbow->wrist link is 134.9 mm along and **5.2 mm**
   across; like the upper arm it is an L-shaped link (offset angle ~2.2 deg).
3. **Radial distance.** The planar sub-problem uses r = sqrt(Px^2 + Py^2) (minus the
   tiny lateral offset), not Px.
4. **beta without division by c1.** beta = atan2(-r33, r13 c1 + r23 s1) instead of
   atan2(-r33, r13 / c1), which is singular at q1 = +-90 deg (the data reaches -98.6 deg).
5. **TCP.** ``gripper_frame_link`` is 7.9 mm off the wrist-roll axis (along the
   jaw-closing axis) and the gripper is mounted with a 2.8 deg twist; both are
   constant tool transforms removed before decoupling.
6. **Frame origin.** The paper's origin is on the pan axis; ``base_link`` is 38.8 mm
   behind it.

All lengths are extracted from the URDF, so the solution is exact for the FK used
elsewhere in this package (verified to < 1e-9 m in the tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from so101_tpgmm.kinematics.fk import SO101Kinematics


def _rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s, C = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


@dataclass
class ArmGeometry:
    """Planar-arm parameters (paper notation where it exists), extracted from the URDF."""

    pan_xy: np.ndarray  # pan axis position in base_link [m]
    a1: float  # = l0 + l2: shoulder-lift axis ahead of the pan axis
    d1: float  # = l1 + l3: shoulder-lift axis height above base_link z = 0
    lateral: float  # arm-plane offset from the pan axis (tiny)
    l5: float  # upper arm, component along the arm at q2 = 0
    l4: float  # upper arm, vertical component at q2 = 0
    l6: float  # forearm, along
    l6b: float  # forearm, vertical offset (missing in the paper)
    l78: float  # = l7 + l8: wrist-flex axis -> TCP depth along the roll axis
    tcp_offset: np.ndarray  # TCP - (point on roll axis), in the TCP frame [m]
    roll_dir_in_tcp: np.ndarray  # roll axis pointing from the wrist towards the TCP, TCP frame
    roll_joint_axis_in_tcp: np.ndarray  # wrist_roll joint axis (URDF sign), TCP frame
    tool_rotation: np.ndarray  # C = R_tcp(q = 0): R_tcp = Rz(-q1) Ry(q2+q3+q4) Rx(-q5) C

    @property
    def L1(self) -> float:
        return float(np.hypot(self.l4, self.l5))

    @property
    def L2(self) -> float:
        return float(np.hypot(self.l6, self.l6b))

    @property
    def phi(self) -> float:
        """Paper's phi = atan2(l4, l5): angle of the upper arm at q2 = 0."""
        return float(np.arctan2(self.l4, self.l5))

    @property
    def phi2(self) -> float:
        """Forearm offset angle atan2(l6b, l6)."""
        return float(np.arctan2(self.l6b, self.l6))

    @classmethod
    def from_urdf(cls, kin: SO101Kinematics) -> "ArmGeometry":
        T, o, a = kin.fk_with_joint_frames(np.zeros(5))
        pan = o[0]
        roll_pt = o[4] + a[4] * np.dot(T[:3, 3] - o[4], a[4])  # TCP depth on the roll axis
        if not np.allclose(a[1:4], a[1], atol=1e-6) or abs(a[0][2]) < 0.999:
            raise ValueError("URDF does not have the expected SO-101 axis layout.")
        return cls(
            pan_xy=pan[:2].copy(),
            a1=float(o[1][0] - pan[0]),
            d1=float(o[1][2]),
            lateral=float(roll_pt[1] - pan[1]),
            l5=float(o[2][0] - o[1][0]),
            l4=float(o[2][2] - o[1][2]),
            l6=float(o[3][0] - o[2][0]),
            l6b=float(o[3][2] - o[2][2]),
            l78=float(roll_pt[0] - o[3][0]),
            tcp_offset=T[:3, :3].T @ (T[:3, 3] - roll_pt),
            roll_dir_in_tcp=T[:3, :3].T @ (a[4] * np.sign(np.dot(a[4], T[:3, 3] - o[3]))),
            roll_joint_axis_in_tcp=T[:3, :3].T @ a[4],
            tool_rotation=T[:3, :3].copy(),
        )


class AnalyticIK:
    """Exact closed-form IK for a full TCP pose; both elbow branches.

    Joint angles are in the **LeRobot/URDF convention** (radians), i.e. directly
    usable with :class:`SO101Kinematics` and the robot.
    """

    def __init__(self, kin: SO101Kinematics) -> None:
        self.kin = kin
        self.g = ArmGeometry.from_urdf(kin)
        # sign of shoulder_pan: +pan turns the arm clockwise seen from above (axis points down)
        _, _, a = kin.fk_with_joint_frames(np.zeros(5))
        self._pan_sign = float(np.sign(a[0][2]))  # azimuth = pan_sign * q_pan (axis points down => -1)
        self._lift_sign = self._planar_sign()

    def _planar_sign(self) -> float:
        """+1 if increasing q2 lowers the upper arm (paper convention), from the URDF."""
        q = np.zeros(5)
        q[1] = 0.1
        _, o, _ = self.kin.fk_with_joint_frames(q)
        return 1.0 if o[2][2] < self.g.d1 + self.g.l4 else -1.0

    def solve_pose(self, T_target: np.ndarray) -> list[np.ndarray]:
        """All closed-form solutions (up to 2 shoulder x 2 elbow branches) for a TCP pose (4x4, base frame).

        Joint limits are **not** applied here; callers filter.
        """
        g = self.g
        R = T_target[:3, :3]
        p_roll = T_target[:3, 3] - R @ g.tcp_offset  # point on the wrist-roll axis
        P = p_roll[:2] - g.pan_xy
        rho2 = P @ P
        if rho2 <= g.lateral**2:
            return []
        r0 = np.sqrt(rho2 - g.lateral**2)
        sols = []
        # Shoulder branches: arm in front of the pan axis (r > 0) or folded back over it.
        for r in (r0, -r0):
            # The roll-axis point is at (radial = r, lateral) in the rotated arm frame.
            az = np.arctan2(P[1], P[0]) - np.arctan2(g.lateral, r)
            q1 = az / self._pan_sign
            # beta: elevation of the roll axis below the horizontal, in the arm plane (no 1/c1).
            u = R @ g.roll_dir_in_tcp
            radial = u[0] * np.cos(az) + u[1] * np.sin(az)
            beta = np.arctan2(-u[2], radial)
            # Wrist-flex axis position in the arm plane.
            xp = r - g.a1 - g.l78 * np.cos(beta)
            zp = p_roll[2] - g.d1 + g.l78 * np.sin(beta)
            c = (xp**2 + zp**2 - g.L1**2 - g.L2**2) / (2 * g.L1 * g.L2)
            if abs(c) > 1 + 1e-4:  # tolerance: stretched arm + URDF rounding (non-parallel axes)
                continue
            c = float(np.clip(c, -1, 1))
            for elbow in (+1.0, -1.0):  # elbow-up / elbow-down (paper: +- arccos)
                delta = elbow * np.arccos(c)
                ang_u = np.arctan2(zp, xp) - np.arctan2(g.L2 * np.sin(delta), g.L1 + g.L2 * np.cos(delta))
                ang_f = ang_u + delta
                # Upper-arm direction = phi - s q2 ; forearm = phi2 - s (q2 + q3); roll axis = -s beta.
                sgn = self._lift_sign
                q2 = (g.phi - ang_u) / sgn
                q3 = (g.phi2 - ang_f) / sgn - q2
                q4 = beta / sgn - q2 - q3
                q = np.array([q1, q2, q3, q4, 0.0])
                # q5: R0^T R is a rotation about the roll joint axis n (TCP frame) by q5 (Rodrigues).
                M = self.kin.fk(q)[:3, :3].T @ R
                n = g.roll_joint_axis_in_tcp
                w = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
                q[4] = np.arctan2(n @ w, np.trace(M) - 1.0)
                sols.append(np.arctan2(np.sin(q), np.cos(q)))
        return sols

    def solve_task(self, p: np.ndarray, yaw: float, pitch: float) -> list[np.ndarray]:
        """Closed-form seeds for the task variables (TCP position, wrist yaw, approach pitch).

        The full orientation is built from (yaw, pitch, azimuth of p); because the TCP is
        7.9 mm off the roll axis this is approximate at the mm level, so the result is
        meant as a seed for :class:`SO101IK` (which polishes it exactly).
        """
        d = np.asarray(p[:2]) - self.g.pan_xy
        rhat = np.r_[d / max(np.linalg.norm(d), 1e-9), 0.0]
        z = np.r_[np.sin(pitch) * rhat[:2], -np.cos(pitch)]
        h = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        x = h - (h @ z) * z
        if np.linalg.norm(x) < 1e-6:
            return []
        x /= np.linalg.norm(x)
        R = np.c_[x, np.cross(z, x), z]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return self.solve_pose(T)


def rivas_eq1_position(q_paper: np.ndarray, g: ArmGeometry, forearm_offset: bool = True) -> np.ndarray:
    """Position block of eq. (1) of the paper (pan-axis frame, wrist-roll-axis point).

    Args:
        q_paper: joints in the **paper's** convention (q1 = -pan_LeRobot).
        g: geometry; ``forearm_offset=False`` reproduces the printed equation, which
            omits the 5.2 mm forearm offset (l6b).
    """
    q1, q2, q3, q4 = q_paper[:4]
    c1, s1, c2, s2 = np.cos(q1), np.sin(q1), np.cos(q2), np.sin(q2)
    c23, s23 = np.cos(q2 + q3), np.sin(q2 + q3)
    c234, s234 = np.cos(q2 + q3 + q4), np.sin(q2 + q3 + q4)
    l6b = g.l6b if forearm_offset else 0.0
    r = g.a1 + g.l6 * c23 + l6b * s23 + g.l5 * c2 + g.l4 * s2 + g.l78 * c234
    z = g.d1 - g.l6 * s23 + l6b * c23 + g.l4 * c2 - g.l5 * s2 - g.l78 * s234
    return np.array([c1 * r, s1 * r, z])


def _rz(a: float) -> np.ndarray:
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, 0, s_], [0, 1.0, 0], [-s_, 0, c]])


def _rx(a: float) -> np.ndarray:
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s_], [0, s_, c]])


def closed_form_fk(q: np.ndarray, g: ArmGeometry) -> np.ndarray:
    """Closed-form FK (LeRobot/URDF joint convention, radians) -> TCP pose (4x4, base_link).

    ::

        az   = -q1                      (the pan axis points down)
        beta = q2 + q3 + q4
        r    = a1 + l5 c2 + l4 s2 + l6 c23 + l6b s23 + l78 c(beta)
        z    = d1 + l4 c2 - l5 s2 - l6 s23 + l6b c23 - l78 s(beta)
        p    = (x_pan + r cos az - e sin az,  y_pan + r sin az + e cos az,  z)   # on the roll axis
        R    = Rz(az) Ry(beta) Rx(-q5) C                                        # C = R_tcp(0)
        p_tcp = p + R d_tool                                                     # d_tool ~ (7.9, -0.2, 0) mm

    The approach (TCP z) axis is (c(beta) cos az, c(beta) sin az, -s(beta)), so the
    approach pitch of :func:`so101_tpgmm.kinematics.ik.approach_pitch` is 90 deg - beta.
    """
    q1, q2, q3, q4, q5 = np.asarray(q, dtype=np.float64)[:5]
    az = -q1
    c2, s2 = np.cos(q2), np.sin(q2)
    c23, s23 = np.cos(q2 + q3), np.sin(q2 + q3)
    beta = q2 + q3 + q4
    cb, sb = np.cos(beta), np.sin(beta)
    r = g.a1 + g.l5 * c2 + g.l4 * s2 + g.l6 * c23 + g.l6b * s23 + g.l78 * cb
    z = g.d1 + g.l4 * c2 - g.l5 * s2 - g.l6 * s23 + g.l6b * c23 - g.l78 * sb
    ca, sa = np.cos(az), np.sin(az)
    p = np.array([g.pan_xy[0] + r * ca - g.lateral * sa, g.pan_xy[1] + r * sa + g.lateral * ca, z])
    R = _rz(az) @ _ry(beta) @ _rx(-q5) @ g.tool_rotation
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p + R @ g.tcp_offset
    return T
