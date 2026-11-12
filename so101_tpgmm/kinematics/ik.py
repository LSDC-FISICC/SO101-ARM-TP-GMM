"""Numerical inverse kinematics for the 5-DOF SO-101: TCP position + wrist yaw.

Task variables
--------------
* **Position** of the TCP (``gripper_frame_link``) in the base frame [m].
* **Wrist yaw** ``psi``: heading of the jaw-closing axis projected on the table,
  ``psi = atan2(x_y, x_x)`` with ``x`` the TCP x axis. The moving jaw rotates
  about the TCP y axis, so the jaws close along TCP x; the TCP z axis is the
  approach direction (pointing down when grasping).

Position (3) + yaw (1) = 4 constraints for 5 joints: one redundant DOF. For a
5-DOF arm the approach axis (TCP z) always lies in the vertical plane through the
arm, so the redundancy is the **approach pitch**: the tilt of TCP z from the
downward vertical, positive when it leans radially outward. Phase 1 analysis:
median pitch ~9 deg at grasp, ~27 deg near the table, ~20 deg at rest.

Leaving it free makes the solver drift to postures far from the demonstrations
(up to 80 deg per joint with identical TCP position/yaw) or to configurations
where the jaw axis is vertical and the yaw is undefined. It is therefore resolved
by a **soft pitch target** (explicit, or ``default_pitch``), plus a weak pull
towards the previous posture for smoothness.

Solver
------
Bounded nonlinear least squares (``scipy.optimize.least_squares``, method
``trf``) with the analytic geometric Jacobian, restarted from several seeds if
needed. Joint limits are hard bounds; a solution is **only** reported as
successful if the task errors are below tolerance — targets outside the
reachable workspace come back flagged, never silently clipped.

Joint limits
------------
:func:`joint_limits_from_config` builds the union of the URDF limits and the
range observed in the demonstrations (Phase 0: the rest pose has
``shoulder_lift`` = -104.4 deg, outside the URDF limit of -100 deg).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from so101_tpgmm.kinematics.fk import SO101Kinematics


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap to (-pi, pi]."""
    return -((-np.asarray(a) + np.pi) % (2 * np.pi) - np.pi)


def tool_yaw(R: np.ndarray) -> np.ndarray:
    """Wrist yaw from TCP rotation(s) ``(..., 3, 3)``: heading of the TCP x axis."""
    R = np.asarray(R)
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def yaw_observability(R: np.ndarray) -> np.ndarray:
    """Horizontal norm of the TCP x axis; yaw is ill-defined when it approaches 0."""
    R = np.asarray(R)
    return np.hypot(R[..., 0, 0], R[..., 1, 0])


def approach_pitch(T: np.ndarray, pan_axis_xy: np.ndarray = np.zeros(2)) -> np.ndarray:
    """Signed tilt of the approach axis (TCP z) from straight down, + = leaning outward.

    Args:
        T: ``(..., 4, 4)`` TCP poses.
        pan_axis_xy: XY of the shoulder_pan axis (radial direction origin).
    """
    T = np.asarray(T)
    z = T[..., :3, 2]
    r = T[..., :2, 3] - pan_axis_xy
    r = r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-9)
    return np.arctan2(np.sum(z[..., :2] * r, axis=-1), -z[..., 2])


def joint_limits_from_config(kin: SO101Kinematics, cfg: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Joint limits in radians according to ``kinematics.joint_limits`` in the YAML config.

    ``source``: ``urdf`` or ``urdf_union_observed`` (default). ``observed_deg`` maps joint
    name -> [min, max] observed in the data; ``margin_deg`` widens the observed range.
    """
    lo, hi = kin.lower.copy(), kin.upper.copy()
    cfg = cfg or {}
    if cfg.get("source", "urdf_union_observed") == "urdf":
        return lo, hi
    margin = np.deg2rad(cfg.get("margin_deg", 1.0))
    for i, name in enumerate(kin.joint_names):
        rng = (cfg.get("observed_deg") or {}).get(name)
        if rng is not None:
            lo[i] = min(lo[i], np.deg2rad(rng[0]) - margin)
            hi[i] = max(hi[i], np.deg2rad(rng[1]) + margin)
    return lo, hi


@dataclass
class IKResult:
    """Outcome of a single IK solve."""

    q: np.ndarray  # (5,) radians
    success: bool
    pos_err: float  # [m]
    yaw_err: float  # [rad], 0 if yaw was not requested
    within_limits: bool
    nfev: int


@dataclass
class TrajectoryIKResult:
    """Outcome of :meth:`SO101IK.solve_trajectory`."""

    q: np.ndarray  # (N, 5)
    success: np.ndarray  # (N,) bool
    pos_err: np.ndarray  # (N,)
    yaw_err: np.ndarray  # (N,)
    max_joint_step: np.ndarray  # (N,) max |q_t - q_{t-1}| [rad]

    @property
    def feasible_fraction(self) -> float:
        return float(np.mean(self.success))


class SO101IK:
    """Position + wrist-yaw IK with hard joint limits.

    Args:
        kin: Forward kinematics.
        lower, upper: Joint limits [rad] (default: URDF limits).
        pos_tol: Success tolerance on the position error [m].
        yaw_tol: Success tolerance on the yaw error [rad].
        yaw_weight: Metres of residual per radian of yaw error (balances units).
        reg_weight: Weight of the pull towards the reference posture [m/rad];
            small, only for smoothness.
        pitch_weight: Weight of the soft approach-pitch target [m/rad].
        default_pitch: Pitch target [rad] used when none is given (None = no pitch term).
        min_yaw_observability: Below this horizontal norm of the TCP x axis the
            yaw constraint is dropped (yaw undefined) and the result is position-only.
    """

    def __init__(
        self,
        kin: SO101Kinematics,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        pos_tol: float = 1e-3,
        yaw_tol: float = np.deg2rad(1.0),
        yaw_weight: float = 0.05,
        reg_weight: float = 1e-3,
        min_yaw_observability: float = 0.2,
        pitch_weight: float = 0.02,
        default_pitch: float | None = np.deg2rad(15.0),
    ) -> None:
        self.kin = kin
        self.lower = np.asarray(kin.lower if lower is None else lower, dtype=np.float64)
        self.upper = np.asarray(kin.upper if upper is None else upper, dtype=np.float64)
        if np.any(self.lower >= self.upper):
            raise ValueError("Invalid joint limits.")
        self.pos_tol = pos_tol
        self.yaw_tol = yaw_tol
        self.yaw_weight = yaw_weight
        self.reg_weight = reg_weight
        self.min_yaw_observability = min_yaw_observability
        self.pitch_weight = pitch_weight
        self.default_pitch = default_pitch
        self.pan_axis_xy = kin.chain[0].origin[:2, 3].copy()
        # Canonical (lift, elbow, wrist_flex) postures [deg] for target-aware seeds:
        # reaching far/low, mid, near/high, folded.
        self._postures = np.deg2rad([[30, -40, 90], [0, 0, 70], [-40, 40, 80], [-80, 90, 70], [60, -90, 60]])
        # Linearised pan -> azimuth map (sign/offset from the URDF).
        q = np.zeros(5)
        q[1:4] = self._postures[1]
        q2 = q.copy()
        q2[0] = 0.1
        self._pan_offset = self._azimuth(q)
        self._pan_sign = float(np.sign(wrap_angle(self._azimuth(q2) - self._pan_offset)))

    # ------------------------------------------------------------------ task map
    def task(self, q: np.ndarray) -> tuple[np.ndarray, float]:
        """(TCP position, wrist yaw) for joints ``q`` (5,) in radians."""
        T = self.kin.fk(q)
        return T[:3, 3], float(tool_yaw(T[:3, :3]))

    def _residual_and_jac(self, q, p_t, yaw_t, q_ref, use_yaw, z_t=None):
        T, o, a = self.kin.fk_with_joint_frames(q)
        p = T[:3, 3]
        Jp = np.cross(a, p - o).T  # (3, 5)
        rows_r = [p - p_t]
        rows_J = [Jp]
        if use_yaw:
            x = T[:3, 0]
            dx = np.cross(a, x)  # (5, 3) derivative of the x axis w.r.t. each joint
            den = x[0] ** 2 + x[1] ** 2
            dyaw = (x[0] * dx[:, 1] - x[1] * dx[:, 0]) / den
            rows_r.append([self.yaw_weight * wrap_angle(np.arctan2(x[1], x[0]) - yaw_t)])
            rows_J.append(self.yaw_weight * dyaw[None, :])
        if z_t is not None:  # soft approach-direction target
            z = T[:3, 2]
            rows_r.append(self.pitch_weight * (z - z_t))
            rows_J.append(self.pitch_weight * np.cross(a, z).T)
        rows_r.append(self.reg_weight * (q - q_ref))
        rows_J.append(self.reg_weight * np.eye(5))
        return np.concatenate([np.ravel(r) for r in rows_r]), np.vstack(rows_J)

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        p_target: np.ndarray,
        yaw_target: float | None = None,
        q_seed: np.ndarray | None = None,
        q_ref: np.ndarray | None = None,
        max_restarts: int = 8,
        pitch: float | None = None,
    ) -> IKResult:
        """Solve for one target.

        Args:
            p_target: (3,) TCP position [m], base frame.
            yaw_target: Wrist yaw [rad] or None for position-only.
            q_seed: Initial guess [rad] (clipped into the limits).
            q_ref: Reference posture for the redundant DOF (default: the seed).
            max_restarts: Extra seeds tried if the first solve fails.
            pitch: Soft approach-pitch target [rad] (default: ``default_pitch``).
        """
        p_target = np.asarray(p_target, dtype=np.float64)
        pitch = self.default_pitch if pitch is None else pitch
        z_t = None
        if pitch is not None:
            r = p_target[:2] - self.pan_axis_xy
            r = r / max(np.linalg.norm(r), 1e-9)
            z_t = np.r_[np.sin(pitch) * r, -np.cos(pitch)]
        seeds = ([np.asarray(q_seed, dtype=np.float64)] if q_seed is not None else []) + self._target_seeds(
            p_target, yaw_target)
        best: IKResult | None = None
        for k, seed in enumerate(seeds[: max_restarts + 1]):
            x0 = np.clip(seed, self.lower + 1e-9, self.upper - 1e-9)
            ref = x0 if q_ref is None else np.asarray(q_ref, dtype=np.float64)
            use_yaw = yaw_target is not None

            def fun(q):
                return self._residual_and_jac(q, p_target, yaw_target, ref, use_yaw, z_t)[0]

            def jac(q):
                return self._residual_and_jac(q, p_target, yaw_target, ref, use_yaw, z_t)[1]

            sol = least_squares(fun, x0, jac=jac, bounds=(self.lower, self.upper), method="trf",
                                xtol=1e-10, ftol=1e-12, gtol=1e-10, max_nfev=200)
            nfev = sol.nfev
            if z_t is not None:
                # Stage 2 (task priority): the pitch only chose the posture; now drive the
                # position/yaw error to zero from there, staying close to that posture.
                q1 = np.clip(sol.x, self.lower + 1e-9, self.upper - 1e-9)

                def fun2(q, q1=q1):
                    return self._residual_and_jac(q, p_target, yaw_target, q1, use_yaw)[0]

                def jac2(q, q1=q1):
                    return self._residual_and_jac(q, p_target, yaw_target, q1, use_yaw)[1]

                sol = least_squares(fun2, q1, jac=jac2, bounds=(self.lower, self.upper), method="trf",
                                    xtol=1e-12, ftol=1e-14, gtol=1e-12, max_nfev=100)
                nfev += sol.nfev
            res = self._evaluate(sol.x, p_target, yaw_target, nfev)
            if best is None or (res.success, -res.pos_err - res.yaw_err * self.yaw_weight) > (
                best.success, -best.pos_err - best.yaw_err * self.yaw_weight
            ):
                best = res
            if res.success:
                break
        assert best is not None
        return best

    def _azimuth(self, q: np.ndarray) -> float:
        d = self.kin.fk_position(q)[:2] - self.pan_axis_xy
        return float(np.arctan2(d[1], d[0]))

    def _target_seeds(self, p_target: np.ndarray, yaw_target: float | None) -> list[np.ndarray]:
        """Seeds with the pan pointing at the target and the roll matching the requested yaw."""
        d = p_target[:2] - self.pan_axis_xy
        pan = float(wrap_angle(self._pan_sign * (np.arctan2(d[1], d[0]) - self._pan_offset)))
        seeds = []
        for lef in self._postures:
            q = np.r_[pan, lef, 0.0]
            if yaw_target is not None:
                y0 = float(tool_yaw(self.kin.fk(q)[:3, :3]))
                q_r = q.copy()
                q_r[4] = 0.1
                dyaw = float(wrap_angle(tool_yaw(self.kin.fk(q_r)[:3, :3]) - y0)) / 0.1
                if abs(dyaw) > 0.3:
                    q[4] = float(wrap_angle(yaw_target - y0)) / dyaw
            for roll in (q[4], q[4] - 2 * np.pi, q[4] + 2 * np.pi):
                if self.lower[4] <= roll <= self.upper[4]:
                    qq = q.copy()
                    qq[4] = roll
                    seeds.append(np.clip(qq, self.lower, self.upper))
        seeds.append(0.5 * (self.lower + self.upper))
        return seeds

    def _evaluate(self, q, p_target, yaw_target, nfev) -> IKResult:
        T = self.kin.fk(q)
        pos_err = float(np.linalg.norm(T[:3, 3] - p_target))
        yaw_err = 0.0
        yaw_ok = True
        if yaw_target is not None:
            if yaw_observability(T[:3, :3]) < self.min_yaw_observability:
                yaw_ok = False  # yaw undefined at the solution: do not claim success
            yaw_err = float(abs(wrap_angle(tool_yaw(T[:3, :3]) - yaw_target)))
        inside = bool(np.all(q >= self.lower - 1e-9) and np.all(q <= self.upper + 1e-9))
        ok = pos_err <= self.pos_tol and yaw_err <= self.yaw_tol and yaw_ok and inside
        return IKResult(q=q, success=ok, pos_err=pos_err, yaw_err=yaw_err, within_limits=inside, nfev=int(nfev))

    def solve_trajectory(
        self,
        P: np.ndarray,
        yaw: np.ndarray | None,
        q0: np.ndarray,
        max_restarts: int = 3,
        pitch: np.ndarray | None = None,
    ) -> TrajectoryIKResult:
        """Solve a sequence of targets with warm starts (each solution seeds the next).

        Args:
            P: (N, 3) positions [m].
            yaw: (N,) yaw [rad] or None.
            q0: Initial posture [rad] (e.g. the current robot state).
            pitch: (N,) soft approach-pitch targets [rad] or None (default pitch).
        """
        P = np.asarray(P, dtype=np.float64)
        n = len(P)
        Q = np.zeros((n, 5))
        succ = np.zeros(n, bool)
        pe, ye = np.zeros(n), np.zeros(n)
        q_prev = np.asarray(q0, dtype=np.float64)
        for t in range(n):
            r = self.solve(P[t], None if yaw is None else float(yaw[t]), q_seed=q_prev, q_ref=q_prev,
                           max_restarts=max_restarts, pitch=None if pitch is None else float(pitch[t]))
            Q[t], succ[t], pe[t], ye[t] = r.q, r.success, r.pos_err, r.yaw_err
            q_prev = r.q
        steps = np.r_[np.max(np.abs(Q[0] - np.asarray(q0))), np.max(np.abs(np.diff(Q, axis=0)), axis=1)]
        return TrajectoryIKResult(Q, succ, pe, ye, steps)


def ik_from_config(cfg: dict) -> SO101IK:
    """Build FK + IK from the full YAML config (``kinematics`` section)."""
    from so101_tpgmm.config import resolve_path

    k = cfg["kinematics"]
    kin = SO101Kinematics(resolve_path(k["urdf"]), k["base_link"], k["tcp_link"], k["joint_names"])
    lo, hi = joint_limits_from_config(kin, k.get("joint_limits"))
    p = k.get("ik", {})
    default_pitch = p.get("default_pitch_deg", 15.0)
    return SO101IK(
        kin, lo, hi,
        pos_tol=p.get("pos_tol", 1e-3),
        yaw_tol=np.deg2rad(p.get("yaw_tol_deg", 1.0)),
        yaw_weight=p.get("yaw_weight", 0.05),
        reg_weight=p.get("reg_weight", 1e-3),
        min_yaw_observability=p.get("min_yaw_observability", 0.2),
        pitch_weight=p.get("pitch_weight", 0.02),
        default_pitch=None if default_pitch is None else np.deg2rad(default_pitch),
    )
