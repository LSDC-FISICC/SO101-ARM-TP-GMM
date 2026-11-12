"""Geometric baseline: waypoint planner on the same primitives and frames, executed by IK.

Instead of a probabilistic model, each primitive is a short list of Cartesian
waypoints whose positions are expressed in the task frame that matters for them and
whose values are **medians over the demonstrations** (so it is learned from the same
data, with no Gaussians):

* ``approach_grasp`` (from the robot's current pose):
  start → apex (a fraction of the way to the cube, at the median apex height) →
  pre-grasp (cube frame, z_contact + 3 cm) → contact (cube frame).
* ``transport_release``: contact → lift (cube frame) → apex (a fraction of the way from the
  lift point to the release point, at the median transport apex height) → release (box frame).

Heading, pitch and gripper are interpolated between the median values at the same
waypoints. Between waypoints the path is a shape-preserving PCHIP spline, traversed
with a minimum-jerk time law (zero velocity and acceleration at both ends of each
primitive). Output format and time base are those of :class:`TPGMMPolicy`, so both
are evaluated identically and converted to joints by the same IK.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from so101_tpgmm.tpgmm.demos import Demo, choose_cube_heading, rot_z
from so101_tpgmm.tpgmm.policy import Rollout

KEY_HEIGHT = 0.03  # same definition as the segmentation keypoints


def min_jerk(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0, 1)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _local(p: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    return rot_z(yaw).T @ (p - origin)


def _global(p: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    return rot_z(yaw) @ p + origin


@dataclass
class PrimitivePlan:
    """Median waypoint parameters of one primitive."""

    s: list[float]  # path parameters of the waypoints (normalized time in the demos)
    pos: list[tuple[str, list[float]]]  # (frame name, local position) per waypoint ("start" = given)
    h: list[float | None]  # heading offset relative to the cube heading (None = keep start)
    g: list[float | None]
    pitch: list[float | None]


class IKPlanner:
    """Waypoint planner learned from demonstrations (see module docstring)."""

    def __init__(self) -> None:
        self.plans: dict[str, PrimitivePlan] = {}
        self.meta: dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, demos: dict[str, list[Demo]], ctx: dict, durations: dict[str, float]) -> "IKPlanner":
        med = lambda a: float(np.median(np.asarray(a), axis=0)) if np.ndim(a) == 1 else np.median(np.asarray(a), axis=0).tolist()  # noqa: E731
        ap, tr = demos["approach_grasp"], demos["transport_release"]
        # --- approach
        s_apex, lam_apex, z_apex, s_pre, pre_loc, end_loc, h_end, g_open, s_open, g_end, p_end = ([] for _ in range(11))
        for d in ap:
            X = d.X
            z = X[:, 3]
            i_apex = int(np.argmax(z))
            above = np.flatnonzero(z >= z[-1] + KEY_HEIGHT)
            i_pre = int(above[-1]) if above.size else len(z) - 2
            # fraction of the horizontal way from start to contact at the apex
            tot = np.linalg.norm(X[-1, 1:3] - X[0, 1:3]) + 1e-9
            s_apex.append(X[i_apex, 0]); z_apex.append(z[i_apex])
            lam_apex.append(np.clip(np.linalg.norm(X[i_apex, 1:3] - X[0, 1:3]) / tot, 0, 1))
            s_pre.append(X[i_pre, 0])
            pre_loc.append(_local(X[i_pre, 1:4], d.cube[:3], d.cube[3]))
            end_loc.append(_local(X[-1, 1:4], d.cube[:3], d.cube[3]))
            h_end.append(np.angle(np.exp(1j * (X[-1, 4] - d.cube[3]))))
            g_open.append(X[:, 5].max()); s_open.append(X[int(np.argmax(X[:, 5] > 0.5 * X[:, 5].max())), 0])
            g_end.append(X[-1, 5]); p_end.append(X[-1, 6] if X.shape[1] > 6 else 0.0)
        s_apex_m = min(med(s_apex), med(s_pre) - 0.05)
        self.plans["approach_grasp"] = PrimitivePlan(
            s=[0.0, s_apex_m, med(s_pre), 1.0],
            pos=[("start", []), ("apex", [med(lam_apex), med(z_apex), 0, 2]), ("cube", med(pre_loc)), ("cube", med(end_loc))],
            h=[None, None, med(h_end), med(h_end)],
            g=[None, med(g_open), med(g_open), med(g_end)],
            pitch=[None, None, med(p_end), med(p_end)],
        )
        self.meta["approach_open_s"] = med(s_open)
        # --- transport
        s_lift, lift_loc, s_top, lam_top, z_top, end_box, h_rel, g_hold, s_rel, g_rel, p_rel = ([] for _ in range(11))
        for d in tr:
            X = d.X
            z = X[:, 3]
            up = np.flatnonzero(z >= z[0] + KEY_HEIGHT)
            i_lift = int(up[0]) if up.size else 1
            i_top = int(np.argmax(z))
            s_lift.append(X[i_lift, 0]); lift_loc.append(_local(X[i_lift, 1:4], d.cube[:3], d.cube[3]))
            s_top.append(X[i_top, 0]); z_top.append(z[i_top])
            tot = np.linalg.norm(X[-1, 1:3] - X[i_lift, 1:3]) + 1e-9
            lam_top.append(np.clip(np.linalg.norm(X[i_top, 1:3] - X[i_lift, 1:3]) / tot, 0, 1))
            end_box.append(_local(X[-1, 1:4], d.box[:3], d.box[3]))
            h_rel.append(np.angle(np.exp(1j * (X[-1, 4] - d.box[3]))))
            g_hold.append(X[0, 5])
            opening = np.flatnonzero(X[:, 5] > X[0, 5] + 3)
            s_rel.append(X[int(opening[0]), 0] if opening.size else 0.9)
            g_rel.append(X[-1, 5]); p_rel.append(X[-1, 6] if X.shape[1] > 6 else 0.0)
        s_lift_m, s_top_m = med(s_lift), med(s_top)
        s_top_m = float(np.clip(s_top_m, s_lift_m + 0.05, 0.95))
        self.plans["transport_release"] = PrimitivePlan(
            s=[0.0, s_lift_m, s_top_m, 1.0],
            pos=[("start", []), ("cube", med(lift_loc)), ("apex", [med(lam_top), med(z_top), 1, 3]), ("box", med(end_box))],
            h=[None, None, ("box", med(h_rel)), ("box", med(h_rel))],
            g=[None, med(g_hold), med(g_hold), med(g_rel)],
            pitch=[None, None, med(p_rel), med(p_rel)],
        )
        self.meta.update(transport_release_open_s=med(s_rel), durations=durations, box=ctx["box"],
                         heading_offset=ctx["heading_offset"], pan_xy=ctx["pan_xy"], var_names=ctx["var_names"])
        return self

    # ------------------------------------------------------------------ rollout
    def _primitive(self, name: str, start: np.ndarray, cube: np.ndarray, box: np.ndarray, n: int) -> np.ndarray:
        plan = self.plans[name]
        O = len(start)
        W = np.zeros((len(plan.s), O))
        for i, (frame, loc) in enumerate(plan.pos):
            if frame == "start":
                W[i, :3] = start[:3]
            elif frame == "cube":
                W[i, :3] = _global(np.asarray(loc), cube[:3], cube[3])
            elif frame == "box":
                W[i, :3] = _global(np.asarray(loc), box[:3], box[3])
        for i, (frame, loc) in enumerate(plan.pos):  # apex: a fraction of the way between two waypoints
            if frame == "apex":
                lam, z, a, b = loc
                W[i, :2] = W[int(a), :2] + lam * (W[int(b), :2] - W[int(a), :2])
                W[i, 2] = max(z, W[int(a), 2], W[int(b), 2])
        for col, values, base in ((3, plan.h, cube[3]), (4, plan.g, 0.0), (5, plan.pitch, 0.0)):
            if col >= O:
                continue
            for i, v in enumerate(values):
                if v is None:
                    W[i, col] = W[i - 1, col] if i else start[col]
                elif isinstance(v, (tuple, list)):  # (frame, offset)
                    W[i, col] = box[3] + v[1]
                else:
                    W[i, col] = base + v if col == 3 else v
        # express every heading waypoint within +-pi of the current heading (no 2*pi jumps)
        W[:, 3] = start[3] + np.angle(np.exp(1j * (W[:, 3] - start[3])))
        tau = min_jerk(np.linspace(0, 1, n))
        return np.stack([PchipInterpolator(plan.s, W[:, j])(tau) for j in range(O)], axis=1)

    def rollout(self, cube_xyz: np.ndarray, cube_yaw: float, start: np.ndarray, box: np.ndarray | None = None,
                dt: float = 1 / 30, n_per_primitive: int | None = None, durations: dict[str, float] | None = None,
                heading: float | None = None, **_: object) -> Rollout:
        """Same interface as :meth:`TPGMMPolicy.rollout` (``start`` is required here)."""
        from so101_tpgmm.tpgmm.demos import canonical_cube_yaw

        if heading is None:
            phi_c, k = choose_cube_heading(cube_yaw, np.asarray(cube_xyz)[:2], np.asarray(self.meta["pan_xy"]),
                                           self.meta["heading_offset"])
        else:
            phi_c, k = canonical_cube_yaw(cube_yaw, heading)
        cube = np.r_[np.asarray(cube_xyz, dtype=float)[:3], phi_c]
        box = np.asarray(self.meta["box"] if box is None else box, dtype=float)
        durations = durations or self.meta["durations"]
        Ys, Ts, P, bounds = [], [], [], []
        t0, cur = 0.0, np.asarray(start, dtype=float)
        for i, name in enumerate(("approach_grasp", "transport_release")):
            n = n_per_primitive or max(2, int(round(durations[name] / dt)) + 1)
            Y = self._primitive(name, cur, cube, box, n)
            tt = t0 + np.linspace(0, durations[name], n)
            if i > 0:
                Y, tt = Y[1:], tt[1:]
            Ys.append(Y), Ts.append(tt), P.append(np.full(len(Y), i))
            bounds.append(sum(len(y) for y in Ys) - 1)
            t0, cur = tt[-1], Y[-1]
        Y = np.vstack(Ys)
        return Rollout(np.concatenate(Ts), Y, np.zeros((len(Y), Y.shape[1], Y.shape[1])), np.concatenate(P),
                       self.meta["var_names"][1:], bounds, float(phi_c), k)

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": self.meta, "plans": {k: v.__dict__ for k, v in self.plans.items()}}, f, indent=1)

    @classmethod
    def load(cls, path: str | Path) -> "IKPlanner":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        p = cls()
        p.meta = d["meta"]
        p.plans = {k: PrimitivePlan(**v) for k, v in d["plans"].items()}
        return p
