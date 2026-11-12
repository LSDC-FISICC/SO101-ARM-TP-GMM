"""LPV-DS policy on the same primitives and task frames as the TP-GMM.

* ``approach_grasp``: position LPV-DS in the **cube frame** (attractor = median contact point),
  heading and pitch as stable scalar DS scheduled by the same γ(x), gripper as a state-dependent
  map g = E[g | max(‖x − x*‖, gate)] (GMR): it opens on the way, stays open inside the last
  ``close_gate`` and closes only on arrival — because the arm arrived, not because time ran out.
* ``transport_release``: the same in the **box frame** (attractor = median release point).

Speed: the learned field is scaled by a positive, state-dependent factor — a floor equal to the
demonstrated median speed 5–30 mm from the end (fading out within the last ~5 mm) and a ceiling
at the 99th percentile of demonstrated speeds. Positive scaling does not change the flow
direction, so the Lyapunov decrease (stability) is preserved; it removes the slow exponential
tail of a linear DS near its attractor.

A primitive ends when the TCP is within ``end_tolerance`` of its attractor (or after
``max_duration_factor`` × the median demonstrated duration), followed by ``settle_s`` for the
gripper. The commanded speed is saturated at the 99th percentile of the demonstrated speeds.

This is a closed-loop controller: :meth:`step` returns the state derivative for the current
state, which is what the perturbation benchmark integrates; :meth:`rollout` integrates it from a
start state for the offline evaluation and the inference pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from so101_tpgmm.lpvds.lpvds import LPVDS, ScalarDS, fit_lpvds
from so101_tpgmm.tpgmm.demos import Demo, canonical_cube_yaw, choose_cube_heading, rot_z
from so101_tpgmm.tpgmm.ds import fit_gmm
from so101_tpgmm.tpgmm.gmm import GMM, gmr
from so101_tpgmm.tpgmm.policy import Rollout

FRAME = {"approach_grasp": 0, "transport_release": 1}  # 0 = cube, 1 = box


@dataclass
class PrimitiveDS:
    pos: LPVDS
    heading: ScalarDS
    pitch: ScalarDS
    gripper: GMM  # joint GMM on (distance [m], g)
    v_max: float
    median_duration: float
    v_min: float = 0.0  # speed floor near the target (demonstrated median speed 5-30 mm from the end)
    g_final: float | None = None  # gripper command on arrival: median demonstrated final value (hold / release)

    def to_dict(self) -> dict:
        return {"pos": self.pos.to_dict(), "heading": [self.heading.log_gain.tolist(), self.heading.target],
                "pitch": [self.pitch.log_gain.tolist(), self.pitch.target],
                "gripper": [self.gripper.priors.tolist(), self.gripper.means.tolist(), self.gripper.covs.tolist()],
                "v_max": self.v_max, "median_duration": self.median_duration, "v_min": self.v_min,
                "g_final": self.g_final}

    @classmethod
    def from_dict(cls, d: dict) -> "PrimitiveDS":
        g = d["gripper"]
        return cls(LPVDS.from_dict(d["pos"]), ScalarDS(np.array(d["heading"][0]), d["heading"][1]),
                   ScalarDS(np.array(d["pitch"][0]), d["pitch"][1]), GMM(np.array(g[0]), np.array(g[1]), np.array(g[2])),
                   d["v_max"], d["median_duration"], d.get("v_min", 0.0), d.get("g_final"))

    def gripper_at(self, dist: float) -> float:
        m, _ = gmr(self.gripper, np.array([[dist]]), [0], [1])
        return float(m[0, 0])

    def gripper_on_arrival(self) -> float:
        """Value commanded once the attractor is reached. Not g(0) of the smooth map: near the contact the
        demonstrations mix still-open and already-closed jaws, and the regression averages them (~43
        instead of the ~20 holding level)."""
        return self.g_final if self.g_final is not None else self.gripper_at(0.0)


def fit_primitive(demos: list[Demo], name: str, K: int | None = None) -> tuple[PrimitiveDS, dict]:
    fr = FRAME[name]
    P, V, H, Hd, T, Td, E, Gd = [], [], [], [], [], [], [], []
    for d in demos:
        L = d.local()[fr]
        dt = d.duration / (len(L) - 1)
        P.append(L[:, 1:4]), V.append(np.gradient(L[:, 1:4], dt, axis=0))
        H.append(L[:, 4]), Hd.append(np.gradient(L[:, 4], dt))
        T.append(L[:, 6]), Td.append(np.gradient(L[:, 6], dt))
        E.append(L[-1])
    P, V = np.vstack(P), np.vstack(V)
    E = np.array(E)
    target = np.median(E[:, 1:4], axis=0)
    model, info = fit_lpvds(P, V, target, k_range=(K,) if K else (3, 4, 5, 6, 8))
    G = model.gamma(P)
    heading = ScalarDS.fit(G, np.concatenate(H), np.concatenate(Hd), float(np.median(E[:, 4])))
    pitch = ScalarDS.fit(G, np.concatenate(T), np.concatenate(Td), float(np.median(E[:, 6])))
    dist = np.linalg.norm(P - target, axis=1)
    g_all = np.concatenate([d.X[:, 5] for d in demos])
    gm, _ = fit_gmm(np.c_[dist, g_all], 6, np.array([1e-6, 1.0]))
    v_max = float(np.percentile(np.linalg.norm(V, axis=1), 99))
    near = (dist > 0.005) & (dist < 0.03)
    v_min = float(np.median(np.linalg.norm(V[near], axis=1))) if near.any() else 0.0
    info.update(target=target.tolist())
    g_final = float(np.median([d.X[-1, 5] for d in demos]))
    return (PrimitiveDS(model, heading, pitch, gm, v_max, float(np.median([d.duration for d in demos])), v_min, g_final),
            info)


class GripperLatch:
    """Once the jaws have opened during a primitive, keep them open until arrival.

    The state-dependent map g(d) is smooth: if a perturbation pushes the arm away when it is already
    near the cube, g(d) at the new distance can be almost closed and the jaws would close in the air.
    The operator never did that (open once, close on the cube), so the command is latched at 90 % of
    the largest opening reached so far; arrival (settle phase) releases the latch.
    """

    def __init__(self, threshold: float = 25.0) -> None:
        self.peak, self.threshold = 0.0, threshold

    def __call__(self, g: float) -> float:
        self.peak = max(self.peak, g)
        return max(g, 0.9 * self.peak) if self.peak >= self.threshold else g


class LPVDSPolicy:
    """Closed-loop LPV-DS policy (see module docstring)."""

    primitives = ["approach_grasp", "transport_release"]

    def __init__(self, prims: dict[str, PrimitiveDS], meta: dict) -> None:
        self.prims = prims
        self.meta = meta

    @classmethod
    def fit(cls, demos: dict[str, list[Demo]], ctx: dict, k_fixed: dict[str, int] | None = None,
            end_tolerance: float = 0.004, settle_s: float = 0.3, max_duration_factor: float = 2.5,
            close_gate: float = 0.02) -> "LPVDSPolicy":
        prims, infos = {}, {}
        for name in cls.primitives:
            prims[name], infos[name] = fit_primitive(demos[name], name, (k_fixed or {}).get(name))
        meta = {**ctx, "fit_info": infos, "end_tolerance": end_tolerance, "settle_s": settle_s,
                "max_duration_factor": max_duration_factor, "close_gate": close_gate}
        return cls(prims, meta)

    @property
    def var_names(self) -> list[str]:
        return self.meta["var_names"][1:]

    # ------------------------------------------------------------------ frames
    def frames(self, cube_xyz, cube_yaw, heading=None, box=None):
        if heading is None:
            phi, k = choose_cube_heading(cube_yaw, np.asarray(cube_xyz)[:2], np.asarray(self.meta["pan_xy"]),
                                         self.meta["heading_offset"])
        else:
            phi, k = canonical_cube_yaw(cube_yaw, heading)
        box = np.asarray(self.meta["box"] if box is None else box, float)
        return {"approach_grasp": (np.asarray(cube_xyz, float)[:3], phi), "transport_release": (box[:3], box[3])}, k

    # ------------------------------------------------------------------ dynamics
    def step(self, name: str, state: np.ndarray, frame: tuple[np.ndarray, float]) -> tuple[np.ndarray, float, float, float]:
        """(ẋ, ḣ, θ̇) in the base frame and the commanded gripper, for state [x, y, z, h, g, pitch]."""
        p = self.prims[name]
        o, yaw = frame
        R = rot_z(yaw)
        xl = R.T @ (state[:3] - o)
        G = p.pos.gamma(xl)
        v = R @ p.pos.velocity(xl)[0]
        n = np.linalg.norm(v)
        dist = float(np.linalg.norm(xl - p.pos.target))
        # Positive, state-dependent speed scaling keeps the flow direction, hence V̇ < 0 (stability):
        # a floor that fades out in the last millimetres (no overshoot) and a ceiling.
        floor = min(p.v_min, dist / 0.15)
        if 0 < n < floor:
            v *= floor / n
        elif n > p.v_max:
            v *= p.v_max / n
        hdot = float(p.heading.rate(G, state[3] - yaw)[0])
        tdot = float(p.pitch.rate(G, state[5])[0])
        # Gripper: state-dependent, but it only closes on arrival. Within ``close_gate`` of the attractor it
        # keeps the value it had at the gate (open); the closing happens in the settle phase at the target.
        gate = self.meta.get("close_gate", 0.02)
        g = p.gripper_at(max(dist, gate))
        return v, hdot, tdot if np.isfinite(tdot) else 0.0, g

    def distance(self, name: str, state: np.ndarray, frame) -> float:
        o, yaw = frame
        return float(np.linalg.norm(rot_z(yaw).T @ (state[:3] - o) - self.prims[name].pos.target))

    def rollout(self, cube_xyz, cube_yaw, start, box=None, dt: float = 1 / 30, heading=None, **_) -> Rollout:
        frames, k = self.frames(cube_xyz, cube_yaw, heading, box)
        state = np.asarray(start, float).copy()
        Ys, P, bounds, t = [state.copy()], [0], [], 0.0
        for i, name in enumerate(self.primitives):
            p = self.prims[name]
            t_max = self.meta["max_duration_factor"] * p.median_duration
            t0, settle = t, None
            latch = GripperLatch() if name == "approach_grasp" else None
            while True:
                v, hd, td, g = self.step(name, state, frames[name])
                if latch is not None:
                    g = latch(g)
                if settle is not None:  # arrived: stop and actuate the gripper (close / open)
                    v, hd, td, g = np.zeros(3), 0.0, 0.0, p.gripper_on_arrival()
                state[:3] += dt * v
                state[3] += dt * hd
                state[5] += dt * td
                state[4] += np.clip(g - state[4], -dt * 200, dt * 200)  # gripper rate limit 200 units/s
                t += dt
                Ys.append(state.copy()), P.append(i)
                if settle is None and (self.distance(name, state, frames[name]) < self.meta["end_tolerance"]
                                       or t - t0 > t_max):
                    settle = t
                if settle is not None and t - settle >= self.meta["settle_s"]:
                    break
            bounds.append(len(Ys) - 1)
        Y = np.array(Ys)
        return Rollout(np.arange(len(Y)) * dt, Y, np.zeros((len(Y), 6, 6)), np.array(P), self.var_names, bounds,
                       float(frames["approach_grasp"][1]), k)

    # ------------------------------------------------------------------ io
    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": self.meta, "prims": {k: v.to_dict() for k, v in self.prims.items()}}, f)

    @classmethod
    def load(cls, path) -> "LPVDSPolicy":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls({k: PrimitiveDS.from_dict(v) for k, v in d["prims"].items()}, d["meta"])
