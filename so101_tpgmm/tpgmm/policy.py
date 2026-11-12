"""Policies that turn a cube pose into a Cartesian trajectory [x, y, z, h, g, pitch].

* :class:`TPGMMPolicy` — one TP-GMM per primitive (segmented, main model) or a single
  TP-GMM over the whole cycle (monolithic, baseline). Primitives are chained; the start of
  each one is blended into the end of the previous one (and the first one into the
  robot's current pose) over ``boundary_blend`` of its duration, so the trajectory is
  continuous.
* Durations: median demonstrated duration per primitive (time is normalized inside
  the models).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from so101_tpgmm.tpgmm.demos import choose_cube_heading, make_frame
from so101_tpgmm.tpgmm.tpgmm import TPGMM, load_models, save_models


@dataclass
class Rollout:
    """A generated trajectory (outputs exclude time)."""

    t: np.ndarray  # (N,) seconds
    Y: np.ndarray  # (N, O) [x, y, z, h, g, (pitch)]
    cov: np.ndarray  # (N, O, O)
    primitive: np.ndarray  # (N,) primitive index per sample
    names: list[str]
    boundaries: list[int] = field(default_factory=list)  # sample index where each primitive ends
    cube_heading: float = 0.0
    cube_k: int = 0

    def segment(self, i: int) -> np.ndarray:
        return self.Y[self.primitive == i]


def blend_start(Y: np.ndarray, start: np.ndarray, frac: float, angular: tuple[int, ...] = ()) -> np.ndarray:
    """Shift the beginning of ``Y`` so it starts at ``start``, fading the offset out over ``frac``."""
    off = start - Y[0]
    for a in angular:
        off[a] = (off[a] + np.pi) % (2 * np.pi) - np.pi
    n = len(Y)
    w = np.clip(1.0 - np.arange(n) / max(frac * (n - 1), 1e-9), 0.0, 1.0)
    w = w * w * (3 - 2 * w)  # smoothstep
    return Y + w[:, None] * off[None, :]


class TPGMMPolicy:
    """Chained TP-GMM primitives (segmented) or a single one (monolithic)."""

    def __init__(self, models: dict[str, TPGMM], meta: dict) -> None:
        self.models = models
        self.meta = meta
        self.primitives: list[str] = meta["primitives"]
        self.durations: dict[str, float] = meta["durations"]

    @property
    def var_names(self) -> list[str]:
        return self.meta["var_names"][1:]

    def cube_frame(self, cube_xyz: np.ndarray, cube_yaw: float, heading: float | None = None) -> tuple[np.ndarray, int]:
        """Canonical cube frame (x, y, z, phi') for execution; ``heading`` overrides the k choice."""
        if heading is None:
            phi_c, k = choose_cube_heading(cube_yaw, np.asarray(cube_xyz)[:2], np.asarray(self.meta["pan_xy"]),
                                           self.meta["heading_offset"])
        else:
            from so101_tpgmm.tpgmm.demos import canonical_cube_yaw

            phi_c, k = canonical_cube_yaw(cube_yaw, heading)
        return np.r_[np.asarray(cube_xyz, dtype=float)[:3], phi_c], k

    def predict_primitive(self, name: str, cube: np.ndarray, box: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        m = self.models[name]
        frames = [make_frame(cube[:3], cube[3], m.dim), make_frame(box[:3], box[3], m.dim)]
        return m.reproduce(frames, np.linspace(0, 1, n))

    def rollout(self, cube_xyz: np.ndarray, cube_yaw: float, start: np.ndarray | None = None,
                box: np.ndarray | None = None, dt: float = 1 / 30, n_per_primitive: int | None = None,
                durations: dict[str, float] | None = None, heading: float | None = None,
                blend: float | None = None) -> Rollout:
        """Generate the full trajectory for one cube.

        Args:
            cube_xyz, cube_yaw: cube centre [m] and yaw [rad] (any of the 4 equivalent yaws).
            start: current [x, y, z, h, g, (pitch)] of the robot; the first primitive is blended into it.
            box: box frame (x, y, z, yaw); default: the training box.
            dt: sample period when ``n_per_primitive`` is None.
            durations: per-primitive durations [s] (default: training medians).
            heading: known grasp heading (only for evaluation of the k choice); default: predicted.
            blend: boundary blending fraction (default: from training config).
        """
        cube, k = self.cube_frame(cube_xyz, cube_yaw, heading)
        box = np.asarray(self.meta["box"] if box is None else box, dtype=float)
        durations = durations or self.durations
        blend = self.meta.get("boundary_blend", 0.3) if blend is None else blend
        ang = (self.var_names.index("h"),)
        Ys, Cs, Ts, P, bounds = [], [], [], [], []
        t0, prev_end = 0.0, start
        for i, name in enumerate(self.primitives):
            n = n_per_primitive or max(2, int(round(durations[name] / dt)) + 1)
            Y, C = self.predict_primitive(name, cube, box, n)
            if prev_end is not None:
                Y = blend_start(Y, np.asarray(prev_end, dtype=float), blend, ang)
            tt = t0 + np.linspace(0, durations[name], n)
            if i > 0:  # drop the duplicated boundary sample
                Y, C, tt = Y[1:], C[1:], tt[1:]
            Ys.append(Y), Cs.append(C), Ts.append(tt), P.append(np.full(len(Y), i))
            bounds.append(sum(len(y) for y in Ys) - 1)
            t0, prev_end = tt[-1], Y[-1]
        return Rollout(np.concatenate(Ts), np.vstack(Ys), np.concatenate(Cs), np.concatenate(P), self.var_names,
                       bounds, float(cube[3]), k)

    def save(self, path: str | Path) -> None:
        save_models(path, self.models, self.meta)

    @classmethod
    def load(cls, path: str | Path) -> "TPGMMPolicy":
        models, meta = load_models(path)
        return cls(models, meta)
