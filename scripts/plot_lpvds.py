#!/usr/bin/env python
"""Figure of the learned LPV-DS: scheduling Gaussians, field and rollouts vs demonstrations.

For each primitive (approach_grasp in the cube frame, transport_release in the box frame):

* left: x–z slice through the attractor with the streamlines of ẋ = Σ_k γ_k(x) A_k (x − x*) and the
  scheduling Gaussians (position marginals, 2σ ellipses projected on x–z, numbered); each
  component's linear system A_k is what the field follows where γ_k dominates;
* middle / right: the DS integrated in 3-D (with the policy's speed floor and ceiling) from the start
  of each demonstration, against that demonstration, in x–z and x–y.

Uses the model saved by ``scripts/train_lpvds.py`` (outputs/models/lpv_ds.json).
Writes docs/figures/lpvds_gaussians_rollouts.png.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402
from so101_tpgmm.data.lerobot_dataset import LeRobotV3Dataset  # noqa: E402
from so101_tpgmm.kinematics.fk import SO101Kinematics  # noqa: E402
from so101_tpgmm.lpvds.policy import FRAME, LPVDSPolicy, PrimitiveDS  # noqa: E402
from so101_tpgmm.tpgmm.demos import build_demos, training_table  # noqa: E402

DEMO_C, DS_C, GAUSS_C = "0.62", "#2a6fdb", "#d1495b"


def integrate(p: PrimitiveDS, x0: np.ndarray, dt: float = 1 / 30, tol: float = 0.004, t_max: float = 12.0) -> np.ndarray:
    """Integrate the position DS with the policy's speed floor / ceiling (local frame)."""
    x, out = x0.copy(), [x0.copy()]
    for _ in range(int(t_max / dt)):
        v = p.pos.velocity(x)[0]
        n, dist = np.linalg.norm(v), np.linalg.norm(x - p.pos.target)
        floor = min(p.v_min, dist / 0.15)
        if 0 < n < floor:
            v *= floor / n
        elif n > p.v_max:
            v *= p.v_max / n
        x = x + dt * v
        out.append(x.copy())
        if dist < tol:
            break
    return np.array(out)


def ellipse(ax, mean, cov, i, j, label):
    w, V = np.linalg.eigh(cov[np.ix_([i, j], [i, j])])
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    w2, h2 = 2 * 2 * np.sqrt(np.maximum(w, 0)) * 100  # 2σ, cm
    ax.add_patch(Ellipse((mean[i] * 100, mean[j] * 100), w2, h2, angle=ang, fill=False, ec=GAUSS_C, lw=1.4))
    ax.text(mean[i] * 100, mean[j] * 100, label, color=GAUSS_C, fontsize=8, ha="center", va="center",
            fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n-demos", type=int, default=25, help="demonstrations / rollouts drawn per primitive")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    ds = LeRobotV3Dataset(resolve_path(cfg["dataset"]["local_dir"]), cfg["dataset"]["state_key"])
    kin = SO101Kinematics(resolve_path(cfg["kinematics"]["urdf"]))
    df = training_table(pd.read_csv(resolve_path(cfg["segmentation"]["output_csv"])),
                        pd.read_csv(resolve_path(cfg["vision"]["frames_csv"])), cfg["tpgmm"]["exclude_aborted"])
    demos, _ = build_demos(df, ds, kin, cfg, ["approach_grasp", "transport_release"])
    pol = LPVDSPolicy.load(resolve_path(cfg["tpgmm"]["models_dir"]) / "lpv_ds.json")

    fig, axs = plt.subplots(2, 3, figsize=(17, 10.5), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    rng = np.random.default_rng(0)
    for row, name in enumerate(pol.primitives):
        p, m = pol.prims[name], pol.prims[name].pos
        L = [d.local()[FRAME[name]][:, 1:4] for d in demos[name]]
        sel = rng.choice(len(L), min(args.n_demos, len(L)), replace=False)
        frame = "cube" if name == "approach_grasp" else "box"
        allp = np.vstack(L)
        lo, hi = np.percentile(allp, 1, axis=0) - 0.03, np.percentile(allp, 99, axis=0) + 0.03

        # field slice + Gaussians
        ax = axs[row, 0]
        xs, zs = np.linspace(lo[0], hi[0], 60), np.linspace(lo[2], hi[2], 60)
        XX, ZZ = np.meshgrid(xs, zs)
        G = np.tile(m.target, (XX.size, 1))
        G[:, 0], G[:, 2] = XX.ravel(), ZZ.ravel()
        V = m.velocity(G)
        ax.streamplot(xs * 100, zs * 100, V[:, 0].reshape(XX.shape), V[:, 2].reshape(XX.shape), color="0.55",
                      density=1.1, linewidth=0.6, arrowsize=0.8)
        for k in range(m.gmm.n_components):
            ellipse(ax, m.gmm.means[k], m.gmm.covs[k], 0, 2, str(k + 1))
        ax.plot(m.target[0] * 100, m.target[2] * 100, "*", c="gold", ms=15, mec="k", zorder=5)
        ax.set_title(f"{name} ({frame} frame): field slice y = y* and the {m.gmm.n_components} scheduling "
                     "Gaussians (2σ)", fontsize=9)
        ax.set_xlabel("x [cm]"), ax.set_ylabel("z [cm]")
        ax.set_xlim(xs[0] * 100, xs[-1] * 100), ax.set_ylim(zs[0] * 100, zs[-1] * 100)

        # rollouts vs demonstrations
        err = []
        for c, (i, j, lab) in zip((1, 2), ((0, 2, ("x", "z")), (0, 1, ("x", "y")))):
            ax = axs[row, c]
            for n_, s in enumerate(sel):
                tr = L[s]
                ro = integrate(p, tr[0])
                if c == 1:
                    err.append(np.linalg.norm(ro[-1] - m.target))
                ax.plot(tr[:, i] * 100, tr[:, j] * 100, c=DEMO_C, lw=0.8,
                        label="demonstration" if n_ == 0 else None)
                ax.plot(ro[:, i] * 100, ro[:, j] * 100, c=DS_C, lw=1.0, alpha=0.85,
                        label="LPV-DS from the same start" if n_ == 0 else None)
                ax.plot(tr[0, i] * 100, tr[0, j] * 100, "o", c="k", ms=3)
            ax.plot(m.target[i] * 100, m.target[j] * 100, "*", c="gold", ms=15, mec="k", zorder=5)
            ax.set_xlabel(f"{lab[0]} [cm]"), ax.set_ylabel(f"{lab[1]} [cm]")
            ax.set_title(f"{name}: 3-D rollouts, projected on {lab[0]}–{lab[1]}", fontsize=9)
            if lab == ("x", "y"):
                ax.set_aspect("equal", adjustable="datalim")
            ax.legend(fontsize=8, loc="best", frameon=False)
        print(f"{name}: K={m.gmm.n_components}, rollouts ending < 4 mm: {np.mean(np.array(err) < 0.0045):.0%}")
        for ax in axs[row]:
            ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("Learned LPV-DS: each Gaussian schedules one stable linear system A_k (x − x*); "
                 "the field blends them by γ_k(x). ★ = attractor, • = start of the demonstration", fontsize=10)
    fig.tight_layout()
    out = resolve_path("docs/figures/lpvds_gaussians_rollouts.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
