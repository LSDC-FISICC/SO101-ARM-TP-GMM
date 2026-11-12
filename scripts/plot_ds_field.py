#!/usr/bin/env python
"""Vector field and divergence of a GMR dynamical system learned from the demonstrations.

For each primitive, in the frame that matters for it (approach_grasp: cube frame;
transport_release: box frame) and for two 2-D projections, a GMM is fitted on
[position, velocity] and the field f(x) = E[ẋ | x] is drawn as streamlines over its
divergence. Regions where the position density of the model is low (no data) are masked.
The field is also integrated from every demonstration's start to check convergence.

Writes docs/figures/ds_field.png and docs/ds_field.md.
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
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402
from so101_tpgmm.data.lerobot_dataset import LeRobotV3Dataset  # noqa: E402
from so101_tpgmm.kinematics.fk import SO101Kinematics  # noqa: E402
from so101_tpgmm.tpgmm.demos import build_demos, training_table  # noqa: E402
from so101_tpgmm.tpgmm.ds import density, divergence, field, fit_ds, integrate  # noqa: E402

AX = {"x": 1, "y": 2, "z": 3}


def local_pos_vel(demos, frame: int, dims: tuple[str, str]):
    P, V, starts, ends = [], [], [], []
    for d in demos:
        L = d.local()[frame][:, [AX[a] for a in dims]]
        dt = d.duration / (len(L) - 1)
        P.append(L), V.append(np.gradient(L, dt, axis=0)), starts.append(L[0]), ends.append(L[-1])
    return np.vstack(P), np.vstack(V), np.array(starts), np.array(ends), P


def panel(ax, demos, frame, dims, title, results):
    pos, vel, starts, ends, trajs = local_pos_vel(demos, frame, dims)
    g, bic = fit_ds(pos, vel)
    target = np.median(ends, axis=0)
    lo, hi = np.percentile(pos, 1, axis=0) - 0.03, np.percentile(pos, 99, axis=0) + 0.03
    xs, ys = np.linspace(lo[0], hi[0], 120), np.linspace(lo[1], hi[1], 120)
    XX, YY = np.meshgrid(xs, ys)
    G = np.c_[XX.ravel(), YY.ravel()]
    F = field(g, G)
    U, V = F[:, 0].reshape(XX.shape), F[:, 1].reshape(XX.shape)
    div = divergence(U, V, xs, ys)
    dens = density(g, G).reshape(XX.shape)
    mask = dens < 0.01 * dens.max()
    lim = np.nanpercentile(np.abs(div[~mask]), 98)
    im = ax.pcolormesh(xs * 100, ys * 100, np.ma.array(div, mask=mask), cmap="RdBu_r",
                       norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
    ax.contourf(xs * 100, ys * 100, mask, levels=[0.5, 1.5], colors=["0.92"])
    for tr in trajs[::6]:
        ax.plot(tr[:, 0] * 100, tr[:, 1] * 100, c="0.55", lw=0.4)
    ax.streamplot(xs * 100, ys * 100, np.where(mask, np.nan, U), np.where(mask, np.nan, V),
                  color="k", density=1.1, linewidth=0.6, arrowsize=0.8)
    conv = []
    for s in starts:
        x = integrate(g, s, T=10.0)
        conv.append(np.linalg.norm(x[-1] - target))
    for s in starts[::40]:
        x = integrate(g, s, T=10.0)
        ax.plot(x[:, 0] * 100, x[:, 1] * 100, c="#2a9d4b", lw=1.6)
    ax.plot(*(target * 100), "*", c="gold", ms=14, mec="k")
    ax.set_xlabel(f"{dims[0]} [cm]"), ax.set_ylabel(f"{dims[1]} [cm]")
    if dims == ("x", "y"):
        ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8, label="divergence [1/s]")
    conv = np.array(conv)
    near = div[~mask][np.linalg.norm(G[~mask.ravel()] - target, axis=1) < 0.02]
    results.append(dict(panel=title, K=g.n_components, converge_2cm=float(np.mean(conv < 0.02)),
                        converge_5cm=float(np.mean(conv < 0.05)), median_end_dist_cm=float(np.median(conv) * 100),
                        div_near_target=float(np.median(near)) if near.size else float("nan")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    ds = LeRobotV3Dataset(resolve_path(cfg["dataset"]["local_dir"]), cfg["dataset"]["state_key"])
    kin = SO101Kinematics(resolve_path(cfg["kinematics"]["urdf"]))
    df = training_table(pd.read_csv(resolve_path(cfg["segmentation"]["output_csv"])),
                        pd.read_csv(resolve_path(cfg["vision"]["frames_csv"])), cfg["tpgmm"]["exclude_aborted"])
    demos, _ = build_demos(df, ds, kin, cfg, ["approach_grasp", "transport_release"])
    results = []
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    panel(axs[0, 0], demos["approach_grasp"], 0, ("x", "z"), "approach_grasp, cube frame: x (outward) – z", results)
    panel(axs[0, 1], demos["approach_grasp"], 0, ("x", "y"), "approach_grasp, cube frame: x – y (top view)", results)
    panel(axs[1, 0], demos["transport_release"], 1, ("x", "y"), "transport_release, box frame: x – y (top view)", results)
    panel(axs[1, 1], demos["transport_release"], 1, ("x", "z"), "transport_release, box frame: x – z", results)
    fig.suptitle("GMR dynamical system f(x) = E[ẋ | x]: streamlines over divergence (red: expansion, blue: "
                 "contraction); grey = no data; green = field integrated from demo starts; ★ = target", fontsize=10)
    fig.tight_layout(); fig.savefig(resolve_path("docs/figures/ds_field.png"), dpi=100); plt.close(fig)

    r = pd.DataFrame(results)
    L = ["# Learned vector field (GMR dynamical system)\n",
         "_Generated by `scripts/plot_ds_field.py`._\n",
         "The TP-GMM of this project is time-indexed: its Gaussians are joint in (t, position) and the regression is "
         "on time, so they do not define a vector field. A dynamical system is obtained from the **same "
         "demonstrations in the same task frames** by fitting a GMM on [position, velocity] and conditioning on the "
         "position: f(x) = E[ẋ | x] (GMR-DS; SEDS is the same model with stability constraints). One 2-D model per "
         "panel, K by BIC.\n",
         "![](figures/ds_field.png)\n",
         "| panel | K | starts converging within 2 cm / 5 cm | median final distance | median divergence within 2 cm of the target |",
         "|---|---|---|---|---|"]
    for x in r.itertuples():
        L.append(f"| {x.panel} | {x.K} | {100 * x.converge_2cm:.0f} % / {100 * x.converge_5cm:.0f} % | "
                 f"{x.median_end_dist_cm:.1f} cm | {x.div_near_target:.2f} 1/s |")
    L += ["",
          "**Reading.** Negative divergence (blue) marks where the flow contracts — trajectories from nearby states "
          "merge — which is what a reaching motion towards the cube or the box should show near its target. The "
          "table checks it by integrating the field from every demonstrated start. Because GMR-DS has no stability "
          "constraint, convergence is an empirical property here, not a guarantee: where the integration stops "
          "short or drifts, the field has a spurious equilibrium or leaves the data (grey). Enforcing convergence "
          "would require SEDS's constrained fit or a stabilizing term (e.g. a Lyapunov-based correction).",
          "",
          "**Observations.** In the top views the field funnels every demonstrated start into the target (blue, contracting region around it). In the vertical planes all integrations end within 5 cm but the field's equilibrium sits 2–3 cm from the median target (slightly below the contact point for the approach): with no stability constraint, nothing forces the equilibrium onto the target. The thin horizontal bands of extreme divergence at table height (z ≈ 0 in the cube frame, z ≈ −8 cm in the box frame) are an artifact: there the data form a thin layer (rest and lift-off from the table) across which the vertical velocity changes abruptly, so ∂ż/∂z is large; they are not a real expansion/contraction of the flow.",
          "",
          "This field is a diagnostic of what the demonstrations encode as a state-dependent policy; the controller "
          "used by the pipeline remains the time-indexed TP-GMM (whose evaluation is in `evaluation_report.md`)."]
    resolve_path("docs/ds_field.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(r.round(3).to_string())


if __name__ == "__main__":
    main()
