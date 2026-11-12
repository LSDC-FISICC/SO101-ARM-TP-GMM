#!/usr/bin/env python
"""Phase 4: train the segmented TP-GMM (main), the monolithic TP-GMM (baseline) and the
IK waypoint planner (geometric baseline) on all usable cycles.

Outputs: outputs/models/{segmented,monolithic,ik_planner}.json, docs/tpgmm_report.md and
figures. Out-of-sample evaluation is scripts/evaluate_offline.py (leave-one-out).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402
from so101_tpgmm.data.lerobot_dataset import LeRobotV3Dataset  # noqa: E402
from so101_tpgmm.kinematics.fk import SO101Kinematics  # noqa: E402
from so101_tpgmm.planner.ik_planner import IKPlanner  # noqa: E402
from so101_tpgmm.tpgmm.demos import build_demos, training_table  # noqa: E402
from so101_tpgmm.tpgmm.policy import TPGMMPolicy  # noqa: E402
from so101_tpgmm.tpgmm.tpgmm import TPGMM, select_by_bic  # noqa: E402


def stack_local(demos) -> np.ndarray:
    return np.concatenate([d.local() for d in demos], axis=1)


def fit_policies(demos: dict, ctx: dict, cfg: dict, k_fixed: dict[str, int] | None = None,
                 verbose: bool = False) -> tuple[TPGMMPolicy, TPGMMPolicy, IKPlanner, dict]:
    """Fit the three policies. With ``k_fixed`` the BIC search is skipped (used in leave-one-out)."""
    tcfg = cfg["tpgmm"]
    reg = np.asarray(tcfg["reg"], dtype=float)[: demos["cycle"][0].X.shape[1]]
    ang = (4,)
    prims = list(tcfg["primitives"])
    durations = {p: float(np.median([d.duration for d in demos[p]])) for p in prims + ["cycle"]}
    models, bic = {}, {}
    for p in prims + ["cycle"]:
        X = stack_local(demos[p])
        if k_fixed is not None:
            models[p] = TPGMM(k_fixed[p], X.shape[0], X.shape[2], reg, ang).fit(X, max_iter=tcfg["max_iter"])
        else:
            t0 = time.time()
            models[p], bic[p] = select_by_bic(X, tcfg["k_range"], reg, ang, tcfg["max_iter"])
            if verbose:
                print(f"  {p}: K={models[p].n_components} (BIC over {tcfg['k_range']}, {time.time() - t0:.0f} s)")
        models[p].var_names = ctx["var_names"]
    meta = {**ctx, "durations": durations, "boundary_blend": tcfg["boundary_blend"]}
    seg = TPGMMPolicy({p: models[p] for p in prims}, {**meta, "primitives": prims})
    mono = TPGMMPolicy({"cycle": models["cycle"]}, {**meta, "primitives": ["cycle"]})
    planner = IKPlanner().fit(demos, ctx, durations)
    return seg, mono, planner, bic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    tcfg = cfg["tpgmm"]
    ds = LeRobotV3Dataset(resolve_path(cfg["dataset"]["local_dir"]), cfg["dataset"]["state_key"])
    kin = SO101Kinematics(resolve_path(cfg["kinematics"]["urdf"]))
    df = training_table(pd.read_csv(resolve_path(cfg["segmentation"]["output_csv"])),
                        pd.read_csv(resolve_path(cfg["vision"]["frames_csv"])), tcfg["exclude_aborted"])
    prims = list(tcfg["primitives"])
    demos, ctx = build_demos(df, ds, kin, cfg, prims + ["cycle"])
    print(f"{len(df)} training cycles; cube frames from {tcfg['frames']['cube_source']}")
    seg, mono, planner, bic = fit_policies(demos, ctx, cfg, verbose=True)
    mdir = resolve_path(tcfg["models_dir"])
    seg.save(mdir / "segmented.json")
    mono.save(mdir / "monolithic.json")
    planner.save(mdir / "ik_planner.json")

    # ------------------------------------------------------------------ figures
    fig_dir = resolve_path("docs/figures")
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for p, sc in bic.items():
        ks = sorted(sc)
        ax.plot(ks, [sc[k] for k in ks], "o-", label=p)
    ax.set_xlabel("K"); ax.set_ylabel("BIC"); ax.legend(fontsize=8); ax.set_title("Model selection (BIC)")
    fig.tight_layout(); fig.savefig(fig_dir / "tpgmm_bic.png", dpi=100); plt.close(fig)

    # components of the approach model in the cube frame (x-z and x-y, local)
    m = seg.models["approach_grasp"]
    L = np.stack([d.local()[0] for d in demos["approach_grasp"]])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax_, (i, j, lab) in zip(axs[:2], ((1, 2, "y"), (1, 3, "z"))):
        for tr in L[::4]:
            ax_.plot(tr[:, i] * 100, tr[:, j] * 100, c="0.7", lw=0.5)
        for k in range(m.n_components):
            mu, S = m.means[k, 0][[i, j]], m.covs[k, 0][np.ix_([i, j], [i, j])]
            w, V = np.linalg.eigh(S)
            ang_ = np.linspace(0, 2 * np.pi, 60)
            ell = (V @ np.diag(2 * np.sqrt(np.maximum(w, 0))) @ np.vstack([np.cos(ang_), np.sin(ang_)])).T + mu
            ax_.plot(ell[:, 0] * 100, ell[:, 1] * 100, c=plt.cm.viridis(k / m.n_components), lw=1.5)
        ax_.set_xlabel("cube-frame x [cm] (outward)"); ax_.set_ylabel(f"cube-frame {lab} [cm]")
        ax_.set_title(f"approach_grasp in the cube frame (K={m.n_components}, 2σ)")
    Lb = np.stack([d.local()[1] for d in demos["transport_release"]])
    mb = seg.models["transport_release"]
    for tr in Lb[::4]:
        axs[2].plot(tr[:, 1] * 100, tr[:, 2] * 100, c="0.7", lw=0.5)
    for k in range(mb.n_components):
        mu, S = mb.means[k, 1][[1, 2]], mb.covs[k, 1][np.ix_([1, 2], [1, 2])]
        w, V = np.linalg.eigh(S)
        ang_ = np.linspace(0, 2 * np.pi, 60)
        ell = (V @ np.diag(2 * np.sqrt(np.maximum(w, 0))) @ np.vstack([np.cos(ang_), np.sin(ang_)])).T + mu
        axs[2].plot(ell[:, 0] * 100, ell[:, 1] * 100, c=plt.cm.viridis(k / mb.n_components), lw=1.5)
    axs[2].set_aspect("equal"); axs[2].set_xlabel("box-frame x [cm]"); axs[2].set_ylabel("box-frame y [cm]")
    axs[2].set_title(f"transport_release in the box frame (K={mb.n_components})")
    fig.tight_layout(); fig.savefig(fig_dir / "tpgmm_components.png", dpi=100); plt.close(fig)

    # in-sample reproductions of three demonstrations
    ex = [0, len(df) // 2, len(df) - 1]
    fig, axs = plt.subplots(len(ex), 3, figsize=(15, 3.6 * len(ex)))
    for row, i in enumerate(ex):
        dA, dT = demos["approach_grasp"][i], demos["transport_release"][i]
        r = df.iloc[i]
        cube = np.array([r.vision_x, r.vision_y, r.vision_z])
        demo = np.vstack([dA.X[:, 1:], dT.X[1:, 1:]])
        tdem = np.r_[dA.X[:, 0] * dA.duration, dA.duration + dT.X[1:, 0] * dT.duration]
        start = dA.X[0, 1:]
        ros = {"segmented TP-GMM": seg.rollout(cube, r.vision_yaw, start=start),
               "monolithic TP-GMM": mono.rollout(cube, r.vision_yaw, start=start),
               "IK planner": planner.rollout(cube, r.vision_yaw, start=start)}
        axs[row, 0].plot(demo[:, 1] * 100, demo[:, 0] * 100, "k", lw=2, label="demonstration")
        axs[row, 1].plot(tdem, demo[:, 2] * 100, "k", lw=2)
        axs[row, 2].plot(tdem, demo[:, 4], "k", lw=2)
        for (name, ro), c in zip(ros.items(), ("#3b6ea5", "#c03b3b", "#2a9d4b")):
            axs[row, 0].plot(ro.Y[:, 1] * 100, ro.Y[:, 0] * 100, c=c, label=name)
            axs[row, 1].plot(ro.t, ro.Y[:, 2] * 100, c=c)
            axs[row, 2].plot(ro.t, ro.Y[:, 4], c=c)
        axs[row, 0].plot(cube[1] * 100, cube[0] * 100, "s", c="orange", ms=8)
        axs[row, 0].invert_xaxis(); axs[row, 0].set_aspect("equal")
        axs[row, 0].set_title(f"episode {int(r.episode)} ({r.color}): top view [cm]", fontsize=9)
        axs[row, 1].set_title("TCP z [cm] vs time [s]", fontsize=9)
        axs[row, 2].set_title("gripper vs time [s]", fontsize=9)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "tpgmm_reproductions.png", dpi=90); plt.close(fig)

    # ------------------------------------------------------------------ report
    L = []
    A = L.append
    A("# TP-GMM training report (Phase 4)\n")
    A("_Generated by `scripts/train.py`. In-sample figures only; out-of-sample results are in "
      "`docs/evaluation_report.md` (leave-one-out)._\n")
    A("## Data\n")
    A(f"- {len(df)} cycles (usable vision frame, no missed/aborted grasp), pooled over colours: "
      + ", ".join(f"{c} {n}" for c, n in df.color.value_counts().items()) + ".")
    A(f"- Variables `{ctx['var_names']}`: normalized time (input) → TCP position, outward jaw heading h, gripper, "
      "approach pitch. Every primitive is resampled to "
      f"{tcfg['samples_per_primitive']} points (equal weight per demonstration).")
    A(f"- Frames: **cube** from `{tcfg['frames']['cube_source']}` (position + yaw; the yaw among the 4 equivalent "
      "ones closest to the grasp heading), **box** "
      f"({tcfg['frames']['box']['mode']}): ({', '.join(f'{v * 100:.1f}' for v in ctx['box'][:3])}) cm.")
    A(f"- Learned heading offset at the grasp (circular mean of h − cube azimuth): "
      f"{np.degrees(ctx['heading_offset']):.1f}° — used to pick the cube yaw at execution.")
    A("- Median durations: " + ", ".join(f"{p} {v:.2f} s" for p, v in seg.meta["durations"].items()) + ".\n")
    A("## Model selection\n")
    A("| model | selected K | BIC per K |\n|---|---|---|")
    for p, sc in bic.items():
        mk = seg.models.get(p) or mono.models.get(p)
        A(f"| {p} | {mk.n_components} | " + ", ".join(f"{k}: {v:.0f}" for k, v in sorted(sc.items())) + " |")
    A("\nBIC keeps decreasing up to the largest K tried: the 50 samples of a demonstration are strongly "
      "correlated, so counting each as an independent datapoint under-penalizes complexity. The search range is "
      "therefore capped (`tpgmm.k_range`); the leave-one-out evaluation (Phase 5) is the real check against "
      "over-fitting.\n")
    A("![](figures/tpgmm_bic.png)\n")
    A("![](figures/tpgmm_components.png)\n")
    A("## In-sample reproductions\n")
    A("Black: demonstration; the three policies are given the same cube pose (vision) and start pose.\n")
    A("![](figures/tpgmm_reproductions.png)\n")
    A("## Design decisions\n")
    A("- **Segmented model** (one TP-GMM per primitive) is the main model; the **monolithic** one (a single TP-GMM "
      "over the whole cycle, same frames and variables) is the baseline asked for. Time is normalized per "
      "primitive in the segmented model, per cycle in the monolithic one.")
    A("- **Yaw**: handled as an angular offset of the frame (A = 1, b = frame yaw) and aligned modulo 2π before "
      "the product of Gaussians; the outward heading h avoids the ±180° cut of the raw tool yaw.")
    A("- **Regularization**: per-dimension diagonal term (`tpgmm.reg`), in each variable's units.")
    A("- **IK planner baseline**: median waypoints in the same frames, PCHIP + minimum-jerk; no probabilistic "
      "model (`so101_tpgmm/planner/ik_planner.py`).")
    resolve_path("docs/tpgmm_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("models and report written")


if __name__ == "__main__":
    main()
