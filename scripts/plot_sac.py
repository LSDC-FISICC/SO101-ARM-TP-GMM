#!/usr/bin/env python
"""Learning curves of the SAC runs (evaluation success and return vs environment steps).

Reads ``rl.models_dir/<run>/eval.csv`` and ``progress.csv`` for every run and writes
``docs/figures/sac_learning_curves.png``. The residual run's step 0 is the TP-GMM itself (zero
correction), drawn as a reference line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402

LABEL = {"scratch": "SAC from scratch", "residual": "SAC residual on TP-GMM"}
COLOR = {"scratch": "#c0392b", "residual": "#2a6fdb"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", nargs="*", default=["residual_s0", "scratch_s0"])
    args = ap.parse_args()
    cfg = load_config(args.config, [])
    root = resolve_path(cfg["rl"]["models_dir"])
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for run in args.runs:
        ev = root / run / "eval.csv"
        if not ev.exists():
            continue
        mode = run.split("_")[0]
        e = pd.read_csv(ev)
        c = COLOR.get(mode, "0.3")
        axs[0].plot(e.timesteps / 1e3, 100 * e.success_rate, "-o", ms=3, c=c, label=LABEL.get(mode, run))
        axs[1].plot(e.timesteps / 1e3, e.mean_return, "-o", ms=3, c=c, label=LABEL.get(mode, run))
        if mode == "residual" and len(e):
            axs[0].axhline(100 * e.success_rate.iloc[0], c=c, ls=":", lw=1, label="TP-GMM alone (residual at step 0)")
        pr = root / run / "progress.csv"
        if pr.exists():
            p = pd.read_csv(pr).dropna(subset=["rollout/success_rate"])
            axs[2].plot(p["time/total_timesteps"] / 1e3, 100 * p["rollout/success_rate"].rolling(5, min_periods=1).mean(),
                        c=c, label=LABEL.get(mode, run))
    axs[0].set_ylabel("evaluation success [%]")
    axs[1].set_ylabel("evaluation return")
    axs[2].set_ylabel("success while collecting [%] (stochastic)")
    for ax in axs:
        ax.set_xlabel("environment steps [×1000] (10 Hz: 1000 steps = 100 s of robot time)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axs[0].set_ylim(-3, 103)
    axs[2].set_ylim(-3, 103)
    fig.suptitle("SAC on the SO-101 pick-and-place (MuJoCo, domain randomization; evaluation: 30 episodes, "
                 "deterministic policy)", fontsize=10)
    fig.tight_layout()
    out = resolve_path("docs/figures/sac_learning_curves.png")
    fig.savefig(out, dpi=95)
    print("wrote", out)


if __name__ == "__main__":
    main()
