#!/usr/bin/env python
"""Paired comparison of two ACT offline evaluations (same held-out episodes and frames).

Reads ``outputs/act/offline_<a>.csv`` and ``offline_<b>.csv`` (from eval_act_offline.py) and, for each
metric, reports the medians, how often B beats A on the same episode/frame, and a Wilcoxon signed-rank
p-value. Writes ``outputs/act/compare_<a>_vs_<b>.md``.

    python scripts/act/compare_act_offline.py A_50k B_50k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from so101_tpgmm.config import resolve_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    root = resolve_path("outputs/act")
    da, db = (pd.read_csv(root / f"offline_{n}.csv") for n in (args.a, args.b))
    key = ["episode", "frame", "horizon", "kind"]
    m = da.merge(db, on=key, suffixes=("_a", "_b"))
    L = [f"## ACT offline: `{args.a}` vs `{args.b}` (paired, {m.episode.nunique()} held-out episodes)\n",
         "| metric | horizon | median A | median B | B better in | Wilcoxon p | pairs |", "|---|---|---|---|---|---|---|"]
    metrics = [("chunk", "tcp_err_mm", "TCP error [mm]"), ("chunk", "joint_err_deg", "joint error [°]"),
               ("chunk", "grip_err", "gripper error"), ("grasp", "tcp_err_mm", "grasp anticipation [mm]"),
               ("grasp_cmd", "tcp_err_mm", "grasp anticipation, command space [mm]"),
               ("grasp_cmd_closest", "tcp_err_mm", "closest approach to the grasp [mm]")]
    for kind, col, label in metrics:
        for h, g in m[m.kind == kind].groupby("horizon"):
            a, b = g[f"{col}_a"], g[f"{col}_b"]
            ok = a.notna() & b.notna()
            a, b = a[ok], b[ok]
            if len(a) < 5:
                continue
            p = wilcoxon(a, b).pvalue if (a != b).any() else 1.0
            L.append(f"| {label} | {h / 30:.2f} s | {a.median():.2f} | {b.median():.2f} | {100 * (b < a).mean():.0f} % | "
                     f"{p:.2g} | {len(a)} |")
    text = "\n".join(L) + "\n"
    (root / f"compare_{args.a}_vs_{args.b}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
