#!/usr/bin/env python
"""Phase 5: leave-one-out evaluation of the segmented TP-GMM, the monolithic TP-GMM and the
IK waypoint planner.

For each held-out cycle: refit the three policies on the other cycles (fixed K from
outputs/models, box frame re-estimated without the held-out cycle), generate the cycle from
its vision cube frame and its start pose, and measure trajectory RMSE, grasp and release
errors, heading error, choice of the cube's equivalent yaw and IK feasibility.

    python scripts/evaluate_offline.py                 # stratified subset (evaluation.n_folds)
    python scripts/evaluate_offline.py --folds all     # every usable cycle
    python scripts/evaluate_offline.py --workers 4

Outputs: outputs/evaluation/loo_results.csv, docs/evaluation_report.md, docs/figures/eval_*.png.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one BLAS thread per worker
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402
from so101_tpgmm.data.lerobot_dataset import LeRobotV3Dataset  # noqa: E402
from so101_tpgmm.evaluation.metrics import cycle_metrics, ik_feasibility  # noqa: E402
from so101_tpgmm.kinematics.fk import SO101Kinematics  # noqa: E402
from so101_tpgmm.kinematics.ik import ik_from_config  # noqa: E402
from so101_tpgmm.tpgmm.cube_symmetry import decision_margin, heading_target  # noqa: E402
from so101_tpgmm.tpgmm.demos import build_demos, canonical_cube_yaw, choose_cube_heading, training_table  # noqa: E402
from so101_tpgmm.tpgmm.policy import TPGMMPolicy  # noqa: E402
from train import fit_policies  # noqa: E402

MODELS = ("segmented", "monolithic", "ik_planner")
_G: dict = {}


def _init(cfg_args: tuple) -> None:
    cfg = load_config(*cfg_args)
    _G["cfg"] = cfg
    _G["ds"] = LeRobotV3Dataset(resolve_path(cfg["dataset"]["local_dir"]), cfg["dataset"]["state_key"])
    _G["kin"] = SO101Kinematics(resolve_path(cfg["kinematics"]["urdf"]))
    _G["ik"] = ik_from_config(cfg)
    _G["df"] = training_table(pd.read_csv(resolve_path(cfg["segmentation"]["output_csv"])),
                              pd.read_csv(resolve_path(cfg["vision"]["frames_csv"])), cfg["tpgmm"]["exclude_aborted"])
    mdir = resolve_path(cfg["tpgmm"]["models_dir"])
    seg = TPGMMPolicy.load(mdir / "segmented.json")
    mono = TPGMMPolicy.load(mdir / "monolithic.json")
    _G["K"] = {**{p: m.n_components for p, m in seg.models.items()}, "cycle": mono.models["cycle"].n_components}


def run_fold(i: int) -> list[dict]:
    cfg, ds, kin, ik, df = _G["cfg"], _G["ds"], _G["kin"], _G["ik"], _G["df"]
    prims = list(cfg["tpgmm"]["primitives"])
    train, held = df.drop(index=i).reset_index(drop=True), df.iloc[[i]].reset_index(drop=True)
    demos, ctx = build_demos(train, ds, kin, cfg, prims + ["cycle"])
    hd, _ = build_demos(held, ds, kin, cfg, prims, box=np.asarray(ctx["box"]))
    dA, dT = hd["approach_grasp"][0], hd["transport_release"][0]
    t0 = time.time()
    seg, mono, planner, _ = fit_policies(demos, ctx, cfg, k_fixed=_G["K"])
    fit_s = time.time() - t0
    r = held.iloc[0]
    cube = np.array([r.vision_x, r.vision_y, r.vision_z])
    start = dA.X[0, 1:]
    q0 = np.deg2rad(ds.episode(int(r.episode)).state[int(r.approach_start), :5])
    durs = {"approach_grasp": dA.duration, "transport_release": dT.duration, "cycle": dA.duration + dT.duration}
    _, k_pred = choose_cube_heading(float(r.vision_yaw), cube[:2], np.asarray(ctx["pan_xy"]), ctx["heading_offset"])
    _, k_true = canonical_cube_yaw(float(r.vision_yaw), float(dA.cube[3]))
    margin = decision_margin(float(r.vision_yaw), heading_target(cube[:2], np.asarray(ctx["pan_xy"]), ctx["heading_offset"]))
    rows = []
    for name, pol in (("segmented", seg), ("monolithic", mono), ("ik_planner", planner)):
        ro = pol.rollout(cube, float(r.vision_yaw), start=start, durations=durs)
        m = cycle_metrics(ro, dA, dT, monolithic=(name == "monolithic"))
        m.update(ik_feasibility(ro, ik, q0))
        # oracle: the equivalent cube yaw the operator actually used (upper bound for any choice rule)
        ro_o = pol.rollout(cube, float(r.vision_yaw), start=start, durations=durs, heading=float(dA.cube[3]))
        mo = cycle_metrics(ro_o, dA, dT, monolithic=(name == "monolithic"))
        m.update({f"{k}_oracle_k": mo[k] for k in ("grasp_err_mm", "rmse_pos_mm", "grasp_heading_err_deg")})
        m["k_margin_deg"] = float(np.degrees(margin))
        m.update(model=name, episode=int(r.episode), color=r.color, cube_x=cube[0], cube_y=cube[1],
                 k_correct=bool(k_pred == k_true), fit_s=fit_s)
        rows.append(m)
    return rows


def pick_folds(df: pd.DataFrame, n: int | str) -> list[int]:
    if n == "all" or int(n) >= len(df):
        return list(range(len(df)))
    key = df.color.astype(str) + "_" + pd.cut(df.vision_y, 4, labels=False).astype(str)
    order = df.assign(key=key).sort_values(["key", "vision_x"]).index.to_numpy()
    return sorted(order[np.linspace(0, len(order) - 1, int(n)).astype(int)].tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--folds", default=None, help="number of held-out cycles or 'all'")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--report-only", action="store_true", help="rewrite the report from loo_results.csv")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    ecfg = cfg["evaluation"]
    _init((args.config, args.set))
    if args.report_only:
        write_report(pd.read_csv(resolve_path("outputs/evaluation/loo_results.csv")), cfg, len(_G["df"]), _G["K"])
        print("report rewritten")
        return
    df = _G["df"]
    folds = pick_folds(df, args.folds or ecfg["n_folds"])
    workers = args.workers or ecfg["workers"]
    print(f"LOO on {len(folds)}/{len(df)} cycles with {workers} workers; K = {_G['K']}")
    t0 = time.time()
    rows = []
    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=((args.config, args.set),)) as pool:
        for j, out in enumerate(pool.imap_unordered(run_fold, folds)):
            rows += out
            if (j + 1) % 10 == 0:
                print(f"  {j + 1}/{len(folds)} folds, {time.time() - t0:.0f} s")
    res = pd.DataFrame(rows)
    out_dir = resolve_path("outputs/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "loo_results.csv", index=False)
    write_report(res, cfg, len(df), _G["K"])
    print(f"done in {time.time() - t0:.0f} s")


def write_report(res: pd.DataFrame, cfg: dict, n_total: int, K: dict) -> None:
    fig_dir = resolve_path("docs/figures")
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})
    labels = {"segmented": "TP-GMM segmented", "monolithic": "TP-GMM monolithic", "ik_planner": "IK planner"}
    metrics = [("rmse_pos_mm", "trajectory RMSE [mm]"), ("grasp_err_mm", "grasp point error [mm]"),
               ("release_err_mm", "release point error [mm]"), ("rmse_heading_deg", "heading RMSE [deg]"),
               ("rmse_gripper", "gripper RMSE [0-100]"), ("ik_feasible", "IK-feasible fraction")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (col, title) in zip(axs.flat, metrics):
        data = [res[res.model == m][col].dropna().to_numpy() for m in MODELS]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(MODELS) + 1), [labels[m] for m in MODELS], fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "eval_boxplots.png", dpi=100); plt.close(fig)

    seg = res[res.model == "segmented"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in ((axs[0], "grasp_err_mm", "segmented: grasp error [mm]"),
                           (axs[1], "rmse_pos_mm", "segmented: trajectory RMSE [mm]")):
        sc = ax.scatter(seg.cube_y * 100, seg.cube_x * 100, c=seg[col], cmap="viridis", s=25)
        ax.plot(0, 0, "k^"); ax.invert_xaxis(); ax.set_aspect("equal")
        ax.set_xlabel("cube y [cm]"); ax.set_ylabel("cube x [cm]"); ax.set_title(title, fontsize=10)
        plt.colorbar(sc, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(fig_dir / "eval_by_position.png", dpi=100); plt.close(fig)

    def cell(m, col, fmt="{:.1f}"):
        v = res[res.model == m][col].dropna()
        return f"{fmt.format(v.median())} [{fmt.format(v.quantile(0.25))}, {fmt.format(v.quantile(0.75))}]"

    n = res.episode.nunique()
    L = []
    A = L.append
    A("# Offline evaluation (Phase 5): leave-one-out\n")
    A("_Generated by `scripts/evaluate_offline.py`. Raw results: `outputs/evaluation/loo_results.csv` (copy in `docs/evaluation_loo_results.csv`)._\n")
    A("## Protocol\n")
    A(f"- Held-out cycles: **{n}** of {n_total} usable cycles "
      f"({'all' if n == n_total else 'stratified by colour and lateral position'}). For each, the three policies "
      "are refitted on the remaining cycles and the box frame is re-estimated without the held-out cycle.")
    A(f"- Number of components fixed to the BIC choice on all data: {K}. The pixel→base homography was also "
      "calibrated on all cycles (one correspondence out of ~340 inliers: negligible, but stated).")
    A("- Each held-out cycle is generated from its **vision** cube frame (the yaw among the 4 equivalent ones is "
      "chosen with the execution rule, not with the demonstration), its start pose and its own primitive "
      "durations (time-aligned comparison).")
    A("- Metrics: position RMSE over the cycle (200 time-aligned samples); grasp error = predicted TCP at the "
      "predicted grasp instant (segmented/planner: end of the approach primitive; monolithic: where its gripper "
      "closes) vs the demonstrated contact; release error = end of the cycle vs the demonstrated end of release; "
      "IK feasibility = fraction of points with an IK solution inside the joint limits (pitch output used).\n")
    A("## Results (median [IQR])\n")
    cols = [("rmse_pos_mm", "traj. RMSE [mm]"), ("rmse_approach_mm", "approach RMSE [mm]"),
            ("rmse_transport_mm", "transport RMSE [mm]"), ("grasp_err_mm", "grasp err [mm]"),
            ("grasp_err_oracle_timing_mm", "grasp err, oracle timing [mm]"), ("release_err_mm", "release err [mm]"),
            ("grasp_heading_err_deg", "grasp heading err [°]"), ("rmse_gripper", "gripper RMSE")]
    A("| model | " + " | ".join(c[1] for c in cols) + " |")
    A("|---|" + "---|" * len(cols))
    for m in MODELS:
        A(f"| {labels[m]} | " + " | ".join(cell(m, c) for c, _ in cols) + " |")
    A("\n| model | IK-feasible fraction (mean) | cycles 100 % feasible | max joint step [°] (median) |\n|---|---|---|---|")
    for m in MODELS:
        s = res[res.model == m]
        A(f"| {labels[m]} | {s.ik_feasible.mean():.3f} | {100 * (s.ik_feasible == 1).mean():.0f} % | "
          f"{s.ik_max_step_deg.median():.1f} |")
    kc = res[res.model == "segmented"].k_correct.mean()
    A(f"\nCube-yaw choice at execution (azimuth + learned offset) matches the demonstrated grasp in "
      f"**{100 * kc:.0f} %** of held-out cycles; a wrong choice rotates the grasp by 90° (it is still a valid grasp "
      "of a cube, but the heading error then counts 90°).\n")
    A("![](figures/eval_boxplots.png)\n")
    A("![](figures/eval_by_position.png)\n")
    from scipy.stats import wilcoxon

    by = {m: res[res.model == m].set_index("episode") for m in MODELS}
    seg_ = by["segmented"]

    def test(col, other):
        o = by[other].loc[seg_.index, col]
        return wilcoxon(seg_[col], o).pvalue, float((seg_[col] < o).mean())

    A("## Reading\n")
    A("Paired comparison on the same held-out cycles (Wilcoxon signed-rank; share of cycles where the segmented "
      "model is better):\n")
    A("| metric | segmented vs monolithic | segmented vs IK planner |\n|---|---|---|")
    for col, lab in (("rmse_pos_mm", "trajectory RMSE"), ("grasp_err_mm", "grasp error"),
                     ("release_err_mm", "release error")):
        p1, f1 = test(col, "monolithic")
        p2, f2 = test(col, "ik_planner")
        A(f"| {lab} | p = {p1:.1g}, better in {100 * f1:.0f} % | p = {p2:.1g}, better in {100 * f2:.0f} % |")
    mono = by["monolithic"]
    wrong = seg_[~seg_.k_correct.astype(bool)]
    right = seg_[seg_.k_correct.astype(bool)]
    A("")
    A(f"- **Segmented vs monolithic.** The clearest difference is the trajectory, mostly the transport "
      f"({seg_.rmse_transport_mm.median():.0f} vs {mono.rmse_transport_mm.median():.0f} mm RMSE): segmentation "
      "aligns every demonstration at the contact instant, so each primitive is learned on consistent phases. At the "
      f"grasp the gap is smaller ({seg_.grasp_err_mm.median():.1f} vs {mono.grasp_err_mm.median():.1f} mm) because "
      "the monolithic model is evaluated where its own gripper closes; its grasp instant is off by "
      f"{mono.grasp_time_err_s.abs().median():.2f} s (median) from the demonstration, and with the demonstration's "
      f"contact time its position error is {mono.grasp_err_oracle_timing_mm.median():.0f} mm — i.e. it reaches the "
      "cube, but not when the gripper of a time-indexed controller would expect it.")
    A(f"- **TP-GMM vs IK planner.** Same grasp accuracy within ~1 mm (both use the same cube frame and the planner's "
      f"contact point is the median of the same data), but the planner's straight segments are far from the "
      f"demonstrated motion (trajectory RMSE {by['ik_planner'].rmse_pos_mm.median():.0f} vs "
      f"{seg_.rmse_pos_mm.median():.0f} mm), its gripper timing is worse, its joint steps are larger, and "
      f"{100 * (1 - (by['ik_planner'].ik_feasible == 1).mean()):.0f} % of its cycles contain an IK-infeasible point.")
    A(f"- **Release:** no significant difference; all three are bounded by the spread of the demonstrated releases "
      "in the box (~2 cm), which the box frame cannot explain (the operator dropped the cube anywhere in it).")
    A(f"- **Cube-yaw choice (90° symmetry):** see the next section.")
    A("- **Error floors:** the grasp error is bounded below by the vision error (~8 mm median vs FK, "
      "`vision_report.md`) and by how consistently the operator grasped. Large errors are more frequent on the "
      "robot's right side close to the base (`eval_by_position.png`), where vision is also least accurate and cubes "
      "were pushed more often, but they also occur elsewhere.")
    A("\n## Choice among the 4 equivalent cube yaws\n")
    A("A cube can be grasped with the jaws along any of its 4 face normals. The execution rule picks the candidate "
      "closest to (cube azimuth + learned heading offset); its *margin* is the distance to the 45° decision "
      "boundary. Alternatives tried on the 366 training cycles (cross-validated): optimized offset 91.3 %, "
      "offset linear in the azimuth 92.1 %, k-NN offset by position 89.3 %, wrist heading at rest 71.6 %, vs "
      "91.8 % for the rule used — none is meaningfully better (`so101_tpgmm/tpgmm/cube_symmetry.py`).\n")
    seg_m = by["segmented"]
    A("| decision margin | cycles | agreement with the operator | grasp err, rule [mm] | grasp err, oracle yaw [mm] |")
    A("|---|---|---|---|---|")
    for lo, hi in ((0, 10), (10, 20), (20, 46)):
        s_ = seg_m[(seg_m.k_margin_deg >= lo) & (seg_m.k_margin_deg < hi)]
        if len(s_):
            A(f"| {lo}–{min(hi, 45)}° | {len(s_)} | {100 * s_.k_correct.mean():.0f} % | {s_.grasp_err_mm.median():.1f} | "
              f"{s_.grasp_err_mm_oracle_k.median():.1f} |")
    A(f"| all | {len(seg_m)} | {100 * seg_m.k_correct.mean():.0f} % | {seg_m.grasp_err_mm.median():.1f} | "
      f"{seg_m.grasp_err_mm_oracle_k.median():.1f} |")
    A("\nThe disagreements concentrate where the margin is small: there two grasps are equally plausible and the "
      "operator's choice is not predictable from the geometry. A grasp rotated by 90° is still a valid grasp of a "
      "cube, but it differs from the demonstration (contact offset and approach rotate with it), which is why the "
      "grasp error grows. At inference, when the margin is below `inference.ambiguity_margin_deg`, both options are "
      "planned and the IK-feasible one with the least joint motion is executed.")
    resolve_path("docs/evaluation_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
