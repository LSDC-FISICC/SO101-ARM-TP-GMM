#!/usr/bin/env python
"""Phase 2: segment every episode into grasp cycles and movement primitives.

Writes the per-cycle table (``segmentation.output_csv``, plus a copy in
docs/segmentation_cycles.csv), docs/segmentation_report.md and figures.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402
from so101_tpgmm.data.lerobot_dataset import LeRobotV3Dataset  # noqa: E402
from so101_tpgmm.kinematics.fk import SO101Kinematics  # noqa: E402
from so101_tpgmm.kinematics.ik import approach_pitch, tool_yaw  # noqa: E402
from so101_tpgmm.segmentation.gripper import GripperParams  # noqa: E402
from so101_tpgmm.segmentation.primitives import SegmentationParams, segment_dataset  # noqa: E402

EXAMPLES = (150, 2, 77, 106, 374, 334)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    ds = LeRobotV3Dataset(resolve_path(cfg["dataset"]["local_dir"]), cfg["dataset"]["state_key"])
    kin = SO101Kinematics(resolve_path(cfg["kinematics"]["urdf"]))
    fps = ds.fps
    gp = GripperParams.from_config(cfg["gripper"])
    sp = SegmentationParams.from_config(cfg["segmentation"])
    results, df = segment_dataset(ds, kin, cfg)

    # Per-cycle geometry (FK) for the report and for the IK planner.
    rows = []
    for r in df.itertuples():
        st = ds.episode(int(r.episode)).state
        T = kin.fk_from_dataset(st, "degrees")
        pax = kin.chain[0].origin[:2, 3]
        rows.append({
            "grasp_xyz": T[r.close_idx, :3, 3], "pregrasp_xyz": T[r.pregrasp_idx, :3, 3],
            "lift_xyz": T[r.lift_idx, :3, 3], "release_xyz": T[r.open_idx, :3, 3],
            "approach_apex_z": float(T[r.approach_start:r.close_idx + 1, 2, 3].max()),
            "transport_apex_z": float(T[r.close_idx:r.release_end + 1, 2, 3].max()),
            "grasp_yaw": float(tool_yaw(T[r.close_idx, :3, :3])),
            "grasp_pitch": float(approach_pitch(T[r.close_idx], pax)),
            "release_pitch": float(approach_pitch(T[r.open_idx], pax)),
        })
    for key in ("grasp", "pregrasp", "lift", "release"):
        arr = np.array([x[f"{key}_xyz"] for x in rows])
        for j, a in enumerate("xyz"):
            df[f"{key}_{a}"] = arr[:, j]
    for key in ("grasp_yaw", "grasp_pitch", "release_pitch", "approach_apex_z", "transport_apex_z"):
        df[key] = [x[key] for x in rows]
    df["approach_s"] = (df.close_idx - df.approach_start) / fps
    df["transport_s"] = (df.release_end - df.close_idx) / fps
    out = resolve_path(cfg["segmentation"]["output_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(out, index=False)
    df.round(5).to_csv(resolve_path("docs/segmentation_cycles.csv"), index=False)

    # ------------------------------------------------------------------ figures
    fig_dir = resolve_path("docs/figures")
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})
    by_ep = {r.episode: r for r in results}
    fig, axs = plt.subplots(len(EXAMPLES), 1, figsize=(10, 1.9 * len(EXAMPLES)), sharex=True)
    for a, k in zip(axs, EXAMPLES):
        st = ds.episode(k).state
        z = kin.fk_from_dataset(st, "degrees")[:, 2, 3]
        a.plot(st[:, 5], c="#3b6ea5", label="gripper")
        a.plot(z * 500, c="#e08a2c", alpha=0.8, label="TCP z ×500 [m]")
        r = by_ep[k]
        for c in r.cycles:
            a.axvspan(*c.primitive("approach_grasp"), color="#3b6ea5", alpha=0.12)
            a.axvspan(*c.primitive("transport_release"), color="#c03b3b", alpha=0.12)
            a.plot(c.pregrasp_idx, z[c.pregrasp_idx] * 500, "kv", ms=5)
            a.plot(c.lift_idx, z[c.lift_idx] * 500, "k^", ms=5)
        for idx, why in r.rejected:
            a.axvline(idx, c="k", ls="--", lw=1)
            a.text(idx + 3, 85, why, fontsize=7)
        flags = [f for c in r.cycles for f, v in (("quiet release", c.quiet_release), ("aborted before", c.aborted_before)) if v]
        a.set_title(f"episode {k}" + (f" — {', '.join(flags)}" if flags else ""), fontsize=9)
        a.set_ylim(-10, 100)
    axs[0].legend(loc="upper right", fontsize=7)
    axs[-1].set_xlabel("frame — blue: approach_grasp, red: transport_release, ▼ pre-grasp, ▲ lift, dashed: rejected")
    fig.tight_layout(); fig.savefig(fig_dir / "segmentation_examples.png", dpi=100); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    axs[0].hist(df.approach_s, bins=40, color="#3b6ea5"); axs[0].set_title("approach_grasp duration [s]")
    axs[1].hist(df.transport_s, bins=40, color="#c03b3b"); axs[1].set_title("transport_release duration [s]")
    fig.tight_layout(); fig.savefig(fig_dir / "segmentation_durations.png", dpi=100); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    sel = df.iloc[:: max(1, len(df) // 60)]
    for r in sel.itertuples():
        P = kin.fk_from_dataset(ds.episode(int(r.episode)).state, "degrees")[:, :3, 3]
        for (s, e), col, ax_t in (((r.approach_start, r.close_idx), "#3b6ea5", axs[1]),
                                  ((r.close_idx, r.release_end), "#c03b3b", axs[2])):
            axs[0].plot(P[s:e + 1, 1] * 100, P[s:e + 1, 0] * 100, c=col, lw=0.6, alpha=0.5)
            tn = np.linspace(0, 1, e - s + 1)
            ax_t.plot(tn, P[s:e + 1, 2] * 100, c=col, lw=0.6, alpha=0.4)
    axs[0].plot(0, 0, "k^"); axs[0].invert_xaxis(); axs[0].set_aspect("equal")
    axs[0].set_xlabel("base y [cm]"); axs[0].set_ylabel("base x [cm]"); axs[0].set_title("TCP paths (top view)")
    axs[1].set_title("approach_grasp: TCP z vs normalised time"); axs[1].set_ylabel("z [cm]")
    axs[2].set_title("transport_release: TCP z vs normalised time")
    fig.tight_layout(); fig.savefig(fig_dir / "segmentation_paths.png", dpi=100); plt.close(fig)

    # ------------------------------------------------------------------ report
    nc = Counter(len(r.cycles) for r in results)
    rej = Counter(x[1] for r in results for x in r.rejected)
    pre_dz = (df.pregrasp_z - df.grasp_z) * 100
    pre_dh = np.hypot(df.pregrasp_x - df.grasp_x, df.pregrasp_y - df.grasp_y) * 100
    lift_dz = (df.lift_z - df.grasp_z) * 100
    pct = lambda a: f"{np.percentile(a, 5):.2f} / {np.median(a):.2f} / {np.percentile(a, 95):.2f}"  # noqa: E731
    L = []
    A = L.append
    A("# Segmentation report (Phase 2)\n")
    A("_Generated by `scripts/segment_dataset.py`. Per-cycle table: `docs/segmentation_cycles.csv`._\n")
    A("## Rules\n")
    A(f"1. **Gripper events** (`segmentation/gripper.py`): median filter ({gp.smooth_window} frames), hysteresis "
      f"(OPEN > {gp.open_threshold}, CLOSED < {gp.close_threshold}), OPEN debounce {gp.min_open_s} s, hold "
      f"plateau ∈ [{gp.hold_min}, {gp.close_threshold}] lasting ≥ {gp.min_hold_s} s. Cycle = OPEN → HOLD → OPEN.")
    A(f"2. **Quiet releases:** a hold that ends straight in *closed-empty* (opening below the threshold) "
      f"counts as released at the end of its plateau if the TCP is above {sp.release_min_z * 100:.0f} cm.")
    A(f"3. **Physical gates** (no stacking): contact at TCP z ≤ {sp.grasp_max_z * 100:.0f} cm "
      f"(else `contact_in_air`); release at TCP z > {sp.release_min_z * 100:.0f} cm (else `release_at_table` = "
      "aborted grasp).")
    A(f"4. **Primitives:** `approach_grasp` = [motion onset or previous cycle end, contact]; `transport_release` "
      f"= [contact, end of the release opening]. Motion onset = joint speed > {sp.onset_joint_speed_deg_s}°/s "
      f"for ≥ {sp.onset_min_s} s. The return to home is discarded. Minimum primitive length {sp.min_primitive_s} s.")
    A(f"5. **Keypoints:** pre-grasp = last frame before contact with z ≥ z_contact + {sp.keypoint_height * 100:.0f} cm; "
      f"lift = first frame after contact above the same height.\n")
    A("## Results\n")
    A("| valid cycles per episode | episodes |\n|---|---|")
    for k in sorted(nc):
        A(f"| {k} | {nc[k]} |")
    A(f"\nRejected candidates: {dict(rej)} — " +
      ", ".join(f"ep {r.episode}: {why} @ frame {idx}" for r in results for idx, why in r.rejected) + ".\n")
    A(f"- Quiet releases recovered: {df[df.quiet_release].episode.tolist()}.")
    A(f"- Cycles with a missed/aborted grasp inside the approach (`aborted_before`): "
      f"{df[df.aborted_before].episode.tolist()} — kept in the table, excluded from training by default.")
    A(f"- Durations (p5 / median / p95): approach_grasp {pct(df.approach_s)} s, transport_release "
      f"{pct(df.transport_s)} s.")
    A(f"- Release points: all within {np.max(np.hypot(df.release_x - df.release_x.median(), df.release_y - df.release_y.median())) * 100:.1f} cm "
      f"of their median ({df.release_x.median() * 100:.1f}, {df.release_y.median() * 100:.1f}, {df.release_z.median() * 100:.1f}) cm → the box.\n")
    A("![](figures/segmentation_examples.png)\n")
    A("![](figures/segmentation_durations.png)\n")
    A("![](figures/segmentation_paths.png)\n")
    A("## Keypoint statistics (input for the IK-based planner)\n")
    A("Relative to the contact pose (FK; p5 / median / p95):\n")
    A("| quantity | value |\n|---|---|")
    A(f"| pre-grasp height above contact (≈ threshold by definition) | {pct(pre_dz)} cm |")
    A(f"| highest TCP point during approach_grasp | {pct(df.approach_apex_z * 100)} cm |")
    A(f"| highest TCP point during transport_release | {pct(df.transport_apex_z * 100)} cm |")
    A(f"| pre-grasp horizontal offset from contact | {pct(pre_dh)} cm |")
    A(f"| pre-grasp → contact time | {pct((df.close_idx - df.pregrasp_idx) / fps)} s |")
    A(f"| lift height (first frame ≥ threshold) | {pct(lift_dz)} cm |")
    A(f"| contact → lift time | {pct((df.lift_idx - df.close_idx) / fps)} s |")
    A(f"| approach pitch at contact | {pct(np.degrees(df.grasp_pitch))}° |")
    A(f"| approach pitch at release | {pct(np.degrees(df.release_pitch))}° |")
    A(f"| release height above the table plane z = 0 | {pct(df.release_z * 100)} cm |")
    A("\n## Two solutions on the same primitives\n")
    A("Both consume this table and are evaluated with the same leave-one-out protocol (Phase 5):\n")
    A("1. **TP-GMM** (Phase 4): one model per primitive, in the cube and box frames; output "
      "[x, y, z, yaw, gripper] (+ pitch, see `kinematics_report.md`), converted to joints by IK.")
    A("2. **IK planner** (geometric baseline): per primitive, a few Cartesian waypoints expressed in the same "
      "frames — approach: start → pre-grasp (above the cube) → contact; transport: contact → lift → above "
      "box → release — with heights, pitch and timing learned as medians of the keypoints above, minimum-jerk "
      "interpolation and the same IK. It needs no probabilistic model and makes the value added by TP-GMM "
      "measurable.")
    resolve_path("docs/segmentation_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{len(df)} cycles; report written")


if __name__ == "__main__":
    main()
