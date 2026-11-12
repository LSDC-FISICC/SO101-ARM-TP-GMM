#!/usr/bin/env python
"""Phase 1: verify FK/IK and map the reachable workspace.

* FK -> IK round trips on random task-relevant configurations.
* IK on recorded trajectories (FK of observation.state), with the demonstrated
  approach pitch and with the default pitch: feasibility and joint recovery.
* Reachability map on the table: fraction of cube yaws graspable at each XY,
  using the 90-degree symmetry of the cube (any of yaw + k*90 deg is a valid grasp).

Writes docs/kinematics_report.md and docs/figures/reachability.png.
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
from so101_tpgmm.kinematics.ik import approach_pitch, ik_from_config, tool_yaw, yaw_observability  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--grid-step", type=float, default=0.02)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    ik = ik_from_config(cfg)
    kin = ik.kin
    pax = ik.pan_axis_xy
    L: list[str] = []
    A = L.append

    # ------------------------------------------------------------- random round trips
    rng = np.random.default_rng(0)
    rows = []
    for mode in ("demonstrated pitch", "default pitch"):
        rng = np.random.default_rng(0)
        n = ok = 0
        t0 = time.time()
        qerr = []
        while n < args.n_random:
            q = rng.uniform(ik.lower, ik.upper)
            T = kin.fk(q)
            if not (T[0, 3] > 0.03 and -0.01 < T[2, 3] < 0.25 and yaw_observability(T[:3, :3]) > 0.3
                    and abs(approach_pitch(T, pax)) < np.deg2rad(60)):
                continue
            n += 1
            r = ik.solve(T[:3, 3], tool_yaw(T[:3, :3]), pitch=approach_pitch(T, pax) if mode.startswith("dem") else None)
            ok += r.success
            if r.success:
                qerr.append(np.max(np.abs(r.q - q)))
        rows.append({"pitch": mode, "targets": n, "success": f"{100 * ok / n:.1f} %",
                     "same joints (< 1°)": f"{100 * np.mean(np.rad2deg(qerr) < 1):.0f} %",
                     "ms / solve": f"{(time.time() - t0) / n * 1e3:.0f}"})
    print("[1] random round trips done")

    # ------------------------------------------------------------- recorded trajectories
    ds = LeRobotV3Dataset(resolve_path(cfg["dataset"]["local_dir"]), cfg["dataset"]["state_key"])
    eps = np.linspace(0, ds.num_episodes - 1, args.n_episodes).astype(int)
    traj = {"demonstrated pitch": [], "default pitch": []}
    t_traj = {k: 0.0 for k in traj}
    n_pts = 0
    for k in eps:
        Q = np.deg2rad(ds.episode(int(k)).state[::3, :5])
        T = kin.fk(Q)
        n_pts += len(Q)
        for mode in traj:
            t0 = time.time()
            res = ik.solve_trajectory(T[:, :3, 3], tool_yaw(T[:, :3, :3]), Q[0],
                                      pitch=approach_pitch(T, pax) if mode.startswith("dem") else None)
            t_traj[mode] += time.time() - t0
            traj[mode].append((res.feasible_fraction, np.rad2deg(np.abs(res.q - Q).max()),
                               np.rad2deg(res.max_joint_step.max()), res.pos_err.max()))
    print("[2] trajectories done")

    # ------------------------------------------------------------- reachability map
    z_grasp = -0.007  # median grasp TCP height measured in Phase 0 (dataset_report §4)
    xs = np.arange(0.0, 0.46, args.grid_step)
    ys = np.arange(-0.26, 0.32, args.grid_step)
    cube_yaws = np.deg2rad(np.arange(-45, 45, 15))
    reach = np.full((len(ys), len(xs)), np.nan)
    t0 = time.time()
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if np.hypot(x - pax[0], y - pax[1]) < 0.06:
                continue  # inside the robot base
            p = np.array([x, y, z_grasp])
            good = 0
            for cy in cube_yaws:
                for k90 in range(4):
                    r = ik.solve(p, cy + k90 * np.pi / 2, pitch=np.deg2rad(10), max_restarts=4)
                    if r.success:
                        good += 1
                        break
            reach[iy, ix] = good / len(cube_yaws)
    print(f"[3] reachability map done ({time.time() - t0:.0f} s)")

    summary = pd.read_csv(resolve_path("docs/episodes_summary.csv"))
    g = summary.dropna(subset=["grasp_x"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(ys * 100, xs * 100, reach.T, cmap="viridis", vmin=0, vmax=1, shading="nearest")
    ax.scatter(g.grasp_y * 100, g.grasp_x * 100, s=4, c="w", alpha=0.6, label="demonstrated grasps (TCP)")
    ax.scatter(g.release_y * 100, g.release_x * 100, s=4, c="r", alpha=0.6, label="releases")
    ax.plot(0, 0, "k^", ms=9, label="base_link origin")
    ax.invert_xaxis(); ax.set_aspect("equal")
    ax.set_xlabel("base y [cm]"); ax.set_ylabel("base x [cm]")
    ax.set_title(f"Fraction of cube yaws graspable at TCP z = {z_grasp * 100:.1f} cm\n(pitch 10°, cube 90° symmetry)",
                 fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(resolve_path("docs/figures/reachability.png"), dpi=110); plt.close(fig)

    # grasp points outside the full-yaw region
    def lookup(x, y):
        ix = int(np.clip(np.round((x - xs[0]) / args.grid_step), 0, len(xs) - 1))
        iy = int(np.clip(np.round((y - ys[0]) / args.grid_step), 0, len(ys) - 1))
        return reach[iy, ix]
    gr = np.array([lookup(x, y) for x, y in zip(g.grasp_x, g.grasp_y)])

    # ------------------------------------------------------------- report
    A("# Kinematics report (Phase 1)\n")
    A("_Generated by `scripts/check_kinematics.py`._\n")
    A("## Definitions\n")
    A("- **FK:** URDF `so101_new_calib.urdf`, chain `base_link → gripper_frame_link`, radians "
      "(dataset degrees converted explicitly).")
    A("- **Wrist yaw:** heading on the table of the TCP x axis, which is the jaw-closing direction (the "
      "moving jaw rotates about TCP y; TCP z is the approach axis, pointing down when grasping).")
    A("- **Approach pitch:** tilt of TCP z from straight down, positive when leaning radially outward. For a "
      "5-DOF arm the approach axis always lies in the vertical plane through the arm (measured tangential "
      "component ≈ 0), so position + yaw leave exactly this one DOF free.")
    A(f"- **Joint limits:** union of URDF and observed range (+{cfg['kinematics']['joint_limits']['margin_deg']}°): "
      + ", ".join(f"{n} [{np.rad2deg(lo):.1f}, {np.rad2deg(hi):.1f}]°"
                  for n, lo, hi in zip(kin.joint_names, ik.lower, ik.upper)) + ".")
    A(f"- **Success:** position error ≤ {ik.pos_tol * 1e3:.1f} mm, yaw error ≤ {np.rad2deg(ik.yaw_tol):.1f}°, "
      "inside limits, yaw defined at the solution. Otherwise the target is flagged — never clipped.\n")
    A("## Solver\n")
    A("Bounded least squares (`scipy` TRF) with the analytic geometric Jacobian (checked against finite "
      "differences in the tests). Two stages: (1) position + yaw + **soft pitch target** choose the posture; "
      "(2) task-priority polish drives position/yaw error to zero from there. Seeds point the pan at the "
      "target and set the roll from the requested yaw; restarts are tried only if needed. Trajectories are "
      "solved with warm starts (each solution seeds the next and regularizes the redundant DOF).\n")
    A("## FK → IK round trips (random task-relevant configurations)\n")
    A("TCP in front of the robot (x > 3 cm), −1 cm < z < 25 cm, |pitch| < 60°, yaw defined.\n")
    A("| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]))
    for r in rows:
        A("| " + " | ".join(str(v) for v in r.values()) + " |")
    A("\nWithout the true pitch the solver finds *a* valid posture (task exact) but not necessarily the "
      "original joints: that is the redundant DOF.\n")
    A(f"## Recorded trajectories ({len(eps)} episodes, every 3rd frame, {n_pts} targets)\n")
    A("| pitch | feasible (mean / min per episode) | max \\|q − q_recorded\\| (median / max) | max joint step | ms / point |\n|---|---|---|---|---|")
    for mode, v in traj.items():
        v = np.array(v)
        A(f"| {mode} | {100 * v[:, 0].mean():.1f} % / {100 * v[:, 0].min():.1f} % | {np.median(v[:, 1]):.1f}° / "
          f"{v[:, 1].max():.1f}° | {v[:, 2].max():.1f}° | {t_traj[mode] / n_pts * 1e3:.1f} |")
    A("\n- With the demonstrated pitch, IK reproduces the recorded joint trajectories (the human postures).")
    A("- With a fixed default pitch the TCP path is still exact, but the posture can differ by tens of degrees, "
      "mostly near the folded rest pose where a small pitch change needs a very different folding. "
      "**Design consequences:** (a) the return to home is executed in joint space, not in Cartesian space; "
      "(b) Phase 4 should also regress the approach pitch (one extra GMR output) so IK can follow the "
      "demonstrated posture.\n")
    A("## Reachable workspace on the table\n")
    A(f"Grid step {args.grid_step * 100:.0f} cm at TCP z = {z_grasp * 100:.1f} cm (median demonstrated grasp "
      "height), pitch 10°. A cube with yaw φ is graspable if any gripper yaw φ + k·90° is feasible.\n")
    A("![](figures/reachability.png)\n")
    A(f"- Demonstrated grasps: {100 * np.mean(gr == 1):.1f} % lie where all cube yaws are graspable, "
      f"{100 * np.mean(gr > 0):.1f} % where at least one is.")
    A("- Positions outside the map (or partially graspable) are a **silent failure** for inference: "
      "`infer.py` must check IK feasibility of each target before execution and report the cube as "
      "unreachable rather than clip it.")
    resolve_path("docs/kinematics_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("report written")


if __name__ == "__main__":
    main()
