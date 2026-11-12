#!/usr/bin/env python
"""Metric camera calibration from the taped grid + registration to the robot base.

1. Median of the first frames (arm at home, cubes removed by the median).
2. Sub-pixel inner tape edges -> pixel->grid homography (+ tape width).
3. Camera pose in the grid frame from the homography and the measured camera height.
4. Grid -> robot-base rigid transform from (cube in first frame, FK at grasp) pairs,
   jointly with the cube-centre offset in the TCP frame; a free-scale variant
   checks the metric consistency of FK against the tape.

Writes docs/camera_calibration.md, docs/figures/grid_calibration.png and
outputs/camera_calibration.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_dataset import quick_cube_detector  # noqa: E402

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402
from so101_tpgmm.data.lerobot_dataset import LeRobotV3Dataset  # noqa: E402
from so101_tpgmm.kinematics.fk import SO101Kinematics  # noqa: E402
from so101_tpgmm.segmentation.gripper import GripperParams, detect_gripper_events  # noqa: E402
from so101_tpgmm.vision.camera_pose import (  # noqa: E402
    CameraPose,
    apply_homography,
    decompose_homography,
    focal_from_height,
)
from so101_tpgmm.vision.grid_calibration import GridSpec, extract_inner_edges, fit_grid, line_value  # noqa: E402


def rot2(th: float) -> np.ndarray:
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def decompose_with_pp(H: np.ndarray, height: float, size: tuple[int, int], pp_shift: tuple[float, float]) -> CameraPose:
    """Decompose after shifting the assumed principal point (sensitivity analysis)."""
    S = np.array([[1, 0, pp_shift[0]], [0, 1, pp_shift[1]], [0, 0, 1.0]])
    Hs = H @ S  # pixels expressed relative to a shifted centre
    f = focal_from_height(Hs, height, size)
    return decompose_homography(Hs, f, size)


def backproject_to_plane(pose: CameraPose, uv: np.ndarray, z: float) -> np.ndarray:
    rays = (np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(pose.K).T) @ pose.R
    lam = (z - pose.center[2]) / rays[:, 2]
    return pose.center + lam[:, None] * rays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    gcfg, sc = cfg["grid"], cfg["scene"]
    spec = GridSpec(gcfg["n_lanes"], gcfg["lane_width"], gcfg["lane_length"], gcfg["tape_width_guess"])
    size = tuple(cfg["camera"]["image_size"])
    h_cam = sc["camera_height_above_table"]
    cube = sc["cube_side"]

    ds = LeRobotV3Dataset(resolve_path(cfg["dataset"]["local_dir"]), cfg["dataset"]["state_key"])
    cam_key = cfg["dataset"]["fixed_camera_key"]
    kin = SO101Kinematics(resolve_path(cfg["kinematics"]["urdf"]))
    gp = GripperParams.from_config(cfg["gripper"])
    out_dir = resolve_path("outputs")
    out_dir.mkdir(exist_ok=True)

    # 1-2. background + grid fit --------------------------------------------------------
    bg_path = out_dir / "bg_first_frames.npy"
    if bg_path.exists():
        bg = np.load(bg_path)
    else:
        bg = np.median([ds.first_frame(cam_key, k) for k in range(0, ds.num_episodes, 4)], axis=0).astype(np.uint8)
        np.save(bg_path, bg)
    edges = extract_inner_edges(bg, spec, exclude=tuple(map(tuple, gcfg["exclude_boxes"])))
    grid = fit_grid(edges, spec, size, fit_distortion=False)
    grid_d = fit_grid(edges, spec, size, fit_distortion=True)
    per_line = {int(ln): float(np.mean(grid.residuals_m[edges.line == ln]) * 1e3) for ln in sorted(set(edges.line))}
    print(f"grid: {len(edges.uv)} edge points, tape {grid.tape_width * 1e3:.1f} mm, rms {grid.rms_mm:.2f} mm")

    # 3. camera pose in grid frame + sensitivity --------------------------------------
    f = focal_from_height(grid.H, h_cam, size)
    pose = decompose_homography(grid.H, f, size)
    sens = []
    for dpp in [(0, -20), (0, 20), (-20, 0), (20, 0)]:
        p = decompose_with_pp(grid.H, h_cam, size, dpp)
        sens.append((dpp, p.center.copy(), p.tilt_deg))
    sens_h = []
    for dh in (-0.05, 0.05):
        fh = focal_from_height(grid.H, h_cam + dh, size)
        sens_h.append((dh, fh, decompose_homography(grid.H, fh, size)))

    # 4. grid -> base ------------------------------------------------------------------
    uv, T = [], []
    for k in range(ds.num_episodes):
        ep = ds.episode(k)
        ev = detect_gripper_events(ep.state[:, 5], ds.fps, gp)
        if len(ev.cycles) != 1:
            continue
        c = quick_cube_detector(ds.first_frame(cam_key, k))
        if c is None:
            continue
        uv.append(c)
        T.append(kin.fk_from_dataset(ep.state[ev.cycles[0].close_idx], "degrees"))
    uv, T = np.array(uv), np.array(T)
    top = backproject_to_plane(pose, uv, cube)[:, :2]  # cube top-face centre (same XY as its centre)

    def res(x: np.ndarray, scale: bool = False, offset: bool = True) -> np.ndarray:
        s = x[6] if scale else 1.0
        o = x[3:6] if offset else np.zeros(3)
        pb = s * top @ rot2(x[0]).T + x[1:3]
        cc = T[:, :3, :3] @ o + T[:, :3, 3]
        return (pb - cc[:, :2]).ravel()

    best = None
    for th0 in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        s0 = least_squares(res, np.r_[th0, 0.3, 0, 0, 0, 0], loss="soft_l1", f_scale=0.005)
        best = s0 if best is None or s0.cost < best.cost else best
    x = best.x
    err = np.linalg.norm(res(x).reshape(-1, 2), axis=1)
    sS = least_squares(lambda z: res(z, scale=True), np.r_[x, 1.0], loss="soft_l1", f_scale=0.005)
    errS = np.linalg.norm(res(sS.x, scale=True).reshape(-1, 2), axis=1)
    s0 = least_squares(lambda z: res(np.r_[z, 0, 0, 0], offset=False), x[:3], loss="soft_l1", f_scale=0.005)
    err0 = np.linalg.norm(res(np.r_[s0.x, 0, 0, 0], offset=False).reshape(-1, 2), axis=1)

    R_gb, t_gb = rot2(x[0]), x[1:3]
    cube_c = T[:, :3, :3] @ x[3:6] + T[:, :3, 3]
    table_z = float(np.median(cube_c[:, 2]) - cube / 2)
    cam_base = np.r_[R_gb @ pose.center[:2] + t_gb, table_z + h_cam]
    pan_x = float(kin.chain[0].origin[0, 3])
    R_cam_base = np.eye(3)
    R_cam_base[:2, :2] = R_gb
    R_wc_base = pose.R @ R_cam_base.T  # base -> camera rotation

    # outputs -------------------------------------------------------------------------
    result = {
        "source": "scripts/calibrate_camera_grid.py (taped grid + measured height); principal point assumed at image centre",
        "image_size": list(size),
        "K": np.round(pose.K, 3).tolist(),
        "distortion_k1_diagnostic": round(grid_d.k1, 4),
        "camera_height_above_table": h_cam,
        "camera_center_base": np.round(cam_base, 4).tolist(),
        "R_base_to_camera": np.round(R_wc_base, 6).tolist(),
        "table_z_base": round(table_z, 4),
        "grid_to_base": {"theta_rad": round(float(x[0]), 6), "t": np.round(t_gb, 4).tolist()},
        "H_pixel_to_grid": np.round(grid.H, 10).tolist(),
        "tape_width": round(grid.tape_width, 4),
        "grasp_offset_tcp": np.round(x[3:6], 4).tolist(),
        "fk_vs_tape_scale": round(float(sS.x[6]), 4),
    }
    with open(out_dir / "camera_calibration.yaml", "w") as fh:
        yaml.safe_dump(result, fh, sort_keys=False)

    # figure
    img = bg.copy()
    Hinv = np.linalg.inv(grid.H)
    p = spec.lane_width + grid.tape_width
    for i in range(spec.n_lanes):
        X0, X1 = i * p, i * p + spec.lane_width
        poly = apply_homography(Hinv, np.array([[X0, 0], [X1, 0], [X1, spec.lane_length], [X0, spec.lane_length]]))
        cv2.polylines(img, [poly.astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), 1, cv2.LINE_AA)
    for (u, v), ln in zip(edges.uv[::3], edges.line[::3]):
        img[int(round(v)), int(round(u))] = (0, 255, 0)
    # base frame grid (5 cm) mapped to the image through base -> grid -> pixel
    def base_to_px(P: np.ndarray) -> np.ndarray:
        g_xy = (P - t_gb) @ R_gb  # inverse rigid
        return apply_homography(Hinv, g_xy)
    for xb in np.arange(0.0, 0.46, 0.05):
        pts = base_to_px(np.c_[np.full(40, xb), np.linspace(-0.2, 0.3, 40)])
        cv2.polylines(img, [pts.astype(np.int32).reshape(-1, 1, 2)], False, (255, 200, 0), 1, cv2.LINE_AA)
    for yb in np.arange(-0.2, 0.31, 0.05):
        pts = base_to_px(np.c_[np.linspace(0.0, 0.45, 40), np.full(40, yb)])
        cv2.polylines(img, [pts.astype(np.int32).reshape(-1, 1, 2)], False, (255, 200, 0), 1, cv2.LINE_AA)
    o_px = base_to_px(np.zeros((1, 2)))[0]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].imshow(img)
    axs[0].plot(*o_px, "r^", ms=9)
    axs[0].set_title("Inner tape edges (green), fitted 10x40 cm lanes (magenta),\n"
                     "robot base frame 5 cm grid (orange), base_link origin (red)", fontsize=9)
    axs[0].set_xlim(0, size[0]); axs[0].set_ylim(size[1], 0); axs[0].grid(False)
    r = grid.residuals_m * 1e3
    sc_ = axs[1].scatter(edges.uv[:, 0], edges.uv[:, 1], c=np.clip(r, -4, 4), cmap="coolwarm", s=3)
    axs[1].invert_yaxis(); axs[1].set_aspect("equal")
    axs[1].set_title("Edge residual (mm, point-to-line on the table plane)", fontsize=9)
    plt.colorbar(sc_, ax=axs[1], shrink=0.8)
    fig.tight_layout(); fig.savefig(resolve_path("docs/figures/grid_calibration.png"), dpi=110); plt.close(fig)

    # report --------------------------------------------------------------------------
    lane_w = {i: spec.lane_width * 1e3 + per_line[2 * i + 1] - per_line[2 * i] for i in range(spec.n_lanes)}
    L = []
    A = L.append
    A("# Camera calibration from the taped grid\n")
    A("_Generated by `scripts/calibrate_camera_grid.py`._\n")
    A(f"Input: {spec.n_lanes} taped lanes, inner size **{spec.lane_width * 100:.0f} × {spec.lane_length * 100:.0f} cm** "
      f"(measured), camera height **{h_cam * 100:.0f} cm** (measured, approx.), cube side {cube * 100:.0f} cm.\n")
    A("![](figures/grid_calibration.png)\n")
    A("## 1. Pixel → table homography\n")
    A(f"- {len(edges.uv)} sub-pixel inner-edge samples (the robot and the box hide parts of the grid, so edges "
      "are used instead of corners).")
    A(f"- Point-to-line residual: median {np.median(np.abs(r)):.2f} mm, RMS {grid.rms_mm:.2f} mm.")
    A(f"- Estimated tape width: {grid.tape_width * 1e3:.1f} mm.")
    A("- Per-lane width implied by the residuals (hand-taped tolerance): " +
      ", ".join(f"lane {i}: {w:.1f} mm" for i, w in lane_w.items()) + ".")
    A(f"- One-coefficient radial distortion fitted as a diagnostic: k1 = {grid_d.k1:+.3f} (normalised by f = 500 px), "
      f"RMS {grid_d.rms_mm:.2f} mm vs {grid.rms_mm:.2f} mm without it. Not significant enough to be separated "
      "from lane-taping errors with parallel lines only → **not used**.\n")
    A("## 2. Camera pose (principal point assumed at the image centre)\n")
    A(f"- Focal length from the measured height: **{f:.0f} px** (HFOV {pose.hfov_deg(size[0]):.1f}°). "
      "Agrees with the dataset-only estimate (493–504 px, `dataset_report.md` §6).")
    A(f"- Optical-axis tilt: **{pose.tilt_deg:.1f}°**; directly visible as keystone (the far end of the grid "
      "is wider in the image than the end next to the robot).")
    A(f"- Lens above the table at base (x, y) = **({cam_base[0] * 100:.1f}, {cam_base[1] * 100:.1f}) cm**, i.e. "
      f"**{(cam_base[0] - pan_x) * 100:.1f} cm** in front of the `shoulder_pan` axis; z = {cam_base[2] * 100:.1f} cm "
      f"in the base frame (table at z = {table_z * 100:.1f} cm).\n")
    A("Sensitivity (the dominant uncertainty is the unknown principal point):\n")
    A("| assumption | lens x from pan axis | lens y | tilt |\n|---|---|---|---|")
    A(f"| nominal | {(cam_base[0] - pan_x) * 100:.1f} cm | {cam_base[1] * 100:.1f} cm | {pose.tilt_deg:.1f}° |")
    for dpp, cg, tl in sens:
        cb = R_gb @ cg[:2] + t_gb
        A(f"| principal point shifted ({dpp[0]:+d}, {dpp[1]:+d}) px | {(cb[0] - pan_x) * 100:.1f} cm | "
          f"{cb[1] * 100:.1f} cm | {tl:.1f}° |")
    for dh, fh, ph in sens_h:
        cb = R_gb @ ph.center[:2] + t_gb
        A(f"| camera height {(h_cam + dh) * 100:.0f} cm (f = {fh:.0f} px) | {(cb[0] - pan_x) * 100:.1f} cm | "
          f"{cb[1] * 100:.1f} cm | {ph.tilt_deg:.1f}° |")
    A("\nThe dataset-only estimate (FK homography) gave 18–24 cm from the pan axis; the grid, being "
      "sub-millimetre on the table, constrains the keystone much better, so this estimate supersedes it "
      "**if** the principal point is near the centre. A tape measurement of the lens footprint (or a "
      "checkerboard intrinsic calibration) settles it.\n")
    A("## 3. Grid ↔ robot base\n")
    A(f"Registration from {len(uv)} pairs (cube top-face centre back-projected to z = {cube * 100:.0f} cm ↔ FK at grasp), "
      "rigid 2-D transform + cube-centre offset in the TCP frame:\n")
    A("| model | median error | p90 |\n|---|---|---|")
    A(f"| rigid, TCP origin | {np.median(err0) * 1e3:.1f} mm | {np.percentile(err0, 90) * 1e3:.1f} mm |")
    A(f"| rigid + cube offset in TCP frame | {np.median(err) * 1e3:.1f} mm | {np.percentile(err, 90) * 1e3:.1f} mm |")
    A(f"| similarity (free scale) + offset | {np.median(errS) * 1e3:.1f} mm | {np.percentile(errS, 90) * 1e3:.1f} mm |")
    A(f"\n- Grid rotation w.r.t. base: {np.degrees(x[0]):.1f}°; grid origin at base ({t_gb[0] * 100:.1f}, {t_gb[1] * 100:.1f}) cm.")
    base_in_grid = R_gb.T @ (-t_gb)
    total_w = spec.n_lanes * spec.lane_width + (spec.n_lanes - 1) * grid.tape_width
    A(f"- `base_link` origin in grid coordinates: ({base_in_grid[0] * 100:.1f}, {base_in_grid[1] * 100:.1f}) cm, i.e. "
      f"{(base_in_grid[1] - spec.lane_length) * 100:.1f} cm beyond the inner edge of the tape next to the robot and "
      f"{(base_in_grid[0] - total_w / 2) * 100:+.1f} cm from the grid's centre line (grid inner width "
      f"{total_w * 100:.1f} cm). **Physical check:** measure these two distances on the table.")
    A(f"- Cube-centre offset in the TCP frame: ({x[3] * 1e3:.1f}, {x[4] * 1e3:.1f}, {x[5] * 1e3:.1f}) mm. It differs "
      "from the dataset-only fit (−7.5, −17.2, −16.9) mm mainly along x: this component is weakly identifiable "
      "and partly traded against the rigid transform; Phase 3 re-estimates it.")
    A(f"- **FK vs tape scale: {sS.x[6]:.3f}** — FK distances are {abs(1 - sS.x[6]) * 100:.1f} % "
      f"{'shorter' if sS.x[6] < 1 else 'longer'} than tape distances. Candidates: URDF link lengths vs the "
      "real arm, follower joint calibration (gear backlash/offsets), or lanes not exactly 10 × 40 cm. "
      "To check on the robot: command two poses ~30 cm apart and measure the TCP displacement.")
    resolve_path("docs/camera_calibration.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"camera at base ({cam_base[0]:.3f}, {cam_base[1]:.3f}, {cam_base[2]:.3f}) m, f={f:.0f}, tilt={pose.tilt_deg:.1f}")


if __name__ == "__main__":
    main()
