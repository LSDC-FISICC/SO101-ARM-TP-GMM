#!/usr/bin/env python
"""Phase 3: cube detection, pixel->base calibration, cycle↔cube association.

For every segmented cycle: decode the reference frame (just before motion onset),
detect cubes, validate the scene, calibrate the pixel->base homography on
(detected centre, FK cube centre at contact) pairs with RANSAC, express every cube
in the base frame (x, y, yaw folded to ±45°), associate each cycle with the nearest
cube and compare with independent checks (grid calibration, gripper yaw).

Outputs: outputs/vision_calibration.yaml, outputs/cube_frames.csv (+ docs copy),
docs/vision_report.md, docs/figures/vision_*.png|jpg.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402
from so101_tpgmm.data.lerobot_dataset import LeRobotV3Dataset  # noqa: E402
from so101_tpgmm.kinematics.fk import SO101Kinematics  # noqa: E402
from so101_tpgmm.kinematics.ik import tool_yaw  # noqa: E402
from so101_tpgmm.vision.calibration import calibrate_homography  # noqa: E402
from so101_tpgmm.vision.detection import DetectorParams, detect_cubes, fold_yaw_90  # noqa: E402
from so101_tpgmm.vision.draw import COLORS_RGB, draw_detections  # noqa: E402
from so101_tpgmm.vision.geometry import PixelToBase  # noqa: E402
from so101_tpgmm.vision.scene import SceneParams, associate, changed_fraction, reference_frame_index, validate_detections  # noqa: E402


def load_intrinsics(vcfg: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    intr = vcfg.get("intrinsics")
    if not intr:
        return None, None
    return np.array(intr["K"], dtype=np.float64), np.array(intr["dist"], dtype=np.float64)


def backproject(cam: dict, uv: np.ndarray, z: float) -> np.ndarray:
    """Pixel rays of the grid-calibrated camera intersected with the plane base z = ``z``."""
    K = np.array(cam["K"])
    R = np.array(cam["R_base_to_camera"])
    C = np.array(cam["camera_center_base"])
    rays = (np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T) @ R
    lam = (z - C[2]) / rays[:, 2]
    return (C + lam[:, None] * rays)[:, :2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    vcfg = cfg["vision"]
    ds = LeRobotV3Dataset(resolve_path(cfg["dataset"]["local_dir"]), cfg["dataset"]["state_key"])
    cam_key = cfg["dataset"]["fixed_camera_key"]
    kin = SO101Kinematics(resolve_path(cfg["kinematics"]["urdf"]))
    seg = pd.read_csv(resolve_path(cfg["segmentation"]["output_csv"]))
    dp = DetectorParams.from_config(vcfg)
    sp = SceneParams.from_config(vcfg["scene"], cfg["scene"]["cube_side"])
    K, dist = load_intrinsics(vcfg)
    fig_dir = resolve_path("docs/figures")

    # ------------------------------------------------------------- detection per cycle
    recs, frames = [], {}
    win = int(round(sp.static_window_s * ds.fps))
    for r in seg.itertuples():
        ep = ds.episode(int(r.episode))
        i = reference_frame_index(r.approach_start, sp)
        j = max(0, i - win)
        fr, _ = ds.read_frames(cam_key, int(r.episode), ep.timestamps[sorted({i, j})])
        ref, prev = fr[-1], fr[0]
        frames[int(r.episode)] = ref
        dets = detect_cubes(ref, dp)
        T_ep = kin.fk_from_dataset(ep.state, "degrees")
        recs.append(dict(r=r, dets=dets, ref_idx=i, tcp_ref=T_ep[i, :3, 3], T_grasp=T_ep[int(r.close_idx)],
                         static=changed_fraction(ref, prev, sp.static_diff_threshold, [tuple(sp.arm_home_roi_px)])))
    tcp_home = np.median([x["tcp_ref"] for x in recs], axis=0)
    n_det = Counter(len(x["dets"]) for x in recs)
    print(f"[1] detections per cycle: {dict(n_det)}")

    # ------------------------------------------------------------- calibration
    def und(uv):
        return PixelToBase(np.eye(3), K, dist)._und(uv)

    one = [x for x in recs if len(x["dets"]) == 1]
    uv = und(np.array([x["dets"][0].center_px for x in one]))
    T = np.array([x["T_grasp"] for x in one])
    ccfg = vcfg["calibration"]
    cal = calibrate_homography(uv, T, ccfg["ransac_threshold"], ccfg["fit_tcp_offset"])
    cal_no = calibrate_homography(uv, T, ccfg["ransac_threshold"], False)
    cal.save(resolve_path(ccfg["output"]))
    p2b = PixelToBase(cal.H, K, dist)
    print(f"[2] homography: {cal.inliers.sum()}/{len(uv)} inliers, median {cal.inlier_median_mm:.1f} mm")

    # ------------------------------------------------------------- frames + association
    rows = []
    for x in recs:
        r, dets = x["r"], x["dets"]
        scene_flags = validate_detections(dets, p2b, sp, x["tcp_ref"], tcp_home, x["static"])
        Tg = x["T_grasp"]
        cube_fk = Tg[:3, :3] @ cal.tcp_offset + Tg[:3, 3]
        xy = p2b.to_base(np.array([d.center_px for d in dets])) if dets else np.zeros((0, 2))
        a = associate(cube_fk[:2], xy, vcfg["association_max_distance"])
        g_yaw = float(tool_yaw(Tg[:3, :3]))
        row = dict(episode=int(r.episode), cycle=int(r.cycle), ref_frame=x["ref_idx"], n_detections=len(dets),
                   scene_flags=";".join(scene_flags), static_fraction=round(x["static"], 5),
                   fk_x=cube_fk[0], fk_y=cube_fk[1], fk_z=cube_fk[2], fk_yaw=float(fold_yaw_90(g_yaw)),
                   gripper_yaw=g_yaw, assoc_distance=a.distance_m)
        if a.detection_index is not None:
            d = dets[a.detection_index]
            yaw_b = p2b.yaw_to_base(d.center_px, d.yaw_img)
            row.update(color=d.color, u=d.center_px[0], v=d.center_px[1], yaw_img=d.yaw_img,
                       vision_x=xy[a.detection_index, 0], vision_y=xy[a.detection_index, 1], vision_yaw=yaw_b,
                       det_flags=";".join(d.flags), area_px=d.area_px,
                       yaw_residual=float(fold_yaw_90(g_yaw - yaw_b)),
                       yaw_residual_image_direct=float(fold_yaw_90(g_yaw - d.yaw_img)))
        else:
            row.update(color=None, det_flags="no_association")
        rows.append(row)
    df = pd.DataFrame(rows)
    df["vision_z"] = cal.plane_z
    df["position_error_mm"] = np.hypot(df.vision_x - df.fk_x, df.vision_y - df.fk_y) * 1e3
    max_err = vcfg["max_vision_fk_error_mm"]
    pushed = df.position_error_mm > max_err
    df.loc[pushed, "det_flags"] = [";".join([f for f in (str(x) if isinstance(x, str) else "").split(";") if f] + ["cube_pushed"])
                                   for x in df.loc[pushed, "det_flags"]]
    usable = (df.color.notna() & (df.scene_flags == "") & ~df.det_flags.fillna("").str.contains("partial_blob")
              & ~pushed)
    df["usable"] = usable
    out = resolve_path(vcfg["frames_csv"])
    df.round(6).to_csv(out, index=False)
    df.round(6).to_csv(resolve_path("docs/cube_frames.csv"), index=False)
    print(f"[3] associated {df.color.notna().sum()}/{len(df)}, usable {usable.sum()}")

    # ------------------------------------------------------------- independent checks
    checks = {}
    cam_path = resolve_path("outputs/camera_calibration.yaml")
    if cam_path.exists():
        cam = yaml.safe_load(open(cam_path))
        uv_all = und(df.loc[df.color.notna(), ["u", "v"]].to_numpy())
        tz = cam["table_z_base"]
        top = backproject(cam, uv_all, tz + cfg["scene"]["cube_side"])
        table = backproject(cam, uv_all, tz)
        fk = df.loc[df.color.notna(), ["fk_x", "fk_y"]].to_numpy()
        vis = df.loc[df.color.notna(), ["vision_x", "vision_y"]].to_numpy()
        checks["grid_top_vs_dataset_H_mm"] = float(np.median(np.linalg.norm(top - vis, axis=1)) * 1e3)
        checks["grid_top_vs_fk_mm"] = float(np.median(np.linalg.norm(top - fk, axis=1)) * 1e3)
        checks["grid_table_plane_vs_fk_mm"] = float(np.median(np.linalg.norm(table - fk, axis=1)) * 1e3)
        checks["grid_table_plane_vs_fk_p90_mm"] = float(np.percentile(np.linalg.norm(table - fk, axis=1), 90) * 1e3)
    print(f"[4] checks: {checks}")

    # ------------------------------------------------------------- figures
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})
    ok = df[df.color.notna()]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for c, sub in ok.groupby("color"):
        col = np.array(COLORS_RGB.get(c, (0, 0, 0))) / 255
        axs[0].scatter(sub.vision_y * 100, sub.vision_x * 100, s=10, color=col, label=f"{c} ({len(sub)})")
    axs[0].plot(0, 0, "k^", ms=9)
    axs[0].invert_xaxis(); axs[0].set_aspect("equal"); axs[0].legend(fontsize=8)
    axs[0].set_xlabel("base y [cm]"); axs[0].set_ylabel("base x [cm]"); axs[0].set_title("Cube positions from vision (base frame)")
    inl = np.zeros(len(ok), bool)
    inl_set = {int(e) for e, i in zip([x["r"].episode for x in one], cal.inliers) if i}
    inl = ok.episode.isin(inl_set).to_numpy()
    for (vx, vy, fx, fy), good in zip(ok[["vision_x", "vision_y", "fk_x", "fk_y"]].to_numpy(), inl):
        axs[1].plot([vy * 100, fy * 100], [vx * 100, fx * 100], c="#2a9d4b" if good else "#c03b3b", lw=0.8)
    axs[1].invert_xaxis(); axs[1].set_aspect("equal")
    axs[1].set_title(f"Vision → FK cube centre (green: inlier ≤ {ccfg['ransac_threshold'] * 100:.1f} cm)")
    axs[1].set_xlabel("base y [cm]")
    axs[2].hist(np.degrees(ok.yaw_residual), bins=45, range=(-45, 45), alpha=0.8, label="via homography (used)")
    axs[2].hist(np.degrees(ok.yaw_residual_image_direct), bins=45, range=(-45, 45), alpha=0.5, label="image angle used directly")
    axs[2].set_xlabel("gripper yaw − cube yaw, folded to ±45° [deg]"); axs[2].legend(fontsize=8)
    axs[2].set_title("Yaw check against the grasp")
    fig.tight_layout(); fig.savefig(fig_dir / "vision_calibration.png", dpi=100); plt.close(fig)

    # montage: hand episodes (first frame vs reference), flagged and normal examples
    tiles = []
    F0 = {e: ds.first_frame(cam_key, e) for e in (59, 168, 170)}
    for e in (59, 168, 170):
        d0 = detect_cubes(F0[e], dp)
        tiles.append(draw_detections(F0[e], d0, exclude_boxes=dp.exclude_boxes, header=f"ep {e}: first frame ({len(d0)} cube)"))
        dr = detect_cubes(frames[e], dp)
        tiles.append(draw_detections(frames[e], dr, exclude_boxes=dp.exclude_boxes,
                                     header=f"ep {e}: reference frame ({len(dr)} cube)"))
    flagged = df[(df.det_flags.fillna("") != "") | (df.scene_flags != "")].episode.tolist()[:4]
    for e in [0, 100, 250, *flagged][:6]:
        dr = detect_cubes(frames[e], dp)
        validate_detections(dr, p2b, sp)
        tiles.append(draw_detections(frames[e], dr, exclude_boxes=dp.exclude_boxes, arm_roi=sp.arm_home_roi_px,
                                     header=f"ep {e}"))
    tiles = [cv2.resize(t, (320, 240), interpolation=cv2.INTER_AREA) for t in tiles]
    # cube pushed by the gripper before the grasp: reference / pre-grasp / contact
    rows_p = []
    for e in df.sort_values("position_error_mm", ascending=False).episode.head(4):
        r = seg[seg.episode == e].iloc[0]
        d = df[df.episode == e].iloc[0]
        fr, _ = ds.read_frames(cam_key, int(e), ds.episode(int(e)).timestamps[[int(d.ref_frame), int(r.pregrasp_idx), int(r.close_idx)]])
        row = []
        for f, lab in zip(fr, ("reference", "pre-grasp", "contact")):
            f = f.copy()
            cv2.circle(f, (int(d.u), int(d.v)), 7, (255, 0, 255), 2)
            cv2.putText(f, f"ep {e} {lab}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            row.append(cv2.resize(f, (320, 240), interpolation=cv2.INTER_AREA))
        rows_p.append(np.hstack(row))
    cv2.imwrite(str(fig_dir / "vision_pushed.jpg"), cv2.cvtColor(np.vstack(rows_p), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(fig_dir / "vision_detections.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])

    # ------------------------------------------------------------- report
    write_report(cfg, df, cal, cal_no, checks, n_det, recs, sp, dp, K)
    print("report written")


def write_report(cfg, df, cal, cal_no, checks, n_det, recs, sp, dp, K) -> None:
    ok = df[df.color.notna()]
    scene = Counter(f for s in df.scene_flags for f in s.split(";") if f)
    det = Counter(f for s in df.det_flags.fillna("") for f in s.split(";") if f)
    yr = np.degrees(np.abs(ok.yaw_residual))
    yd = np.degrees(np.abs(ok.yaw_residual_image_direct))
    L = []
    A = L.append
    A("# Vision report (Phase 3)\n")
    A("_Generated by `scripts/vision_pipeline.py`. Per-cycle frames: `docs/cube_frames.csv`._\n")
    A("## Detection\n")
    A("Fixed camera only. Per colour: HSV mask (red = two hue intervals, OpenCV hue wraps at 180) → exclusion "
      "boxes (the box, which holds a yellow object, and the camera stand) → opening/closing → connected components → "
      f"area {dp.area_px[0]}–{dp.area_px[1]} px, aspect ≤ {dp.max_aspect}, fill ≥ {dp.min_fill} → `minAreaRect` "
      "(centre, orientation folded to [−45°, 45°) by the cube's 90° symmetry). Ranges in `configs/default.yaml` "
      "(`vision.colors`).\n")
    A("| colour | HSV ranges | cycles |\n|---|---|---|")
    for c in dp.colors:
        A(f"| {c.name} | H {c.hue}, S ≥ {c.min_s}, V ≥ {c.min_v} | {int((ok.color == c.name).sum())} |")
    A(f"\nDetections per cycle at the reference frame: {dict(n_det)}. **No blue cube appears in this dataset** "
      "(the blue range is kept, with a value floor that rejects the floor tape: tape blobs have S ≈ 110, V ≈ 96).\n")
    A("### Reference frame: why not the first frame\n")
    A("In episodes 59, 168 and 170 the first frame shows the operator's **hand still placing the cube**: in 168/170 "
      "the cube is still in the hand (no detection); in 59 the skin passes the red hue/saturation test and is only "
      "rejected by the area filter. The scene is observed in the last frame "
      f"before motion onset (onset − {sp.reference_margin_frames} frames): the operator is then on the leader arm "
      "and the follower is at rest. With it every cycle has exactly one detection; elsewhere the cube does not "
      "move between frame 0 and the reference frame (max 3 px).\n")
    A("![](figures/vision_detections.jpg)\n")
    A("### Scene validation\n")
    A(f"- Arm at home: TCP within {sp.home_tolerance_m * 100:.0f} cm of the median home TCP (the rest posture varies "
      "by up to ~9° per joint between episodes, the TCP by ~3 cm). Flags: "
      f"`arm_not_home` × {scene.get('arm_not_home', 0)}.")
    A(f"- Static scene: fraction of pixels changing over {sp.static_window_s} s before the reference frame "
      f"> {sp.static_max_fraction}: `scene_not_static` × {scene.get('scene_not_static', 0)} "
      f"({df[df.scene_flags.str.contains('scene_not_static')].episode.tolist()}).")
    A(f"- Cube next to the arm at rest: `near_arm` × {det.get('near_arm', 0)} (informative; areas are normal).")
    A(f"- Partial blob (area < {sp.expected_area_ratio[0]} × expected footprint): `partial_blob` × "
      f"{det.get('partial_blob', 0)} ({df[df.det_flags.fillna('').str.contains('partial_blob')].episode.tolist()}). "
      "In these the cube is not hidden by the arm but lies on the dark tape, where part of a dark red cube drops "
      "out of the mask; the centre can be biased, so they are excluded from training.")
    err = df.position_error_mm.dropna()
    thr = cfg["vision"]["max_vision_fk_error_mm"]
    A(f"- **Cube pushed before the grasp** (`cube_pushed`): the vision position (reference frame) and the FK cube "
      f"centre at contact differ by more than {thr} mm in {int((err > thr).sum())} cycles "
      f"({int((err > 15).sum())} at 15 mm, {int((err > 25).sum())} at 25 mm). The frames below show the gripper "
      "dragging the cube before closing (episode 2: moved by an earlier failed attempt). For these demonstrations "
      "the initial cube pose — the frame TP-GMM is conditioned on — is not where the grasp happened, so they are "
      "excluded from training.")
    A(f"- Usable cycles (associated, no scene flag, no partial blob, not pushed): **{int(df.usable.sum())} / {len(df)}**.\n")
    A("![](figures/vision_pushed.jpg)\n")
    A("## Pixel → base calibration from the dataset\n")
    A("Correspondences: detected cube centre (reference frame) ↔ cube centre at contact from FK, "
      "`p_tcp + R_tcp · offset`, with the offset estimated jointly. RANSAC on the homography, then a soft-L1 refit "
      f"and re-classification (threshold {cal.ransac_threshold * 100:.1f} cm).\n")
    A("| model | inliers | median error (inliers) | p90 (inliers) | median (all) |\n|---|---|---|---|---|")
    for name, c in (("H + cube offset in TCP frame (used)", cal), ("H, TCP origin", cal_no)):
        A(f"| {name} | {c.inliers.sum()}/{len(c.errors_m)} | {c.inlier_median_mm:.1f} mm | {c.inlier_p90_mm:.1f} mm | "
          f"{np.median(c.errors_m) * 1e3:.1f} mm |")
    o = cal.tcp_offset * 1e3
    A(f"\n- Cube centre in the TCP frame: ({o[0]:.1f}, {o[1]:.1f}, {o[2]:.1f}) mm; cube-centre plane at base "
      f"z = {cal.plane_z * 100:.1f} cm.")
    A(f"- Lens distortion: {'intrinsics provided, pixels undistorted before the homography' if K is not None else 'no intrinsics configured (`vision.intrinsics: null`); the homography absorbs only the projective part'}.")
    d_ok = df[df.color.notna()]
    out15 = d_ok.position_error_mm > cal.ransac_threshold * 1e3
    right, left = d_ok.vision_y < -0.05, d_ok.vision_y > 0.05
    A(f"- Outliers are **not uniform**: {100 * out15[right].mean():.0f} % of the cubes on the robot's right "
      f"(base y < −5 cm) vs {100 * out15[left].mean():.0f} % on its left (y > 5 cm); median error "
      f"{d_ok.position_error_mm[right].median():.1f} vs {d_ok.position_error_mm[left].median():.1f} mm. Two causes "
      "add up there and cannot be separated with the data alone: the cube is pushed more often when the arm "
      "reaches across (see the frames below), and lens distortion grows towards the image edge. Intrinsics "
      "(`vision.intrinsics`) would remove the second one. All cycles are kept in the frames table with their error "
      "(`position_error_mm`).\n")
    A("![](figures/vision_calibration.png)\n")
    if checks:
        A("### Independent checks with the taped-grid camera calibration (`camera_calibration.md`)\n")
        A("| check | median |\n|---|---|")
        A(f"| grid camera, ray ∩ cube-top plane vs this homography | {checks['grid_top_vs_dataset_H_mm']:.1f} mm |")
        A(f"| grid camera, ray ∩ cube-top plane vs FK | {checks['grid_top_vs_fk_mm']:.1f} mm |")
        A(f"| grid camera, ray ∩ **table** plane vs FK (wrong plane) | {checks['grid_table_plane_vs_fk_mm']:.1f} mm "
          f"(p90 {checks['grid_table_plane_vs_fk_p90_mm']:.1f} mm) |")
        A("\nThe wrong plane is a silent failure: the mapping still looks smooth and plausible, but every cube is "
          "pulled towards the camera nadir by ~h/H. The dataset homography is fitted on the detections themselves, "
          "so its plane is consistent by construction. The residual difference between the grid camera and FK "
          "includes the 4.7 % FK-vs-tape scale discrepancy (`camera_calibration.md`).\n")
    A("## Yaw\n")
    A("The cube yaw is the `minAreaRect` angle pushed through the homography (two points on the edge direction), "
      "then folded to [−45°, 45°). Check: the jaw-closing axis at contact should be aligned with a cube face, i.e. "
      "gripper yaw − cube yaw ≈ 0 modulo 90°.\n")
    A("| method | median \\|residual\\| | p90 |\n|---|---|---|")
    A(f"| via homography (used) | {np.median(yr):.1f}° | {np.percentile(yr, 90):.1f}° |")
    A(f"| image angle used directly (wrong: the image is mirrored w.r.t. the table) | {np.median(yd):.1f}° | "
      f"{np.percentile(yd, 90):.1f}° |")
    A("\nThe remaining residual reflects how precisely the teleoperator aligned the gripper (and the cube rotating "
      "while the jaws close), not only vision error.\n")
    A("## Frames for TP-GMM\n")
    A("`outputs/cube_frames.csv` holds, per cycle, the **vision** cube frame (`vision_x, vision_y, vision_z, "
      "vision_yaw`, the default) and the **FK** alternative (`fk_x, fk_y, fk_z` = cube centre at contact, `fk_yaw` = "
      "gripper yaw folded to ±45°), the colour, flags and `usable`. Note for evaluation: the homography is fitted on "
      "all episodes, including the one left out in leave-one-out; with ~340 inliers one correspondence has a "
      "negligible influence, and this is stated in the evaluation report.")
    resolve_path("docs/vision_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
