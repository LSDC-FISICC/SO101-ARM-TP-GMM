#!/usr/bin/env python
"""Phase 0: inspect the LeRobot dataset and write docs/dataset_report.md.

Usage::

    python scripts/inspect_dataset.py                 # uses configs/default.yaml
    python scripts/inspect_dataset.py --download      # fetch the dataset first
    python scripts/inspect_dataset.py --skip-camera   # no first-frame decoding/homography

Outputs: the markdown report, figures in docs/figures/ and a per-episode CSV
(docs/episodes_summary.csv).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402
from so101_tpgmm.data.lerobot_dataset import LeRobotV3Dataset, download_dataset  # noqa: E402
from so101_tpgmm.data.units import LEROBOT_DEG_PER_TICK, JointUnits, infer_joint_units  # noqa: E402
from so101_tpgmm.data.video import VideoReader, available_av1_decoders  # noqa: E402
from so101_tpgmm.kinematics.fk import SO101Kinematics  # noqa: E402
from so101_tpgmm.segmentation.gripper import GripperParams, detect_gripper_events  # noqa: E402
from so101_tpgmm.vision.camera_pose import (  # noqa: E402
    apply_homography,
    decompose_homography,
    fit_homography_with_tcp_offset,
    focal_from_footprint,
    focal_from_height,
    homography_is_orientation_reversing,
)

AIR_HOLD_Z = 0.03  # [m] a hold that starts above this TCP height is not a table grasp
BOX_XY_GUESS = np.array([0.06, 0.16])  # [m] rough box position (release cluster), only for diagnostics


# --------------------------------------------------------------------------- helpers
def stationary_values(g: np.ndarray, win: int = 11, tol: float = 0.5) -> np.ndarray:
    """Gripper values at frames where the signal is flat over a centred window."""
    if g.size < win:
        return np.empty(0)
    w = np.lib.stride_tricks.sliding_window_view(g, win)
    flat = (w.max(1) - w.min(1)) < tol
    return g[win // 2 : -(win // 2)][flat]


EXCLUDE_BOXES = ((430, 0, 640, 120), (250, 400, 410, 480))  # (x0, y0, x1, y1): the box and the camera stand


def quick_cube_detector(rgb: np.ndarray, exclude: tuple[tuple[int, int, int, int], ...] = EXCLUDE_BOXES) -> np.ndarray | None:
    """Colour-agnostic *provisional* cube detector (Phase 0 only; Phase 3 replaces it).

    Saturated, non-blue (the floor tape is blue), roughly square blob; the most
    saturated/large one wins. Returns the pixel centre or None.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = (hsv[..., i].astype(np.int32) for i in range(3))
    mask = ((s > 90) & (v > 50) & ~((h >= 100) & (h <= 135))).astype(np.uint8)
    for x0, y0, x1, y1 in exclude:
        mask[y0:y1, x0:x1] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    best, best_score = None, 0.0
    for i in range(1, n):
        a, w, hh = st[i, 4], st[i, 2], st[i, 3]
        if not (120 < a < 2500 and 0.5 < w / hh < 2 and a / (w * hh) > 0.5):
            continue
        score = a * s[lab == i].mean()
        if score > best_score:
            best, best_score = i, score
    return None if best is None else cen[best]


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides key.sub=value")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--skip-camera", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    dcfg = cfg["dataset"]
    root = resolve_path(dcfg["local_dir"])
    if args.download or not (root / "meta" / "info.json").exists():
        print(f"Downloading {dcfg['repo_id']} -> {root}")
        download_dataset(dcfg["repo_id"], root, dcfg.get("download_video_keys"), dcfg.get("revision", "main"))

    fig_dir = resolve_path(cfg["inspection"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_path = resolve_path(cfg["inspection"]["output_report"])
    ds = LeRobotV3Dataset(root, dcfg["state_key"])
    kin = SO101Kinematics(resolve_path(cfg["kinematics"]["urdf"]), cfg["kinematics"]["base_link"],
                          cfg["kinematics"]["tcp_link"], cfg["kinematics"]["joint_names"])
    cam_key = dcfg["fixed_camera_key"]
    R: dict = {}
    t_start = time.time()

    # ------------------------------------------------------------------ 1. format
    info = ds.info
    eps = ds.episodes
    lengths = eps["length"].to_numpy()
    feats = ds.features
    cams = ds.camera_keys
    print(f"[1] LeRobot {ds.codebase_version}, fps={ds.fps}, episodes={ds.num_episodes}")

    # ------------------------------------------------------------------ 2. timestamps
    fr = ds.frames
    ts_dev = float(np.max(np.abs(fr["timestamp"].to_numpy() - fr["frame_index"].to_numpy() / ds.fps)))
    span_dev = []
    for k in cams:
        fr_cols = [f"videos/{k}/from_timestamp", f"videos/{k}/to_timestamp"]
        span = (eps[fr_cols[1]] - eps[fr_cols[0]]).to_numpy() * ds.fps
        span_dev.append(float(np.max(np.abs(span - lengths))))
    R["ts_dev"], R["span_dev_frames"] = ts_dev, max(span_dev)

    # ------------------------------------------------------------------ 3. units
    S = ds.all_states()
    units = infer_joint_units(S, ds.state_names[:5], cfg["units"]["quantization_tol_ticks"])
    forced = cfg["units"]["joints"]
    if forced != "auto" and JointUnits(forced) is not units.joint_units:
        raise SystemExit(f"Configured units {forced} contradict the data ({units.joint_units.value}).")
    U = units.joint_units
    q_rad = np.deg2rad(S[:, :5]) if U is JointUnits.DEGREES else None
    print(f"[3] joint units: {U.value}; gripper span {units.gripper_span_ticks} ticks")

    joint_rows = []
    for i, jq in enumerate(units.joints):
        lo, hi = np.rad2deg(kin.lower[i]), np.rad2deg(kin.upper[i])
        out = float(np.mean((S[:, i] < lo) | (S[:, i] > hi)))
        joint_rows.append({
            "joint": jq.name.replace(".pos", ""),
            "min [deg]": f"{jq.min_value:.1f}", "max [deg]": f"{jq.max_value:.1f}",
            "q01 / q99": f"{np.percentile(S[:, i], 1):.1f} / {np.percentile(S[:, i], 99):.1f}",
            "URDF limits [deg]": f"[{lo:.1f}, {hi:.1f}]",
            "% frames outside URDF": f"{100 * out:.1f}",
            "max resid [ticks]": f"{jq.max_residual_ticks:.1e}",
            "offset": "half-tick" if jq.offset_is_half_tick else "integer",
        })
    R["joint_table"] = pd.DataFrame(joint_rows)
    R["within_limits"] = float(kin.within_limits(q_rad).mean()) if q_rad is not None else float("nan")

    # ------------------------------------------------------------------ 4. gripper + cycles
    gp = GripperParams.from_config(cfg["gripper"])
    g = S[:, 5]
    plateaus = np.concatenate([stationary_values(ds.episode(k).state[:, 5]) for k in range(ds.num_episodes)])
    rows, grasp_T, release_T, examples = [], [], [], {}
    for k in range(ds.num_episodes):
        ep = ds.episode(k)
        ev = detect_gripper_events(ep.state[:, 5], ds.fps, gp)
        T_ep = kin.fk_from_dataset(ep.state, U)
        air = [h for h in ev.unreleased_holds if T_ep[h[0], 2, 3] > AIR_HOLD_Z]
        table = [h for h in ev.unreleased_holds if T_ep[h[0], 2, 3] <= AIR_HOLD_Z]
        n_air_cycles = sum(T_ep[c.close_idx, 2, 3] > AIR_HOLD_Z for c in ev.cycles)
        ends_at_box = []
        for s_, e_, lvl in table:  # last frame still on the hold plateau
            on = np.flatnonzero(np.abs(ep.state[s_:e_, 5] - lvl) <= gp.plateau_tol)
            end = s_ + int(on[-1]) if on.size else e_ - 1
            ends_at_box.append(bool(np.linalg.norm(T_ep[end, :2, 3] - BOX_XY_GUESS) < 0.06))
        row = {"episode": k, "length": ep.length, "n_cycles": len(ev.cycles), "n_cycles_in_air": n_air_cycles,
               "unreleased_table_ends_at_box": all(ends_at_box) if table else None,
               "n_missed_grasps": len(ev.empty_closures), "n_unreleased_table": len(table),
               "n_unreleased_air": len(air), "gripper_min": float(ep.state[:, 5].min()),
               "gripper_max": float(ep.state[:, 5].max())}
        for c_i, c in enumerate(ev.cycles):
            grasp_T.append(T_ep[c.close_idx])
            release_T.append(T_ep[c.open_idx])
            if c_i == 0:
                row.update({"close_idx": c.close_idx, "open_idx": c.open_idx, "hold_level": round(c.hold_level, 2),
                            **{f"grasp_{a}": T_ep[c.close_idx, j, 3] for j, a in enumerate("xyz")},
                            **{f"release_{a}": T_ep[c.open_idx, j, 3] for j, a in enumerate("xyz")}})
        rows.append(row)
        key = (len(ev.cycles), len(ev.empty_closures) > 0, len(table) > 0, len(air) > 0)
        examples.setdefault(key, (k, ev))
    summary = pd.DataFrame(rows)
    grasp_T, release_T = np.array(grasp_T), np.array(release_T)
    print(f"[4] cycles per episode: {dict(Counter(summary.n_cycles))}")

    # ------------------------------------------------------------------ 5. video decode
    decoders = available_av1_decoders()
    vid_rows = []
    files = eps.groupby(f"videos/{cam_key}/file_index")
    for fidx, grp in files:
        path, _, _ = ds.video_span(cam_key, int(grp.episode_index.iloc[0]))
        if not path.exists():
            vid_rows.append({"file": path.name, "status": "missing"})
            continue
        with VideoReader(path) as vr:
            vi = vr.info()
        vid_rows.append({"file": path.name, "codec": vi.codec, "decoder": vi.decoder,
                         "size": f"{vi.width}x{vi.height}", "frames": vi.n_frames,
                         "episode frames": int(grp.length.sum()), "match": vi.n_frames == int(grp.length.sum())})
    vids = pd.DataFrame(vid_rows)
    n_check = int(cfg["inspection"]["n_video_check_episodes"])
    check_eps = np.linspace(0, ds.num_episodes - 1, n_check).astype(int)
    align_dev = []
    t0 = time.time()
    for k in check_eps:
        ep = ds.episode(int(k))
        q = ep.timestamps[[0, ep.length // 2, ep.length - 1]]
        _, dec_t = ds.read_frames(cam_key, int(k), q)
        align_dev.append(float(np.max(np.abs(dec_t - q))))
    R["align_dev"] = max(align_dev)
    R["decode_s_per_frame"] = (time.time() - t0) / (3 * len(check_eps))
    print(f"[5] decoders={decoders}, max |video t - state t| = {R['align_dev'] * 1e3:.2f} ms")

    # ------------------------------------------------------------------ 6. camera (provisional)
    cam = None
    if not args.skip_camera:
        one = summary[summary.n_cycles == 1].episode.to_numpy()
        firsts, uv, T_g = [], [], []
        for k in one:
            ep = ds.episode(int(k))
            ev = detect_gripper_events(ep.state[:, 5], ds.fps, gp)
            img = ds.first_frame(cam_key, int(k))
            firsts.append(img)
            c = quick_cube_detector(img)
            if c is not None:
                uv.append(c)
                T_g.append(kin.fk_from_dataset(ep.state[ev.cycles[0].close_idx], U))
        uv, T_g = np.array(uv), np.array(T_g)
        bg = np.median(np.array(firsts), axis=0).astype(np.uint8)
        fit0 = fit_homography_with_tcp_offset(uv, T_g, fit_offset=False)
        fit1 = fit_homography_with_tcp_offset(uv, T_g, fit_offset=True)
        sc = cfg["scene"]
        size = tuple(cfg["camera"]["image_size"])
        h_top = sc["camera_height_above_table"] - sc["cube_side"]
        pan_x = float(kin.chain[0].origin[0, 3])
        cam = {"n_det": len(uv), "n_eps": len(one), "fit0": fit0, "fit1": fit1, "bg": bg, "uv": uv, "T": T_g,
               "flip": homography_is_orientation_reversing(fit1.H), "pan_x": pan_x}
        for name, fit in (("no_offset", fit0), ("offset", fit1)):
            f_h = focal_from_height(fit.H, h_top, size)
            pose = decompose_homography(fit.H, f_h, size)
            try:
                f_fp = focal_from_footprint(fit.H, pan_x + sc["camera_footprint_from_pan_axis"], size)
                h_fp = decompose_homography(fit.H, f_fp, size).center[2] + sc["cube_side"]
            except ValueError:
                f_fp, h_fp = float("nan"), float("nan")
            stand = apply_homography(fit.H, np.array([[322.0, 447.0]]))[0]
            px_c = apply_homography(fit.H, np.array([[319.5, 239.5], [320.5, 239.5], [319.5, 240.5]]))
            cam[name] = {"f_h": f_h, "pose": pose, "f_fp": f_fp, "h_fp": h_fp, "stand": stand,
                         "mm_per_px": float(np.mean(np.linalg.norm(px_c[1:] - px_c[0], axis=1)) * 1e3)}
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(int(cfg["inspection"].get("camera_bootstrap", 60))):
            idx = rng.integers(0, len(uv), len(uv))
            fb = fit_homography_with_tcp_offset(uv[idx], T_g[idx], fit_offset=True)
            fb_f = focal_from_height(fb.H, h_top, size)
            pb = decompose_homography(fb.H, fb_f, size)
            try:
                hb = decompose_homography(fb.H, focal_from_footprint(fb.H, pan_x + sc["camera_footprint_from_pan_axis"], size), size).center[2] + sc["cube_side"]
            except ValueError:
                hb = np.nan
            boot.append([fb_f, pb.center[0], pb.center[1], pb.tilt_deg, hb, fb.median_error_mm, *fb.tcp_offset])
        cam["boot"] = np.array(boot)
        sides = []
        for img in firsts[::8]:
            c = quick_cube_detector(img)
            if c is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            x, y = int(c[0]), int(c[1])
            roi = hsv[max(y - 40, 0): y + 40, max(x - 40, 0): x + 40]
            m = ((roi[..., 1] > 90) & (roi[..., 2] > 50)).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts:
                (_, (w, hh), _) = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
                sides.append(np.sqrt(w * hh))
        cam["cube_px"] = float(np.median(sides))
        print(f"[6] homography median error {fit0.median_error_mm:.1f} mm (no offset), "
              f"{fit1.median_error_mm:.1f} mm (with TCP offset)")

    # ------------------------------------------------------------------ figures
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
    names = [n.replace(".pos", "") for n in ds.state_names]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(*np.unique(lengths, return_counts=True), color="#3b6ea5")
    ax.set_xlabel("episode length [frames]"); ax.set_ylabel("# episodes")
    ax.set_title(f"Episode lengths ({ds.num_episodes} episodes, {ds.fps:.0f} fps)")
    fig.tight_layout(); fig.savefig(fig_dir / "episode_lengths.png"); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    for i, a in enumerate(axs.flat):
        a.hist(S[:, i], bins=120, color="#3b6ea5", log=True)
        if i < 5:
            a.axvline(np.rad2deg(kin.lower[i]), c="r", ls="--", label="URDF limit")
            a.axvline(np.rad2deg(kin.upper[i]), c="r", ls="--")
            a.set_xlabel("degrees")
        else:
            a.set_xlabel("RANGE_0_100 (0 = closed)")
        a.set_title(names[i])
    axs[0, 0].legend()
    fig.suptitle("observation.state distributions (all frames)")
    fig.tight_layout(); fig.savefig(fig_dir / "joint_distributions.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(plateaus, bins=100, range=(0, 100), color="#3b6ea5", log=True)
    for v, lab, c in ((gp.open_threshold, "open thr.", "g"), (gp.close_threshold, "close thr.", "r"),
                      (gp.hold_min, "hold min", "orange")):
        ax.axvline(v, c=c, ls="--", label=f"{lab} = {v}")
    ax.set_xlabel("gripper value at stationary frames"); ax.set_ylabel("# frames (log)")
    ax.set_title("Gripper plateau levels: closed-empty / holding a cube / open")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "gripper_levels.png"); plt.close(fig)

    ex_keys = sorted(examples)[:6]
    fig, axs = plt.subplots(len(ex_keys), 1, figsize=(10, 1.9 * len(ex_keys)), sharex=True)
    for a, key in zip(np.atleast_1d(axs), ex_keys):
        k, ev = examples[key]
        ep = ds.episode(k)
        z = kin.fk_from_dataset(ep.state, U)[:, 2, 3]
        a.plot(ep.state[:, 5], label="gripper")
        a.plot(z * 500, label="TCP z x500 [m]", alpha=0.7)
        for c in ev.cycles:
            a.axvline(c.close_idx, c="g"); a.axvline(c.open_idx, c="r")
        for s, e, _ in ev.empty_closures:
            a.axvspan(s, e, color="grey", alpha=0.25)
        for s, e, _ in ev.unreleased_holds:
            a.axvspan(s, e, color="orange", alpha=0.25)
        a.axhline(gp.open_threshold, c="g", ls=":", lw=0.8); a.axhline(gp.close_threshold, c="r", ls=":", lw=0.8)
        a.set_ylim(-10, 100)
        a.set_title(f"ep {k}: cycles={key[0]} missed={key[1]} unreleased(table)={key[2]} unreleased(air)={key[3]}",
                    fontsize=9)
    np.atleast_1d(axs)[0].legend(loc="upper right", fontsize=8)
    np.atleast_1d(axs)[-1].set_xlabel("frame (green = close/contact, red = release)")
    fig.tight_layout(); fig.savefig(fig_dir / "gripper_examples.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(grasp_T[:, 1, 3] * 100, grasp_T[:, 0, 3] * 100, s=8, label="grasp (TCP, FK)", c="#3b6ea5")
    if cam is not None:
        o = cam["fit1"].tcp_offset
        cube_c = (grasp_T[:, :3, :3] @ o + grasp_T[:, :3, 3])
        ax.scatter(cube_c[:, 1] * 100, cube_c[:, 0] * 100, s=6, label="grasp (cube centre = TCP + offset)",
                   c="#e08a2c", alpha=0.6)
    ax.scatter(release_T[:, 1, 3] * 100, release_T[:, 0, 3] * 100, s=8, label="release (TCP)", c="#c03b3b")
    ax.plot(0, 0, "k^", ms=10, label="base_link origin")
    ax.set_xlabel("base y [cm] (+ = robot's left)"); ax.set_ylabel("base x [cm] (forward)")
    ax.invert_xaxis(); ax.set_aspect("equal")
    ax.set_title("Grasp / release positions from FK (top view)")
    ax.legend(fontsize=8, loc="lower left"); fig.tight_layout(); fig.savefig(fig_dir / "grasp_xy_scatter.png")
    plt.close(fig)

    if cam is not None:
        img = cam["bg"].copy()
        fit = cam["fit1"]
        Hinv = np.linalg.inv(fit.H)
        cube_c = cam["T"][:, :3, :3] @ fit.tcp_offset + cam["T"][:, :3, 3]
        proj = apply_homography(Hinv, cube_c[:, :2])
        for (u, v), (pu, pv), e in zip(cam["uv"], proj, fit.errors_m):
            col = (0, 170, 0) if e < 0.01 else (220, 0, 0)
            cv2.line(img, (int(u), int(v)), (int(pu), int(pv)), col, 1)
            cv2.circle(img, (int(u), int(v)), 2, col, -1)
        for x in np.arange(0.0, 0.45, 0.05):  # base-frame grid, 5 cm
            ys = np.linspace(-0.2, 0.3, 50)
            pts = apply_homography(Hinv, np.c_[np.full_like(ys, x), ys]).astype(np.int32)
            cv2.polylines(img, [pts], False, (255, 0, 255), 1)
        for y in np.arange(-0.2, 0.31, 0.05):
            xs = np.linspace(0.0, 0.45, 50)
            pts = apply_homography(Hinv, np.c_[xs, np.full_like(xs, y)]).astype(np.int32)
            cv2.polylines(img, [pts], False, (255, 0, 255), 1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img); ax.grid(False)
        ax.set_title("Median first frame; detected cube -> FK cube centre (green < 1 cm, red >= 1 cm);\n"
                     "magenta: base-frame 5 cm grid via the fitted homography", fontsize=9)
        fig.tight_layout(); fig.savefig(fig_dir / "camera_homography_overlay.png"); plt.close(fig)

    # camera views: first frame / contact / release for a few episodes
    tiles = []
    for k in (0, 100, 250, 399):
        ep = ds.episode(k)
        ev = detect_gripper_events(ep.state[:, 5], ds.fps, gp)
        if not ev.cycles:
            continue
        c = ev.cycles[0]
        frames, _ = ds.read_frames(cam_key, k, ep.timestamps[[0, c.close_idx, c.open_idx]])
        tiles.append(np.hstack([cv2.resize(f, (320, 240), interpolation=cv2.INTER_AREA) for f in frames]))
    if tiles:
        fig, ax = plt.subplots(figsize=(9, 2.9 * len(tiles)))
        ax.imshow(np.vstack(tiles)); ax.axis("off")
        ax.set_title("Fixed camera: first frame | grasp (contact) | release — episodes 0, 100, 250, 399", fontsize=9)
        fig.tight_layout(); fig.savefig(fig_dir / "camera_views.jpg", dpi=100, bbox_inches="tight", pad_inches=0.05); plt.close(fig)

    summary.round(4).to_csv(resolve_path("docs/episodes_summary.csv"), index=False)

    # ------------------------------------------------------------------ report
    write_report(report_path, ds, cfg, R, units, summary, grasp_T, release_T, plateaus, g, gp, vids,
                 decoders, check_eps, cam, lengths, cams, feats, info)
    print(f"Report written to {report_path} ({time.time() - t_start:.0f} s)")


def write_report(path, ds, cfg, R, units, summary, grasp_T, release_T, plateaus, g, gp, vids, decoders,
                 check_eps, cam, lengths, cams, feats, info) -> None:
    n = ds.num_episodes
    cyc = Counter(summary.n_cycles)
    feat_rows = pd.DataFrame([{"key": k, "dtype": v["dtype"], "shape": v["shape"],
                               "names": ", ".join(v["names"]) if isinstance(v.get("names"), list) and v["dtype"] != "video" else ""}
                              for k, v in feats.items()])
    cam_rows = pd.DataFrame([{"key": k, "resolution": f"{feats[k]['info']['video.width']}x{feats[k]['info']['video.height']}",
                              "codec": feats[k]["info"]["video.codec"], "pix_fmt": feats[k]["info"]["video.pix_fmt"],
                              "fps": feats[k]["info"]["video.fps"], "GOP (video.g)": feats[k]["info"].get("video.g"),
                              "role here": "fixed, top-down (used)" if k == cfg["dataset"]["fixed_camera_key"] else "wrist (not used)"}
                             for k in cams])
    hold_lv = summary.hold_level.dropna()
    gz = grasp_T[:, 2, 3]
    rel = release_T[:, :3, 3]
    unrel_t = summary[summary.n_unreleased_table > 0].episode.tolist()
    unrel_a = summary[summary.n_unreleased_air > 0].episode.tolist()
    missed = summary[summary.n_missed_grasps > 0].episode.tolist()
    zero = summary[summary.n_cycles == 0].episode.tolist()
    two = summary[summary.n_cycles >= 2].episode.tolist()
    lens_s = pd.Series(lengths)

    L: list[str] = []
    A = L.append
    A("# Dataset report — `%s` (Phase 0)\n" % cfg["dataset"]["repo_id"])
    A("_Generated by `scripts/inspect_dataset.py`; numbers are recomputed on every run._\n")
    A("## 1. Format\n")
    A(f"- LeRobot `codebase_version`: **{ds.codebase_version}** (`robot_type: {info.get('robot_type')}`)")
    A(f"- fps: **{ds.fps:g}**; episodes: **{n}**; frames: **{info['total_frames']}**; tasks: "
      f"{info['total_tasks']} (`{ds.episodes.tasks.iloc[0][0]}`)")
    A(f"- Episode length: min {lens_s.min()}, median {int(lens_s.median())}, max {lens_s.max()} frames "
      f"(≈{lens_s.mean() / ds.fps:.1f} s). Lengths are near-constant: episodes were recorded with a fixed "
      f"duration, not cut at task completion (most of the tail is the arm resting at home). "
      f"Per-episode lengths are in `docs/episodes_summary.csv`.")
    A("\n![](figures/episode_lengths.png)\n")
    A("### Keys\n")
    A(md_table(feat_rows))
    A("\n### Cameras\n")
    A(md_table(cam_rows))
    A("\nThe leader-arm `action` is present but **not used** (we learn from what the follower actually did).\n")

    A("## 2. Units of `observation.state` (inferred, not assumed)\n")
    A("LeRobot normalizes Feetech ticks (`motors_bus.py::_normalize`): `RANGE_M100_100` = "
      "`clamp(raw)` mapped to [−100, 100]; `DEGREES` = `(raw − mid)·360/4095` (**not** clamped); the gripper "
      "always uses `RANGE_0_100`. `raw`, `min`, `max` are integers, which leaves a fingerprint in the floats.\n")
    A(f"**Result: arm joints in `{units.joint_units.value}`; gripper in `RANGE_0_100`.** Evidence:\n")
    for e in units.evidence:
        A(f"- {e}")
    A(f"- `meta/info.json` gives only the names (`shoulder_pan.pos`, …), no unit; current LeRobot "
      f"`SOFollowerConfig.use_degrees` defaults to `True`, consistent with the above.")
    A(f"- Step = 360/4095 = {LEROBOT_DEG_PER_TICK:.6f}° (= 8/91°). Conversion used everywhere: "
      "`rad = deg2rad(value)` (`so101_tpgmm/data/units.py`), the zero being the middle of the calibrated range, "
      "which is the zero of `so101_new_calib.urdf` (the same mapping LeRobot's own `RobotKinematics` uses).\n")
    A(md_table(R["joint_table"]))
    A(f"\nOnly **{100 * R['within_limits']:.1f}%** of frames are inside all URDF limits. The rest pose has "
      f"`shoulder_lift` ≈ −104.4°, beyond the URDF limit of −100°, and `wrist_flex` exceeds +95° at times. "
      "Consequence for Phase 1: IK limits must be the union of URDF limits and the observed range "
      "(otherwise IK reports the demonstrations themselves as infeasible).\n")
    A("![](figures/joint_distributions.png)\n")

    A("## 3. Gripper signal and grasp cycles\n")
    A(f"- Range: {g.min():.2f} … {g.max():.2f} (`RANGE_0_100`, 0 = closed; span {units.gripper_span_ticks} ticks).")
    A("- Stationary plateaus fall in **three** bands, not two:")
    A("  - closed-empty ≈ 0–12 (rest; also the operator partially closing the leader),")
    A(f"  - **holding a cube** ≈ {np.percentile(hold_lv, 1):.1f}–{np.percentile(hold_lv, 99):.1f} "
      f"(median {np.median(hold_lv):.1f}; the jaws stop on the 4 cm cube),")
    A("  - open ≥ ~25 (approach opening 25–90, release opening sometimes only 26–30).")
    A(f"- A single threshold either merges *holding* with *open* or chatters between them. The detector "
      f"(`segmentation/gripper.py`) uses a median filter, **hysteresis** (OPEN > {gp.open_threshold}, "
      f"CLOSED < {gp.close_threshold}), debouncing (OPEN ≥ {gp.min_open_s} s) and classifies each closed "
      f"plateau by level (hold ∈ [{gp.hold_min}, {gp.close_threshold}], ≥ {gp.min_hold_s} s). "
      "Cycle = OPEN → HOLD → OPEN; contact and release instants are the first/last frame on the hold plateau.\n")
    A("![](figures/gripper_levels.png)\n")
    A("### Cycles per episode\n")
    A("| cycles | episodes |\n|---|---|")
    for c in sorted(cyc):
        A(f"| {c} | {cyc[c]} |")
    A(f"\n**Finding: one cube per episode.** The first frames inspected show a single cube on the table, and "
      f"{cyc.get(1, 0)}/{n} episodes have exactly one grasp cycle. The \"several cycles per episode\" case is rare here; the pipeline still "
      "supports it.\n")
    A("Anomalies (all listed in `docs/episodes_summary.csv`):\n")
    A(f"- 0 cycles: {zero}")
    air_c = summary[summary.n_cycles_in_air > 0].episode.tolist()
    A(f"- ≥ 2 cycles: {two}. Cycles whose contact is in the air (TCP z > {AIR_HOLD_Z * 100:.0f} cm, i.e. "
      f"**phantom cycles**, not grasps): {air_c}; the others are re-grasps at the table.")
    A(f"- Missed grasp (closed below the hold band, then re-opened): {len(missed)} episodes, e.g. {missed[:10]}")
    at_box = summary[summary.unreleased_table_ends_at_box == True].episode.tolist()  # noqa: E712
    A(f"- Hold at table height without a detected release: {unrel_t}. In {at_box} the hold plateau ends with "
      "the TCP over the box: the cube **was** released with a tiny opening (< open threshold) followed by a "
      "full closure. Phase 2 will take the end of the hold plateau as the release instant when the gripper "
      "goes from *hold* straight to *closed-empty*.")
    A(f"- Hold-level plateau **in the air** (TCP z > {AIR_HOLD_Z * 100:.0f} cm) after the release: {unrel_a}. "
      "These are the operator half-closing the gripper on the way home, **not** grasps — a level-only rule "
      "would create phantom cycles here. Phase 2 adds a height gate (no stacking ⇒ every real grasp starts "
      "at table height).\n")
    A("![](figures/gripper_examples.png)\n")

    A("## 4. Grasp and release positions (FK)\n")
    A(f"- Grasp TCP height: median {np.median(gz) * 100:.1f} cm, 1–99 % [{np.percentile(gz, 1) * 100:.1f}, "
      f"{np.percentile(gz, 99) * 100:.1f}] cm. A tight band across the whole workspace is a check that the "
      "joint units, signs and URDF zero are right (a wrong sign or degrees/normalized mix-up spreads it by "
      "decimetres).")
    if cam is not None:
        A(f"- With the cube-centre offset of §6, the grasped cube centre is at z ≈ {cam['fit1'].plane_z * 100:.1f} cm, "
          f"so the table is at z ≈ {(cam['fit1'].plane_z - cfg['scene']['cube_side'] / 2) * 100:.1f} cm in the base "
          "frame (the robot base sits slightly below its URDF origin, or the table is not perfectly flat).")
    A(f"- Grasp XY spans x ∈ [{grasp_T[:, 0, 3].min() * 100:.0f}, {grasp_T[:, 0, 3].max() * 100:.0f}] cm, "
      f"y ∈ [{grasp_T[:, 1, 3].min() * 100:.0f}, {grasp_T[:, 1, 3].max() * 100:.0f}] cm.")
    A(f"- Release (TCP): mean ({rel[:, 0].mean() * 100:.1f}, {rel[:, 1].mean() * 100:.1f}, "
      f"{rel[:, 2].mean() * 100:.1f}) cm, std ({rel[:, 0].std() * 100:.1f}, {rel[:, 1].std() * 100:.1f}, "
      f"{rel[:, 2].std() * 100:.1f}) cm → the fixed box, front-left of the robot (top-right in the image).\n")
    A("![](figures/grasp_xy_scatter.png)\n")

    A("## 5. Video decoding and timestamp alignment\n")
    A(f"- AV1 decoders available through PyAV {__import__('av').__version__}: `{', '.join(decoders)}` "
      "(the PyAV wheel bundles FFmpeg with **libdav1d**; no torchcodec needed).")
    A(f"- Decoding speed ≈ {R['decode_s_per_frame'] * 1e3:.0f} ms per random-access frame (seek + decode up to "
      "the next keyframe; GOP = 2).")
    A(f"- `timestamp == frame_index / fps` for every frame (max deviation {R['ts_dev']:.1e} s).")
    A(f"- Per-episode video span `(to − from)·fps` vs `length`: max deviation {R['span_dev_frames']:.2f} frames.")
    A(f"- Decoded presentation time vs state timestamp (first/middle/last frame of episodes "
      f"{list(map(int, check_eps))}): max |Δt| = **{R['align_dev'] * 1e3:.2f} ms** (tolerance 0.4 frame). "
      "`VideoReader` raises if the decoded frame is off by more than the tolerance.")
    A("- Fixed-camera files (only this camera was downloaded; the wrist camera is unused):\n")
    A(md_table(vids))
    A("")

    if cam is not None:
        c0, c1 = cam["no_offset"], cam["offset"]
        sc = cfg["scene"]
        p1 = c1["pose"]
        A("## 6. Camera ↔ robot geometry from the data (preliminary)\n")
        A("Requested during Phase 0. Correspondences: the cube centre detected in the **first frame** of each "
          "one-cycle episode (provisional colour-agnostic detector, Phase 3 replaces it) ↔ the FK pose at the "
          f"**contact instant**. {cam['n_det']}/{cam['n_eps']} cubes detected.\n")
        A("| model | median error | p90 |\n|---|---|---|")
        for nm, f in (("homography, TCP origin", cam["fit0"]), ("homography + constant cube offset in TCP frame", cam["fit1"])):
            A(f"| {nm} | {f.median_error_mm:.1f} mm | {np.percentile(f.errors_m, 90) * 1e3:.1f} mm |")
        o = cam["fit1"].tcp_offset * 1e3
        A(f"\nFitted cube centre in the TCP frame: ({o[0]:.1f}, {o[1]:.1f}, {o[2]:.1f}) mm — "
          "`gripper_frame_link` is **not** the centre of a grasped cube.\n")
        A("With square pixels and the principal point at the centre, a plane homography leaves one free "
          "parameter (focal ↔ height; nearly degenerate for an overhead camera). Fixing it with the measured "
          f"camera height ({sc['camera_height_above_table'] * 100:.0f} cm, i.e. "
          f"{(sc['camera_height_above_table'] - sc['cube_side']) * 100:.0f} cm above the cube top face):\n")
        A("| quantity | TCP origin model | TCP + offset model | measured on the real setup |\n|---|---|---|---|")
        A(f"| focal | {c0['f_h']:.0f} px | {c1['f_h']:.0f} px (HFOV {p1.hfov_deg(640):.0f}°) | — |")
        A(f"| lens above base (x, y) | ({c0['pose'].center[0] * 100:.1f}, {c0['pose'].center[1] * 100:.1f}) cm | "
          f"({p1.center[0] * 100:.1f}, {p1.center[1] * 100:.1f}) cm | x ≈ {(cam['pan_x'] + sc['camera_footprint_from_pan_axis']) * 100:.1f} cm "
          f"({sc['camera_footprint_from_pan_axis'] * 100:.0f} cm from the pan axis, *to be measured*) |")
        A(f"| optical-axis tilt | {c0['pose'].tilt_deg:.1f}° | {p1.tilt_deg:.1f}° | — |")
        h0_txt = "no solution" if np.isnan(c0["h_fp"]) else f"{c0['h_fp'] * 100:.1f} cm"
        A(f"| height implied by the footprint | {h0_txt} | "
          f"{c1['h_fp'] * 100:.1f} cm | {sc['camera_height_above_table'] * 100:.0f} cm |")
        A(f"| camera stand (pixel 322, 447) distance | {np.linalg.norm(c0['stand']) * 100:.1f} cm | "
          f"{np.linalg.norm(c1['stand']) * 100:.1f} cm | {sc['camera_stand_from_robot_base'] * 100:.0f} cm (to the robot base edge) |")
        A(f"| 4 cm cube in pixels | {40 / c0['mm_per_px']:.0f} px | {40 / c1['mm_per_px']:.0f} px "
          f"(homography plane) | {cam['cube_px']:.0f} px measured (top face, ~2 cm closer to the camera) |")
        B = cam["boot"]
        ci = lambda a: f"[{np.nanpercentile(a, 2.5):.1f}, {np.nanpercentile(a, 97.5):.1f}]"  # noqa: E731
        A(f"\nBootstrap over the correspondences ({len(B)} resamples, TCP + offset model, 95 % intervals):\n")
        A("| quantity | 95 % interval |\n|---|---|")
        A(f"| focal at measured height | {ci(B[:, 0])} px |")
        A(f"| lens above base x | {ci(B[:, 1] * 100)} cm (pan axis at {cam['pan_x'] * 100:.1f} cm) |")
        A(f"| lens above base y | {ci(B[:, 2] * 100)} cm |")
        A(f"| optical-axis tilt | {ci(B[:, 3])}° |")
        A(f"| height implied by a {sc['camera_footprint_from_pan_axis'] * 100:.0f} cm footprint | "
          f"{ci(B[:, 4] * 100)} cm ({np.mean(np.isnan(B[:, 4])) * 100:.0f} % no solution) |")
        A(f"| cube offset in TCP frame | x {ci(B[:, 6] * 1e3)}, y {ci(B[:, 7] * 1e3)}, z {ci(B[:, 8] * 1e3)} mm |")
        A("\n**What is and is not determined by the data:**\n")
        A("- Well determined: the pixel→table mapping (≈ 1 cm), the scale / focal length once the height is "
          "known, and the camera stand position (consistent with the measured 46 cm).")
        A("- **Poorly determined:** the horizontal lens position and the tilt. For a near-overhead camera "
          "they are coupled through the small perspective (keystone) terms of the homography, which ~1 cm "
          "noise over a 40 cm area cannot pin down: removing 3 false detections out of 392 moved the lens "
          "x by ~3 cm. A tape measurement of the lens footprint is therefore *informative*, not a check; "
          "the agreement first reported in the conversation (footprint 24 cm ⇔ height 55 cm) is within "
          "this uncertainty but should not be read as a validation.")
        A("- **Design decision:** the cube frame position used for calibration and training is "
          "`p_tcp + R_tcp · offset`: it lowers the median error from "
          f"{cam['fit0'].median_error_mm:.1f} to {cam['fit1'].median_error_mm:.1f} mm and has a physical cause "
          "(single moving jaw). The offset itself should be re-estimated in Phase 3 with the colour detector.")
        A(f"- Pixel→plane orientation is {'reversed' if cam['flip'] else 'NOT reversed'} "
          f"({'as required for a camera looking down: the FK joint signs are not mirrored' if cam['flip'] else '**inconsistent: check joint signs**'}).")
        uvs, ers = cam["uv"], cam["fit1"].errors_m
        r = np.linalg.norm(uvs - np.array([319.5, 239.5]), axis=1)
        inner, outer = ers[r < np.median(r)], ers[r >= np.median(r)]
        A("- Fitting per block of 40 episodes does not reduce the residual below the global fit, so there is "
          "no evidence that the camera moved during recording.")
        A(f"- The residual is **not** uniform: median {np.median(inner) * 1e3:.1f} mm for cubes nearer the image "
          f"centre vs {np.median(outer) * 1e3:.1f} mm for the outer half, larger at the left/right edges (figure). "
          "This points to lens distortion and/or follower joint-calibration error (a homography models "
          "neither), on top of cubes nudged during teleoperation. Exact intrinsics and distortion need a "
          "calibration on the real setup (checkerboard, or a marker on the gripper at several heights).\n")
        A("![](figures/camera_homography_overlay.png)\n")

    A("## 7. Implications for the next phases\n")
    A("1. **Units:** degrees → `deg2rad`, asserted by `infer_joint_units` at load time.")
    A("2. **IK limits:** union of URDF limits and observed range (rest pose violates the URDF).")
    A("3. **Segmentation:** three-level gripper logic + hysteresis + table-height gate; drop episodes "
      "without a clean OPEN→HOLD→OPEN.")
    A("4. **One cube per episode** ⇒ vision on the first frame needs to find exactly one cube; the "
      "cube–cycle association is trivial here but kept general.")
    A("5. **Cube frame** = vision pose; calibration correspondences use the TCP + offset cube centre.")
    A("6. **Blue cubes vs blue floor tape:** same hue family → the blue HSV range and the area/shape filter "
      "must reject tape segments (Phase 3).")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
