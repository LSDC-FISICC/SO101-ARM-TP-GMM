#!/usr/bin/env python
"""MuJoCo simulation of the pick-and-place task with the learned policies.

Each episode places ``sim.n_cubes`` cubes (red, yellow, green) at random positions drawn from the
demonstrated workspace, with random yaw; their poses are known (no camera). The robot starts at
home and, for each cube in the configured colour order, runs the deployment pipeline: policy
rollout from the measured joints → IK → joint trajectory to the simulated servos → back home.
Every policy sees the same cube layouts (same seeds).

Grasp modes (``--grasp-modes``): ``contact`` = pure physics; ``assist`` = a cube touched by both
fingers is attached while the gripper is closed (removes the dependence on friction / mass, which
were not measured; the approach still has to put the cube between the fingers).

Outputs: ``outputs/sim/results.csv`` (one row per cube and run; copy in ``docs/mujoco_sim_results.csv``), ``docs/mujoco_sim.md``,
``docs/figures/sim_*.png`` and, with ``--video``, MP4/GIF files of the first episodes.

Requires MuJoCo (``pip install mujoco``) and the robot model (``scripts/fetch_so101_mujoco.sh``).
Off-screen rendering: ``MUJOCO_GL=egl``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402
from so101_tpgmm.policies import POLICIES  # noqa: E402

LABEL = {"segmented": "TP-GMM segmented", "monolithic": "TP-GMM monolithic", "ik_planner": "IK planner",
         "lpv_ds": "LPV-DS"}
_W: dict = {}


def _setup(config, overrides):
    from so101_tpgmm.kinematics.ik import ik_from_config
    from so101_tpgmm.sim.episode import FingerGeometry
    from so101_tpgmm.sim.scene import SceneSpec

    cfg = load_config(config, overrides)
    xml = resolve_path(cfg["sim"]["robot_dir"]) / "so101_new_calib.xml"
    if not xml.exists():
        raise SystemExit(f"{xml} not found: run scripts/fetch_so101_mujoco.sh")
    f = pd.read_csv(resolve_path(cfg["vision"]["frames_csv"]))
    _W.update(cfg=cfg, xml=xml, ik=ik_from_config(cfg), demo_xy=f[["fk_x", "fk_y"]].dropna().to_numpy(),
              geom=FingerGeometry(SceneSpec.from_config(cfg, xml, [])), policies={})


def _layout(seed: int):
    from so101_tpgmm.sim.episode import sample_cubes
    from so101_tpgmm.sim.scene import SceneSpec

    cubes = sample_cubes(np.random.default_rng(seed), _W["demo_xy"], _W["cfg"])
    return SceneSpec.from_config(_W["cfg"], _W["xml"], cubes)


def _policy(name):
    from so101_tpgmm.policies import load_policy

    if name not in _W["policies"]:
        _W["policies"][name] = load_policy(_W["cfg"], name)
    return _W["policies"][name]


def _job(args):
    name, seed, mode, noise, slots = args
    from so101_tpgmm.sim.episode import run_episode

    t0 = time.time()
    spec = _layout(seed)
    outs, _ = run_episode(_policy(name), _W["ik"], _W["cfg"], spec, _W["geom"], np.random.default_rng(10_000 + seed),
                          noise_mm=noise, grasp_mode=mode, box_offsets=_W["cfg"]["sim"]["box_slots"] if slots else None)
    return [dict(policy=name, seed=seed, grasp_mode=mode, noise_mm=noise, slots=slots, cube=i, wall_s=time.time() - t0,
                 **asdict(o)) for i, o in enumerate(outs)]


def _cameras():
    import mujoco

    persp = mujoco.MjvCamera()
    persp.lookat[:] = [0.2, 0.06, 0.0]
    persp.distance, persp.azimuth, persp.elevation = 0.8, -140.0, -38.0
    top = mujoco.MjvCamera()  # roughly the fixed camera: above the table, looking down
    top.lookat[:] = [0.25, 0.03, 0.0]
    top.distance, top.azimuth, top.elevation = 0.75, 180.0, -89.0
    return persp, top


def render_video(name: str, seed: int, mode: str, out_dir: Path, size=(480, 640), fps: int = 15,
                 slots: bool = False) -> Path:
    """Render one episode from two viewpoints (perspective + overhead) to MP4 and a small GIF."""
    import av
    from PIL import Image

    from so101_tpgmm.sim.episode import run_episode

    spec = _layout(seed)
    persp, top = _cameras()
    frames: list = []

    def grab(sim):  # both viewpoints side by side
        imgs = []
        for cam in (persp, top):
            sim.renderer.update_scene(sim.data, camera=cam)
            imgs.append(sim.renderer.render().copy())
        frames.append(np.concatenate(imgs, axis=1))

    outs, _ = run_episode(_policy(name), _W["ik"], _W["cfg"], spec, _W["geom"], np.random.default_rng(10_000 + seed),
                          grasp_mode=mode, render=size, frame_callback=grab,
                          box_offsets=_W["cfg"]["sim"]["box_slots"] if slots else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"sim_{name}_{mode}{'_slots' if slots else ''}_seed{seed}"
    with av.open(str(stem.with_suffix(".mp4")), "w") as c:
        st = c.add_stream("libx264", rate=fps)
        st.width, st.height, st.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        st.options = {"crf": "26"}
        for f in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    small = [Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2)) for f in frames[::3]]
    small = [im.quantize(colors=96, method=Image.Quantize.MEDIANCUT) for im in small]
    small[0].save(stem.with_suffix(".gif"), save_all=True, append_images=small[1:], duration=int(3000 / fps), loop=0,
                  optimize=True)
    status = ", ".join(f"{o.color}: {'in box' if o.in_box else 'lifted' if o.lifted else o.status if o.status != 'planned' else 'missed'}"
                       for o in outs)
    print(f"video {stem}.mp4 ({len(frames)} frames) — {status}")
    return stem


KEYS = ["slots", "grasp_mode", "policy"]


def render_montage(policies: list[str], seed: int, mode: str, path: Path, slots: bool = False,
                   tile=(240, 320), stride: int = 4, fps: int = 8) -> Path:
    """All policies on the same cube layout, side by side (2 × 2 for four), as a GIF."""
    from PIL import Image, ImageDraw, ImageFont

    from so101_tpgmm.sim.episode import run_episode

    persp, _ = _cameras()
    persp.lookat[:] = [0.2, 0.04, 0.0]
    persp.distance, persp.elevation = 0.55, -42.0
    clips, captions = [], []
    for name in policies:
        frames: list = []

        def grab(sim, frames=frames):
            sim.renderer.update_scene(sim.data, camera=persp)
            frames.append(sim.renderer.render().copy())

        outs, _ = run_episode(_policy(name), _W["ik"], _W["cfg"], _layout(seed), _W["geom"],
                              np.random.default_rng(10_000 + seed), grasp_mode=mode, render=tile,
                              frame_callback=grab, render_every=stride,
                              box_offsets=_W["cfg"]["sim"]["box_slots"] if slots else None)
        n_ok = sum(o.in_box or o.released_over_box for o in outs)
        clips.append(frames)
        captions.append(f"{LABEL[name]}: {n_ok}/{len(outs)} delivered")
    n = max(len(c) for c in clips)
    cols = 2 if len(clips) > 1 else 1
    rows = int(np.ceil(len(clips) / cols))
    h, w = tile
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    out_frames = []
    for k in range(n):
        canvas = Image.new("RGB", (cols * w, rows * h), "white")
        for j, c in enumerate(clips):
            im = Image.fromarray(c[min(k, len(c) - 1)])
            d = ImageDraw.Draw(im)
            text = LABEL[policies[j]] if k < len(c) - 1 else captions[j]
            d.rectangle([0, 0, w, 22], fill=(255, 255, 255))
            d.text((6, 3), text, fill=(20, 20, 20), font=font)
            canvas.paste(im, ((j % cols) * w, (j // cols) * h))
        out_frames.append(canvas.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    out_frames += [out_frames[-1]] * 15  # hold the final state
    path.parent.mkdir(parents=True, exist_ok=True)
    out_frames[0].save(path, save_all=True, append_images=out_frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"montage {path}: " + "; ".join(captions))
    return path


def summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(delivered=df.in_box | df.released_over_box)
    g = df.groupby(KEYS, sort=False)
    ep = df.groupby(KEYS + ["seed"], sort=False).in_box.all().groupby(KEYS, sort=False)
    return pd.DataFrame({
        "delivered": g.delivered.mean(),
        "cubes in box": g.in_box.mean(),
        "cubes lifted": g.lifted.mean(),
        "skipped (IK)": g.status.apply(lambda s: s.str.startswith("skipped").mean()),
        "episodes all 3 in box": ep.mean(),
        "cycle time [s]": g.cycle_s.median(),
        "n episodes": ep.size(),
    })


def failure_kind(r) -> str:
    if str(r.status).startswith("skipped"):
        return "skipped (IK)"
    if not r.lifted:
        return "not grasped"
    if not r.held_at_release:
        return "dropped in transport"
    return "released over the box, fell out" if r.released_over_box else "released outside the box"


def reading(df: pd.DataFrame, s: pd.DataFrame) -> list[str]:
    """Data-driven reading of the results table."""
    def v(sl, m, p, col="cubes in box"):
        try:
            return 100 * float(s.loc[(sl, m, p), col])
        except KeyError:
            return float("nan")

    pols = [p for p in POLICIES if p in set(df.policy)]
    conds = [(sl, m) for sl in (False, True) for m in ("contact", "assist") if (sl, m, pols[0]) in s.index]
    mean = {p: float(np.nanmean([v(sl, m, p) for sl, m in conds])) for p in pols}
    wins = {p: sum(v(sl, m, p) >= max(v(sl, m, q) for q in pols) for sl, m in conds) for p in pols}
    best = max(mean, key=mean.get)
    dl = {p: v(False, "contact", p, "delivered") for p in pols}
    out = ["## Reading\n",
           "- **Delivered to the box** (towers allowed; contact physics, policies as trained): " + ", ".join(
               f"{LABEL[p]} {dl[p]:.0f} %" for p in sorted(pols, key=dl.get, reverse=True)) + ".",
           f"- **Best policy: {LABEL[best]}**: {mean[best]:.0f} % of the cubes in the box on average over the "
           f"{len(conds)} noise-free conditions, best in {wins[best]} of them; from {v(False, 'contact', best):.0f} % "
           f"(contact physics, policies as trained) to {v(True, 'assist', best):.0f} % (box slots, grasp assist). "
           "Average of the others: " + ", ".join(f"{LABEL[p]} {mean[p]:.0f} %" for p in
                                                 sorted(pols, key=mean.get, reverse=True) if p != best) + "."]
    if "lpv_ds" in pols and "segmented" in pols:
        out.append(f"- **LPV-DS grasps fewer cubes** (lifted {v(False, 'contact', 'lpv_ds', 'cubes lifted'):.0f} % vs "
                   f"{v(False, 'contact', 'segmented', 'cubes lifted'):.0f} % for the segmented TP-GMM) although its grasp "
                   "point is as accurate: the rest pose puts the gripper at table height a few cm from the robot, and a "
                   "first-order field of the position, started there, heads for the cube almost horizontally with the "
                   "jaws still closed (they open only near the cube), sweeping or pushing it, where the time-indexed "
                   "TP-GMM reproduces the operator's lift-then-approach. The offline evaluation, which scores the grasp "
                   "point and the path shape, could not show what the arm hits on the way.")
    if "ik_planner" in pols:
        out.append(f"- **IK planner:** {v(False, 'contact', 'ik_planner', 'skipped (IK)'):.0f} % of the cubes are skipped "
                   "because a point of its straight-segment path is unreachable (the TP-GMM paths, learned from feasible "
                   "demonstrations, never are).")
    if "monolithic" in pols:
        out.append(f"- **Monolithic TP-GMM:** lifted {v(False, 'contact', 'monolithic', 'cubes lifted'):.0f} % vs "
                   f"{v(False, 'contact', 'segmented', 'cubes lifted'):.0f} % for the segmented model, consistent with the "
                   "offline finding that, without the primitive boundary, its gripper closes ~0.5 s away from the "
                   "demonstrated contact (`evaluation_report.md`).")
    gains = {p: v(True, "contact", p) - v(False, "contact", p) for p in pols}
    out.append("- **One release point vs box slots:** with every cube released at the same spot they pile up and some fall "
               "out; shifting the box frame per cube (a task parameter changed at execution, never varied in the "
               "demonstrations) changes the cubes in the box by " + ", ".join(
                   f"{LABEL[p]} {gains[p]:+.0f} pts" for p in pols) + ".")
    if "assist" in set(df.grasp_mode):
        da = {p: v(True, "assist", p) - v(True, "contact", p) for p in pols}
        out.append("- **Grasp assist** (removes slipping, not approach errors): " + ", ".join(
            f"{LABEL[p]} {da[p]:+.0f} pts" for p in pols) + " (box slots). What remains with assist is the approach.")
    if (df.noise_mm > 0).any():
        n = sorted(set(df.noise_mm))[-1]
        d = df[(df.slots) & (df.grasp_mode == "assist")].groupby(["policy", "noise_mm"]).in_box.mean() * 100
        out.append(f"- **Perception noise σ = {n:.0f} mm** (box slots, assist): " + ", ".join(
            f"{LABEL[p]} {d.get((p, 0.0), float('nan')):.0f} → {d.get((p, n), float('nan')):.0f} %" for p in pols)
                   + ". With the real camera (~8 mm median error) the success would drop accordingly: better calibration "
                   "pays off as much as a better policy.")
    out.append("")
    return out


def write_report(df: pd.DataFrame, cfg: dict, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = resolve_path("docs/figures")
    s = summary(df[df.noise_mm == 0])
    # outcome map per policy: rows = without / with box slots, in the first grasp mode
    m = "contact" if "contact" in set(df.grasp_mode) else df.grasp_mode.iloc[0]
    rows_sl = [sl for sl in (False, True) if sl in set(df.slots)]
    pols = [p for p in POLICIES if p in set(df.policy)]
    fig, axs = plt.subplots(len(rows_sl), len(pols), figsize=(4.2 * len(pols), 4.4 * len(rows_sl)), squeeze=False)
    bc, bi = np.asarray(cfg["sim"]["box"]["center_xy"]), np.asarray(cfg["sim"]["box"]["inner"]) / 2
    for r, sl in enumerate(rows_sl):
        for c, p in enumerate(pols):
            ax = axs[r, c]
            d = df[(df.grasp_mode == m) & (df.policy == p) & (df.noise_mm == 0) & (df.slots == sl)]
            xy = np.array(d.xyz0.tolist())
            ok, lift = d.in_box.to_numpy(), d.lifted.to_numpy()
            ax.add_patch(plt.Rectangle(((bc - bi)[1] * 100, (bc - bi)[0] * 100), 2 * bi[1] * 100, 2 * bi[0] * 100,
                                       fill=False, ec="0.4", lw=1.2))
            ax.scatter(xy[ok, 1] * 100, xy[ok, 0] * 100, s=14, c="#2a9d55", label="in box")
            m2 = ~ok & lift
            ax.scatter(xy[m2, 1] * 100, xy[m2, 0] * 100, s=22, c="#e09f1f", marker="s", label="lifted, not in box")
            m3 = ~ok & ~lift
            ax.scatter(xy[m3, 1] * 100, xy[m3, 0] * 100, s=26, c="#c0392b", marker="x", label="not grasped / skipped")
            ax.plot(0, 0, "k^", ms=8)
            ax.set_title(f"{LABEL[p]}, {m}, {'box slots' if sl else 'one release point'}: "
                         f"{100 * ok.mean():.0f} % in box", fontsize=8.5)
            ax.set_xlabel("y [cm]"), ax.set_ylabel("x [cm]")
            ax.invert_xaxis(), ax.set_aspect("equal")
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Simulated cube start positions and outcome (robot base ▲; grey rectangle = box)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "sim_outcomes.png", dpi=90)
    plt.close(fig)

    L = ["# MuJoCo simulation of the learned policies\n", "_Generated by `scripts/simulate_mujoco.py`._\n",
         "## Setup\n",
         "- **Robot:** official SO-101 MJCF (TheRobotStudio/SO-ARM100, Apache-2.0), same CAD as our URDF: its TCP "
         "site matches our FK to < 4 µm. Position servos with the model's STS3215 gains and torque limit (3.35 N·m), "
         "commands at 30 Hz in LeRobot units (degrees, gripper 0-100 mapped linearly to the jaw range).",
         "- **Fingers:** the convex hulls of the jaw meshes fill the gap between the fingers (they overlap at every "
         "opening), so each finger is replaced by boxes fitted to 1 cm slices of its mesh. With the linear gripper "
         "map the tips touch at 0 and are 4 cm apart at ~26: the demonstrated holding command (~20) squeezes the cube.",
         "- **Table:** the plane on which the fingers rested at the demonstrated grasps, in FK coordinates "
         f"(z = {cfg['sim']['table_plane'][0]:+.4f} {cfg['sim']['table_plane'][1]:+.3f}·x "
         f"{cfg['sim']['table_plane'][2]:+.3f}·y m). The real table is flat; in FK coordinates it looks tilted by "
         "~3.4° and 2.5 cm lower than the tape-grid estimate — the joint-calibration error of the real arm. The "
         "simulation is consistent with the commanded joints, which is what matters here.",
         f"- **Box:** open container {cfg['sim']['box']['inner'][0] * 100:.0f}×{cfg['sim']['box']['inner'][1] * 100:.0f} cm "
         f"inside, {cfg['sim']['box']['wall_h'] * 100:.1f} cm high, centred at ({bc[0] * 100:.1f}, {bc[1] * 100:.1f}) cm. "
         "**Not measured**: sized to hold every demonstrated release and to stay clear of every demonstrated grasp.",
         f"- **Cubes:** {cfg['sim']['n_cubes']} per episode ({', '.join(cfg['sim']['colors'])}), 4 cm, "
         f"{cfg['sim']['cube_mass'] * 1e3:.0f} g (not measured), friction {cfg['sim']['cube_friction']}; positions = a "
         f"demonstrated cube position + a uniform disc of {cfg['sim']['sample_radius'] * 100:.0f} cm (the trained "
         f"workspace), ≥ {cfg['sim']['min_separation'] * 100:.0f} cm apart and clear of the box; yaw uniform. Picked in "
         f"the order {', '.join(cfg['inference']['color_order'])}. Every policy sees the same layouts.",
         "- **Known position, in the policy's convention.** The camera-based cube positions the policies were "
         "trained with are biased by ~2 cm with respect to the fingers of the FK model. Handing the policies the "
         "physical cube centre would test that calibration bias, not the policy. The position given to each policy "
         "is the one that makes *its own* nominal grasp centre the fingers on the cube with the finger tips 1 cm "
         "above the table (a few fixed-point iterations on its rollout), i.e. a perfectly calibrated perception.",
         "- **Pipeline per cube:** measured joints → policy rollout → IK (pitch soft target; a cube with any "
         "unreachable point is skipped) → joint trajectory → return home (joint space, 2 s). Open loop: the plan "
         "is not corrected during execution (also for the LPV-DS, which is integrated from the measured start).",
         "- **Grasp modes:** `contact` = pure physics. `assist` = a cube touched by both fingers is attached while "
         "the gripper is closed; the approach still has to place the cube between the fingers. It isolates the "
         "policy from the unmeasured friction / mass / pad compliance.",
         "- **Release:** *one release point* = the policies as trained (the box never moved in the demonstrations, "
         "and each demonstration had a single cube): every cube is released at the same place. *Box slots* = the "
         "k-th cube is released with the policy's box frame shifted by `sim.box_slots[k]` (landing points ≥ 5.5 cm "
         "apart): the task parameter is changed at execution, without retraining.",
         "", "## Results\n",
         f"{int(s['n episodes'].max())} episodes × {cfg['sim']['n_cubes']} cubes per policy and condition, same "
         "layouts for all. *Delivered* = released over the box while still held, or inside it at the end (a tower "
         "that falls over still counts: the task is to bring the cubes to the box). *In box* = inside the walls at "
         "the end (resting on another cube counts).\n",
         "| release | grasp mode | policy | delivered | in box at the end | cubes lifted | skipped (IK) | episodes with all cubes in box | cycle time [s] |",
         "|---|---|---|---|---|---|---|---|---|"]
    for (sl, m, p), r in s.iterrows():
        L.append(f"| {'box slots' if sl else 'one point'} | {m} | {LABEL[p]} | {100 * r['delivered']:.0f} % | "
                 f"{100 * r['cubes in box']:.0f} % | "
                 f"{100 * r['cubes lifted']:.0f} % | {100 * r['skipped (IK)']:.0f} % | "
                 f"{100 * r['episodes all 3 in box']:.0f} % | {r['cycle time [s]']:.1f} |")
    L += ["", "![](figures/sim_outcomes.png)\n"]
    vids = sorted(figs.glob("sim_*_seed*.gif"))
    if vids:
        L.append("Example episodes (left: perspective; right: overhead, like the fixed camera; MP4 next to each GIF):\n")
        for v in vids:
            L.append(f"`{v.stem}`\n\n![](figures/{v.name})\n")
    L += reading(df, s)
    nz = df[df.noise_mm > 0]
    if len(nz):
        L += ["### Perception noise\n", "Gaussian noise added to the known cube position (xy, per axis), as a "
              "stand-in for the vision error of the real pipeline (~8 mm median vs FK):\n",
              "| release | grasp mode | policy | noise σ [mm] | cubes in box | cubes lifted |", "|---|---|---|---|---|---|"]
        for (sl, m, p, n), r in df.groupby(KEYS + ["noise_mm"], sort=False).agg(
                ib=("in_box", "mean"), lf=("lifted", "mean")).iterrows():
            L.append(f"| {'box slots' if sl else 'one point'} | {m} | {LABEL[p]} | {n:.0f} | {100 * r.ib:.0f} % | "
                     f"{100 * r.lf:.0f} % |")
        L.append("")
    fails = df[(df.noise_mm == 0) & ~df.in_box]
    L += ["## Failure modes\n"]
    if len(fails):
        cats = fails.apply(failure_kind, axis=1)
        tab = pd.crosstab([fails.slots.map({False: "one point", True: "box slots"}), fails.grasp_mode,
                           fails.policy.map(LABEL)], cats)
        L.append("Cubes not in the box, by cause (counts):\n")
        L.append("| release | grasp mode | policy | " + " | ".join(tab.columns) + " |")
        L.append("|---|---|---|" + "---|" * len(tab.columns))
        for (sl, m, p), r in tab.iterrows():
            L.append(f"| {sl} | {m} | {p} | " + " | ".join(str(int(v)) for v in r) + " |")
        L.append("")
    L += ["## Limits\n",
          "- Kinematic and contact model only as good as its parameters: cube mass and friction, pad compliance and the "
          "box were not measured; the gripper map (0-100 → jaw angle) is linear between the joint limits.",
          "- Open-loop execution of each plan, as in `scripts/infer.py` (no visual feedback while moving).",
          "- The perception is ideal (or Gaussian noise with `--noise-mm`): no occlusions, no detection failures.",
          "- Replaying the human demonstrations in the simulator is not a validation of the policies: the initial "
          "cube pose of the demonstrations is known only to ~1 cm and the operator sometimes nudged the cube.", ""]
    resolve_path("docs/mujoco_sim.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--policies", nargs="*", default=list(POLICIES), choices=list(POLICIES))
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grasp-modes", nargs="*", default=["contact", "assist"], choices=["contact", "assist"])
    ap.add_argument("--noise-mm", type=float, nargs="*", default=[0.0], help="perception noise σ values [mm]")
    ap.add_argument("--box-slots", choices=["off", "on", "both"], default="both",
                    help="release each cube in its own slot of the box (shifted box frame, sim.box_slots)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--video", type=int, default=0, help="render the first N episodes of each policy (first mode) "
                    "to outputs/sim/videos (copy a GIF/MP4 to docs/figures to show it in the report)")
    ap.add_argument("--montage", type=int, nargs="*", default=None,
                    help="seeds: GIF with all --policies side by side on the same layout (docs/figures)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = resolve_path("outputs/sim")
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "results.csv"
    if not args.report_only:
        slot_opts = {"off": [False], "on": [True], "both": [False, True]}[args.box_slots]
        jobs = [(p, args.seed + e, m, n, sl) for sl in slot_opts for m in args.grasp_modes for p in args.policies
                for n in args.noise_mm for e in range(args.episodes)]
        rows, t0 = [], time.time()
        with ProcessPoolExecutor(args.workers, initializer=_setup, initargs=(args.config, args.set)) as ex:
            for k, r in enumerate(ex.map(_job, jobs, chunksize=1)):
                rows += r
                if (k + 1) % 10 == 0:
                    print(f"  {k + 1}/{len(jobs)} episodes, {time.time() - t0:.0f} s", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(csv, index=False)
    df = pd.read_csv(csv)
    for col in ("xyz0", "final_xyz"):
        df[col] = df[col].apply(lambda v: [float(x) for x in str(v).strip("[]").split(",")])
    print(summary(df[df.noise_mm == 0]).round(3).to_string())
    write_report(df, cfg, args)
    if args.montage:
        _setup(args.config, args.set)
        for sd in args.montage:
            render_montage(args.policies, sd, args.grasp_modes[0], resolve_path(f"docs/figures/sim_montage_seed{sd}.gif"),
                           slots=args.box_slots == "on")
    if args.video:
        _setup(args.config, args.set)
        for p in args.policies:
            for e in range(args.video):
                render_video(p, args.seed + e, args.grasp_modes[0], out / "videos", slots=args.box_slots == "on")


if __name__ == "__main__":
    main()
