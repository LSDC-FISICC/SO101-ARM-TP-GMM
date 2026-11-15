#!/usr/bin/env python
"""Offline evaluation of an ACT policy (LeRobot) on held-out episodes of dataset3.

For frames of the held-out episodes (``configs/act_split.json``; the policy must not have been
trained on them), the policy predicts its action chunk from the recorded observation (both
cameras + joint state) and the chunk is compared with what the operator actually commanded:

* joint error [deg] and TCP error [mm] (FK of predicted vs recorded joint actions) at horizons of
  1, 10, 30 and 60 steps (0.03 s – 2 s);
* **grasp anticipation**: from 1 s and 2 s before the demonstrated contact, the predicted TCP at
  the contact instant vs the demonstrated contact TCP (closest analogue of the TP-GMM grasp error,
  10.7 mm, which is predicted from the start of the motion, open loop);
* gripper error.

A zero-motion baseline (repeat the current joint state) is reported alongside: it is what a
policy that has learned nothing would score. These are teacher-forced, open-loop checks; the
closed-loop test is the robot (``lerobot-rollout``).

    python scripts/act/eval_act_offline.py --policy <hf_repo_or_local_dir> --stride 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402
from so101_tpgmm.kinematics.fk import SO101Kinematics  # noqa: E402

HORIZONS = (1, 10, 30, 60)


def load_act(path: str, device: str = "cpu"):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch.set_num_threads(max(1, torch.get_num_threads()))
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    policy = ACTPolicy.from_pretrained(path, config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--policy", required=True, help="HF repo id or local pretrained_model directory")
    ap.add_argument("--name", default=None, help="label for the outputs (default: last part of --policy)")
    ap.add_argument("--stride", type=int, default=15, help="evaluate every N frames")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="default: the held-out split")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--dataset-root", default=None,
                    help="local dataset3 with both cameras (default: dataset.local_dir; on the training machine: "
                         "~/.cache/huggingface/lerobot/SebRivGar/dataset3)")
    args = ap.parse_args()

    import torch
    from lerobot.datasets import LeRobotDataset

    cfg = load_config(args.config, [])
    name = args.name or Path(args.policy.rstrip("/")).name
    split = json.loads(resolve_path("configs/act_split.json").read_text())
    episodes = args.episodes or split["test"]
    kin = SO101Kinematics(resolve_path(cfg["kinematics"]["urdf"]))
    seg_csv = resolve_path(cfg["segmentation"]["output_csv"])
    if not seg_csv.exists():  # outputs/ is not versioned; docs/ holds the same table
        seg_csv = resolve_path("docs/segmentation_cycles.csv")
    seg = pd.read_csv(seg_csv).set_index("episode")
    policy, pre, post, pcfg = load_act(args.policy, args.device)
    chunk = int(pcfg.chunk_size)
    root = Path(args.dataset_root).expanduser() if args.dataset_root else resolve_path(cfg["dataset"]["local_dir"])
    ds = LeRobotDataset(cfg["dataset"]["repo_id"], root=root, episodes=episodes, video_backend="pyav")
    ep_index = np.asarray(ds.hf_dataset["episode_index"])
    act_all = np.asarray(ds.hf_dataset["action"], dtype=float)
    st_all = np.asarray(ds.hf_dataset["observation.state"], dtype=float)

    def tcp(q_deg):
        return kin.fk(np.deg2rad(np.asarray(q_deg)[..., :5]))[..., :3, 3]

    rows = []
    for e in episodes:
        idx = np.flatnonzero(ep_index == e)
        if not len(idx):
            continue
        close = int(seg.loc[e, "close_idx"]) if e in seg.index else -1
        queries = set(range(0, len(idx), args.stride))
        if close >= 0:
            queries |= {close - 30, close - 60}
        for k in sorted(q for q in queries if 0 <= q < len(idx)):
            item = ds[int(idx[k])]
            batch = {key: (v.unsqueeze(0) if torch.is_tensor(v) else v) for key, v in item.items()
                     if key.startswith("observation.") or key == "task"}
            if isinstance(batch.get("task"), str):
                batch["task"] = [batch["task"]]
            with torch.no_grad():
                obs = pre(batch)
                a = policy.predict_action_chunk(obs)
                a = post(a).cpu().numpy()[0]  # (chunk, 6), degrees / gripper 0-100
            for h in HORIZONS:
                if k + h - 1 >= len(idx) or h > chunk:
                    continue
                true = act_all[idx[k + h - 1]]
                cur = st_all[idx[k]]
                pred = a[h - 1]
                rows.append(dict(policy=name, episode=e, frame=k, horizon=h, kind="chunk",
                                 joint_err_deg=float(np.abs(pred[:5] - true[:5]).mean()),
                                 tcp_err_mm=float(np.linalg.norm(tcp(pred) - tcp(true)) * 1e3),
                                 grip_err=float(abs(pred[5] - true[5])),
                                 base_joint_err_deg=float(np.abs(cur[:5] - true[:5]).mean()),
                                 base_tcp_err_mm=float(np.linalg.norm(tcp(cur) - tcp(true)) * 1e3)))
            for lead in (30, 60):
                if close >= 0 and k == close - lead and lead <= chunk:
                    contact = tcp(st_all[idx[close]])  # follower (measured) TCP at contact
                    rows.append(dict(policy=name, episode=e, frame=k, horizon=lead, kind="grasp",
                                     tcp_err_mm=float(np.linalg.norm(tcp(a[lead - 1]) - contact) * 1e3),
                                     base_tcp_err_mm=float(np.linalg.norm(tcp(st_all[idx[k]]) - contact) * 1e3)))
                    # command space: predicted vs recorded leader command at contact. At contact the follower lags
                    # the leader by ~8.5 mm (median, held-out episodes): the cube stops the jaws / the arm sags.
                    contact_cmd = tcp(act_all[idx[close]])
                    rows.append(dict(policy=name, episode=e, frame=k, horizon=lead, kind="grasp_cmd",
                                     tcp_err_mm=float(np.linalg.norm(tcp(a[lead - 1]) - contact_cmd) * 1e3),
                                     base_tcp_err_mm=float(np.linalg.norm(tcp(act_all[idx[k]]) - contact_cmd) * 1e3)))
                    # time-free: closest approach of the whole predicted chunk to the contact command (where, not when),
                    # and when the predicted gripper closes vs the demonstrated contact
                    path = np.array([tcp(q) for q in a])
                    d = np.linalg.norm(path - contact_cmd, axis=1)
                    g = a[:, 5]
                    i_open = int(np.argmax(g)) if g.max() > 25 else 0
                    shut = np.flatnonzero(g[i_open:] < 0.5 * (g[i_open] + g[i_open:].min()))
                    t_close = (i_open + shut[0] + 1 - lead) / 30 if len(shut) and g[i_open] > 25 else np.nan
                    rows.append(dict(policy=name, episode=e, frame=k, horizon=lead, kind="grasp_cmd_closest",
                                     tcp_err_mm=float(d.min() * 1e3), base_tcp_err_mm=float(np.linalg.norm(
                                         tcp(act_all[idx[k]]) - contact_cmd) * 1e3),
                                     closest_dt_s=float((int(np.argmin(d)) + 1 - lead) / 30), close_dt_s=float(t_close)))
        print(f"episode {e}: {sum(r['episode'] == e for r in rows)} rows", flush=True)
    out = resolve_path("outputs/act") / f"offline_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    summarize(df, name)


def summarize(df: pd.DataFrame, name: str) -> None:
    c = df[df.kind == "chunk"].groupby("horizon")[["joint_err_deg", "base_joint_err_deg", "tcp_err_mm",
                                                   "base_tcp_err_mm", "grip_err"]].median()
    g = df[df.kind == "grasp"].groupby("horizon")[["tcp_err_mm", "base_tcp_err_mm"]].median()
    gc = df[df.kind == "grasp_cmd"].groupby("horizon")[["tcp_err_mm", "base_tcp_err_mm"]].median()
    gm = df[df.kind == "grasp_cmd_closest"].groupby("horizon")[["tcp_err_mm", "closest_dt_s", "close_dt_s"]].median() \
        if "closest_dt_s" in df else pd.DataFrame()
    L = [f"## ACT offline: `{name}` ({df.episode.nunique()} held-out episodes)\n",
         "Median errors vs the recorded operator commands (teacher forcing: recorded observation, predicted chunk):\n",
         "| horizon | joint err ACT [°] | joint err zero-motion [°] | TCP err ACT [mm] | TCP err zero-motion [mm] | gripper err ACT |",
         "|---|---|---|---|---|---|"]
    for h, r in c.iterrows():
        L.append(f"| {h} ({h / 30:.2f} s) | {r.joint_err_deg:.2f} | {r.base_joint_err_deg:.2f} | {r.tcp_err_mm:.1f} | "
                 f"{r.base_tcp_err_mm:.1f} | {r.grip_err:.1f} |")
    if len(g):
        L += ["", "Grasp anticipation: predicted TCP at the demonstrated contact instant vs the contact TCP:\n",
              "| ahead | ACT [mm] | zero-motion [mm] |", "|---|---|---|"]
        for h, r in g.iterrows():
            L.append(f"| {h / 30:.0f} s | {r.tcp_err_mm:.1f} | {r.base_tcp_err_mm:.1f} |")
    if len(gc):
        L += ["", "Same, in command space (predicted vs recorded leader command at contact; removes the ~8.5 mm "
              "follower lag at contact from both):\n", "| ahead | ACT [mm] | zero-motion [mm] |", "|---|---|---|"]
        for h, r in gc.iterrows():
            L.append(f"| {h / 30:.0f} s | {r.tcp_err_mm:.1f} | {r.base_tcp_err_mm:.1f} |")
    if len(gm):
        L += ["", "Where vs when (command space): closest approach of the whole predicted chunk to the contact command, "
              "when it happens, and when the predicted gripper closes (both relative to the demonstrated contact; "
              "negative = early):\n", "| ahead | closest approach [mm] | at [s] | predicted closing at [s] |",
              "|---|---|---|---|"]
        for h, r in gm.iterrows():
            L.append(f"| {h / 30:.0f} s | {r.tcp_err_mm:.1f} | {r.closest_dt_s:+.2f} | {r.close_dt_s:+.2f} |")
    text = "\n".join(L) + "\n"
    out = resolve_path("outputs/act") / f"offline_{name}.md"
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
