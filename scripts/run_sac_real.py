#!/usr/bin/env python
"""Run a SAC policy (trained in MuJoCo) on the real SO-101, or keep training it there.

Runs **locally** on the machine connected to the arm and the fixed camera
(``pip install -e ".[robot,rl]"``). Same observation/action as in simulation
(``so101_tpgmm/rl/real_env.py``): the operator places a cube, presses Enter, the camera locates it,
the policy acts at 10 Hz and the operator says whether the cube ended in the box.

    # deploy: 10 evaluation episodes, deterministic policy
    python scripts/run_sac_real.py --model outputs/models/sac/residual_s0/best_model.zip --mode residual \\
        --episodes 10 --port /dev/ttyACM0 --camera 3
    # fine-tune on the robot for 3000 steps (~5 min of motion + resets), saving to rl.models_dir/real_<run>
    python scripts/run_sac_real.py --model ... --mode residual --finetune 3000 --port /dev/ttyACM0
    # without a robot (simulated follower, fixed cube): checks the plumbing
    python scripts/run_sac_real.py --model ... --mode residual --episodes 1 --dry-run

Start with the residual policy (its zero action is the TP-GMM), a free workspace and a hand on
the power switch; the follower's ``max_relative_target`` (``--max-step-deg``) limits every command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["scratch", "residual"], required=True)
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--finetune", type=int, default=0, help="SAC steps on the robot")
    ap.add_argument("--port", default=None)
    ap.add_argument("--id", default="so101_follower")
    ap.add_argument("--camera", default="3", help="fixed top camera (OpenCV index or /dev/videoN)")
    ap.add_argument("--register", action=argparse.BooleanOptionalAction, default=True,
                    help="correct a moved camera view on the taped grid (as scripts/infer.py)")
    ap.add_argument("--max-step-deg", type=float, default=8.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", default="real")
    args = ap.parse_args()

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    from so101_tpgmm.rl.real_env import CameraCubeSource, RealSO101Env
    from so101_tpgmm.robot.so101_runner import SO101Runner

    cfg = load_config(args.config, args.set)
    runner = SO101Runner(args.port, args.id, args.max_step_deg, dry_run=args.dry_run)
    if args.dry_run:
        def cube_source():
            return np.array([0.25, 0.05, 0.0064]), 0.2

        def ask(prompt):
            print(prompt + "(dry run)")
            return "n"
    else:
        cube_source = CameraCubeSource(cfg, int(args.camera) if args.camera.isdigit() else args.camera,
                                       register=args.register)
        ask = input
    env = RealSO101Env(cfg, args.mode, runner=runner, cube_source=cube_source, ask=ask, realtime=not args.dry_run)
    runner.connect()
    out = resolve_path(cfg["rl"]["models_dir"]) / f"{args.run}_{args.mode}"
    out.mkdir(parents=True, exist_ok=True)
    try:
        model = SAC.load(args.model, device="cpu")
        if args.episodes:
            log = []
            for ep in range(args.episodes):
                obs, info = env.reset()
                ret, t0 = 0.0, time.time()
                while True:
                    a, _ = model.predict(obs, deterministic=True)
                    obs, r, te, tu, step_info = env.step(a)
                    ret += r
                    if te or tu:
                        break
                rec = dict(episode=ep, success=bool(step_info["is_success"]), aborted=bool(step_info["aborted"]),
                           ret=ret, seconds=time.time() - t0, cube_vision=np.round(info["cube_vision"], 4).tolist())
                log.append(rec)
                print(rec, flush=True)
            sr = np.mean([r["success"] for r in log])
            print(f"success {100 * sr:.0f} % over {len(log)} episodes")
            (out / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(log, indent=1))
        if args.finetune:
            model.set_env(DummyVecEnv([lambda: env]))
            model.learning_starts = 0
            model.learn(total_timesteps=args.finetune, reset_num_timesteps=False)
            model.save(out / "finetuned")
            print(f"saved {out / 'finetuned.zip'}")
    finally:
        runner.disconnect()


if __name__ == "__main__":
    main()
