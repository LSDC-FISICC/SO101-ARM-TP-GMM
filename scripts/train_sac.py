#!/usr/bin/env python
"""Train SAC on the SO-101 pick-and-place task in MuJoCo (Stable-Baselines3).

Two variants, same environment (``so101_tpgmm/rl``), same observation/action as on the robot:

* ``--mode scratch``: SAC controls the TCP target directly (Δxyz, Δheading, gripper).
* ``--mode residual``: SAC adds bounded corrections to the TP-GMM plan; with a zero action the
  robot executes the TP-GMM. The actor starts at zero correction (its output layer is zeroed), with
  exploration noise ``--init-std`` (the critic must see the effect of corrections), and is frozen for
  ``--actor-delay`` steps while the critic learns; then it learns ``--actor-lr-scale`` times slower
  than the critic. With σ = 0.05, a 30 k warm-up and the full learning rate (first attempt), the actor
  jumped to corrections the critic had never seen and success fell from 93 % to 33 %.

Outputs in ``rl.models_dir/<run>``: ``best_model.zip`` (best evaluation success), ``final.zip``,
checkpoints, ``progress.csv`` (SB3 log) and ``eval.csv`` (success rate over evaluation episodes).

    python scripts/train_sac.py --mode residual --steps 150000 --n-envs 2
    python scripts/train_sac.py --mode scratch --steps 1500000 --n-envs 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_tpgmm.config import load_config, resolve_path  # noqa: E402


def make_env(config, overrides, mode, seed, randomize=True):
    def _init():
        from stable_baselines3.common.monitor import Monitor

        from so101_tpgmm.rl.sim_env import SO101PickPlaceEnv

        cfg = load_config(config, overrides)
        return Monitor(SO101PickPlaceEnv(cfg, mode, randomize=randomize, seed=seed), info_keywords=("is_success",))
    return _init


def evaluate(model, vec_env, n: int) -> tuple[float, float]:
    """Success rate and mean return of the deterministic policy over ``n`` episodes."""
    succ, rets = [], []
    obs = vec_env.reset()
    ret = 0.0
    while len(succ) < n:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = vec_env.step(a)
        ret += float(r[0])
        if done[0]:
            succ.append(bool(info[0].get("is_success", False)))
            rets.append(ret)
            ret = 0.0
    return float(np.mean(succ)), float(np.mean(rets))


def build_sac(mode: str, env, args, run_dir: Path):
    import torch
    from stable_baselines3 import SAC

    class ResidualSAC(SAC):
        """SAC whose actor is frozen (learning rate 0) for the first ``actor_delay`` steps."""

        actor_delay = 0
        actor_lr_scale = 1.0

        def _update_learning_rate(self, optimizers):
            super()._update_learning_rate(optimizers)
            scale = 0.0 if self.num_timesteps < self.actor_delay else self.actor_lr_scale
            for g in self.actor.optimizer.param_groups:
                g["lr"] *= scale

    torch.set_num_threads(args.torch_threads)
    common = dict(learning_rate=args.lr, buffer_size=args.buffer, batch_size=256, gamma=args.gamma, tau=0.005,
                  train_freq=1, gradient_steps=args.gradient_steps, verbose=0, seed=args.seed, device="cpu",
                  policy_kwargs=dict(net_arch=[256, 256]))
    if mode == "scratch":
        model = SAC("MlpPolicy", env, learning_starts=args.learning_starts, **common)
    else:
        # fixed entropy coefficient: the automatic one collapsed to ~0 and left the actor chasing critic noise
        model = ResidualSAC("MlpPolicy", env, learning_starts=0, ent_coef=args.ent_coef, **common)
        model.actor_delay = args.actor_delay
        model.actor_lr_scale = args.actor_lr_scale
        with torch.no_grad():
            # zero correction at the start (the TP-GMM itself) and small exploration noise (σ = --init-std). In SB3,
            # ``log_std_init`` only applies with gSDE: without this the initial σ is ~1 and the noise alone
            # breaks the TP-GMM (4 % success while collecting vs 93 % deterministic).
            model.actor.mu.weight.zero_()
            model.actor.mu.bias.zero_()
            model.actor.log_std.weight.zero_()
            model.actor.log_std.bias.fill_(float(np.log(args.init_std)))
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--mode", choices=["scratch", "residual"], required=True)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--buffer", type=int, default=1_000_000)
    ap.add_argument("--gradient-steps", type=int, default=1, help="updates per vectorized step (residual: 2 recommended)")
    ap.add_argument("--learning-starts", type=int, default=10_000)
    ap.add_argument("--actor-delay", type=int, default=50_000, help="residual: critic-only steps")
    ap.add_argument("--actor-lr-scale", type=float, default=0.1, help="residual: actor learning rate × this")
    ap.add_argument("--init-std", type=float, default=0.2, help="residual: initial exploration σ (action units)")
    ap.add_argument("--ent-coef", type=float, default=0.002, help="residual: fixed entropy coefficient")
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--run", default=None, help="run name (default: <mode>_s<seed>)")
    ap.add_argument("--resume", default=None, help="model .zip to continue from (e.g. a sim policy on the robot)")
    args = ap.parse_args()

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    cfg = load_config(args.config, args.set)
    run = args.run or f"{args.mode}_s{args.seed}"
    run_dir = resolve_path(cfg["rl"]["models_dir"]) / run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=1))
    fns = [make_env(args.config, args.set, args.mode, args.seed * 1000 + i) for i in range(args.n_envs)]
    env = SubprocVecEnv(fns, start_method="fork") if args.n_envs > 1 else DummyVecEnv(fns)
    eval_env = DummyVecEnv([make_env(args.config, args.set, args.mode, 10**6 + args.seed)])
    model = SAC.load(args.resume, env=env, device="cpu") if args.resume else build_sac(args.mode, env, args, run_dir)
    model.set_logger(configure(str(run_dir), ["csv"]))

    class EvalLog(EvalCallback):
        """Evaluation with the success rate written to eval.csv (and printed)."""

        def _on_step(self) -> bool:
            out = super()._on_step()
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                sr = float(np.mean(self._is_success_buffer)) if self._is_success_buffer else float("nan")
                line = f"{self.num_timesteps},{self.last_mean_reward:.2f},{sr:.3f},{time.time() - t0:.0f}\n"
                with open(run_dir / "eval.csv", "a") as f:
                    f.write(line)
                print(f"[{run}] {self.num_timesteps} steps: eval return {self.last_mean_reward:.1f}, "
                      f"success {100 * sr:.0f} % ({time.time() - t0:.0f} s)", flush=True)
            return out

    with open(run_dir / "eval.csv", "w") as f:
        f.write("timesteps,mean_return,success_rate,wall_s\n")
    t0 = time.time()
    cbs = [EvalLog(eval_env, best_model_save_path=str(run_dir), eval_freq=max(1, args.eval_every // args.n_envs),
                   n_eval_episodes=args.eval_episodes, deterministic=True, verbose=0),
           CheckpointCallback(save_freq=max(1, 100_000 // args.n_envs), save_path=str(run_dir / "checkpoints"),
                              name_prefix="sac")]
    sr0, ret0 = evaluate(model, eval_env, args.eval_episodes)  # residual at step 0 = the TP-GMM itself
    with open(run_dir / "eval.csv", "a") as f:
        f.write(f"0,{ret0:.2f},{sr0:.3f},0\n")
    print(f"[{run}] 0 steps: eval return {ret0:.1f}, success {100 * sr0:.0f} %", flush=True)
    model.learn(total_timesteps=args.steps, callback=cbs, progress_bar=False)
    model.save(run_dir / "final")
    print(f"[{run}] done in {time.time() - t0:.0f} s -> {run_dir}")


if __name__ == "__main__":
    main()
