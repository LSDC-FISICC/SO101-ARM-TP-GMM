#!/usr/bin/env python
"""``lerobot-train`` with Weights & Biases logging (LeRobot ≥ 0.6 only logs to Trackio).

LeRobot 0.7 replaced its W&B integration with Trackio, whose API mirrors W&B's. This wrapper puts a
small ``trackio`` stand-in in ``sys.modules`` that forwards ``init / log / Artifact / log_artifact /
Video / finish`` to ``wandb``, then runs LeRobot's own training entry point unchanged. Use it exactly
like ``lerobot-train`` (all arguments are LeRobot's) and enable logging with ``--trackio.enable=true``;
``--trackio.project`` becomes the W&B project, ``--trackio.run_name`` the run name (default
``<job_name>-<id>``), and the run's group is LeRobot's (policy, dataset, seed).

    wandb login                      # once, with your API key
    export WANDB_ENTITY=<team>       # optional: W&B team / user
    python scripts/act/lerobot_train_wandb.py --trackio.enable=true --trackio.project=so101-act <lerobot args...>

``WANDB_MODE=offline`` keeps the run on disk (``wandb sync`` later). Set ``LEROBOT_ALSO_TRACKIO=1`` to log
to Trackio as well.
"""

from __future__ import annotations

import os
import sys
import types


def _make_shim() -> types.ModuleType:
    import wandb

    tee = None
    if os.environ.get("LEROBOT_ALSO_TRACKIO") == "1":
        import trackio as tee  # the real one, imported before the stand-in replaces it

    import importlib.machinery

    shim = types.ModuleType("trackio")
    shim.__version__ = "wandb-shim"
    shim.__spec__ = importlib.machinery.ModuleSpec("trackio", None)  # LeRobot probes it with find_spec

    class _Run:
        def __init__(self, run):
            self._run = run
            self.url = run.url if run is not None else None  # https URL: LeRobot prints it as the run link

    def init(project=None, name=None, group=None, space_id=None, private=None, config=None, resume="never",
             capture_console=False, **_):
        if tee is not None:
            tee.init(project=project, name=name, group=group, space_id=space_id, private=private, config=config,
                     resume=resume, capture_console=capture_console)
        run = wandb.init(project=project, name=name, id=name, group=group, config=config,
                         resume={"never": None, "allow": "allow", "must": "must"}.get(resume, "allow"),
                         job_type="train", save_code=False)
        mode = getattr(getattr(run, "settings", None), "mode", None) or os.environ.get("WANDB_MODE", "online")
        bar = "=" * 72
        if run is not None and run.url:
            print(f"\n{bar}\nW&B run: {run.url}\n(metrics every --log_freq steps: loss, l1_loss, kld_loss, lr, grad norm)\n{bar}\n",
                  flush=True)
        else:
            print(f"\n{bar}\nW&B is in '{mode}' mode: nothing is sent to wandb.ai. Run `wandb login` and unset "
                  f"WANDB_MODE (or `wandb online`).\n{bar}\n", flush=True)
        return _Run(run)

    def log(data, step=None):
        if tee is not None:
            tee.log(data, step=step)
        wandb.log({k: v for k, v in data.items()}, step=step)

    def log_artifact(artifact):
        wandb.log_artifact(artifact)

    def finish():
        if tee is not None:
            tee.finish()
        wandb.finish()

    shim.init, shim.log, shim.log_artifact, shim.finish = init, log, log_artifact, finish
    shim.Artifact = wandb.Artifact
    shim.Video = wandb.Video
    return shim


def main() -> None:
    import importlib.util

    if importlib.util.find_spec("trackio") is None:  # LeRobot checks that the package is installed
        sys.exit("install trackio too (LeRobot checks for it): pip install trackio wandb")
    sys.modules["trackio"] = _make_shim()
    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
