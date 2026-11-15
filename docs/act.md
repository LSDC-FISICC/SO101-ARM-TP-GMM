# ACT on dataset3

Results of the A/B comparison (offline, 40 held-out episodes): `act_report.html`.

ACT (Action Chunking with Transformers, LeRobot implementation) learns joint-space action chunks
from both cameras (`wrist`, `top`) and the joint state. It is trained outside this repository with
`lerobot-train` (on the DGX Spark) and evaluated here against the TP-GMM.

## Split

`configs/act_split.json`: 40 held-out episodes (every 9th usable episode, spread over the whole
recording) and 360 training episodes. ACT must be trained on `train` only, so that the offline
evaluation runs on episodes it has never seen. The TP-GMM numbers for the same episodes come from
its leave-one-out evaluation (`evaluation_report.md`).

## Training (DGX Spark)

```bash
pip install 'lerobot[dataset,training]==0.7.0' wandb trackio
wandb login                              # once; optionally export WANDB_ENTITY=<team>
export HF_USER=<your Hugging Face user>
bash scripts/act/train_act_spark.sh A    # ACT preset: AdamW, constant lr 1e-5
bash scripts/act/train_act_spark.sh B    # linear warm-up (2 k) + cosine decay, peak 3e-5 → 3e-6
```

**Logging:** LeRobot ≥ 0.6 logs only to Trackio; `scripts/act/lerobot_train_wandb.py` runs the same
`lerobot-train` with a stand-in `trackio` module that forwards to Weights & Biases (project
`so101-act`, runs `act_dataset3_const-…` and `act_dataset3_cosine-…`): loss, L1 and KL terms, gradient
norm, learning rate, samples/epochs, data-loading time. `LOGGER=trackio` uses LeRobot's own logger
instead; `LEROBOT_ALSO_TRACKIO=1` logs to both. Checked with a 4-step CPU run (W&B offline mode).

Both: batch 32, 100 k steps (≈ 15 epochs of the 215 k training frames), chunk 100, image
augmentation on, same seed; only the learning-rate schedule differs.

- **Why a schedule (variant B):** the ACT preset (constant 1e-5, no warm-up) was tuned for batch 8.
  With batch 32 a larger peak rate is affordable, but a transformer trained from scratch at a higher
  rate needs a warm-up for the first updates (layer-norm statistics and attention are unstable at
  initialization); the cosine decay then lowers the rate at the end, so the last checkpoint is the
  converged one instead of a point oscillating at a constant rate. The ResNet-18 backbone
  (ImageNet-pretrained) keeps a 3× lower rate (`policy.optimizer_lr_backbone=1e-5`) with the same
  shape. Whether it pays off for ACT here is measured, not assumed: compare A and B with the offline
  evaluation below.
- **The existing policy** (`SebRivGar/train_act_policy2`) was trained on another dataset
  (`training-record2`, 20 episodes, "Grasp green cube") for 25 k steps at batch 8 (200 k samples, less
  than one epoch of dataset3): it is not a dataset3 policy. On two held-out episodes it is worse than
  not moving (TCP error 70–90 mm vs 10–35 mm), as expected.
- **Environment:** LeRobot 0.7.0 in a Python environment whose PyTorch sees the Spark's GPU; check `python -c "import torch; print(torch.cuda.is_available())"`
  after installing (pip may replace a CUDA build by a CPU one). Videos are AV1: `pyav` decodes them
  with dav1d; if data loading is the bottleneck (GPU utilization low), raise `WORKERS`.
- **Image augmentation runs on the GPU** (`AUG_BACKEND=gpu`, not compiled): with LeRobot's default
  (DataLoader workers, CPU), the sharpness filter's `conv2d` crashes on the Spark's ARM CPU inside
  oneDNN's JIT (`RuntimeError: label is too far`, `bad err=3 in Xbyak::Error`). If the GPU augmentation
  fails too, `AUG=false` trains without it (and `AUG_COMPILE=true` tries `torch.compile`, ~3× faster).

## Offline evaluation (here, CPU is enough)

```bash
python scripts/act/eval_act_offline.py --policy <HF_USER>/act_dataset3_const  --name A
python scripts/act/eval_act_offline.py --policy <HF_USER>/act_dataset3_cosine --name B
```

Teacher-forced on the 40 held-out episodes (every 15 frames): predicted chunk vs the operator's
commands (joint error, TCP error by FK, gripper) at 0.03–2 s horizons, against a zero-motion
baseline, plus **grasp anticipation** — from 1 s and 2 s before the demonstrated contact, the
predicted TCP at the contact instant (the closest analogue of the TP-GMM grasp error, 10.7 mm,
which the TP-GMM predicts open loop from the start of the motion). Outputs:
`outputs/act/offline_<name>.{csv,md}`.

## On the robot

Same protocol as the other policies (cube positions, success = cube delivered to the box):

```bash
lerobot-rollout --strategy.type=base --policy.path=<HF_USER>/act_dataset3_cosine \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}}" \
  --task="Grasp cube and put it in the box" --duration=60
```

The task string must be the dataset's ("Grasp cube and put it in the box"), and the cameras the same
as in the recording (wrist + fixed top camera at its recording pose: ACT, unlike the TP-GMM, has no
calibration step that would absorb a moved camera).
