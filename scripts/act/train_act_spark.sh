#!/usr/bin/env bash
# Train ACT (LeRobot) on dataset3 without the 40 held-out episodes of configs/act_split.json, so that
# scripts/act/eval_act_offline.py can score it on episodes it has never seen.
#
#   bash scripts/act/train_act_spark.sh A    # ACT preset: AdamW, constant lr 1e-5 (as the existing policy)
#   bash scripts/act/train_act_spark.sh B    # same, but linear warm-up + cosine decay, peak 3e-5
#
# Both variants are identical except for the learning-rate schedule (same split, batch, steps, image
# augmentation, seed), so the offline evaluation isolates its effect. Environment: HF_USER (Hub user
# for the pushed policy), STEPS (default 100000), BATCH (default 32), WORKERS (default 12),
# LOGGER = wandb (default: Weights & Biases through scripts/act/lerobot_train_wandb.py; run `wandb login`
# once, optionally export WANDB_ENTITY) or trackio (LeRobot's own logger), PROJECT (default so101-act).
set -euo pipefail
VARIANT=${1:?usage: train_act_spark.sh A|B}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STEPS=${STEPS:-100000}
BATCH=${BATCH:-32}
WORKERS=${WORKERS:-12}
HF_USER=${HF_USER:?set HF_USER to your Hugging Face user}
LOGGER=${LOGGER:-wandb}
PROJECT=${PROJECT:-so101-act}
EPISODES=$(python -c "import json; print(json.dumps(json.load(open('$ROOT/configs/act_split.json'))['train']).replace(' ', ''))")

COMMON=(
  --dataset.repo_id=SebRivGar/dataset3
  --dataset.episodes="$EPISODES"
  --dataset.video_backend=pyav            # AV1 via PyAV/dav1d; torchcodec wheels may be missing on ARM64
  --dataset.image_transforms.enable=true  # colour/affine jitter: robustness to lighting on the real robot
  --policy.type=act
  --policy.device=cuda
  --policy.chunk_size=100
  --policy.n_action_steps=100
  --batch_size="$BATCH"
  --steps="$STEPS"
  --num_workers="$WORKERS"
  --save_freq=10000
  --seed=1000
  --trackio.enable=true
  --trackio.project="$PROJECT"
  --policy.push_to_hub=true
)

case "$VARIANT" in
  A)
    NAME=act_dataset3_const
    EXTRA=()   # ACT preset: AdamW lr 1e-5, weight decay 1e-4, backbone lr 1e-5, no scheduler
    ;;
  B)
    NAME=act_dataset3_cosine
    # Warm-up 2 k steps to the peak, cosine decay to 10 % of it by the last step. The scheduler multiplies
    # every parameter group's own lr, so the ResNet backbone (policy.optimizer_lr_backbone) follows the same
    # shape at a third of the transformer's rate. peak_lr must equal optimizer.lr (it only sets the floor ratio).
    EXTRA=(
      --use_policy_training_preset=false
      --optimizer.type=adamw --optimizer.lr=3e-5 --optimizer.weight_decay=1e-4 --optimizer.grad_clip_norm=10
      --policy.optimizer_lr_backbone=1e-5
      --scheduler.type=cosine_decay_with_warmup
      --scheduler.num_warmup_steps=2000 --scheduler.num_decay_steps="$STEPS"
      --scheduler.peak_lr=3e-5 --scheduler.decay_lr=3e-6
    )
    ;;
  *) echo "variant must be A or B" >&2; exit 2 ;;
esac

case "$LOGGER" in
  wandb) TRAIN=(python "$ROOT/scripts/act/lerobot_train_wandb.py") ;;   # LeRobot >= 0.6 only logs to Trackio
  trackio) TRAIN=(lerobot-train) ;;
  *) echo "LOGGER must be wandb or trackio" >&2; exit 2 ;;
esac

"${TRAIN[@]}" "${COMMON[@]}" "${EXTRA[@]}" \
  --output_dir="outputs/train/$NAME" --job_name="$NAME" --policy.repo_id="$HF_USER/$NAME"
