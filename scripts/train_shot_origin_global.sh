#!/usr/bin/env bash
set -euo pipefail

# Reusable launcher for VERTIGO trajectory-generator training with:
# - file-driven split
# - origin_global camera normalization
# - dataset-level robust p99 translation standardization
# - first-frame translation anchor head
# - wandb enabled

ROOT_DIR="/data/GenDoP"
DATA_ROOT="${DATA_ROOT:-/path/to/LenScript_subset/train}"
SPLIT_DIR="${SPLIT_DIR:-/path/to/LenScript_subset/splits/balanced_36k}"
SD_MODEL_PATH="${SD_MODEL_PATH:-stabilityai/stable-diffusion-2-1-base}"

EXP_NAME="${EXP_NAME:-shot_text_36k_origin_global_anchor_p99_bs6}"
WORKSPACE="${WORKSPACE:-/data/GenDoP/workspace}"

TRAIN_SPLIT="${TRAIN_SPLIT:-${SPLIT_DIR}/train.txt}"
TEST_SPLIT="${TEST_SPLIT:-${SPLIT_DIR}/test.txt}"

BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-128}"
LR="${LR:-1e-5}"

# Robust p99 stats from balanced36k train split (33k samples, 30 poses each).
CAMERA_TRANSLATION_MEAN=(
  "${CAMERA_TRANSLATION_MEAN_X:-0.0008104850254545796}"
  "${CAMERA_TRANSLATION_MEAN_Y:--0.007684390541918893}"
  "${CAMERA_TRANSLATION_MEAN_Z:--0.07576538240686936}"
)
CAMERA_TRANSLATION_SCALE=(
  "${CAMERA_TRANSLATION_SCALE_X:-2.354125261025455}"
  "${CAMERA_TRANSLATION_SCALE_Y:-0.9527430714580813}"
  "${CAMERA_TRANSLATION_SCALE_Z:-2.816293688241268}"
)
CAMERA_ANCHOR_MEAN=(
  "${CAMERA_ANCHOR_MEAN_X:-0.012779172054545453}"
  "${CAMERA_ANCHOR_MEAN_Y:-0.016163940687878634}"
  "${CAMERA_ANCHOR_MEAN_Z:-0.07471872632121161}"
)
CAMERA_ANCHOR_SCALE=(
  "${CAMERA_ANCHOR_SCALE_X:-5.1405228406134365}"
  "${CAMERA_ANCHOR_SCALE_Y:-1.5854810593121214}"
  "${CAMERA_ANCHOR_SCALE_Z:-5.313145813205207}"
)
ANCHOR_LOSS_WEIGHT="${ANCHOR_LOSS_WEIGHT:-0.2}"
ANCHOR_SCREEN_LOSS_WEIGHT="${ANCHOR_SCREEN_LOSS_WEIGHT:-0.05}"

# wandb
WANDB_PROJECT="${WANDB_PROJECT:-vertigo}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}}"
export WANDB_PROJECT
export WANDB_NAME="${WANDB_RUN_NAME}"

cd "${ROOT_DIR}"

echo "[INFO] EXP_NAME=${EXP_NAME}"
echo "[INFO] BATCH_SIZE=${BATCH_SIZE}, NUM_WORKERS=${NUM_WORKERS}, NUM_EPOCHS=${NUM_EPOCHS}"
echo "[INFO] TRAIN_SPLIT=${TRAIN_SPLIT}"
echo "[INFO] TEST_SPLIT=${TEST_SPLIT}"
echo "[INFO] CAMERA_TRANSLATION_MEAN=${CAMERA_TRANSLATION_MEAN[*]}"
echo "[INFO] CAMERA_TRANSLATION_SCALE=${CAMERA_TRANSLATION_SCALE[*]}"
echo "[INFO] CAMERA_ANCHOR_MEAN=${CAMERA_ANCHOR_MEAN[*]}"
echo "[INFO] CAMERA_ANCHOR_SCALE=${CAMERA_ANCHOR_SCALE[*]}"

accelerate_cmd=(
  accelerate launch
  --config_file acc_configs/gpu2.yaml
  core/main.py ArAE
  --workspace "${WORKSPACE}"
  --exp-name "${EXP_NAME}"
  --cond-mode text
  --text-key Movement
  --num-cond-tokens 77
  --path "${DATA_ROOT}"
  --train-split-file "${TRAIN_SPLIT}"
  --test-split-file "${TEST_SPLIT}"
  --camera-norm-mode origin_global
  --camera-translation-norm dataset_p99
  --camera-translation-mean "${CAMERA_TRANSLATION_MEAN[@]}"
  --camera-translation-scale "${CAMERA_TRANSLATION_SCALE[@]}"
  --camera-anchor-mean "${CAMERA_ANCHOR_MEAN[@]}"
  --camera-anchor-scale "${CAMERA_ANCHOR_SCALE[@]}"
  --sd-model-path "${SD_MODEL_PATH}"
  --num-workers "${NUM_WORKERS}"
  --batch-size "${BATCH_SIZE}"
  --num-epochs "${NUM_EPOCHS}"
  --lr "${LR}"
  --eval-mode loss
  --use-anchor-head
  --anchor-loss-weight "${ANCHOR_LOSS_WEIGHT}"
  --anchor-screen-loss-weight "${ANCHOR_SCREEN_LOSS_WEIGHT}"
  --use-wandb
)

# typo-safe expansion
"${accelerate_cmd[@]}"
