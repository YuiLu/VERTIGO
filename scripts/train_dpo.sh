#!/usr/bin/env bash
set -euo pipefail

# DPO post-training launcher for VERTIGO trajectory generation.
# The JSONL preference file should contain one pair per line:
# {"prompt": "...", "chosen": "path/to/chosen_transforms.json", "rejected": "path/to/rejected_transforms.json"}

ROOT_DIR="${ROOT_DIR:-/data/GenDoP}"
PREFERENCE_PATH="${PREFERENCE_PATH:-/path/to/vertigo_preferences.jsonl}"
DATA_ROOT="${DATA_ROOT:-}"
POLICY_CKPT="${POLICY_CKPT:-/path/to/sft_or_previous_dpo.safetensors}"
REFERENCE_CKPT="${REFERENCE_CKPT:-${POLICY_CKPT}}"
SD_MODEL_PATH="${SD_MODEL_PATH:-stabilityai/stable-diffusion-2-1-base}"

EXP_NAME="${EXP_NAME:-vertigo_dpo_text_beta01}"
WORKSPACE="${WORKSPACE:-${ROOT_DIR}/workspace}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-128}"
LR="${LR:-1e-6}"
BETA="${BETA:-0.1}"
SFT_WEIGHT="${SFT_WEIGHT:-0.0}"
SAVE_EPOCH="${SAVE_EPOCH:-16}"

WANDB_PROJECT="${WANDB_PROJECT:-vertigo}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}}"
export WANDB_PROJECT
export WANDB_NAME="${WANDB_RUN_NAME}"

cd "${ROOT_DIR}"

cmd=(
  accelerate launch
  --config_file acc_configs/gpu2.yaml
  train_dpo.py ArAE
  --workspace "${WORKSPACE}"
  --exp-name "${EXP_NAME}"
  --resume "${POLICY_CKPT}"
  --dpo-reference "${REFERENCE_CKPT}"
  --dpo-preference-path "${PREFERENCE_PATH}"
  --cond-mode text
  --text-key Movement
  --num-cond-tokens 77
  --sd-model-path "${SD_MODEL_PATH}"
  --camera-norm-mode origin_global
  --camera-translation-norm dataset_p99
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --num-epochs "${NUM_EPOCHS}"
  --lr "${LR}"
  --dpo-beta "${BETA}"
  --dpo-sft-weight "${SFT_WEIGHT}"
  --save-epoch "${SAVE_EPOCH}"
)

if [[ -n "${DATA_ROOT}" ]]; then
  cmd+=(--dpo-data-root "${DATA_ROOT}")
fi

"${cmd[@]}"
