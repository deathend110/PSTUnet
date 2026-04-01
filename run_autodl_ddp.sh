#!/usr/bin/env bash

set -euo pipefail

# =========================
# AutoDL 2-GPU DDP launcher
# Usage:
#   bash run_autodl_ddp.sh
#
# Notes:
# 1. This script is aligned with the current train.py arguments.
# 2. batch-size means per-process / per-GPU batch size.
# 3. global batch size = BATCH_SIZE * NPROC_PER_NODE.
# =========================

# GPUs to expose to torchrun.
# For a 2-GPU AutoDL instance, keep "0,1".
export CUDA_VISIBLE_DEVICES="0,1"

# Project root on the server.
PROJECT_DIR="/root/autodl-tmp/PSTUnet"

# Dataset root directory.
# train.py will read:
#   ${BASE_DIR}/${DOMAIN}/traindata
#   ${BASE_DIR}/${DOMAIN}/testdata
BASE_DIR="/root/autodl-tmp/PST_Dataset"

# Data domain.
# Options:
#   DB
#   Linear
DOMAIN="DB"

# Root output directory.
# train.py will create an experiment subfolder inside it.
OUTPUTS_DIR="./output"

# Per-GPU batch size.
# Start with 1 for safety on the first run.
BATCH_SIZE=1

# DataLoader worker count per process.
NUM_WORKERS=4

# Base learning rate.
# train.py will apply staged LR updates on top of this base value:
#   Epoch 1-15  -> 1e-4
#   Epoch 16-25 -> 5e-5
#   Epoch 26-35 -> 1e-5
#   Epoch 36+   -> 1e-6
# So if you keep LR=1e-4, the schedule matches the prompt exactly.
LR=1e-4

# Total training epochs.
# Recommended default is 35.
# If you want the extra "extreme converge" stage, set this to 40.
NUM_EPOCHS=40

# Early stopping patience.
# With the current train.py this now controls stop behavior only,
# and no longer triggers ad-hoc LR decay.
PATIENCE=20

# TV loss weight.补齐utils里面 /batchsize 的缩小
TV_WEIGHT=2e-3

# SSIM loss weight used after the SSIM stage starts.
# In the current train.py, SSIM is enabled from epoch 26 onward.
SSIM_WEIGHT=0.2

# Scale factor used in model selection score:
#   score = PSNR + SSIM * SCORE_SSIM_SCALE
# A value of 50 is a balanced default for PSNR around 20-22 dB
# and SSIM around 0.5-0.7+.
SCORE_SSIM_SCALE=50.0

# De-quantization divisor.
# Use 255.0 for uint8 data, or 65535.0 for uint16 data.
MAX_VAL=255.0

# Number of DDP processes, usually equal to the number of GPUs used.
NPROC_PER_NODE=2

# Number of steps to accumulate gradients.
# To match the baseline effective batch size of 8 using 2 GPUs (batch_size=1):
# GLOBAL_BATCH_SIZE = BATCH_SIZE(1) * NPROC_PER_NODE(2) * GRAD_ACCUM_STEPS(1) = 2
GRAD_ACCUM_STEPS=1

# Limit OpenMP CPU threads to avoid oversubscription.
export OMP_NUM_THREADS=4

cd "${PROJECT_DIR}"

echo "========================================"
echo "Start AutoDL DDP training"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "BASE_DIR=${BASE_DIR}"
echo "DOMAIN=${DOMAIN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "PER_GPU_BATCH_SIZE=${BATCH_SIZE}"
echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM_STEPS))"
echo "BASE_LR=${LR}"
echo "NUM_EPOCHS=${NUM_EPOCHS}"
echo "SSIM_WEIGHT=${SSIM_WEIGHT}"
echo "SCORE_SSIM_SCALE=${SCORE_SSIM_SCALE}"
echo "========================================"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train.py \
  --base-dir "${BASE_DIR}" \
  --domain "${DOMAIN}" \
  --outputs-dir "${OUTPUTS_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --num-epochs "${NUM_EPOCHS}" \
  --patience "${PATIENCE}" \
  --tv-weight "${TV_WEIGHT}" \
  --ssim-weight "${SSIM_WEIGHT}" \
  --score-ssim-scale "${SCORE_SSIM_SCALE}" \
  --max-val "${MAX_VAL}"
