#!/usr/bin/env bash

set -euo pipefail

# =========================
# AutoDL 2-GPU DDP launcher
# Usage:
#   bash run_autodl_ddp.sh
# =========================

export CUDA_VISIBLE_DEVICES="0,1"

PROJECT_DIR="/root/PSTUnet"

# train.py will read:
#   ${BASE_DIR}/${DOMAIN}/traindata
#   ${BASE_DIR}/${DOMAIN}/testdata
BASE_DIR="/root/autodl-tmp/Sequence_Dataset_AzimuthMix_q3_rt_only"

DOMAIN="DB"

OUTPUTS_DIR="./output"

# Per-GPU batch size
BATCH_SIZE=1

# DataLoader workers per process
NUM_WORKERS=4

# Base learning rate
LR=1e-4

# Total epochs
NUM_EPOCHS=40

# Early stopping patience
PATIENCE=20

# TV loss weight
TV_WEIGHT=2e-3

# De-quantization divisor
MAX_VAL=255.0

# DDP processes
NPROC_PER_NODE=2

# Gradient accumulation
GRAD_ACCUM_STEPS=1

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
echo "TV_WEIGHT=${TV_WEIGHT}"
echo "MAX_VAL=${MAX_VAL}"
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
  --max-val "${MAX_VAL}"