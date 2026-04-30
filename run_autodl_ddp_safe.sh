#!/usr/bin/env bash
set -euo pipefail

# =========================
# AutoDL 2-GPU DDP launcher - safer NCCL/vGPU version
# Usage:
#   bash run_autodl_ddp_safe.sh
# =========================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

PROJECT_DIR="/root/PSTUnet"
BASE_DIR="/root/autodl-tmp/Sequence_Dataset_AzimuthMix_q3_rt_only"
DOMAIN="DB"
OUTPUTS_DIR="./output"

BATCH_SIZE=1
# 首次排错建议先用 0；确认跑通后可以改回 2 或 4。
NUM_WORKERS=4
LR=1e-4
NUM_EPOCHS=80
PATIENCE=20
TV_WEIGHT=2e-3
MAX_VAL=255.0
NPROC_PER_NODE=2
GRAD_ACCUM_STEPS=1

# ---- Stability settings for AutoDL / vGPU / container DDP ----
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="${MASTER_PORT:-29501}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# 如果仍然 SIGSEGV，再临时打开下一行：
# export NCCL_SHM_DISABLE=1

cd "${PROJECT_DIR}"

# 若你把我给你的 train_ddp_safe.py 另存为 train.py，这里保持 train.py 即可。
TRAIN_SCRIPT="train_ddp_safe.py"
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  TRAIN_SCRIPT="train.py"
fi

echo "========================================"
echo "Start AutoDL DDP training - safe launcher"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
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
echo "MASTER_PORT=${MASTER_PORT}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "========================================"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${TRAIN_SCRIPT}" \
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
