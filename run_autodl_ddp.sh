#!/usr/bin/env bash

set -euo pipefail

# =========================
# AutoDL 2-GPU DDP launcher
# Usage:
#   bash run_autodl_ddp.sh
# =========================

export CUDA_VISIBLE_DEVICES="0,1"

PROJECT_DIR="/root/PSTUnet"

# train.py 会读取：
#   ${BASE_DIR}/${DOMAIN}/traindata
#   ${BASE_DIR}/${DOMAIN}/testdata
BASE_DIR="/root/autodl-tmp/Sequence_Dataset_AzimuthMix_q3_rt_only"

DOMAIN="Linear"
OUTPUTS_DIR="./output"

# 每张卡上的 batch size
BATCH_SIZE=1

# 每个训练进程各自的 DataLoader worker 数
NUM_WORKERS=4

# 基础学习率
LR=1e-4

# 总 epoch 数
NUM_EPOCHS=80

# 早停 patience
PATIENCE=20

# TV loss 权重
TV_WEIGHT=2e-3

# 反量化除数
MAX_VAL=255.0

# DDP 进程数
NPROC_PER_NODE=2

# DDP 通信后端
# 当前脚本默认使用 gloo，优先保证在已知 vGPU 环境中的稳定性。
# 如果你更换到了新容器/新实例，想优先测试 nccl，可这样启动：
#   DIST_BACKEND=nccl bash run_autodl_ddp.sh
# 如果 nccl 再次出现 DDP 初始化异常，再退回 gloo 排障。
DIST_BACKEND="${DIST_BACKEND:-nccl}"

# 梯度累积步数
GRAD_ACCUM_STEPS=2

export OMP_NUM_THREADS=4

# 这些环境变量主要用于 NCCL 排障。
# 当前如果使用 gloo，它们基本不会参与训练流程，但保留也无妨。
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1

cd "${PROJECT_DIR}"

echo "========================================"
echo "Start AutoDL DDP training"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "BASE_DIR=${BASE_DIR}"
echo "DOMAIN=${DOMAIN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "DIST_BACKEND=${DIST_BACKEND}"
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
  --dist-backend "${DIST_BACKEND}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --num-epochs "${NUM_EPOCHS}" \
  --patience "${PATIENCE}" \
  --tv-weight "${TV_WEIGHT}" \
  --max-val "${MAX_VAL}"
