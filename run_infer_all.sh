#!/usr/bin/env bash

set -euo pipefail

# =========================
# AutoDL 批量 Linear 推理脚本
# 用途：
#   1. 自动扫描 output/*/best.pth
#   2. 从模型目录名里的 Dataset(...) 自动识别数据集标签
#   3. 自动匹配 /root/autodl-tmp/Sequence_Dataset_<name>_rt_only
#   4. 输出到 inference_output/linear_test_<name>
# 用法：
#   bash run_autodl_infer_all.sh
# =========================

PROJECT_DIR="/root/PSTUnet"

# infer_linux.py 会在该目录下扫描 */best.pth
CHECKPOINTS_DIR="./output"

# SAR 数据集根目录。
# 例如会自动匹配成：
#   /root/autodl-tmp/Sequence_Dataset_AzimuthMix_q3_rt_only
#   /root/autodl-tmp/Sequence_Dataset_RangeMix_q7_rt_only
DATASET_ROOT="/root/autodl-tmp"

# 推理输出根目录。
# 实际输出仍保持原结构，例如：
#   ./inference_output/linear_test_AzimuthMix_q3
INFERENCE_ROOT="./inference_output"

# 推理设备与 DataLoader 参数
DEVICE="cuda"
BATCH_SIZE=1
NUM_WORKERS=4

# 归一化上限，保持与训练/现有推理脚本一致
MAX_VAL=255.0

# 仅在需要随机抽样测试集时设置为正整数。
# 留空表示跑完整个 Linear test 集。
NUM_SAMPLES=""
SEED=42

cd "${PROJECT_DIR}"

echo "========================================"
echo "Start AutoDL auto inference"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "CHECKPOINTS_DIR=${CHECKPOINTS_DIR}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "INFERENCE_ROOT=${INFERENCE_ROOT}"
echo "DEVICE=${DEVICE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "MAX_VAL=${MAX_VAL}"
if [[ -n "${NUM_SAMPLES}" ]]; then
  echo "NUM_SAMPLES=${NUM_SAMPLES}"
  echo "SEED=${SEED}"
else
  echo "NUM_SAMPLES=ALL"
fi
echo "========================================"

EXTRA_ARGS=()
if [[ -n "${NUM_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--num-samples "${NUM_SAMPLES}" --seed "${SEED}")
fi

python infer_linux.py \
  --auto-run-all \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --dataset-root "${DATASET_ROOT}" \
  --inference-root "${INFERENCE_ROOT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-val "${MAX_VAL}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"
