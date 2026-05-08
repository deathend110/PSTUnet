# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAR 序列复原项目，基于 PST-UNet（mask-aware 序列融合网络）。Python 负责训练与推理，Matlab 负责后处理评估与可视化。

## 常用命令

```bash
# 训练（Windows 单卡/DataParallel）
python train.py --base-dir G:\VSCODE-G\PST_Dataset --domain DB --outputs-dir .\output

# 训练（Linux DDP）
bash run_autodl_ddp.sh

# 推理（Windows）
python infer.py --checkpoint .\output\<run>\best.pth --base-dir G:\VSCODE-G\PST_Dataset --output-dir .\inference_output\db_test_sample

# 推理（Linux）
python infer_linux.py --checkpoint ./output/<run>/best.pth --base-dir /root/autodl-tmp/PST_Dataset --output-dir ./inference_output/db_test

# 验证：架构/数据路径变更后
python check_gradients.py

# 验证：DataLoader 变更后
python benchmark_dataloader.py --base-dir G:\VSCODE-G\PST_Dataset --domain DB --mode train

# 验证：推理变更后（单样本快速检查）
python infer.py --checkpoint .\output\<run>\best.pth --base-dir G:\VSCODE-G\PST_Dataset --output-dir .\inference_output\db_test_sample --num-samples 1
```

## 架构总览

### 训练管线

[train.py](train.py) 支持单卡、`DataParallel`、DDP 三种模式。核心流程：
- 用 `SARDataset` 加载 HDF5 `.mat` 文件 → 输入 `[B, 2, T, H, W]`（ch0: 归一化 SAR 序列, ch1: mode mask），目标 `[B, 1, T, H, W]`
- `PST_UNet(in_channels=2, out_channels=1)` 做序列复原
- 损失 `Seq_SAR_L1TVLoss`，验证用 PSNR 选最优模型
- 余弦退火学习率 + 梯度累积 + AMP 混合精度
- Early stopping 耐心值 15 epoch

### 推理管线

- [infer.py](infer.py): Windows 端，双域（DB + Linear）推理
- [infer_linux.py](infer_linux.py): Linux/AutoDL 端推理
- 推理输出 `.mat` 文件到 `output-dir/predictions/`，同时写 `metrics.csv` 和 `summary.json`
- Matlab 评估脚本依赖 `seq_pred_DB` 作为 `.mat` 载荷键名

### 数据约定

数据集根目录结构：
```
base_dir/
├── DB/
│   ├── traindata/   (*.mat, HDF5 keys: /seq_input, /seq_GT)
│   └── testdata/
└── Linear/
    ├── traindata/   (*.mat, HDF5 keys: /seq_input_L, /seq_GT_L)
    └── testdata/
```

- `datasets.py` 优先读取 `/mode_mask_512_all`，缺失时回退到 `/frame_mode_id`
- 这不是加性白噪声去噪任务；SAR 特有的归一化和域转换假设贯穿全管线

### Matlab 评估

- [Inference_compute.m](inference_output/Inference_compute.m): 按帧组聚合 PSNR/SSIM
- [Inference_draw.m](inference_output/Inference_draw.m): 选定帧可视化
- [Inference_compute_draw.m](inference_output/Inference_compute_draw.m): 聚合指标 + 逐帧分析
- Linear 域评估依赖逆归一化，需要 `V_MAX_GT`, `V_MIN_GT`, `V_MAX_GT_L`, `V_MIN_GT_L` 等元数据变量
- 帧分组固定为 16 帧序列，含 `180MHz`、`Mixed`、`60MHz` 子集

## 用户偏好

- 所有面向用户的输出使用中文
- 文件编码统一 UTF-8
- 代码注释用中文，密度应足够让用户快速理解代码流程和意图
- 非平凡函数/代码块需注释：处理阶段/意图、输入输出含义、分支或变换存在的原因

## 编辑守则

- 优先修改现有管线，不要引入并行管线
- 保持 `[B, C, T, H, W]` 张量约定，除非整个管线有意迁移
- 修改训练代码时检查是否影响 DDP、梯度累积或 checkpoint 兼容性
- 修改推理代码时保持 `seq_pred_DB` 作为 `.mat` 主载荷键名（除非同步更新 Matlab 消费者）
- 保持推理输出路径和文件名稳定 — Matlab 脚本依赖它们
- Python 输出格式变更时，同一任务中同步更新 Matlab 消费者
- 不要改动 `output/` 和 `inference_output/` 中的生成结果
- 不要改动 [t.m](t.m) 等临时分析脚本，除非任务明确针对它们
- 不要在未读文件的情况下提出代码修改
