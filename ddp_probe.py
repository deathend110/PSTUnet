#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最小 DDP 探针：
- 只构造一个很小的线性层模型
- 验证 process group 初始化、DDP 包装、一次前向和反向传播
- 用于区分“训练脚本问题”和“容器 / NCCL / vGPU 环境问题”

示例：
    BACKEND=gloo torchrun --nproc_per_node=2 ddp_probe.py
    BACKEND=nccl torchrun --nproc_per_node=2 ddp_probe.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def log(message: str) -> None:
    """
    统一打印格式，便于区分不同 rank 的启动阶段。
    """
    rank = int(os.environ.get("RANK", "0"))
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} - [rank {rank}] - {message}", flush=True)


def main() -> None:
    """
    最小验证路径：
    1. 读取 torchrun 注入的 rank / local_rank / world_size
    2. 先绑定当前进程对应 GPU
    3. 初始化 process group
    4. 构造极小模型并包成 DDP
    5. 做一次前向、loss、反向传播
    """
    backend = os.environ.get("BACKEND", "nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch.cuda.is_available():
        raise RuntimeError("当前探针要求 CUDA 可用，因为目标是验证多卡 DDP 环境。")

    visible_gpu_count = torch.cuda.device_count()
    if local_rank >= visible_gpu_count:
        raise ValueError(
            f"LOCAL_RANK={local_rank} 超出可见 GPU 数量范围，visible_gpu_count={visible_gpu_count}"
        )

    log(
        f"probe start | backend={backend} | world_size={world_size} | "
        f"local_rank={local_rank} | visible_gpu_count={visible_gpu_count}"
    )

    # 先绑定设备，再初始化 process group。
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    log(f"device bound to {device}")

    dist.init_process_group(backend=backend, init_method="env://")
    log("process group initialized")

    # 构造一个极小模型，避免把 SAR 主模型的复杂度带进排障。
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
    ).to(device)
    log("minimal model created on device")

    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    log("DDP wrapper created")

    # 做一次最小前反传，验证参数同步和梯度回传路径。
    inputs = torch.randn(8, 16, device=device)
    targets = torch.randn(8, 16, device=device)
    outputs = ddp_model(inputs)
    loss = torch.mean((outputs - targets) ** 2)
    log(f"forward ok | loss={loss.item():.6f}")

    loss.backward()
    log("backward ok")

    dist.barrier()
    log("barrier ok")

    dist.destroy_process_group()
    log("probe finished successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"probe failed: {type(exc).__name__}: {exc}")
        raise
