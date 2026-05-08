#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
环境检查脚本：
- 用于排查 AutoDL / 容器中的 CUDA、NCCL、PyTorch、GPU 可见性等问题
- 运行后会把结果保存到仓库根目录的 env_check.log

用法：
    python check_env.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
LOG_PATH = ROOT_DIR / "env_check.log"


def append_line(text: str = "") -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def append_section(title: str) -> None:
    append_line("=" * 80)
    append_line(title)
    append_line("=" * 80)


def run_command(cmd: list[str] | str, title: str, shell: bool = False) -> None:
    """
    执行外部命令，并把标准输出/错误输出完整写入日志。
    这里不在失败时中断，目的是尽可能多收集环境证据。
    """
    append_section(title)
    append_line(f"command: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        append_line(f"exit_code: {result.returncode}")
        append_line("--- stdout ---")
        append_line(result.stdout.rstrip())
        append_line("--- stderr ---")
        append_line(result.stderr.rstrip())
    except Exception as exc:
        append_line(f"command execution failed: {type(exc).__name__}: {exc}")


def write_basic_env() -> None:
    """
    记录最基础的容器/进程环境信息。
    这部分有助于判断当前脚本究竟运行在哪个 Python、哪个工作目录、哪些环境变量下。
    """
    append_section("Basic Environment")
    append_line(f"time: {datetime.now().isoformat(timespec='seconds')}")
    append_line(f"python_executable: {sys.executable}")
    append_line(f"python_version: {sys.version}")
    append_line(f"cwd: {os.getcwd()}")
    append_line(f"script_root: {ROOT_DIR}")

    watched_env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "PATH",
        "CONDA_DEFAULT_ENV",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "TORCH_NCCL_BLOCKING_WAIT",
        "OMP_NUM_THREADS",
    ]
    for key in watched_env_keys:
        append_line(f"{key}={os.environ.get(key, '')}")


def write_torch_env() -> None:
    """
    通过内嵌 Python 片段采集 torch / CUDA / cuDNN / GPU 可见性信息。
    即便当前容器没有安装 torch，也会把异常写入日志，而不是直接退出。
    """
    code = r"""
import sys
try:
    import torch
    print("torch_version:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("torch_cuda_version:", torch.version.cuda)
    print("cudnn_version:", torch.backends.cudnn.version())
    print("device_count:", torch.cuda.device_count())
    if hasattr(torch.cuda, "nccl"):
        try:
            print("torch_nccl_version:", torch.cuda.nccl.version())
        except Exception as exc:
            print("torch_nccl_version_error:", repr(exc))
    else:
        print("torch_nccl_version: unavailable")
    for i in range(torch.cuda.device_count()):
        print(f"gpu_{i}_name:", torch.cuda.get_device_name(i))
except Exception as exc:
    print("torch_probe_error:", type(exc).__name__, exc)
    sys.exit(1)
"""
    run_command([sys.executable, "-c", code], "Torch / CUDA Probe")


def main() -> None:
    # 每次运行先覆盖旧日志，保证你拿到的是这一次检查的完整结果。
    LOG_PATH.write_text("", encoding="utf-8")

    write_basic_env()
    run_command(["nvidia-smi"], "nvidia-smi")
    run_command(["bash", "-lc", "ldconfig -p | grep nccl"], "ldconfig grep nccl")
    run_command(["bash", "-lc", "which all_reduce_perf || true"], "which all_reduce_perf")
    run_command(["bash", "-lc", "python -m pip show torch"], "pip show torch")
    write_torch_env()

    # 如果容器里安装了 nccl-tests，这里只记录可执行路径，不默认做重型 all_reduce 压测。
    # 真正做 all_reduce_perf 时通常需要确认当前实例允许双卡占用较长时间。
    if shutil.which("all_reduce_perf"):
        append_section("NCCL Test Hint")
        append_line("Detected all_reduce_perf in PATH.")
        append_line("If you want to run NCCL bandwidth test manually, use:")
        append_line("all_reduce_perf -b 8 -e 128M -f 2 -g 2")

    append_section("Done")
    append_line(f"log_saved_to: {LOG_PATH}")
    print(f"环境检查已完成，日志已保存到: {LOG_PATH}")


if __name__ == "__main__":
    main()
