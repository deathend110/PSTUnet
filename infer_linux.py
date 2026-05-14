import argparse
import csv
import json
import os
import random
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import scipy.io
except ImportError:
    scipy = None

from datasets import SARDataset
from model import PST_UNet
from utils import SSIMLoss, calc_psnr


class LinearInferenceDataset(Dataset):
    def __init__(self, base_dir, max_val=255.0, num_samples=None, seed=42):
        # 这里固定读取 Linear 域 test 集，保持现有 Linux 推理流程不变
        self.base_dir = base_dir
        self.max_val = max_val
        self.num_samples = num_samples
        self.seed = seed
        self.dataset = SARDataset(base_dir=base_dir, domain="Linear", mode="test", max_val=max_val)
        self.linear_dir = self.dataset.data_dir
        all_linear_paths = list(self.dataset.file_paths)

        if num_samples is not None:
            self.linear_paths = self._sample_linear_paths(all_linear_paths, num_samples=num_samples, seed=seed)
        else:
            self.linear_paths = all_linear_paths

        # 直接覆盖底层 file_paths，保证后续 __getitem__ 与采样结果严格一致
        self.dataset.file_paths = self.linear_paths

    def __len__(self):
        return len(self.linear_paths)

    @staticmethod
    def _get_prefix(linear_path):
        # 根据文件名前缀分组，确保随机抽样时每类目标至少覆盖一次
        linear_name = os.path.basename(linear_path)
        if "_L_seq_" not in linear_name:
            raise ValueError(f"Linear file name does not match expected pattern '*_L_seq_*.mat': {linear_name}")
        return linear_name.split("_L_seq_")[0]

    def _sample_linear_paths(self, all_linear_paths, num_samples, seed):
        # 按前缀分组抽样，避免只抽到少数场景，破坏 SAR 序列评估代表性
        if num_samples < 1:
            raise ValueError("--num-samples must be >= 1 when provided")
        if num_samples > len(all_linear_paths):
            raise ValueError(
                f"--num-samples={num_samples} exceeds available Linear test samples ({len(all_linear_paths)})."
            )

        grouped_paths = {}
        for linear_path in all_linear_paths:
            prefix = self._get_prefix(linear_path)
            grouped_paths.setdefault(prefix, []).append(linear_path)

        num_groups = len(grouped_paths)
        if num_samples < num_groups:
            raise ValueError(
                f"--num-samples={num_samples} is too small to cover every image type once; "
                f"need at least {num_groups} samples."
            )

        rng = random.Random(seed)
        selected_paths = []
        remaining_paths = []

        for prefix in sorted(grouped_paths):
            group_paths = list(grouped_paths[prefix])
            chosen_path = rng.choice(group_paths)
            selected_paths.append(chosen_path)
            remaining_paths.extend(path for path in group_paths if path != chosen_path)

        extra_needed = num_samples - len(selected_paths)
        if extra_needed > 0:
            selected_paths.extend(rng.sample(remaining_paths, extra_needed))

        return sorted(selected_paths)

    def __getitem__(self, idx):
        # 显式返回原始样本索引，保证多 batch 推理时 CSV 与 .mat 保存顺序不发生错位
        linear_path = self.linear_paths[idx]
        linear_name = os.path.basename(linear_path)
        input_tensor, target_tensor = self.dataset[idx]
        return input_tensor, target_tensor, linear_name, idx


def parse_args():
    parser = argparse.ArgumentParser(description="PST-UNet Linear-only inference script for Linux")
    default_checkpoint = (
        "./output/Model(PST_UNet)-Dataset(AzimuthMix_q7)-Loss(L1+TV+tv0.002000)-Epochs80-Batch_size1-lr0.000100-domainLinear/best.pth"
    )
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint, help="Path to a model state_dict checkpoint.")

    dataset_path = "/root/autodl-tmp/Sequence_Dataset_AzimuthMix_q7_rt_only"
    parser.add_argument(
        "--base-dir",
        type=str,
        default=dataset_path,
        help="Root directory of the dataset, e.g., /root/autodl-tmp/Sequence_Dataset_AzimuthMix_q7_rt_only",
    )

    save_dir = "./inference_output/linear_test_" + os.path.basename(dataset_path.rstrip("/\\")).replace(
        "Sequence_Dataset_", ""
    ).replace("_rt_only", "")
    parser.add_argument("--output-dir", type=str, default=save_dir)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Randomly sample N Linear test files for inference. If not set, infer all Linear test files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --num-samples is provided.")
    parser.add_argument("--max-val", type=float, default=255.0)
    parser.add_argument("--device", type=str, default="cuda", help='Examples: "cuda", "cuda:0", "cpu"')

    parser.add_argument(
        "--auto-run-all",
        action="store_true",
        help="Automatically scan output/*/best.pth and run the matched Linear test set for each checkpoint.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="./output",
        help="Root directory used by --auto-run-all to scan model checkpoints.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/root/autodl-tmp",
        help="Dataset root used by --auto-run-all, e.g. /root/autodl-tmp.",
    )
    parser.add_argument(
        "--inference-root",
        type=str,
        default="./inference_output",
        help="Inference output root used by --auto-run-all.",
    )
    return parser.parse_args()


def resolve_device(device_text):
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_text)


def maybe_unwrap_state_dict(checkpoint_obj):
    # 同时兼容直接保存 state_dict 和保存成 dict 包装后的 checkpoint 结构
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return checkpoint_obj[key]
    return checkpoint_obj


def validate_args(args):
    if scipy is None:
        raise RuntimeError("scipy is required to save .mat inference results, but it is not installed.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.num_samples is not None and args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")

    if args.auto_run_all:
        checkpoints_dir = Path(args.checkpoints_dir)
        if not checkpoints_dir.is_dir():
            raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir.resolve()}")
        dataset_root = Path(args.dataset_root)
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root.resolve()}")
    else:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {os.path.abspath(args.checkpoint)}")


def save_prediction_file(save_path, mat_dict):
    # Matlab 后处理仍依赖 .mat 文件，因此这里保持原有输出格式不变
    save_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(save_path.with_suffix(".mat")), mat_dict)


def extract_dataset_name_from_checkpoint(checkpoint_path):
    # 从训练输出目录名里的 Dataset(...) 片段提取数据集标签
    checkpoint_dir_name = Path(checkpoint_path).resolve().parent.name
    matched = re.search(r"Dataset\(([^)]+)\)", checkpoint_dir_name)
    if matched is None:
        raise ValueError(f"Checkpoint directory name does not contain Dataset(...): {checkpoint_dir_name}")
    return matched.group(1)


def build_dataset_dir(dataset_root, dataset_name):
    # 训练与推理统一使用 Sequence_Dataset_<name>_rt_only 这套命名规则
    dataset_dir = f"{str(dataset_root).rstrip('/\\')}/Sequence_Dataset_{dataset_name}_rt_only"
    return dataset_dir.replace("\\", "/")


def build_output_dir(inference_root, dataset_name):
    # 保留原 infer_linux 目录结构，只按数据集名称区分输出目录
    return (Path(inference_root) / f"linear_test_{dataset_name}").as_posix()


def build_auto_inference_jobs(checkpoints_dir, dataset_root, inference_root):
    # 扫描 output/*/best.pth，并生成一组一一对应的“模型-数据集-输出目录”任务
    checkpoint_paths = sorted(Path(checkpoints_dir).glob("*/best.pth"), key=lambda path: str(path))
    jobs = []
    for checkpoint_path in checkpoint_paths:
        dataset_name = extract_dataset_name_from_checkpoint(checkpoint_path)
        jobs.append(
            {
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "dataset_name": dataset_name,
                "base_dir": build_dataset_dir(dataset_root, dataset_name),
                "output_dir": build_output_dir(inference_root, dataset_name),
            }
        )
    return jobs


def build_dataloader(dataset, batch_size, num_workers, use_cuda):
    # 单独封装 DataLoader，确保单模型模式和自动批量模式共享同一份加载配置
    if len(dataset) == 0:
        raise RuntimeError(f"No .mat files found for inference under {dataset.linear_dir}.")

    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "drop_last": False,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **dataloader_kwargs)


def create_shared_runtime(args):
    # 批量模式下模型结构、设备、SSIM 计算器都可以复用，只需切换 checkpoint 权重
    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    model = PST_UNet(in_channels=2, out_channels=1, base_dim=64).to(device)
    ssim_calculator = SSIMLoss().to(device)
    autocast_context = (lambda: torch.amp.autocast(device_type="cuda", enabled=True)) if use_cuda else nullcontext
    return {
        "device": device,
        "use_cuda": use_cuda,
        "model": model,
        "ssim_calculator": ssim_calculator,
        "autocast_context": autocast_context,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }


def run_inference_job(job_config, shared_runtime):
    # 这里执行单个推理任务，自动批量模式只是对这个函数做循环调度
    checkpoint_path = job_config["checkpoint"]
    base_dir = job_config["base_dir"]
    output_dir = job_config["output_dir"]
    num_samples = job_config["num_samples"]
    seed = job_config["seed"]
    max_val = job_config["max_val"]

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {os.path.abspath(checkpoint_path)}")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Dataset directory not found: {os.path.abspath(base_dir)}")

    device = shared_runtime["device"]
    use_cuda = shared_runtime["use_cuda"]
    model = shared_runtime["model"]
    ssim_calculator = shared_runtime["ssim_calculator"]
    autocast_context = shared_runtime["autocast_context"]

    os.makedirs(output_dir, exist_ok=True)
    predictions_dir = Path(output_dir) / "predictions"

    dataset = LinearInferenceDataset(
        base_dir=base_dir,
        max_val=max_val,
        num_samples=num_samples,
        seed=seed,
    )
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=shared_runtime["batch_size"],
        num_workers=shared_runtime["num_workers"],
        use_cuda=use_cuda,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(maybe_unwrap_state_dict(checkpoint), strict=True)
    model.eval()

    # 预分配指标槽位，确保 CSV 行顺序与测试集原始顺序完全一致
    metric_rows = [None] * len(dataset)
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    with torch.no_grad():
        progress = tqdm(
            dataloader,
            desc=f"Infer[{Path(output_dir).name}]",
            total=len(dataloader),
        )
        for batch in progress:
            # DataLoader 在 shuffle=False 下返回稳定顺序，因此多 batch 不会破坏序列对应关系
            inputs, targets, file_names, sample_indices = batch

            inputs = inputs.to(device, non_blocking=use_cuda)
            targets = targets.to(device, non_blocking=use_cuda)

            with autocast_context():
                outputs = model(inputs)

            outputs = outputs.clamp(0.0, 1.0)
            targets = targets.clamp(0.0, 1.0)

            outputs_cpu = outputs.cpu()
            targets_cpu = targets.cpu()
            batch_psnr_values = []
            batch_ssim_values = []

            for local_idx, file_name in enumerate(file_names):
                # 使用数据集原始索引落盘，避免最后一个不完整 batch 时发生指标错位
                sample_index = int(sample_indices[local_idx])

                pred_2d_sample = outputs_cpu[local_idx].transpose(0, 1).contiguous()
                tgt_2d_sample = targets_cpu[local_idx].transpose(0, 1).contiguous()
                sample_psnr = calc_psnr(pred_2d_sample, tgt_2d_sample).item()
                sample_ssim = 1.0 - ssim_calculator(
                    pred_2d_sample.to(device),
                    tgt_2d_sample.to(device),
                ).item()

                batch_psnr_values.append(sample_psnr)
                batch_ssim_values.append(sample_ssim)
                total_psnr += sample_psnr
                total_ssim += sample_ssim
                total_samples += 1

                metric_rows[sample_index] = {
                    "index": sample_index,
                    "file_name": file_name,
                    "psnr": sample_psnr,
                    "ssim": sample_ssim,
                }

                # 继续沿用 seq_pred_Linear 作为 Matlab/后处理兼容的主键
                mat_pred_linear = outputs_cpu[local_idx, 0].numpy().transpose(1, 2, 0).astype(
                    np.float32, copy=False
                )
                save_prediction_file(predictions_dir / file_name, {"seq_pred_Linear": mat_pred_linear})

            progress.set_postfix(
                psnr=f"{(sum(batch_psnr_values) / max(len(batch_psnr_values), 1)):.2f}",
                ssim=f"{(sum(batch_ssim_values) / max(len(batch_ssim_values), 1)):.4f}",
            )

    mean_psnr = total_psnr / max(total_samples, 1)
    mean_ssim = total_ssim / max(total_samples, 1)

    metrics_csv_path = Path(output_dir) / "metrics.csv"
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "file_name", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "base_dir": os.path.abspath(base_dir),
        "domain": "Linear",
        "mode": "test",
        "sampled": num_samples is not None,
        "requested_num_samples": num_samples,
        "seed": seed,
        "num_samples": total_samples,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "predictions_dir": str(predictions_dir.resolve()),
        "prediction_format": "mat",
        "saved_keys": ["seq_pred_Linear"],
        "sample_prefixes": sorted({dataset._get_prefix(path) for path in dataset.linear_paths}),
    }

    summary_path = Path(output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"Checkpoint: {os.path.abspath(checkpoint_path)}")
    print(f"Dataset: {os.path.abspath(base_dir)}")
    print(f"Samples: {total_samples}")
    if num_samples is not None:
        print(f"Sampling: random {num_samples} files (seed={seed})")
    print(f"Mean PSNR: {mean_psnr:.4f}")
    print(f"Mean SSIM: {mean_ssim:.6f}")
    print(f"Inference results: {predictions_dir.resolve()}")
    print("=" * 60)
    return summary


def write_auto_run_summary(inference_root, run_records):
    # 批量模式额外生成一份总汇总，方便查看每个数据集对应模型是否全部跑完
    summary_path = Path(inference_root) / "auto_infer_summary.json"
    summary_path.write_text(json.dumps(run_records, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def run_auto_inference(args):
    # 自动扫描 output 中全部模型，并按 Dataset(...) 自动匹配对应 Linear 测试集
    jobs = build_auto_inference_jobs(
        checkpoints_dir=args.checkpoints_dir,
        dataset_root=args.dataset_root,
        inference_root=args.inference_root,
    )
    if not jobs:
        raise RuntimeError(f"No checkpoint found under {Path(args.checkpoints_dir).resolve()}.")

    shared_runtime = create_shared_runtime(args)
    run_records = []
    failed_jobs = []

    for job in jobs:
        job["num_samples"] = args.num_samples
        job["seed"] = args.seed
        job["max_val"] = args.max_val
        try:
            summary = run_inference_job(job, shared_runtime)
            run_records.append(
                {
                    "status": "success",
                    "dataset_name": job["dataset_name"],
                    "checkpoint": job["checkpoint"],
                    "base_dir": job["base_dir"],
                    "output_dir": job["output_dir"],
                    "mean_psnr": summary["mean_psnr"],
                    "mean_ssim": summary["mean_ssim"],
                    "num_samples": summary["num_samples"],
                }
            )
        except Exception as exc:
            failed_jobs.append(job["checkpoint"])
            run_records.append(
                {
                    "status": "failed",
                    "dataset_name": job["dataset_name"],
                    "checkpoint": job["checkpoint"],
                    "base_dir": job["base_dir"],
                    "output_dir": job["output_dir"],
                    "error": str(exc),
                }
            )

    summary_path = write_auto_run_summary(args.inference_root, run_records)
    print(f"Auto summary: {summary_path.resolve()}")

    if failed_jobs:
        raise RuntimeError("Auto inference finished with failures:\n" + "\n".join(failed_jobs))

    return run_records


def main():
    args = parse_args()
    validate_args(args)

    if args.auto_run_all:
        run_auto_inference(args)
        return

    shared_runtime = create_shared_runtime(args)
    job_config = {
        "checkpoint": args.checkpoint,
        "base_dir": args.base_dir,
        "output_dir": args.output_dir,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "max_val": args.max_val,
    }
    run_inference_job(job_config, shared_runtime)


if __name__ == "__main__":
    main()
