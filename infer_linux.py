import argparse
import csv
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import scipy.io
except ImportError:
    scipy = None

import numpy as np
from datasets import SARDataset
from model import PST_UNet
from utils import SSIMLoss, calc_psnr


class LinearInferenceDataset(Dataset):
    def __init__(self, base_dir, max_val=255.0, num_samples=None, seed=42):
        # 记录 Linear 域测试集根目录与归一化尺度，供 SAR 序列推理复用
        self.base_dir = base_dir
        self.max_val = max_val
        self.dataset = SARDataset(base_dir=base_dir, domain="Linear", mode="test", max_val=max_val)
        self.linear_dir = self.dataset.data_dir
        all_linear_paths = list(self.dataset.file_paths)

        if num_samples is not None:
            self.linear_paths = self._sample_linear_paths(all_linear_paths, num_samples=num_samples, seed=seed)
        else:
            self.linear_paths = all_linear_paths

        self.dataset.file_paths = self.linear_paths

    def __len__(self):
        return len(self.linear_paths)

    @staticmethod
    def _get_prefix(linear_path):
        linear_name = os.path.basename(linear_path)
        if "_L_seq_" not in linear_name:
            raise ValueError(f"Linear file name does not match expected pattern '*_L_seq_*.mat': {linear_name}")
        return linear_name.split("_L_seq_")[0]

    def _sample_linear_paths(self, all_linear_paths, num_samples, seed):
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
        # 这里显式返回数据集原始索引，后续即使使用多 batch 推理，
        # 也能保证指标表与保存结果严格对应测试集顺序
        linear_path = self.linear_paths[idx]
        linear_name = os.path.basename(linear_path)
        input_tensor, target_tensor = self.dataset[idx]
        return input_tensor, target_tensor, linear_name, idx


def parse_args():
    parser = argparse.ArgumentParser(description="PST-UNet Linear-only inference script for Linux")
    default_checkpoint = (
        "./output/Model(PST_UNet)-Dataset(Sequence_Dataset_AzimuthMix)-Loss(L1+TV+tv0.002000)-Epochs80-Batch_size1-lr0.000100-domainLinear/best.pth"
    )
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint, help="Path to a model state_dict checkpoint.")
    dataset_path = "/root/autodl-tmp/Sequence_Dataset_AzimuthMix_q3_rt_only"
    parser.add_argument("--base-dir", type=str, default=dataset_path, help="Root directory of the dataset, e.g., /root/autodl-tmp/Sequence_Dataset_AzimuthMix_q3_rt_only")
    save_dir = "./inference_output/linear_test_" + os.path.basename(dataset_path.rstrip("/\\")).replace('Sequence_Dataset_', '').replace('_rt_only', '')
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
    return parser.parse_args()


def resolve_device(device_text):
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_text)


def maybe_unwrap_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return checkpoint_obj[key]
    return checkpoint_obj


def validate_args(args):
    if scipy is None:
        raise RuntimeError("scipy is required to save .mat inference results, but it is not installed.")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {os.path.abspath(args.checkpoint)}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.num_samples is not None and args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")


def save_prediction_file(save_path, mat_dict):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(save_path.with_suffix(".mat")), mat_dict)


def main():
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"

    os.makedirs(args.output_dir, exist_ok=True)
    predictions_dir = Path(args.output_dir) / "predictions"

    dataset = LinearInferenceDataset(
        base_dir=args.base_dir,
        max_val=args.max_val,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No .mat files found for inference under {dataset.linear_dir}.")

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
        "drop_last": False,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    model = PST_UNet(in_channels=2, out_channels=1, base_dim=64).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(maybe_unwrap_state_dict(checkpoint), strict=True)
    model.eval()

    ssim_calculator = SSIMLoss().to(device)
    autocast_context = (lambda: torch.amp.autocast(device_type="cuda", enabled=True)) if use_cuda else nullcontext

    # 预先按数据集长度分配指标槽位，保证 CSV 输出顺序与测试集顺序完全一致
    metric_rows = [None] * len(dataset)
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Infer", total=len(dataloader))
        for batch_idx, batch in enumerate(progress):
            # file_names 与 sample_indices 都由 DataLoader 按 shuffle=False 的顺序组 batch，
            # 因此多 batch 只影响吞吐，不会改变 SAR 序列样本的保存顺序
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
                # 直接使用数据集返回的原始索引，而不是由 batch 位置反推，
                # 这样最后一个不完整 batch 或未来更换 collate 方式时也不会错位
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

                mat_pred_linear = outputs_cpu[local_idx, 0].numpy().transpose(1, 2, 0).astype(np.float32, copy=False)
                save_prediction_file(predictions_dir / file_name, {"seq_pred_Linear": mat_pred_linear})

            progress.set_postfix(
                psnr=f"{(sum(batch_psnr_values) / max(len(batch_psnr_values), 1)):.2f}",
                ssim=f"{(sum(batch_ssim_values) / max(len(batch_ssim_values), 1)):.4f}",
            )

    mean_psnr = total_psnr / max(total_samples, 1)
    mean_ssim = total_ssim / max(total_samples, 1)

    metrics_csv_path = Path(args.output_dir) / "metrics.csv"
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "file_name", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "base_dir": os.path.abspath(args.base_dir),
        "domain": "Linear",
        "mode": "test",
        "sampled": args.num_samples is not None,
        "requested_num_samples": args.num_samples,
        "seed": args.seed,
        "num_samples": total_samples,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "predictions_dir": str(predictions_dir.resolve()),
        "prediction_format": "mat",
        "saved_keys": ["seq_pred_Linear"],
        "sample_prefixes": sorted({dataset._get_prefix(path) for path in dataset.linear_paths}),
    }

    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"Checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"Samples: {total_samples}")
    if args.num_samples is not None:
        print(f"Sampling: random {args.num_samples} files (seed={args.seed})")
    print(f"Mean PSNR: {mean_psnr:.4f}")
    print(f"Mean SSIM: {mean_ssim:.6f}")
    print(f"Inference results: {predictions_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
