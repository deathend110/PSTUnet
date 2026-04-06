import argparse
import csv
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import scipy.io
except ImportError:
    scipy = None

from datasets import SARDataset
from model import PST_UNet
from utils import SSIMLoss, calc_psnr


def parse_args():
    parser = argparse.ArgumentParser(description="PST-UNet inference script")
    parser.add_argument("--base-dir", type=str, default=r"G:\VSCODE-G\PST_Dataset")
    parser.add_argument("--domain", type=str, default="DB", choices=["DB", "Linear"])
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"])
    checkpoint = "./output/Model(PST_UNet)-Dataset(PST_Dataset)-Loss(L1+TV+DynSSIM+tv0.002000)-Epochs40-Batch_size1-lr0.000100/best.pth"
    parser.add_argument("--checkpoint", type=str, default=checkpoint, help="Path to a model state_dict checkpoint.")
    parser.add_argument("--output-dir", type=str, default="./inference_output/db_test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
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


def save_prediction_file(save_path, payload):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(save_path.with_suffix(".mat")), payload)
    return str(save_path.with_suffix(".mat"))


def validate_args(args):
    if scipy is None:
        raise RuntimeError("scipy is required to save .mat inference results, but it is not installed.")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {os.path.abspath(args.checkpoint)}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")


def main():
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"

    os.makedirs(args.output_dir, exist_ok=True)
    predictions_dir = Path(args.output_dir) / "predictions"

    dataset = SARDataset(
        base_dir=args.base_dir,
        domain=args.domain,
        mode=args.mode,
        max_val=args.max_val,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No .mat files found for inference under {os.path.join(args.base_dir, args.domain, 'testdata' if args.mode == 'test' else 'traindata')}."
        )

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
    state_dict = maybe_unwrap_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    ssim_calculator = SSIMLoss().to(device)

    metric_rows = []
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    autocast_context = (lambda: torch.amp.autocast(device_type="cuda", enabled=True)) if use_cuda else nullcontext
    sample_offset = 0

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Infer", total=len(dataloader))
        for inputs, targets in progress:
            inputs = inputs.to(device, non_blocking=use_cuda)
            targets = targets.to(device, non_blocking=use_cuda)

            with autocast_context():
                outputs = model(inputs)

            outputs = outputs.clamp(0.0, 1.0)
            targets = targets.clamp(0.0, 1.0)

            batch_size = outputs.size(0)
            batch_psnr_values = []
            batch_ssim_values = []

            outputs_cpu = outputs.cpu()
            targets_cpu = targets.cpu()

            for local_idx in range(batch_size):
                dataset_idx = sample_offset + local_idx
                sample_path = Path(dataset.file_paths[dataset_idx])
                sample_name = sample_path.stem

                pred_2d_sample = outputs_cpu[local_idx].transpose(0, 1).contiguous()
                tgt_2d_sample = targets_cpu[local_idx].transpose(0, 1).contiguous()

                sample_psnr = calc_psnr(pred_2d_sample, tgt_2d_sample).item()
                sample_ssim = 1.0 - ssim_calculator(
                    pred_2d_sample.to(device),
                    tgt_2d_sample.to(device),
                ).item()

                metric_rows.append(
                    {
                        "index": dataset_idx,
                        "file_name": sample_path.name,
                        "psnr": sample_psnr,
                        "ssim": sample_ssim,
                    }
                )
                total_psnr += sample_psnr
                total_ssim += sample_ssim
                total_samples += 1
                batch_psnr_values.append(sample_psnr)
                batch_ssim_values.append(sample_ssim)

                pred_seq = outputs_cpu[local_idx, 0].numpy().astype(np.float32, copy=False)
                payload = {"seq_pred": pred_seq}
                save_prediction_file(predictions_dir / sample_name, payload)

            sample_offset += batch_size
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
        "domain": args.domain,
        "mode": args.mode,
        "num_samples": total_samples,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "prediction_format": "mat",
        "predictions_dir": str(predictions_dir.resolve()),
    }

    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"Checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"Dataset: {args.mode} @ {os.path.join(args.base_dir, args.domain)}")
    print(f"Samples: {total_samples}")
    print(f"Mean PSNR: {mean_psnr:.4f}")
    print(f"Mean SSIM: {mean_ssim:.6f}")
    print(f"Inference results: {predictions_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
