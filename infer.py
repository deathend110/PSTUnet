import argparse
import csv
import glob
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import scipy.io
except ImportError:
    scipy = None

from model import PST_UNet
from utils import SSIMLoss, calc_psnr


class DualDomainInferenceDataset(Dataset):
    def __init__(self, base_dir, max_val=255.0, num_samples=None, seed=42):
        self.base_dir = base_dir
        self.max_val = max_val
        self.db_dir = os.path.join(base_dir, "DB", "testdata")
        self.linear_dir = os.path.join(base_dir, "Linear", "testdata")
        all_db_paths = sorted(glob.glob(os.path.join(self.db_dir, "*.mat")))

        if num_samples is not None:
            self.db_paths = self._sample_db_paths(all_db_paths, num_samples=num_samples, seed=seed)
        else:
            self.db_paths = all_db_paths

    def __len__(self):
        return len(self.db_paths)

    @staticmethod
    def _get_prefix(db_path):
        db_name = os.path.basename(db_path)
        if "_DB_seq_" not in db_name:
            raise ValueError(f"DB file name does not match expected pattern '*_DB_seq_*.mat': {db_name}")
        return db_name.split("_DB_seq_")[0]

    def _sample_db_paths(self, all_db_paths, num_samples, seed):
        if num_samples < 1:
            raise ValueError("--num-samples must be >= 1 when provided")
        if num_samples > len(all_db_paths):
            raise ValueError(
                f"--num-samples={num_samples} exceeds available DB test samples ({len(all_db_paths)})."
            )

        grouped_paths = {}
        for db_path in all_db_paths:
            prefix = self._get_prefix(db_path)
            grouped_paths.setdefault(prefix, []).append(db_path)

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
        db_path = self.db_paths[idx]
        db_name = os.path.basename(db_path)
        linear_name = db_name.replace("_DB_seq_", "_L_seq_")
        linear_path = os.path.join(self.linear_dir, linear_name)

        if linear_name == db_name or not os.path.exists(linear_path):
            raise FileNotFoundError(f"Missing paired Linear file for {db_name}: {linear_path}")

        with h5py.File(db_path, "r") as f_db:
            seq_input_db_raw = np.array(f_db["/seq_input"]).astype(np.float32)
            seq_gt_db_raw = np.array(f_db["/seq_GT"]).astype(np.float32)
            frame_type = np.array(f_db["/frame_type"]).astype(np.float32).flatten()

        seq_input_db = seq_input_db_raw.transpose(0, 2, 1) / self.max_val
        seq_gt_db = seq_gt_db_raw.transpose(0, 2, 1) / self.max_val

        t, h, w = seq_input_db.shape
        prompt_mask = np.broadcast_to(frame_type[:, np.newaxis, np.newaxis], (t, h, w)).astype(np.float32)
        input_tensor = np.stack([seq_input_db, prompt_mask], axis=0).astype(np.float32)
        target_tensor = seq_gt_db[np.newaxis, :, :, :].astype(np.float32)

        mat_input_db = seq_input_db.transpose(1, 2, 0).astype(np.float32)
        mat_gt_db = seq_gt_db.transpose(1, 2, 0).astype(np.float32)

        with h5py.File(linear_path, "r") as f_linear:
            seq_input_linear_raw = np.array(f_linear["/seq_input_L"]).astype(np.float32)
            seq_gt_linear_raw = np.array(f_linear["/seq_GT_L"]).astype(np.float32)

        mat_input_linear = seq_input_linear_raw.transpose(2, 1, 0).astype(np.float32)
        mat_gt_linear = seq_gt_linear_raw.transpose(2, 1, 0).astype(np.float32)

        return (
            torch.from_numpy(input_tensor),
            torch.from_numpy(target_tensor),
            torch.from_numpy(mat_input_db),
            torch.from_numpy(mat_gt_db),
            torch.from_numpy(mat_input_linear),
            torch.from_numpy(mat_gt_linear),
            db_name,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="PST-UNet dual-domain inference script")
    default_checkpoint = (
        "./output/Model(PST_UNet)-Dataset(PST_Dataset)-Loss"
        "(L1+TV+DynSSIM+tv0.002000)-Epochs40-Batch_size1-lr0.000100/best.pth"
    )
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint, help="Path to a model state_dict checkpoint.")
    parser.add_argument("--base-dir", type=str, default="/root/autodl-tmp/PST_Dataset")
    parser.add_argument("--output-dir", type=str, default="./inference_output/db_test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None, help="Randomly sample N DB test files for inference.")
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

    dataset = DualDomainInferenceDataset(
        base_dir=args.base_dir,
        max_val=args.max_val,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No .mat files found for inference under {dataset.db_dir}.")

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

    metric_rows = []
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Infer", total=len(dataloader))
        for batch_idx, batch in enumerate(progress):
            (
                inputs,
                targets,
                mat_input_db_batch,
                mat_gt_db_batch,
                mat_input_linear_batch,
                mat_gt_linear_batch,
                file_names,
            ) = batch

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
                sample_index = batch_idx * args.batch_size + local_idx

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

                metric_rows.append(
                    {
                        "index": sample_index,
                        "file_name": file_name,
                        "psnr": sample_psnr,
                        "ssim": sample_ssim,
                    }
                )

                mat_pred_db = outputs_cpu[local_idx, 0].numpy().transpose(1, 2, 0).astype(np.float32, copy=False)
                mat_dict = {
                    "seq_input_DB": mat_input_db_batch[local_idx].numpy().astype(np.float32, copy=False),
                    "seq_GT_DB": mat_gt_db_batch[local_idx].numpy().astype(np.float32, copy=False),
                    "seq_pred_DB": mat_pred_db,
                    "seq_input_Linear": mat_input_linear_batch[local_idx].numpy().astype(np.float32, copy=False),
                    "seq_GT_Linear": mat_gt_linear_batch[local_idx].numpy().astype(np.float32, copy=False),
                }
                save_prediction_file(predictions_dir / file_name, mat_dict)

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
        "domain": "DB",
        "mode": "test",
        "sampled": args.num_samples is not None,
        "requested_num_samples": args.num_samples,
        "seed": args.seed,
        "num_samples": total_samples,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "predictions_dir": str(predictions_dir.resolve()),
        "linear_dir": os.path.abspath(dataset.linear_dir),
        "prediction_format": "mat",
        "sample_prefixes": sorted({dataset._get_prefix(path) for path in dataset.db_paths}),
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
