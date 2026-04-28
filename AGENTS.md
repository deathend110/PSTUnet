# PSTUnet Repository Guide

## Scope

This repository is a SAR sequence restoration project built around:

- Python training in [train.py](/g:/VSCODE-G/PSTUnet/train.py)
- Python inference in [infer.py](/g:/VSCODE-G/PSTUnet/infer.py) and [infer_linux.py](/g:/VSCODE-G/PSTUnet/infer_linux.py)
- Matlab metric computation and plotting in [inference_output/Inference_compute.m](/g:/VSCODE-G/PSTUnet/inference_output/Inference_compute.m), [inference_output/Inference_draw.m](/g:/VSCODE-G/PSTUnet/inference_output/Inference_draw.m), and [inference_output/Inference_compute_draw.m](/g:/VSCODE-G/PSTUnet/inference_output/Inference_compute_draw.m)

Work in SAR terms, not generic image denoising terms. The project operates on normalized SAR sequence patches and uses Matlab scripts for DB-domain and Linear-domain post-evaluation.

## Repository Layout

- [train.py](/g:/VSCODE-G/PSTUnet/train.py): main training entry, supports single GPU, `DataParallel`, and DDP
- [model.py](/g:/VSCODE-G/PSTUnet/model.py): PST-UNet architecture and mask-aware sequence fusion blocks
- [datasets.py](/g:/VSCODE-G/PSTUnet/datasets.py): HDF5 `.mat` dataset loader for `DB` and `Linear` domains
- [utils.py](/g:/VSCODE-G/PSTUnet/utils.py): loss functions, SSIM, PSNR, early stopping
- [infer.py](/g:/VSCODE-G/PSTUnet/infer.py): Windows-oriented dual-domain inference flow
- [infer_linux.py](/g:/VSCODE-G/PSTUnet/infer_linux.py): Linux/AutoDL inference flow
- [run_autodl_ddp.sh](/g:/VSCODE-G/PSTUnet/run_autodl_ddp.sh): canonical AutoDL DDP launch script
- [benchmark_dataloader.py](/g:/VSCODE-G/PSTUnet/benchmark_dataloader.py): dataloader throughput check
- [check_gradients.py](/g:/VSCODE-G/PSTUnet/check_gradients.py): gradient/debug helper
- [inference_output/](/g:/VSCODE-G/PSTUnet/inference_output): Matlab evaluation and visualization scripts
- [t.m](/g:/VSCODE-G/PSTUnet/t.m): local Matlab analysis scratch script, not the main evaluation entry

## Data Contract

- Dataset root is expected to contain `DB/traindata`, `DB/testdata`, `Linear/traindata`, and `Linear/testdata`.
- Python loaders read HDF5-backed `.mat` files via `h5py`.
- Training input shape is `[B, 2, T, H, W]`.
- Channel 0 is normalized SAR input sequence in `[0, 1]`.
- Channel 1 is the mode mask sequence expanded to `[T, H, W]`.
- Training target shape is `[B, 1, T, H, W]`.
- `datasets.py` prefers `/mode_mask_512_all`; it falls back to `/frame_mode_id` only when the full mask is absent.
- Do not treat this as additive white-noise restoration. The code and Matlab evaluation imply SAR-specific normalization and domain conversion assumptions.

## Training Behavior

- Default dataset path in code: `G:\VSCODE-G\PST_Dataset`
- Default domain: `DB`
- Loss used in training: `Seq_SAR_L1TVLoss`
- Validation model selection is driven by PSNR, not loss.
- Training writes into `./output/Model(...)/` and saves:
  - `best.pth`
  - `best_early_stopping.pth`
  - `train.log`
  - `loss_avg.mat`
  - `psnr_avg.mat`
  - `ssim_avg.mat`

Common commands:

```powershell
python train.py --base-dir G:\VSCODE-G\PST_Dataset --domain DB --outputs-dir .\output
```

```bash
bash run_autodl_ddp.sh
```

When changing training code:

- Preserve `[B, C, T, H, W]` conventions unless the whole pipeline is intentionally migrated.
- Check whether a change affects DDP, gradient accumulation, or output checkpoint compatibility.
- Keep `best.pth` loading compatible with inference unless the user explicitly wants a format break.

## Inference Behavior

- [infer.py](/g:/VSCODE-G/PSTUnet/infer.py) loads paired DB and Linear test samples.
- It writes predicted DB-domain `.mat` files into `output-dir/predictions/`.
- It also writes `metrics.csv` and `summary.json`.
- Matlab scripts expect those prediction `.mat` files and separate metadata under `TH/`.

Common commands:

```powershell
python infer.py --checkpoint .\output\<run>\best.pth --base-dir G:\VSCODE-G\PST_Dataset --output-dir .\inference_output\db_test_sample
```

```bash
python infer_linux.py --checkpoint ./output/<run>/best.pth --base-dir /root/autodl-tmp/PST_Dataset --output-dir ./inference_output/db_test
```

When changing inference code:

- Preserve `seq_pred_DB` as the main `.mat` payload unless Matlab consumers are updated in the same change.
- Keep output paths and filenames stable if Matlab scripts still depend on them.
- Validate whether DB-only and dual-domain inference paths must stay aligned.

## Matlab Evaluation Flow

Primary scripts:

- [inference_output/Inference_compute.m](/g:/VSCODE-G/PSTUnet/inference_output/Inference_compute.m): aggregate PSNR/SSIM by frame group
- [inference_output/Inference_draw.m](/g:/VSCODE-G/PSTUnet/inference_output/Inference_draw.m): visualize selected frames
- [inference_output/Inference_compute_draw.m](/g:/VSCODE-G/PSTUnet/inference_output/Inference_compute_draw.m): aggregate metrics plus per-frame analysis

Assumptions in Matlab scripts:

- Prediction files are DB-domain `.mat` files produced by Python inference.
- Ground truth DB and Linear sequences are loaded from the dataset root separately.
- Linear-domain evaluation depends on inverse normalization using metadata variables like `V_MAX_GT`, `V_MIN_GT`, `V_MAX_GT_L`, and `V_MIN_GT_L`.
- The frame grouping is fixed to 16-frame sequences with `180MHz`, `Mixed`, and `60MHz` subsets.

If Python output format changes, update Matlab consumers in the same task.

## Environment Expectations

Python dependencies inferred from the code:

- `torch`
- `numpy`
- `h5py`
- `scipy`
- `tqdm`

Matlab dependencies inferred from scripts:

- `psnr`
- `ssim`
- HDF5 `.mat` reading support via `h5read`

`scipy` is optional for training history export, but inference requires it for `.mat` output.

## Editing Guardrails

- Prefer modifying the existing pipeline instead of introducing parallel pipelines.
- Keep Windows paths in Windows scripts and Linux paths in Linux scripts unless deliberately unifying them.
- Do not rewrite Matlab evaluation scripts to a different directory layout without updating all path assumptions.
- Treat `output/` and `inference_output/` as experiment artifact areas first; avoid committing generated results unless explicitly requested.
- Avoid touching scratch analysis scripts like [t.m](/g:/VSCODE-G/PSTUnet/t.m) unless the task is specifically about them.

## Validation Checklist

Before claiming a change is complete, run the narrowest relevant checks:

- Architecture/data-path change:

```powershell
python check_gradients.py
```

- Dataloader change:

```powershell
python benchmark_dataloader.py --base-dir G:\VSCODE-G\PST_Dataset --domain DB --mode train
```

- Inference change:

```powershell
python infer.py --checkpoint .\output\<run>\best.pth --base-dir G:\VSCODE-G\PST_Dataset --output-dir .\inference_output\db_test_sample --num-samples 1
```

- Matlab metrics or plotting change:
  - run the relevant `.m` script after confirming `pred_dir`, dataset paths, and metadata paths

Do not report success without stating what was or was not executed.
