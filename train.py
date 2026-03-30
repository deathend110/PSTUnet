import argparse
import copy
import logging
import os
import time
from contextlib import nullcontext

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

try:
    import scipy.io
except ImportError:
    scipy = None

from datasets import SARDataset
from model import PST_UNet
from utils import AverageMeter, EarlyStopping, SSIMLoss, Seq_SAR_HybridLoss, calc_psnr


class DPModelWrapper(nn.Module):
    """
    Compute loss inside forward so DataParallel only gathers small scalars
    during training instead of the full 5D predictions.
    """

    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, inputs, targets=None, return_outputs=False):
        outputs = self.model(inputs)
        if targets is None:
            return outputs

        loss = self.criterion(outputs, targets)
        if return_outputs:
            return outputs, loss
        return loss


def setup_logger(log_dir):
    log_file = os.path.join(log_dir, "train.log")
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(message)s")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def parse_gpu_ids(gpu_ids_text):
    if gpu_ids_text is None:
        return None
    gpu_ids = [int(gid.strip()) for gid in gpu_ids_text.split(",") if gid.strip()]
    return gpu_ids or None


def save_history_mat(path, key, values):
    if scipy is None:
        return
    scipy.io.savemat(path, mdict={key: values})


if __name__ == "__main__":
    model_name = "PST_UNet"
    dataset_name = "PST_Dataset"
    loss_name = "L1+TV+DynSSIM"

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=r"G:\VSCODE-G\PST_Dataset")
    parser.add_argument("--domain", type=str, default="DB", choices=["DB", "Linear"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-val", type=float, default=255.0)
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help='Comma-separated GPU ids, for example "0,1". Default uses all visible GPUs.',
    )
    parser.add_argument("--outputs-dir", type=str, default="./output/")
    parser.add_argument("--tv-weight", type=float, default=1e-3)
    parser.add_argument("--ssim-weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    requested_gpu_ids = parse_gpu_ids(args.gpu_ids)

    file_name = (
        "Model({:s})-Dataset({:s})-Loss({:s}+tv{:f})-Epochs{:d}-Batch_size{:d}-lr{:f}".format(
            model_name,
            dataset_name,
            loss_name,
            args.tv_weight,
            args.num_epochs,
            args.batch_size,
            args.lr,
        )
    )

    args.outputs_dir = os.path.join(args.outputs_dir, file_name)
    os.makedirs(args.outputs_dir, exist_ok=True)

    logger = setup_logger(args.outputs_dir)
    logger.info("=" * 50)
    logger.info(f"Start training job: {file_name}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info("=" * 50)
    if scipy is None:
        logger.warning("scipy is not installed. Training will continue, but .mat history files will not be written.")

    use_cuda = torch.cuda.is_available()
    cudnn.benchmark = use_cuda
    if use_cuda:
        visible_gpu_count = torch.cuda.device_count()
        if requested_gpu_ids is None:
            device_ids = list(range(visible_gpu_count))
        else:
            invalid_gpu_ids = [gid for gid in requested_gpu_ids if gid < 0 or gid >= visible_gpu_count]
            if invalid_gpu_ids:
                raise ValueError(
                    f"Invalid --gpu-ids {invalid_gpu_ids}; visible GPU ids are 0 to {visible_gpu_count - 1}."
                )
            device_ids = requested_gpu_ids

        device = torch.device(f"cuda:{device_ids[0]}")
        gpu_desc = ", ".join(f"cuda:{gid}({torch.cuda.get_device_name(gid)})" for gid in device_ids)
        logger.info(f"Device: {device} | selected GPUs: {gpu_desc}")

        if len(device_ids) > 1 and args.batch_size < len(device_ids):
            logger.warning(
                f"batch-size={args.batch_size} is smaller than GPU count={len(device_ids)}; some GPUs will stay idle."
            )
        elif len(device_ids) > 1 and args.batch_size % len(device_ids) != 0:
            logger.warning(
                f"batch-size={args.batch_size} is not divisible by GPU count={len(device_ids)}; per-GPU load will be uneven."
            )
    else:
        device_ids = []
        device = torch.device("cpu")
        logger.info(f"Device: {device}")

    model = PST_UNet(in_channels=2, out_channels=1, base_dim=64)
    criterion = Seq_SAR_HybridLoss(tv_weight=args.tv_weight, ssim_weight=0.0)
    wrapped_model = DPModelWrapper(model, criterion)

    if len(device_ids) > 1:
        logger.info(f"Enable DataParallel on {len(device_ids)} GPUs.")
        wrapped_model = nn.DataParallel(wrapped_model, device_ids=device_ids, output_device=device_ids[0])
    wrapped_model = wrapped_model.to(device)

    amp_enabled = use_cuda
    autocast_context = (
        (lambda: torch.amp.autocast(device_type="cuda", enabled=True)) if amp_enabled else nullcontext
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ssim_calculator = SSIMLoss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    early_stopping = EarlyStopping(args.patience, verbose=False)
    model_save_path = os.path.join(args.outputs_dir, "best_early_stopping.pth")
    early_stopping.path = model_save_path
    trig = 0

    dataloader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True

    logger.info("Loading dataset...")
    train_dataset = SARDataset(base_dir=args.base_dir, domain=args.domain, mode="train", max_val=args.max_val)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **dataloader_kwargs,
    )

    eval_dataset = SARDataset(base_dir=args.base_dir, domain=args.domain, mode="test", max_val=args.max_val)
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )
    logger.info(f"Dataset ready. Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = -float("inf")

    loss_avg = []
    psnr_avg = []
    ssim_avg = []

    logger.info("=" * 50)
    logger.info("Start training loop...")
    for epoch in range(args.num_epochs):
        tc = time.time()

        actual_criterion = wrapped_model.module.criterion if isinstance(wrapped_model, nn.DataParallel) else wrapped_model.criterion

        if epoch < 40:
            actual_criterion.ssim_weight = 0.0
            phase_desc = "[L1+TV | Val: PSNR]"
        else:
            actual_criterion.ssim_weight = args.ssim_weight
            phase_desc = "[L1+TV+SSIM | Val: PSNR+SSIM]"

        wrapped_model.train()
        epoch_losses = AverageMeter()
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as progress:
            progress.set_description(f"epoch: {epoch}/{args.num_epochs - 1} {phase_desc}")

            for inputs, targets in train_dataloader:
                inputs = inputs.to(device, non_blocking=use_cuda)
                targets = targets.to(device, non_blocking=use_cuda)

                optimizer.zero_grad(set_to_none=True)

                with autocast_context():
                    loss = wrapped_model(inputs, targets)
                    if loss.ndim > 0:
                        loss = loss.mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_losses.update(loss.item(), inputs.size(0))
                progress.set_postfix(loss=f"{epoch_losses.avg:.6f}")
                progress.update(inputs.size(0))

        loss_avg.append(epoch_losses.avg)
        save_history_mat(os.path.join(args.outputs_dir, "loss_avg.mat"), "loss_avg", loss_avg)

        wrapped_model.eval()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()

        for inputs, targets in eval_dataloader:
            inputs = inputs.to(device, non_blocking=use_cuda)
            targets = targets.to(device, non_blocking=use_cuda)

            with torch.no_grad():
                with autocast_context():
                    outputs = wrapped_model(inputs)

                    batch_size, channels, frames, height, width = outputs.shape
                    out_2d = outputs.transpose(1, 2).contiguous().view(batch_size * frames, channels, height, width)
                    tgt_2d = targets.transpose(1, 2).contiguous().view(batch_size * frames, channels, height, width)

                    out_2d = out_2d.clamp(0.0, 1.0)
                    tgt_2d = tgt_2d.clamp(0.0, 1.0)

                    batch_psnr = calc_psnr(out_2d, tgt_2d).item()
                    batch_ssim = 1.0 - ssim_calculator(out_2d, tgt_2d).item()

            epoch_psnr.update(batch_psnr, inputs.size(0))
            epoch_ssim.update(batch_ssim, inputs.size(0))

        psnr_avg.append(epoch_psnr.avg)
        ssim_avg.append(epoch_ssim.avg)
        save_history_mat(os.path.join(args.outputs_dir, "psnr_avg.mat"), "psnr_avg", psnr_avg)
        save_history_mat(os.path.join(args.outputs_dir, "ssim_avg.mat"), "ssim_avg", ssim_avg)

        if epoch < 40:
            current_score = epoch_psnr.avg
        else:
            current_score = epoch_psnr.avg + (epoch_ssim.avg * 100.0)

        if current_score > best_score:
            best_epoch = epoch
            best_score = current_score
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(args.outputs_dir, "best.pth"))
            logger.info(
                "New best model. "
                f"Score: {best_score:.2f} "
                f"(PSNR: {epoch_psnr.avg:.2f}, SSIM: {epoch_ssim.avg:.4f}) -> saved to best.pth"
            )

        model_to_save = model
        early_stopping(current_score * -1, model_to_save)

        if early_stopping.early_stop:
            trig += 1
            model_to_save.load_state_dict(torch.load(model_save_path, map_location=device))
            args.lr = args.lr / 10
            logger.warning(f"Early stopping triggered (trigger={trig}). Reduce lr to {args.lr}.")

            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            if trig > 1:
                logger.error("Learning rate already decayed once and early stopping triggered again. Stop training.")
                break
            early_stopping.early_stop = False
            early_stopping.counter = 0

        te = time.time()
        logger.info(
            "Epoch [{}/{}] | eval psnr: {:.2f} | eval ssim: {:.4f} | score: {:.2f} | Time: {:.2f}s".format(
                epoch,
                args.num_epochs - 1,
                epoch_psnr.avg,
                epoch_ssim.avg,
                current_score,
                te - tc,
            )
        )

    logger.info("=" * 50)
    logger.info(f"Training finished. Best epoch: {best_epoch}, best score: {best_score:.2f}")
