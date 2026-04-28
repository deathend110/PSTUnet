'''
V2：Mask-aware PST-UNet
- 训练损失：L1 + TV
- 第二通道：真实空间 mode mask
- 模型内部 PST 门控已对齐空间 mask
'''

import argparse
import copy
import logging
import os
import time
from contextlib import nullcontext

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    import scipy.io
except ImportError:
    scipy = None

from datasets import SARDataset
from model import PST_UNet
from utils import AverageMeter, EarlyStopping, SSIMLoss, Seq_SAR_L1TVLoss, calc_psnr


class TrainModelWrapper(nn.Module):
    """
    Compute loss inside forward so multi-GPU training only needs to gather
    small scalars during the training step.
    """
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, inputs, targets=None):
        outputs = self.model(inputs)
        if targets is None:
            return outputs
        return self.criterion(outputs, targets)


def setup_logger(log_dir, is_main_process_flag):
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if is_main_process_flag:
        log_file = os.path.join(log_dir, "train.log")
        formatter = logging.Formatter("%(asctime)s - %(message)s")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    else:
        logger.addHandler(logging.NullHandler())

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


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_epoch_schedule(epoch_index, base_lr):
    epoch_id = epoch_index + 1

    if epoch_id <= 15:
        return {
            "stage_name": "WarmupAndBaseFit",
            "phase_desc": "[L1+TV | LR=1e-4 | Epoch 1-15]",
            "lr": base_lr,
        }
    if epoch_id <= 25:
        return {
            "stage_name": "StructureStabilize",
            "phase_desc": "[L1+TV | LR=5e-5 | Epoch 16-25]",
            "lr": base_lr * 0.5,
        }
    if epoch_id <= 35:
        return {
            "stage_name": "Finetune",
            "phase_desc": "[L1+TV | LR=1e-5 | Epoch 26-35]",
            "lr": base_lr * 0.1,
        }
    return {
        "stage_name": "ExtremeConverge",
        "phase_desc": "[L1+TV | LR=1e-6 | Epoch 36+]",
        "lr": base_lr * 0.01,
    }


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_dist_initialized()) or dist.get_rank() == 0


def get_world_size():
    return dist.get_world_size() if is_dist_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_initialized() else 0


def reduce_average(sum_value, count_value, device):
    if not is_dist_initialized():
        return sum_value / max(count_value, 1)

    stats = torch.tensor([sum_value, count_value], dtype=torch.float64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[1].clamp_min(1.0)).item()


def broadcast_flags(flags, device):
    tensor = torch.tensor(flags, dtype=torch.int64, device=device)
    if is_dist_initialized():
        dist.broadcast(tensor, src=0)
    return tensor.tolist()


def init_runtime(args):
    requested_gpu_ids = parse_gpu_ids(args.gpu_ids)
    use_cuda = torch.cuda.is_available()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = use_cuda and world_size > 1

    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")

        visible_gpu_count = torch.cuda.device_count()
        if requested_gpu_ids is None:
            if local_rank >= visible_gpu_count:
                raise ValueError(
                    f"LOCAL_RANK={local_rank} is out of range for visible GPU count={visible_gpu_count}."
                )
            device_id = local_rank
        else:
            if len(requested_gpu_ids) != world_size:
                raise ValueError(
                    f"In DDP mode, --gpu-ids count ({len(requested_gpu_ids)}) must match WORLD_SIZE ({world_size})."
                )
            invalid_gpu_ids = [gid for gid in requested_gpu_ids if gid < 0 or gid >= visible_gpu_count]
            if invalid_gpu_ids:
                raise ValueError(
                    f"Invalid --gpu-ids {invalid_gpu_ids}; visible GPU ids are 0 to {visible_gpu_count - 1}."
                )
            device_id = requested_gpu_ids[local_rank]

        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        device_ids = [device_id]
    elif use_cuda:
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
        rank = 0
        local_rank = 0
    else:
        device_ids = []
        device = torch.device("cpu")
        rank = 0
        local_rank = 0

    cudnn.benchmark = use_cuda

    return {
        "use_cuda": use_cuda,
        "distributed": distributed,
        "device": device,
        "device_ids": device_ids,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": get_world_size(),
    }


def cleanup_distributed():
    if is_dist_initialized():
        dist.destroy_process_group()


def main():
    model_name = "PST_UNet_MaskAware"
    dataset_name = "Sequence_Dataset_AzimuthMix"
    loss_name = "L1+TV"

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=r"G:\VSCODE-G\PST_Dataset")
    parser.add_argument("--domain", type=str, default="DB", choices=["DB", "Linear"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-val", type=float, default=255.0)
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help='Single-process mode: "0,1" for DataParallel. DDP mode: should match WORLD_SIZE order.',
    )
    parser.add_argument("--outputs-dir", type=str, default="./output/")
    parser.add_argument("--tv-weight", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-process batch size. Under DDP, global batch size = batch-size * WORLD_SIZE.",
    )
    parser.add_argument("--num-epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients before updating. Global Batch Size = batch-size * WORLD_SIZE * grad-accum-steps.",
    )
    args = parser.parse_args()

    runtime = init_runtime(args)
    use_cuda = runtime["use_cuda"]
    distributed = runtime["distributed"]
    device = runtime["device"]
    device_ids = runtime["device_ids"]
    world_size = runtime["world_size"]

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

    logger = setup_logger(args.outputs_dir, is_main_process())
    logger.info("=" * 50)
    logger.info(f"Start training job: {file_name}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info("=" * 50)

    if scipy is None:
        logger.warning("scipy is not installed. Training will continue, but .mat history files will not be written.")

    if use_cuda:
        gpu_desc = ", ".join(
            f"cuda:{gid}({torch.cuda.get_device_name(gid)})" for gid in device_ids
        ) if device_ids else "none"

        if distributed:
            logger.info(
                f"Runtime: DDP | rank={get_rank()} | world_size={world_size} | local device={device} | visible selection={gpu_desc}"
            )
            logger.info(f"Global batch size: {args.batch_size * world_size} ({args.batch_size} per process)")
        else:
            logger.info(
                f"Runtime: {'DataParallel' if len(device_ids) > 1 else 'Single GPU'} | device={device} | selected GPUs: {gpu_desc}"
            )
    else:
        logger.info(f"Runtime: CPU | device={device}")

    model = PST_UNet(in_channels=2, out_channels=1, base_dim=64)
    criterion = Seq_SAR_L1TVLoss(tv_weight=args.tv_weight)
    wrapped_model = TrainModelWrapper(model, criterion).to(device)

    if distributed:
        wrapped_model = DistributedDataParallel(
            wrapped_model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
        )
    elif len(device_ids) > 1:
        wrapped_model = DataParallel(
            wrapped_model,
            device_ids=device_ids,
            output_device=device_ids[0]
        )

    amp_enabled = use_cuda
    autocast_context = (
        (lambda: torch.amp.autocast(device_type="cuda", enabled=True))
        if amp_enabled else nullcontext
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ssim_calculator = SSIMLoss().to(device)

    base_lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    early_stopping = EarlyStopping(args.patience, verbose=False)
    model_save_path = os.path.join(args.outputs_dir, "best_early_stopping.pth")
    early_stopping.path = model_save_path

    dataloader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True

    logger.info("Loading dataset...")
    train_dataset = SARDataset(base_dir=args.base_dir, domain=args.domain, mode="train", max_val=args.max_val)
    eval_dataset = SARDataset(base_dir=args.base_dir, domain=args.domain, mode="test", max_val=args.max_val)

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) if distributed else None
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False) if distributed else None

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **dataloader_kwargs,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=eval_sampler,
        drop_last=False,
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

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        schedule = get_epoch_schedule(epoch, base_lr)
        set_optimizer_lr(optimizer, schedule["lr"])
        phase_desc = schedule["phase_desc"]

        wrapped_model.train()
        epoch_losses = AverageMeter()

        progress = None
        if is_main_process():
            progress = tqdm(total=len(train_dataloader) * args.batch_size)
            progress.set_description(f"epoch: {epoch}/{args.num_epochs - 1} {phase_desc}")

        optimizer.zero_grad(set_to_none=True)

        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device, non_blocking=use_cuda)
            targets = targets.to(device, non_blocking=use_cuda)

            is_accumulating = (i + 1) % args.grad_accum_steps != 0 and (i + 1) != len(train_dataloader)
            context = wrapped_model.no_sync() if (is_accumulating and distributed) else nullcontext()

            with context:
                with autocast_context():
                    loss = wrapped_model(inputs, targets)
                    if loss.ndim > 0:
                        loss = loss.mean()
                    scaled_loss = loss / args.grad_accum_steps

                scaler.scale(scaled_loss).backward()

            if not is_accumulating:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_losses.update(loss.item(), inputs.size(0))
            if progress is not None:
                progress.set_postfix(loss=f"{epoch_losses.avg:.6f}")
                progress.update(inputs.size(0))

        if progress is not None:
            progress.close()

        train_loss_avg = reduce_average(epoch_losses.sum, epoch_losses.count, device)

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

        epoch_psnr_avg = reduce_average(epoch_psnr.sum, epoch_psnr.count, device)
        epoch_ssim_avg = reduce_average(epoch_ssim.sum, epoch_ssim.count, device)

        current_score = epoch_psnr_avg

        should_stop = [0]
        if is_main_process():
            loss_avg.append(train_loss_avg)
            psnr_avg.append(epoch_psnr_avg)
            ssim_avg.append(epoch_ssim_avg)

            save_history_mat(os.path.join(args.outputs_dir, "loss_avg.mat"), "loss_avg", loss_avg)
            save_history_mat(os.path.join(args.outputs_dir, "psnr_avg.mat"), "psnr_avg", psnr_avg)
            save_history_mat(os.path.join(args.outputs_dir, "ssim_avg.mat"), "ssim_avg", ssim_avg)

            if current_score > best_score:
                best_epoch = epoch
                best_score = current_score
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, os.path.join(args.outputs_dir, "best.pth"))
                logger.info(
                    "New best model. "
                    f"PSNR score: {best_score:.4f} "
                    f"(PSNR: {epoch_psnr_avg:.4f}, SSIM: {epoch_ssim_avg:.4f}) -> saved to best.pth"
                )

            early_stopping(-current_score, model)
            if early_stopping.early_stop:
                logger.warning("Early stopping triggered. Stop training with the current staged LR schedule.")
                should_stop[0] = 1

            te = time.time()
            logger.info(
                "Epoch [{}/{}] | stage: {} | lr: {:.2e} | train loss: {:.6f} | eval psnr: {:.4f} | eval ssim: {:.4f} | psnr_score: {:.4f} | Time: {:.2f}s".format(
                    epoch,
                    args.num_epochs - 1,
                    schedule["stage_name"],
                    schedule["lr"],
                    train_loss_avg,
                    epoch_psnr_avg,
                    epoch_ssim_avg,
                    current_score,
                    te - tc,
                )
            )

        should_stop = broadcast_flags(should_stop, device if use_cuda else torch.device("cpu"))
        if should_stop[0] == 1:
            break

    if is_main_process():
        logger.info("=" * 50)
        logger.info(f"Training finished. Best epoch: {best_epoch}, best PSNR score: {best_score:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()