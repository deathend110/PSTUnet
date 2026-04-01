import argparse
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import SARDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark dataloader throughput without model compute.")
    parser.add_argument("--base-dir", type=str, default=r"G:\VSCODE-G\PST_Dataset")
    parser.add_argument("--domain", type=str, default="DB", choices=["DB", "Linear"])
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-val", type=float, default=255.0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep worker processes alive after the first epoch.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="How many full dataloader passes to benchmark.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="If > 0, stop each epoch early after this many batches.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=0,
        help="Iterate this many batches first and exclude them from timing.",
    )
    return parser.parse_args()


def build_dataloader(args):
    dataset = SARDataset(
        base_dir=args.base_dir,
        domain=args.domain,
        mode=args.mode,
        max_val=args.max_val,
    )

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": False,
    }
    if args.num_workers > 0 and args.persistent_workers:
        dataloader_kwargs["persistent_workers"] = True

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataset, dataloader


def consume_batches(dataloader, limit_batches, show_progress, desc):
    first_batch_time = None
    sample_shape = None
    target_shape = None
    total_batches = 0
    total_samples = 0

    iterator = tqdm(dataloader, desc=desc, total=min(len(dataloader), limit_batches) if limit_batches > 0 else len(dataloader)) if show_progress else dataloader
    start_time = time.perf_counter()

    for batch_idx, (inputs, targets) in enumerate(iterator):
        if first_batch_time is None:
            first_batch_time = time.perf_counter() - start_time
            sample_shape = tuple(inputs.shape)
            target_shape = tuple(targets.shape)

        total_batches += 1
        total_samples += inputs.size(0)

        # Touch tensor metadata so timing reflects actual batch materialization.
        _ = inputs.shape, targets.shape

        if limit_batches > 0 and total_batches >= limit_batches:
            break

    elapsed = time.perf_counter() - start_time
    return {
        "elapsed": elapsed,
        "batches": total_batches,
        "samples": total_samples,
        "first_batch_time": first_batch_time,
        "sample_shape": sample_shape,
        "target_shape": target_shape,
    }


def main():
    args = parse_args()

    init_start = time.perf_counter()
    dataset, dataloader = build_dataloader(args)
    init_elapsed = time.perf_counter() - init_start

    print("=" * 60)
    print("Dataloader benchmark")
    print("=" * 60)
    print(f"base_dir            : {args.base_dir}")
    print(f"domain / mode       : {args.domain} / {args.mode}")
    print(f"dataset size        : {len(dataset)} samples")
    print(f"batch_size          : {args.batch_size}")
    print(f"num_workers         : {args.num_workers}")
    print(f"pin_memory          : {args.pin_memory}")
    print(f"persistent_workers  : {args.persistent_workers}")
    print(f"epochs              : {args.epochs}")
    print(f"limit_batches       : {args.limit_batches if args.limit_batches > 0 else 'full epoch'}")
    print(f"warmup_batches      : {args.warmup_batches}")
    print(f"init time           : {init_elapsed:.2f}s")
    print("=" * 60)

    if args.warmup_batches > 0:
        print(f"Running warmup for {args.warmup_batches} batches...")
        warmup_stats = consume_batches(
            dataloader=dataloader,
            limit_batches=args.warmup_batches,
            show_progress=False,
            desc="warmup",
        )
        print(
            f"Warmup done: {warmup_stats['batches']} batches, "
            f"{warmup_stats['samples']} samples, {warmup_stats['elapsed']:.2f}s"
        )

    epoch_times = []
    total_samples = 0
    total_batches = 0
    first_batch_time = None
    sample_shape = None
    target_shape = None

    for epoch in range(args.epochs):
        stats = consume_batches(
            dataloader=dataloader,
            limit_batches=args.limit_batches,
            show_progress=True,
            desc=f"epoch {epoch + 1}/{args.epochs}",
        )
        epoch_times.append(stats["elapsed"])
        total_samples += stats["samples"]
        total_batches += stats["batches"]

        if first_batch_time is None:
            first_batch_time = stats["first_batch_time"]
            sample_shape = stats["sample_shape"]
            target_shape = stats["target_shape"]

        samples_per_sec = stats["samples"] / stats["elapsed"] if stats["elapsed"] > 0 else 0.0
        print(
            f"[epoch {epoch + 1}] elapsed={stats['elapsed']:.2f}s | "
            f"batches={stats['batches']} | samples={stats['samples']} | "
            f"throughput={samples_per_sec:.2f} samples/s"
        )

    total_elapsed = sum(epoch_times)
    avg_epoch_time = total_elapsed / max(len(epoch_times), 1)
    avg_samples_per_sec = total_samples / total_elapsed if total_elapsed > 0 else 0.0
    avg_batch_time = total_elapsed / total_batches if total_batches > 0 else 0.0

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"sample tensor shape : {sample_shape}")
    print(f"target tensor shape : {target_shape}")
    print(f"first batch time    : {first_batch_time:.2f}s" if first_batch_time is not None else "first batch time    : N/A")
    print(f"avg epoch time      : {avg_epoch_time:.2f}s")
    print(f"avg batch time      : {avg_batch_time:.2f}s")
    print(f"avg throughput      : {avg_samples_per_sec:.2f} samples/s")
    print(f"total samples read  : {total_samples}")
    print("=" * 60)


if __name__ == "__main__":
    main()
