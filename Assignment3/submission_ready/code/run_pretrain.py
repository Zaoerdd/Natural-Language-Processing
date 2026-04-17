# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Script for pretraining a small GPT-2 124M parameter model
on Chinese Wikipedia text data.

Before running this script, make sure you:
1. Extracted and preprocessed the text data
2. Trained a BPE tokenizer on the text data
"""

import argparse
from collections import deque
import json
import os
import time
import math
from pathlib import Path

from tokenizers import Tokenizer
import torch
from utils import (
    GPTModel,
    calc_loss_batch,
    create_window_dataloader,
    evaluate_model,
    estimate_windows_from_token_counts,
    list_text_files,
    list_token_files,
    load_token_manifest,
    plot_losses,
)


def create_dataloaders(
    data_path,
    tokenizer,
    train_ratio,
    batch_size,
    max_length,
    stride,
    num_workers=0,
    train_shuffle_seed=None,
):
    """Create streaming training and validation dataloaders from text or token shards."""
    data_path = Path(data_path)
    if data_path.is_file() and data_path.suffix.lower() == ".npy":
        source_type = "tokens"
        file_paths = list_token_files(data_path)
        manifest = load_token_manifest(data_path.with_name("manifest.json"))
    elif data_path.is_dir() and list(data_path.rglob("*.npy")):
        source_type = "tokens"
        file_paths = list_token_files(data_path)
        manifest = load_token_manifest(data_path / "manifest.json")
    else:
        source_type = "text"
        file_paths = list_text_files(data_path)
        manifest = None

    if len(file_paths) == 1:
        split_idx = 1
    else:
        split_idx = int(train_ratio * len(file_paths))
        split_idx = min(max(split_idx, 1), len(file_paths) - 1)

    train_paths = file_paths[:split_idx]
    val_paths = file_paths[split_idx:] if split_idx < len(file_paths) else file_paths[-1:]

    pin_memory = torch.cuda.is_available()
    train_loader = create_window_dataloader(
        train_paths,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=True,
        num_workers=num_workers,
        source_type=source_type,
        pin_memory=pin_memory,
        shuffle_seed=train_shuffle_seed,
    )
    val_loader = create_window_dataloader(
        val_paths,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
        num_workers=num_workers,
        source_type=source_type,
        pin_memory=pin_memory,
    )

    stats = {
        "source_type": source_type,
        "train_paths": [str(p) for p in train_paths],
        "val_paths": [str(p) for p in val_paths],
        "train_file_count": len(train_paths),
        "val_file_count": len(val_paths),
        "train_batch_count": None,
        "val_batch_count": None,
        "train_usable_tokens": None,
        "val_usable_tokens": None,
    }

    if manifest and manifest.get("files"):
        token_count_map = {
            item["relative_path"]: item["token_count"]
            for item in manifest["files"]
        }
        train_counts = []
        val_counts = []
        base_dir = data_path if data_path.is_dir() else data_path.parent
        for file_path in train_paths:
            rel = str(file_path.relative_to(base_dir)).replace("\\", "/")
            if rel in token_count_map:
                train_counts.append(token_count_map[rel])
        for file_path in val_paths:
            rel = str(file_path.relative_to(base_dir)).replace("\\", "/")
            if rel in token_count_map:
                val_counts.append(token_count_map[rel])

        if train_counts:
            train_est = estimate_windows_from_token_counts(train_counts, max_length, stride, batch_size)
            val_est = estimate_windows_from_token_counts(val_counts, max_length, stride, batch_size)
            stats["train_batch_count"] = train_est["batch_count"]
            stats["val_batch_count"] = val_est["batch_count"]
            stats["train_usable_tokens"] = train_est["usable_tokens"]
            stats["val_usable_tokens"] = val_est["usable_tokens"]

    return train_loader, val_loader, stats


def convert_time(seconds):
    """Convert seconds to hours, minutes, seconds."""
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def train_model_simple(
    model,
    optimizer,
    device,
    n_epochs,
    eval_freq,
    eval_iter,
    output_dir,
    save_ckpt_freq,
    tokenizer,
    data_path,
    batch_size=1024,
    train_ratio=0.90,
    stride=None,
    num_workers=0,
    max_tokens=None,
    warmup_steps=0,
    min_lr_ratio=0.1,
    grad_clip=1.0,
    log_freq=20,
    resume_state=None,
    seed=123,
):
    """
    Simple training loop for GPT model.
    
    Args:
        model: The GPT model to train
        optimizer: The optimizer
        device: Device to train on
        n_epochs: Number of epochs to train
        eval_freq: Evaluate every N steps
        eval_iter: Number of iterations for evaluation
        output_dir: Directory to save checkpoints
        save_ckpt_freq: Save checkpoint every N steps
        tokenizer: Tokenizer for encoding text
        data_path: Path to the training data file or directory
        batch_size: Batch size for training
        train_ratio: Ratio of data to use for training (rest for validation)
        
    Returns:
        Tuple of (train_losses, val_losses, track_tokens_seen, history)
    """
    ### START YOUR CODE ###
    # Initialize tracking variables
    resume_state = resume_state or {}
    train_losses = list(resume_state.get("train_losses", []))
    val_losses = list(resume_state.get("val_losses", []))
    track_tokens_seen = list(resume_state.get("track_tokens_seen", []))
    tokens_seen = int(resume_state.get("tokens_seen", 0))
    global_step = int(resume_state.get("global_step", 0))
    start_epoch = int(resume_state.get("epoch", 0))
    resume_epoch_step = int(resume_state.get("epoch_step", 0))
    elapsed_before = float(resume_state.get("elapsed_sec", 0.0))
    start_time = time.time() - elapsed_before
    eval_steps = list(resume_state.get("eval_steps", []))
    recent_losses = deque(maxlen=max(log_freq, 1))
    stride = stride or model.pos_emb.weight.shape[0]
    context_length = model.pos_emb.weight.shape[0]
    vocab_size = model.out_head.out_features
    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    train_loader, val_loader, data_stats = create_dataloaders(
        data_path=data_path,
        tokenizer=tokenizer,
        train_ratio=train_ratio,
        batch_size=batch_size,
        max_length=context_length,
        stride=stride,
        num_workers=num_workers,
        train_shuffle_seed=seed + start_epoch,
    )

    total_train_batches = data_stats["train_batch_count"]
    if max_tokens is not None:
        steps_from_tokens = math.ceil(max_tokens / (batch_size * context_length))
        planned_steps = (
            min(total_train_batches, steps_from_tokens)
            if total_train_batches is not None
            else steps_from_tokens
        )
    else:
        planned_steps = total_train_batches * n_epochs if total_train_batches is not None else None

    min_lr = optimizer.param_groups[0]["lr"] * min_lr_ratio
    base_lr = optimizer.param_groups[0]["lr"]

    def update_learning_rate(step_index):
        if planned_steps is None or planned_steps <= 0:
            return optimizer.param_groups[0]["lr"]
        if warmup_steps > 0 and step_index < warmup_steps:
            lr = base_lr * float(step_index + 1) / float(warmup_steps)
        else:
            progress_den = max(planned_steps - warmup_steps, 1)
            progress = min(max((step_index - warmup_steps) / progress_den, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = min_lr + (base_lr - min_lr) * cosine
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    print("Training setup")
    print(f"  Data path: {data_path}")
    print(f"  Source type: {data_stats['source_type']}")
    print(f"  Train files: {data_stats['train_file_count']}")
    print(f"  Val files: {data_stats['val_file_count']}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Context length: {context_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Stride: {stride}")
    print(f"  Mixed precision: {use_amp} ({amp_dtype if use_amp else 'fp32'})")
    if data_stats["train_batch_count"] is not None:
        print(f"  Estimated train batches/epoch: {data_stats['train_batch_count']:,}")
    if data_stats["val_batch_count"] is not None:
        print(f"  Estimated val batches: {data_stats['val_batch_count']:,}")
    if data_stats["train_usable_tokens"] is not None:
        print(f"  Estimated train tokens/epoch: {data_stats['train_usable_tokens']:,}")
    if max_tokens is not None:
        print(f"  Max tokens target: {max_tokens:,}")
    if planned_steps is not None:
        print(f"  Planned training steps: {planned_steps:,}")

    log_path = output_dir / "train.log"
    metrics_path = output_dir / "metrics.jsonl"
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    metrics_file = open(metrics_path, "a", encoding="utf-8", buffering=1)

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")

    def write_metric(record):
        metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    log("Training setup")
    log(f"  Data path: {data_path}")
    log(f"  Source type: {data_stats['source_type']}")
    log(f"  Train files: {data_stats['train_file_count']}")
    log(f"  Val files: {data_stats['val_file_count']}")
    log(f"  Vocab size: {vocab_size}")
    log(f"  Context length: {context_length}")
    log(f"  Batch size: {batch_size}")
    log(f"  Stride: {stride}")
    log(f"  Mixed precision: {use_amp} ({amp_dtype if use_amp else 'fp32'})")
    if data_stats["train_batch_count"] is not None:
        log(f"  Estimated train batches/epoch: {data_stats['train_batch_count']:,}")
    if data_stats["val_batch_count"] is not None:
        log(f"  Estimated val batches: {data_stats['val_batch_count']:,}")
    if data_stats["train_usable_tokens"] is not None:
        log(f"  Estimated train tokens/epoch: {data_stats['train_usable_tokens']:,}")
    if max_tokens is not None:
        log(f"  Max tokens target: {max_tokens:,}")
    if planned_steps is not None:
        log(f"  Planned training steps: {planned_steps:,}")
    if resume_state:
        log(f"  Resuming from step {global_step:,}, tokens {tokens_seen:,}, epoch {start_epoch + 1}")
        if resume_epoch_step:
            log(f"  Will skip {resume_epoch_step:,} batch(es) in the resumed epoch")
    log(f"  Log file: {log_path}")
    log(f"  Metrics file: {metrics_path}")

    def save_checkpoint(checkpoint_name):
        checkpoint_path = output_dir / checkpoint_name
        elapsed = time.time() - start_time
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "track_tokens_seen": track_tokens_seen,
            "eval_steps": eval_steps,
            "epoch": current_epoch,
            "epoch_step": current_epoch_step,
            "elapsed_sec": elapsed,
            "seed": seed,
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    current_epoch = start_epoch
    current_epoch_step = resume_epoch_step

    try:
        for epoch in range(start_epoch, n_epochs):
            model.train()
            current_epoch = epoch
            current_epoch_step = 0
            train_loader, _, _ = create_dataloaders(
                data_path=data_path,
                tokenizer=tokenizer,
                train_ratio=train_ratio,
                batch_size=batch_size,
                max_length=context_length,
                stride=stride,
                num_workers=num_workers,
                train_shuffle_seed=seed + epoch,
            )
            log(f"\nEpoch {epoch + 1}/{n_epochs}")
            if epoch == start_epoch and resume_epoch_step:
                log(f"Skipping {resume_epoch_step:,} already-processed batch(es) for resumed epoch {epoch + 1}")
            for input_batch, target_batch in train_loader:
                if epoch == start_epoch and current_epoch_step < resume_epoch_step:
                    current_epoch_step += 1
                    continue
                if max_tokens is not None and tokens_seen >= max_tokens:
                    log(f"Reached max token budget: {tokens_seen:,}")
                    raise StopIteration

                current_lr = update_learning_rate(global_step)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    loss = calc_loss_batch(input_batch, target_batch, model, device)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                loss_value = float(loss.item())
                recent_losses.append(loss_value)
                tokens_seen += input_batch.numel()
                global_step += 1
                current_epoch_step += 1

                elapsed = time.time() - start_time
                tokens_per_sec = tokens_seen / elapsed if elapsed > 0 else 0.0
                steps_per_sec = global_step / elapsed if elapsed > 0 else 0.0
                gpu_mem_gb = (
                    torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available()
                    else 0.0
                )
                eta_seconds = None
                if planned_steps is not None and steps_per_sec > 0:
                    eta_seconds = max(planned_steps - global_step, 0) / steps_per_sec

                if log_freq and global_step % log_freq == 0:
                    eta_str = "n/a"
                    if eta_seconds is not None:
                        eh, em, es = convert_time(eta_seconds)
                        eta_str = f"{eh:02d}:{em:02d}:{es:02d}"
                    h, m, s = convert_time(elapsed)
                    message = (
                        f"train step={global_step:,} tokens={tokens_seen:,} "
                        f"loss={loss_value:.4f} avg_loss={sum(recent_losses)/len(recent_losses):.4f} "
                        f"lr={current_lr:.6g} tok/s={tokens_per_sec:,.0f} "
                        f"step/s={steps_per_sec:.2f} gpu_mem={gpu_mem_gb:.2f}GB "
                        f"elapsed={h:02d}:{m:02d}:{s:02d} eta={eta_str}"
                    )
                    log(message)
                    write_metric(
                        {
                            "event": "train",
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "tokens_seen": tokens_seen,
                            "loss": loss_value,
                            "avg_loss": sum(recent_losses) / len(recent_losses),
                            "lr": current_lr,
                            "tokens_per_sec": tokens_per_sec,
                            "steps_per_sec": steps_per_sec,
                            "gpu_mem_gb": gpu_mem_gb,
                            "elapsed_sec": elapsed,
                            "eta_sec": eta_seconds,
                        }
                    )

                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        eval_iter=eval_iter,
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    eval_steps.append(global_step)
                    h, m, s = convert_time(elapsed)
                    log(
                        f"step={global_step:,} tokens={tokens_seen:,} lr={current_lr:.6g} "
                        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                        f"tok/s={tokens_per_sec:,.0f} gpu_mem={gpu_mem_gb:.2f}GB "
                        f"elapsed={h:02d}:{m:02d}:{s:02d}"
                    )
                    write_metric(
                        {
                            "event": "eval",
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "tokens_seen": tokens_seen,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "lr": current_lr,
                            "tokens_per_sec": tokens_per_sec,
                            "steps_per_sec": steps_per_sec,
                            "gpu_mem_gb": gpu_mem_gb,
                            "elapsed_sec": elapsed,
                        }
                    )

                if save_ckpt_freq and global_step % save_ckpt_freq == 0:
                    ckpt_path = save_checkpoint(f"checkpoint_step_{global_step:07d}.pt")
                    log(f"Saved checkpoint to {ckpt_path}")

    except KeyboardInterrupt:
        ckpt_path = save_checkpoint("checkpoint_interrupted.pt")
        log(f"\nTraining interrupted. Saved checkpoint to {ckpt_path}")
    except StopIteration:
        pass

    final_ckpt_path = save_checkpoint("checkpoint_final.pt")
    history = {
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "track_tokens_seen": track_tokens_seen,
        "eval_steps": eval_steps,
        "train_batch_count": total_train_batches,
        "val_batch_count": data_stats["val_batch_count"],
        "train_usable_tokens": data_stats["train_usable_tokens"],
        "val_usable_tokens": data_stats["val_usable_tokens"],
        "source_type": data_stats["source_type"],
        "final_checkpoint": str(final_ckpt_path),
        "epoch": current_epoch,
        "epoch_step": current_epoch_step,
        "elapsed_sec": time.time() - start_time,
        "seed": seed,
    }
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    log(f"Saved final checkpoint to {final_ckpt_path}")
    log_file.close()
    metrics_file.close()

    ### END YOUR CODE ###

    return train_losses, val_losses, track_tokens_seen, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="GPT Model Training Configuration",
    )

    parser.add_argument(
        "--data_file", "--data",
        type=str,
        required=True,
        help="Path to the training data file or directory containing .txt files (e.g., data/wiki_zh_2019.txt or data/wiki_zh_2019/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_checkpoints",
        help="Directory where the model checkpoints will be saved",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=1, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to the tokenizer JSON file (e.g., tokenizer/wikizh_tokenizer_whitespace.json)",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=100,
        help="Frequency of evaluations during training (in steps)",
    )
    parser.add_argument(
        "--save_ckpt_freq",
        type=int,
        default=100_000,
        help="Frequency of saving model checkpoints during training (in steps)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.90, help="Ratio of data for training (rest for validation)"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=52000, help="Vocabulary size (should match tokenizer)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Uses a very small model for debugging purposes",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding window stride. Defaults to context length for non-overlapping chunks.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Optional token budget. Training stops once this many input tokens are processed.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="Linear warmup steps before cosine decay.",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1,
        help="Final learning rate as a ratio of the initial learning rate for cosine decay.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Set <=0 to disable.",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=20,
        help="Print and log training progress every N optimizer steps.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional checkpoint path for resuming training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used for initialization and deterministic shard shuffling.",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Set model configuration
    if args.debug:
        GPT_CONFIG_124M = {
            "vocab_size": args.vocab_size,
            "context_length": 10,
            "emb_dim": 12,
            "n_heads": 2,
            "n_layers": 2,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
    else:
        GPT_CONFIG_124M = {
            "vocab_size": args.vocab_size,  # Should match tokenizer vocab size
            "context_length": 1024,  # Context length
            "emb_dim": 768,  # Embedding dimension
            "n_heads": 12,  # Number of attention heads
            "n_layers": 12,  # Number of layers
            "drop_rate": 0.1,  # Dropout rate
            "qkv_bias": False,  # Query-key-value bias
        }

    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    # Verify vocab size matches
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != args.vocab_size:
        print(f"Warning: Tokenizer vocab size ({actual_vocab_size}) doesn't match --vocab_size ({args.vocab_size})")
        print(f"Updating model config to use vocab size: {actual_vocab_size}")
        GPT_CONFIG_124M["vocab_size"] = actual_vocab_size

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    torch.manual_seed(args.seed)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    # Setup optimizer
    fused_ok = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        fused=fused_ok,
    )

    resume_state = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        print(f"Loading checkpoint from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resume_state = {
            "global_step": checkpoint.get("global_step", 0),
            "tokens_seen": checkpoint.get("tokens_seen", 0),
            "train_losses": checkpoint.get("train_losses", []),
            "val_losses": checkpoint.get("val_losses", []),
            "track_tokens_seen": checkpoint.get("track_tokens_seen", []),
            "eval_steps": checkpoint.get("eval_steps", []),
            "epoch": checkpoint.get("epoch", 0),
            "epoch_step": checkpoint.get("epoch_step", 0),
            "elapsed_sec": checkpoint.get("elapsed_sec", 0.0),
        }
        if "seed" in checkpoint and checkpoint["seed"] != args.seed:
            print(
                f"Warning: checkpoint seed ({checkpoint['seed']}) differs from --seed ({args.seed}). "
                "Use the checkpoint seed for the most faithful resume."
            )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {output_dir.absolute()}")

    # Train model
    print("\nStarting training...")
    train_losses, val_losses, tokens_seen, history = train_model_simple(
        model=model,
        optimizer=optimizer,
        device=device,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        eval_iter=1,
        output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        tokenizer=tokenizer,
        data_path=args.data_file,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        stride=args.stride,
        num_workers=args.num_workers,
        max_tokens=args.max_tokens,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip,
        log_freq=args.log_freq,
        resume_state=resume_state,
        seed=args.seed,
    )

    ### START YOUR CODE ###
    # TODO: Plot losses if train_losses is not empty
    # Hint: Use torch.linspace to create epochs_tensor and call plot_losses()

    if train_losses:
        if history["train_batch_count"]:
            epochs_tensor = torch.tensor(
                [step / history["train_batch_count"] for step in history["eval_steps"]],
                dtype=torch.float32,
            )
        else:
            epochs_tensor = torch.arange(1, len(train_losses) + 1, dtype=torch.float32)
        plot_losses(
            epochs_seen=epochs_tensor,
            tokens_seen=tokens_seen,
            train_losses=train_losses,
            val_losses=val_losses,
            output_path=output_dir / "loss.pdf",
        )

    # TODO: Save the final model to output_dir / "model_final.pth"
    torch.save(model.state_dict(), output_dir / "model_final.pth")

    ### END YOUR CODE ###
    
    # Print GPU memory usage if CUDA is available
    if torch.cuda.is_available():
        print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("Training completed!")
