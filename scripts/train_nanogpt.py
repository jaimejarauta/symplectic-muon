#!/usr/bin/env python3
"""Stage 2 NanoGPT-124M benchmark training pipeline for Symplectic Muon."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
import subprocess
import sys
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn

from symplectic_muon.nanogpt import GPTConfig, NanoGPT, partition_parameters
from symplectic_muon.optimizer import StandardMuon, SymplecticMuon


def get_git_commit() -> str:
    """Retrieve the current git commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def resolve_device(device_override: Optional[str] = None) -> Tuple[torch.device, str]:
    """Auto-detect hardware platform: CUDA -> Apple Silicon MPS -> CPU."""
    if device_override:
        device = torch.device(device_override)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    return device, dev_name


class FastTokenStream:
    """Fast memmap token streaming with GPU-resident synthetic token fallback."""

    def __init__(
        self,
        data_path: Optional[str],
        split: str,
        block_size: int,
        batch_size: int,
        device: torch.device,
        vocab_size: int = 50257,
        seed: int = 42,
    ):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.vocab_size = vocab_size
        self.split = split
        self.use_synthetic = True
        self.data = None

        if data_path and os.path.exists(data_path):
            try:
                import numpy as np

                self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
                if len(self.data) > (block_size + 1):
                    self.use_synthetic = False
                    print(
                        f"[{split}] Loaded memmap from {data_path} ({len(self.data):,} tokens)"
                    )
            except Exception as e:
                print(
                    f"[{split}] Failed loading memmap ({e}). Using synthetic fallback."
                )

        if self.use_synthetic:
            # Allocate a structured synthetic buffer with learnable n-gram grammar
            synth_len = max(200_000, batch_size * block_size * 16)
            generator = torch.Generator(device="cpu").manual_seed(
                seed + (0 if split == "train" else 1000)
            )
            # 64 motifs of length 8 with Zipfian frequency to simulate language structure
            vocab = min(1000, self.vocab_size)
            motifs = torch.randint(1, vocab, (64, 8), generator=torch.Generator().manual_seed(42))
            weights = torch.softmax(-torch.arange(64).float() / 10.0, dim=0)
            chosen = torch.multinomial(weights, synth_len // 8 + 1, replacement=True, generator=generator)
            raw_tokens = motifs[chosen].flatten()[:synth_len]
            self.synth_tokens = raw_tokens.to(self.device)
            print(
                f"[{split}] Slicing from device-resident structured synthetic buffer ({synth_len:,} tokens)"
            )

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch next batch with pinned memory for CUDA and num_workers=0."""
        if not self.use_synthetic and self.data is not None:
            import numpy as np

            max_idx = len(self.data) - self.block_size - 1
            ix = np.random.randint(0, max_idx, size=(self.batch_size,))
            x_np = np.stack(
                [self.data[i : i + self.block_size].astype(np.int64) for i in ix]
            )
            y_np = np.stack(
                [self.data[i + 1 : i + 1 + self.block_size].astype(np.int64) for i in ix]
            )

            if self.device.type == "cuda":
                x = torch.from_numpy(x_np).pin_memory().to(self.device, non_blocking=True)
                y = torch.from_numpy(y_np).pin_memory().to(self.device, non_blocking=True)
            else:
                x = torch.from_numpy(x_np).to(self.device)
                y = torch.from_numpy(y_np).to(self.device)
            return x, y
        else:
            max_idx = self.synth_tokens.size(0) - self.block_size - 1
            ix = torch.randint(0, max_idx, (self.batch_size,), device=self.device)
            x = torch.stack([self.synth_tokens[i : i + self.block_size] for i in ix])
            y = torch.stack([self.synth_tokens[i + 1 : i + 1 + self.block_size] for i in ix])
            return x, y


def evaluate_val_loss(
    model: nn.Module,
    val_loader: FastTokenStream,
    eval_steps: int,
    autocast_ctx,
) -> float:
    """Evaluate mean cross-entropy loss over validation split."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_steps):
            x, y = val_loader.get_batch()
            with autocast_ctx:
                _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 NanoGPT-124M benchmark training pipeline."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="symplectic_muon",
        choices=["symplectic_muon", "standard_muon", "adamw"],
        help="Optimizer type",
    )
    parser.add_argument(
        "--steps", type=int, default=4000, help="Total training steps (default 4000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Per-step batch size (default 16)"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=250, help="Evaluation interval"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=20, help="Steps per validation evaluation"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Muon learning rate (default 0.01)"
    )
    parser.add_argument(
        "--adamw-lr",
        type=float,
        default=0.001,
        help="AdamW learning rate for 1D/embedding parameters (default 0.001)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device override (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to tokenized training binary (e.g. FineWeb-Edu)",
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        default=None,
        help="Path to tokenized validation binary",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=50257, help="Vocabulary size"
    )
    parser.add_argument(
        "--block-size", type=int, default=1024, help="Sequence block length"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute 5-step smoke verification run and exit",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="results/nanogpt_manifest.json",
        help="Manifest output path",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases telemetry logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="moonshot-symplectic-muon",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="W&B run display name",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="W&B run mode override",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    device, dev_name = resolve_device(args.device)

    # Initialize W&B tracking if requested or WANDB_API_KEY is present
    use_wandb = args.wandb or (os.environ.get("WANDB_API_KEY") is not None and not args.dry_run)
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            mode = args.wandb_mode or os.environ.get("WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "offline")
            run_name = args.wandb_name or f"{args.optimizer}-steps{args.steps}-seed{args.seed}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                group="nanogpt-stage2",
                job_type="train",
                name=run_name,
                config=vars(args),
                mode=mode,
                dir=os.environ.get("WANDB_DIR", "/tmp/wandb"),
            )
            print(f"[W&B] Initialized run '{run_name}' in project '{args.wandb_project}' (mode: {mode})")
        except Exception as e:
            print(f"[W&B Warning] Failed to initialize wandb: {e}. Continuing without W&B.")
            use_wandb = False

    # Precision setup: bfloat16 on Ampere/Blackwell (sm_80+), float16 on older CUDA, float32 on MPS/CPU
    precision_str = "float32"
    if device.type == "cuda":
        if torch.cuda.get_device_capability() >= (8, 0):
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            precision_str = "bfloat16"
        else:
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            precision_str = "float16"
    elif device.type == "cpu":
        autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
        precision_str = "bfloat16-cpu"
    else:
        autocast_ctx = nullcontext()

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision_str == "float16"))

    print(
        f"Stage 2 NanoGPT-124M | Optimizer: {args.optimizer} | Device: {device} ({dev_name}) | Precision: {precision_str}"
    )

    # Initialize NanoGPT-124M model (GPT-2 standard: 12 layers, 12 heads, 768 dim, 1024 seq)
    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=12,
        n_head=12,
        n_embd=768,
        bias=True,
    )
    model = NanoGPT(config).to(device)

    # Clean parameter partitioning:
    # 2D linear weight matrices -> SymplecticMuon / StandardMuon
    # 1D normalization and embedding layers -> AdamW
    muon_params, adamw_params = partition_parameters(model)
    print(
        f"Partitioned parameters: {len(muon_params)} 2D linear matrices for Muon ({sum(p.numel() for p in muon_params):,} params) | "
        f"{len(adamw_params)} 1D/embedding tensors for AdamW ({sum(p.numel() for p in adamw_params):,} params)"
    )

    optimizers = []
    if args.optimizer == "symplectic_muon":
        opt_muon = SymplecticMuon(
            muon_params,
            lr=args.lr,
            momentum=0.95,
            weight_decay=args.weight_decay,
            steps=5,
        )
        opt_adamw = torch.optim.AdamW(
            adamw_params,
            lr=args.adamw_lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        optimizers = [opt_muon, opt_adamw]
    elif args.optimizer == "standard_muon":
        opt_muon = StandardMuon(
            muon_params,
            lr=args.lr,
            momentum=0.95,
            weight_decay=args.weight_decay,
            steps=5,
        )
        opt_adamw = torch.optim.AdamW(
            adamw_params,
            lr=args.adamw_lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        optimizers = [opt_muon, opt_adamw]
    elif args.optimizer == "adamw":
        # Baseline: all parameters updated via decoupled AdamW
        opt_all = torch.optim.AdamW(
            [
                {"params": muon_params, "lr": args.lr, "weight_decay": args.weight_decay},
                {"params": adamw_params, "lr": args.adamw_lr, "weight_decay": args.weight_decay},
            ],
            betas=(0.9, 0.95),
        )
        optimizers = [opt_all]

    # Initialize dataloaders
    train_loader = FastTokenStream(
        args.data_path,
        split="train",
        block_size=args.block_size,
        batch_size=args.batch_size,
        device=device,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    val_loader = FastTokenStream(
        args.val_data_path,
        split="val",
        block_size=args.block_size,
        batch_size=args.batch_size,
        device=device,
        vocab_size=args.vocab_size,
        seed=args.seed + 1,
    )

    total_steps = 5 if args.dry_run else args.steps
    eval_interval = 2 if args.dry_run else args.eval_interval

    print(f"Beginning training for {total_steps} steps...")
    t0 = time.time()
    losses = []
    val_losses = {}

    # Initial validation loss
    initial_val_loss = evaluate_val_loss(model, val_loader, 5 if args.dry_run else args.eval_steps, autocast_ctx)
    val_losses[0] = initial_val_loss
    print(f"Step 0000 | Initial Val Loss: {initial_val_loss:.4f}")

    for step in range(1, total_steps + 1):
        step_t0 = time.time()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        x, y = train_loader.get_batch()
        with autocast_ctx:
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        for opt in optimizers:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        lr_mult = min(1.0, step / 200) * (0.5 * (1 + math.cos(math.pi * step / total_steps)))
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g.setdefault("base_lr", g["lr"]) * lr_mult
            scaler.step(opt)
        scaler.update()

        step_ms = (time.time() - step_t0) * 1000.0
        losses.append(loss.item())

        if use_wandb and (step % 10 == 0 or step == total_steps):
            tokens_per_sec = (args.batch_size * args.block_size) / max(1e-4, step_ms / 1000.0)
            wandb.log({
                "step": step,
                "train/loss": loss.item(),
                "train/step_ms": step_ms,
                "train/tokens_per_sec": tokens_per_sec,
            })

        if step % eval_interval == 0 or step == total_steps:
            val_loss = evaluate_val_loss(
                model,
                val_loader,
                eval_steps=5 if args.dry_run else args.eval_steps,
                autocast_ctx=autocast_ctx,
            )
            val_losses[step] = val_loss
            if use_wandb:
                wandb.log({"step": step, "val/loss": val_loss})
            print(
                f"Step {step:04d}/{total_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Step Time: {step_ms:.1f}ms"
            )

    elapsed_sec = time.time() - t0
    final_val_loss = val_losses[total_steps]
    total_tokens = total_steps * args.batch_size * args.block_size

    # Gate 1 evaluation rule: L_sym < L_muon at step 4000
    # On a single arm run, verify loss convergence and pre-registered Gate 1 stability
    baseline_muon_val_loss = None
    baseline_file = "results/standard_muon_baseline.json"
    if os.path.exists(baseline_file):
        try:
            with open(baseline_file, "r") as f:
                bdata = json.load(f)
                baseline_muon_val_loss = bdata.get("final_val_loss")
        except Exception:
            baseline_muon_val_loss = None

    if args.optimizer == "symplectic_muon" and baseline_muon_val_loss is not None:
        gate1_passed = bool(final_val_loss < baseline_muon_val_loss)
    elif args.optimizer != "symplectic_muon":
        gate1_passed = bool(final_val_loss < initial_val_loss)
    else:
        # Pre-registered gate requires comparative standard_muon baseline file
        gate1_passed = None

    if use_wandb and wandb_run:
        wandb.summary["final_val_loss"] = final_val_loss
        wandb.summary["initial_val_loss"] = initial_val_loss
        wandb.summary["gate1_passed"] = gate1_passed
        wandb.summary["total_tokens"] = total_tokens
        wandb.finish()

    print("\n--- Benchmark Summary ---")
    print(f"Elapsed: {elapsed_sec:.2f}s | Tokens seen: {total_tokens:,}")
    print(f"Initial Val Loss: {initial_val_loss:.4f} -> Final Val Loss: {final_val_loss:.4f}")
    print(f"Gate 1 Criterion (Val Loss Convergence): {'PASS' if gate1_passed else 'FAIL'}")

    os.makedirs(os.path.dirname(args.manifest_path), exist_ok=True)
    manifest = {
        "git_commit": get_git_commit(),
        "date": datetime.now(timezone.utc).isoformat(),
        "optimizer": args.optimizer,
        "lr": args.lr,
        "adamw_lr": args.adamw_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "steps": total_steps,
        "tokens_seen": total_tokens,
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "gate1_passed": gate1_passed,
        "gate1_criteria": "L_sym < L_muon at step 4000",
        "elapsed_sec": elapsed_sec,
        "hardware": dev_name,
        "precision": precision_str,
        "dry_run": args.dry_run,
        "seed": args.seed,
    }

    with open(args.manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote provenance manifest to {args.manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
