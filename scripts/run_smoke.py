#!/usr/bin/env python3
"""Execution script for Symplectic Muon smoke test."""

import argparse
import json
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from symplectic_muon.optimizer import SymplecticMuon
from symplectic_muon.spectral import compute_spectral_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke_test.yaml")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda, mps, cpu)")
    parser.add_argument("--steps", type=int, default=None, help="Step override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Device auto-detection: CUDA -> Apple Silicon MPS -> CPU
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    print(f"Running Symplectic Muon Smoke Test on: {device} ({dev_name})")

    torch.manual_seed(cfg["optimization"]["seed"])
    d_in = cfg["model"]["d_in"]
    d_hidden = cfg["model"]["d_hidden"]
    d_out = cfg["model"]["d_out"]
    batch_size = args.batch_size or cfg["optimization"]["batch_size"]
    n_steps = args.steps or cfg["optimization"]["steps"]

    # Generate ill-conditioned dataset with geometrically decaying singular values
    X = torch.randn(batch_size * 10, d_in, device=device)
    decay = torch.pow(0.8, torch.arange(d_in, device=device).float())
    X = X * decay
    W_true = torch.randn(d_in, d_out, device=device)
    Y = X @ W_true + 0.05 * torch.randn(X.size(0), d_out, device=device)

    model = nn.Sequential(
        nn.Linear(d_in, d_hidden, bias=False),
        nn.GELU(),
        nn.Linear(d_hidden, d_out, bias=False)
    ).to(device)

    opt = SymplecticMuon(
        model.parameters(),
        lr=cfg["optimization"]["lr"],
        momentum=cfg["optimization"]["momentum"],
        weight_decay=cfg["optimization"]["weight_decay"],
        gamma_min=cfg["optimization"]["gamma_min"],
        gamma_max=cfg["optimization"]["gamma_max"],
        ema_alpha=cfg["optimization"]["ema_alpha"],
        steps=cfg["optimization"]["ns_steps"]
    )

    t0 = time.time()
    losses = []
    for step in range(n_steps):
        idx = (step * batch_size) % (X.size(0) - batch_size)
        x_b, y_b = X[idx:idx+batch_size], Y[idx:idx+batch_size]

        pred = model(x_b)
        loss = F.mse_loss(pred, y_b)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if step % 100 == 0 or step == n_steps - 1:
            print(f"Step {step:03d} | Loss: {loss.item():.4f}")

    elapsed = time.time() - t0
    initial_loss = losses[0]
    final_loss = losses[-1]

    # Verify pre-registered rule
    passed = final_loss < initial_loss * 0.10
    print(f"\nElapsed: {elapsed:.2f}s | Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    print(f"Rule evaluation (final_loss < initial_loss * 0.10): {'PASS' if passed else 'FAIL'}")

    os.makedirs("results", exist_ok=True)
    manifest = {
        "config": args.config,
        "date": cfg["date"],
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "passed": passed,
        "elapsed_sec": elapsed,
    }
    with open("results/smoke_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()

