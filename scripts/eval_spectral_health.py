#!/usr/bin/env python3
"""Spectral Health and Phase-Space Geometry Evaluation for Symplectic Muon vs Standard Muon.

Evaluates singular value distributions, condition numbers, stable ranks, and spectral entropy
across all 2D projection matrices in NanoGPT-124M.
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Ensure symplectic_muon package is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from symplectic_muon.nanogpt import GPTConfig, NanoGPT


def compute_matrix_spectral_metrics(W: torch.Tensor, eps: float = 1e-9) -> Dict[str, float]:
    """Computes rigorous spectral metrics for a 2D weight matrix W (out_features, in_features)."""
    assert W.ndim == 2, f"Expected 2D matrix, got shape {W.shape}"
    W_f = W.detach().float()
    
    # SVD singular values
    sv = torch.linalg.svdvals(W_f)
    s_max = sv[0].item()
    s_min = sv[-1].item()
    s_sum = sv.sum().item()
    s_sq_sum = (sv ** 2).sum().item()

    # Condition number kappa
    cond_num = s_max / max(eps, s_min)

    # Stable rank: ||W||_F^2 / ||W||_2^2
    stable_rank = s_sq_sum / max(eps, s_max ** 2)

    # Spectral entropy (isotropy metric): H = -sum(p_i * log2(p_i))
    p = (sv / max(eps, s_sum)).clamp_min(eps)
    spectral_entropy = -float((p * torch.log2(p)).sum().item())
    max_entropy = math.log2(len(sv))
    normalized_entropy = spectral_entropy / max(eps, max_entropy)

    return {
        "rows": W.size(0),
        "cols": W.size(1),
        "min_dim": min(W.size(0), W.size(1)),
        "sigma_max": float(s_max),
        "sigma_min": float(s_min),
        "condition_number": float(cond_num),
        "stable_rank": float(stable_rank),
        "frobenius_norm": float(math.sqrt(s_sq_sum)),
        "nuclear_norm": float(s_sum),
        "spectral_entropy": float(spectral_entropy),
        "normalized_entropy": float(normalized_entropy),
    }


def audit_nanogpt_spectral_health(
    model: nn.Module,
) -> Dict[str, any]:
    """Audits all 2D linear weight matrices in NanoGPT-124M."""
    layer_metrics: Dict[str, Dict[str, float]] = {}
    cond_numbers: List[float] = []
    stable_ranks: List[float] = []
    normalized_entropies: List[float] = []

    for name, param in model.named_parameters():
        if param.ndim == 2 and min(param.shape) > 1 and "wte" not in name and "wpe" not in name:
            metrics = compute_matrix_spectral_metrics(param)
            layer_metrics[name] = metrics
            cond_numbers.append(metrics["condition_number"])
            stable_ranks.append(metrics["stable_rank"])
            normalized_entropies.append(metrics["normalized_entropy"])

    avg_cond = sum(cond_numbers) / max(1, len(cond_numbers))
    avg_stable_rank = sum(stable_ranks) / max(1, len(stable_ranks))
    avg_norm_entropy = sum(normalized_entropies) / max(1, len(normalized_entropies))

    return {
        "num_matrices_audited": len(layer_metrics),
        "mean_condition_number": float(avg_cond),
        "median_condition_number": float(sorted(cond_numbers)[len(cond_numbers) // 2]) if cond_numbers else 0.0,
        "mean_stable_rank": float(avg_stable_rank),
        "mean_normalized_entropy": float(avg_norm_entropy),
        "layers": layer_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate spectral health of trained transformer weights.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output", "--output-manifest", dest="output", type=str, default="results/spectral_health.json", help="Output JSON path")
    parser.add_argument("--wandb", action="store_true", help="Log spectral audit to W&B")
    parser.add_argument("--wandb-project", type=str, default="moonshot-symplectic-muon", help="W&B project")
    parser.add_argument("--wandb-name", type=str, default="spectral-health-audit", help="W&B run name")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    cfg = GPTConfig()
    model = NanoGPT(cfg)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Strip potential _orig_mod prefix from torch.compile
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=False)
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint provided or file not found; auditing initial state configuration.")

    print("\nComputing SVD on all 48 2D hidden linear projections...")
    report = audit_nanogpt_spectral_health(model)

    print("\n=== Spectral Health Summary ===")
    print(f"Matrices Audited:        {report['num_matrices_audited']}")
    print(f"Mean Condition Number:   {report['mean_condition_number']:.2f}")
    print(f"Median Condition Number: {report['median_condition_number']:.2f}")
    print(f"Mean Stable Rank:        {report['mean_stable_rank']:.2f}")
    print(f"Mean Spectral Entropy:   {report['mean_normalized_entropy']:.4f} (1.0 = perfectly isotropic)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {args.output}")

    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={"checkpoint": args.checkpoint},
        )
        wandb.log({
            "spectral/mean_condition_number": report["mean_condition_number"],
            "spectral/median_condition_number": report["median_condition_number"],
            "spectral/mean_stable_rank": report["mean_stable_rank"],
            "spectral/mean_normalized_entropy": report["mean_normalized_entropy"],
        })
        wandb.finish()
        print("Telemetry synced to W&B.")


if __name__ == "__main__":
    main()
