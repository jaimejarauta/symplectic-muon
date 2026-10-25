# Symplectic Muon

Hamiltonian matrix polar retraction optimizer for deep neural network training.

## Overview
Symplectic Muon replaces empirical momentum in matrix-orthogonalized optimizers (such as Keller Jordan's Muon) with a coordinate-free, phase-space symplectic integrator. It scales non-square weight matrices $W \in \mathbb{R}^{n \times m}$ by their intrinsic aspect ratio and enforces spectral energy preservation through a Newton-Schulz matrix polar retraction:

$$G \leftarrow \frac{1}{2} G (3I - G^T G)$$

## Pre-Registered Gate Criteria
- **Gate 1 (Smoke):** Convergence on ill-conditioned spectrum ($\kappa(A) \ge 10^3$) with loss reduction $> 90\%$ in $\le 500$ steps without limit-cycle divergence.
- **Gate 2 (Stage 2 NanoGPT-124M):** At step 4,000 on FineWeb-Edu tokenized data, validation loss $L_{\text{sym}} < L_{\text{standard\_muon}}$.

## Quickstart

### Local (Apple Silicon MPS / NVIDIA CUDA):
```bash
# Run smoke test (500 steps)
python scripts/run_smoke.py --config configs/smoke_test.yaml

# Run Stage 2 Benchmark (NanoGPT-124M on FineWeb-Edu)
python scripts/train_nanogpt.py --optimizer symplectic_muon --steps 4000 --wandb
```

### Google Colab:
Open `notebooks/colab_runner_symplectic_muon.ipynb` in Colab on a T4/L4/A100 GPU. Every notebook includes an automatic machine teardown guard (`runtime.unassign()`) to protect compute credits upon completion.

## Telemetry
Native Weights & Biases support via `--wandb` (project: `moonshot-symplectic-muon`).

## License
MIT
