# Note 01: Hamiltonian Matrix-Orthogonalized Optimization & Thermostatted Symplectic Scaling

**Author:** Jaime Jarauta  
**Status:** Pre-registered active theory note  
**Target:** `src/symplectic_muon/optimizer.py`

---

## 1. Mathematical Derivation

Let $W \in \mathbb{R}^{d_1 \times d_2}$ be a 2D weight matrix and $P \in \mathbb{R}^{d_1 \times d_2}$ its conjugate momentum.
The continuous dissipative Hamiltonian dynamics on the loss surface $\mathcal{L}(W)$ are given by:
$$\frac{dW}{dt} = M^{-1} P, \quad \frac{dP}{dt} = -\nabla \mathcal{L}(W) - \xi P$$

In standard Muon (Keller Jordan, 2024; Bernstein, 2025), the update direction is projected onto the Stiefel manifold via the zeroth matrix power polar factor:
$$R(P) = P (P^\top P)^{-1/2}$$
computed iteratively via the quintic Newton-Schulz polynomial.

### The Problem: Late-Stage River-Valley Oscillations
As documented by Shen et al. (arXiv:2606.21514), because Newton-Schulz normalizes all singular values to $\approx 1$, the update magnitude $\|O_t\|_F = \sqrt{\min(d_1, d_2)}$ remains constant even as the gradient approaches zero in narrow loss valleys. This causes persistent limit-cycle oscillations around minima.

### The Solution: Thermostatted Kinetic Scaling
Naive instantaneous scaling $\gamma_t = \frac{\|P_t\|_F}{\|P_{\text{orth}}\|_F}$ introduces stochastic "energy pumping" due to mini-batch variance $\mathbb{E}[\xi_t \xi_t^\top] \neq 0$.
We introduce a Berendsen-inspired relative thermostat:
$$\bar{K}_{t+1} = (1 - \alpha) \bar{K}_t + \alpha \|P_{t+1}\|_F$$
$$\gamma_{t+1} = \text{clamp}\left( \frac{\|P_{t+1}\|_F}{\bar{K}_{t+1} + \epsilon}, 0.5, 1.8 \right) \cdot \sqrt{\max\left(1, \frac{d_1}{d_2}\right)}$$

### Semi-Implicit Symplectic Euler Integration
To preserve phase-space volume and avoid dissipative bias:
1. Momentum kick: $P_{t+1} = \mu P_t - \eta \nabla \mathcal{L}(W_t)$
2. Polar retraction: $P_{\text{orth}} = \text{NewtonSchulz5}(P_{t+1})$
3. Thermostatted scaling: $\tilde{P}_{t+1} = \gamma_{t+1} P_{\text{orth}}$
4. Coordinate drift: $W_{t+1} = (1 - \eta \lambda) W_t + \eta \tilde{P}_{t+1}$

---

## 2. Invariants & Testable Properties
1. **Spectral Uniformity:** For $P_{\text{orth}} = \text{NS}_5(P)$, the singular value ratio $\sigma_{\max} / \sigma_{\min} \le 1.15$.
2. **Bounded Kinetic Scaling:** $0.5 \le \gamma_t / \sqrt{\max(1, d_1/d_2)} \le 1.8$ holds for all $t$.
3. **Decoupled Partitioning:** Only 2D linear matrices are updated via Newton-Schulz; all 1D biases and normalization vectors remain on standard AdamW.
