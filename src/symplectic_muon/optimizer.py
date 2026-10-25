"""Thermostatted Symplectic Muon Optimizer for 2D weight matrices.
See notes/01_hamiltonian_polar_theory.md for the underlying derivation.
"""

import math
import torch
from torch.optim.optimizer import Optimizer

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz polynomial mapping matrix to its polar factor.
    Optimized with fused addmm operations and hardware-aware precision.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    
    # Fast precision selection: bfloat16 on Ampere/Blackwell, float16 on T4/V100, float32 on CPU/MPS
    if G.is_cuda and torch.cuda.is_bf16_supported():
        X = G.bfloat16()
    elif G.is_cuda:
        X = G.half()
    else:
        X = G.float()

    # Normalize by Frobenius norm to guarantee singular values are strictly in (0, 1] < sqrt(3)
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        # A = X @ X.T
        A = torch.mm(X, X.T)
        # B = b * A + c * (A @ A) -> single-pass fused GEMM: beta * mat + alpha * (mat1 @ mat2)
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        # X = a * X + B @ X -> single-pass fused GEMM
        X = torch.addmm(X, B, X, beta=a, alpha=1.0)

    if transposed:
        X = X.T
    return X.to(G.dtype)

class SymplecticMuon(Optimizer):
    """Thermostatted semi-implicit symplectic Euler optimizer with polar orthogonalization."""
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.01,
                 steps: int = 5, eps: float = 1e-7, gamma_min: float = 0.5, gamma_max: float = 1.8,
                 ema_alpha: float = 0.05):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        steps=steps, eps=eps, gamma_min=gamma_min,
                        gamma_max=gamma_max, ema_alpha=ema_alpha)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']
            steps = group['steps']
            eps = group['eps']
            g_min, g_max = group['gamma_min'], group['gamma_max']
            alpha = group['ema_alpha']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                # State initialization
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['running_norm'] = g.norm().item() + eps
                    state['step'] = 0

                state['step'] += 1
                buf = state['momentum_buffer']
                # Symplectic Euler: Kick momentum first
                buf.mul_(mu).add_(g)

                # Only 2D hidden matrices receive Newton-Schulz orthogonalization
                if p.ndim == 2 and min(p.shape) > 1:
                    p_orth = zeropower_via_newtonschulz5(buf, steps=steps, eps=eps)
                    
                    # Thermostatted Kinetic Scaling (see notes/01_hamiltonian_polar_theory.md)
                    current_norm = buf.norm().item()
                    state['running_norm'] = (1.0 - alpha) * state['running_norm'] + alpha * current_norm
                    rel_scale = current_norm / (state['running_norm'] + eps)
                    gamma = max(g_min, min(g_max, rel_scale))
                    
                    # Symmetric aspect-ratio compensation: preserves invariant under transpose
                    d0, d1 = p.size(0), p.size(1)
                    aspect_scale = math.sqrt(max(d0, d1) / min(d0, d1))
                    update = (gamma * aspect_scale) * p_orth
                else:
                    update = buf

                # Decoupled weight decay + Drift coordinate update
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

        return loss

