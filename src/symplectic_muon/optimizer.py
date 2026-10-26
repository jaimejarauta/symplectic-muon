"""Thermostatted Symplectic Muon Optimizer for 2D weight matrices.
See notes/01_hamiltonian_polar_theory.md for the underlying derivation.
"""

import math
import torch
from torch.optim.optimizer import Optimizer

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7, precision: str = "auto") -> torch.Tensor:
    """Quintic Newton-Schulz polynomial mapping matrix to its polar factor.
    Optimized with fused addmm operations and hardware-aware precision.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    orig_dtype = G.dtype
    
    # Preserve full precision if fp32; use bf16 only if sm_80+ native bf16 support
    if precision == "fp32" or G.dtype == torch.float32:
        X = G.float()
    elif precision == "auto" and G.is_cuda and torch.cuda.get_device_capability(G.device) >= (8, 0):
        X = G.bfloat16()
    elif G.is_cuda and precision != "fp32":
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

class StandardMuon(Optimizer):
    """Standard Muon optimizer (Keller Jordan) with polar orthogonalization and aspect scaling."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        steps: int = 5,
        eps: float = 1e-7,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            steps=steps,
            eps=eps,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            steps = group["steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["step"] = 0
                    if p.ndim == 2 and min(p.shape) > 1:
                        d0, d1 = p.size(0), p.size(1)
                        state["aspect_scale"] = math.sqrt(max(d0, d1) / min(d0, d1))

                state["step"] += 1
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                direction = g.add(buf, alpha=mu) if group["nesterov"] else buf

                if p.ndim == 2 and min(p.shape) > 1:
                    p_orth = zeropower_via_newtonschulz5(direction, steps=steps, eps=eps)
                    aspect_scale = state.get("aspect_scale", math.sqrt(max(p.size(0), p.size(1)) / min(p.size(0), p.size(1))))
                    update = aspect_scale * p_orth
                else:
                    update = direction

                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

        return loss

    @torch.no_grad()
    def get_phase_space_diagnostics(self) -> dict:
        """Computes phase space diagnostics: kinetic energy T, running norm, gamma stats."""
        total_kinetic_energy = 0.0
        gammas = []
        for group in self.param_groups:
            eps = group.get("eps", 1e-7)
            g_min, g_max = group.get("gamma_min", 0.5), group.get("gamma_max", 1.8)
            for p in group["params"]:
                state = self.state.get(p, {})
                buf = state.get("momentum_buffer")
                if buf is not None:
                    total_kinetic_energy += 0.5 * (buf.norm() ** 2).item()
                    if state.get("running_norm") is not None and p.ndim == 2 and min(p.shape) > 1:
                        current_norm = buf.norm()
                        gamma = (state["running_norm"] / (current_norm + eps)).clamp(g_min, g_max).item()
                        gammas.append(gamma)
        mean_gamma = float(sum(gammas) / max(1, len(gammas))) if gammas else 1.0
        return {
            "kinetic_energy": total_kinetic_energy,
            "mean_gamma": mean_gamma,
            "min_gamma": float(min(gammas)) if gammas else 1.0,
            "max_gamma": float(max(gammas)) if gammas else 1.0,
        }


class SymplecticMuon(Optimizer):
    """Thermostatted semi-implicit symplectic Euler optimizer with polar orthogonalization."""
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.01,
                 steps: int = 5, eps: float = 1e-7, gamma_min: float = 0.5, gamma_max: float = 1.8,
                 ema_alpha: float = 0.05, nesterov: bool = True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        steps=steps, eps=eps, gamma_min=gamma_min,
                        gamma_max=gamma_max, ema_alpha=ema_alpha, nesterov=nesterov)
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
                    state['running_norm'] = None  # Seeded from buf after first kick
                    state['step'] = 0
                    if p.ndim == 2 and min(p.shape) > 1:
                        d0, d1 = p.size(0), p.size(1)
                        state['aspect_scale'] = math.sqrt(max(d0, d1) / min(d0, d1))

                state['step'] += 1
                buf = state['momentum_buffer']
                # Symplectic Euler: Kick momentum first
                buf.mul_(mu).add_(g)
                direction = g.add(buf, alpha=mu) if group['nesterov'] else buf

                # Only 2D hidden matrices receive Newton-Schulz orthogonalization
                if p.ndim == 2 and min(p.shape) > 1:
                    p_orth = zeropower_via_newtonschulz5(direction, steps=steps, eps=eps)
                    
                    # Thermostatted Kinetic Scaling (damping when momentum is hotter than running mean)
                    current_norm = buf.norm()  # Device tensor: no host sync
                    if state['running_norm'] is None:
                        state['running_norm'] = current_norm.clone()
                    else:
                        state['running_norm'].lerp_(current_norm, alpha)
                    gamma = (state['running_norm'] / (current_norm + eps)).clamp_(g_min, g_max)
                    # Hold gamma = 1 while the momentum buffer fills (~3/(1-mu) steps); the EMA lags a growing
                    # ||buf|| and would otherwise act as a hidden 0.5x LR warm-up
                    if state['step'] <= math.ceil(3.0 / max(1e-3, 1.0 - mu)):
                        gamma = torch.ones_like(gamma)
                    
                    # Cached symmetric aspect-ratio compensation
                    aspect_scale = state.get('aspect_scale', math.sqrt(max(p.size(0), p.size(1)) / min(p.size(0), p.size(1))))
                    update = (gamma * aspect_scale) * p_orth
                else:
                    update = direction

                # Decoupled weight decay + Drift coordinate update
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

        return loss

    @torch.no_grad()
    def get_phase_space_diagnostics(self) -> dict:
        """Computes phase space diagnostics: kinetic energy T, running norm, gamma stats."""
        total_kinetic_energy = 0.0
        gammas = []
        for group in self.param_groups:
            eps = group.get("eps", 1e-7)
            g_min, g_max = group.get("gamma_min", 0.5), group.get("gamma_max", 1.8)
            for p in group["params"]:
                state = self.state.get(p, {})
                buf = state.get("momentum_buffer")
                if buf is not None:
                    total_kinetic_energy += 0.5 * (buf.norm() ** 2).item()
                    if state.get("running_norm") is not None and p.ndim == 2 and min(p.shape) > 1:
                        current_norm = buf.norm()
                        gamma = (state["running_norm"] / (current_norm + eps)).clamp(g_min, g_max).item()
                        gammas.append(gamma)
        mean_gamma = float(sum(gammas) / max(1, len(gammas))) if gammas else 1.0
        return {
            "kinetic_energy": total_kinetic_energy,
            "mean_gamma": mean_gamma,
            "min_gamma": float(min(gammas)) if gammas else 1.0,
            "max_gamma": float(max(gammas)) if gammas else 1.0,
        }

