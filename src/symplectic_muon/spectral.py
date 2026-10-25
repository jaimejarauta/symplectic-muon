"""Spectral diagnostics and kinetic invariants for matrix updates.
See notes/01_hamiltonian_polar_theory.md for definitions.
"""

from typing import Dict
import torch

def compute_spectral_metrics(W: torch.Tensor, P: torch.Tensor, P_orth: torch.Tensor) -> Dict[str, float]:
    """Computes singular value condition number, off-diagonal RMS, and kinetic ratio."""
    with torch.no_grad():
        W_f = W.float()
        P_f = P.float()
        O_f = P_orth.float()
        
        # Singular values of orthogonalized update
        S = torch.linalg.svdvals(O_f)
        s_max = S[0].item()
        s_min = S[-1].item()
        cond_num = s_max / max(1e-9, s_min)
        
        # Off-diagonal RMS of O * O^T
        d_min = min(O_f.shape)
        prod = O_f @ O_f.T if O_f.size(0) <= O_f.size(1) else O_f.T @ O_f
        diag = torch.diag(prod)
        off_diag = prod - torch.diag(diag)
        off_diag_rms = torch.sqrt(torch.mean(off_diag ** 2)).item()
        
        # Kinetic energy ratio
        kinetic_p = 0.5 * torch.sum(P_f ** 2).item()
        kinetic_o = 0.5 * torch.sum(O_f ** 2).item()
        
        return {
            "s_max": s_max,
            "s_min": s_min,
            "cond_number": cond_num,
            "off_diag_rms": off_diag_rms,
            "kinetic_p": kinetic_p,
            "kinetic_orth": kinetic_o,
        }
