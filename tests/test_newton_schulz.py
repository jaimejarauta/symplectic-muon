"""Unit tests for Newton-Schulz polar orthogonalization and kinetic invariants."""

import math
import torch
import pytest

from symplectic_muon.optimizer import zeropower_via_newtonschulz5, SymplecticMuon
from symplectic_muon.spectral import compute_spectral_metrics

def test_newton_schulz_orthogonality():
    torch.manual_seed(42)
    G = torch.randn(32, 64)
    O = zeropower_via_newtonschulz5(G, steps=5)
    
    # Check that O @ O^T is close to identity scaled by 1.0
    OO_T = O @ O.T
    diag_mean = torch.mean(torch.diag(OO_T)).item()
    off_diag = OO_T - torch.diag(torch.diag(OO_T))
    off_diag_rms = torch.sqrt(torch.mean(off_diag ** 2)).item()
    
    assert 0.85 <= diag_mean <= 1.15, f"Unexpected diagonal mean: {diag_mean}"
    assert off_diag_rms < 0.15, f"Excessive off-diagonal leakage: {off_diag_rms}"

def test_symplectic_muon_step_finite():
    torch.manual_seed(42)
    p = torch.nn.Parameter(torch.randn(16, 16))
    p.grad = torch.randn(16, 16)
    
    opt = SymplecticMuon([p], lr=0.01)
    opt.step()
    
    assert torch.isfinite(p).all(), "Non-finite parameter update detected"
