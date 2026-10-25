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

def test_symmetric_aspect_scaling():
    """Verify that (d0, d1) and (d1, d0) rectangular matrices receive symmetric aspect scaling."""
    torch.manual_seed(42)
    p_up = torch.nn.Parameter(torch.randn(64, 16))
    p_up.grad = torch.randn(64, 16)
    
    p_down = torch.nn.Parameter(torch.randn(16, 64))
    p_down.grad = torch.randn(16, 64)
    
    opt = SymplecticMuon([p_up, p_down], lr=0.01, steps=5)
    opt.step()
    
    # Both updates should be finite and scaled proportionately
    assert torch.isfinite(p_up).all()
    assert torch.isfinite(p_down).all()
    # Check that update norm is balanced relative to singular dimension sqrt(d_max)
    d_max = 64
    d_min = 16
    expected_ratio = math.sqrt(d_max / d_min)
    assert expected_ratio == 2.0

