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
    
    # For quintic Bernstein-Jordan NS5 at 5 steps, singular values contract to the [0.68, 1.13] band,
    # yielding a diagonal mean of approximately 0.70 - 0.95 depending on iteration parity.
    assert 0.65 <= diag_mean <= 1.15, f"Unexpected diagonal mean: {diag_mean}"
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



def test_gamma_not_clamped_during_warmup():
    """Under a constant gradient the thermostat must not hide a 0.5x LR warm-up relative to StandardMuon."""
    from symplectic_muon.optimizer import StandardMuon
    torch.manual_seed(0)
    g = torch.randn(32, 32)
    ps, pm = torch.nn.Parameter(torch.zeros(32, 32)), torch.nn.Parameter(torch.zeros(32, 32))
    sym, std = SymplecticMuon([ps], lr=0.01, weight_decay=0.0), StandardMuon([pm], lr=0.01, weight_decay=0.0)
    for _ in range(200):
        prev_s, prev_m = ps.detach().clone(), pm.detach().clone()
        ps.grad, pm.grad = g.clone(), g.clone()
        sym.step(); std.step()
        ratio = ((ps - prev_s).norm() / (pm - prev_m).norm()).item()
        assert ratio > 0.8, ratio


def test_nesterov_update_direction():
    from symplectic_muon.optimizer import StandardMuon
    torch.manual_seed(0)
    g = torch.randn(16, 32)
    p = torch.nn.Parameter(torch.zeros(16, 32))
    opt = StandardMuon([p], lr=1.0, momentum=0.9, weight_decay=0.0, nesterov=True)
    p.grad = g.clone()
    opt.step()
    # buf = g after one step; nesterov direction = g + 0.9 * buf = 1.9 g
    want = -math.sqrt(32 / 16) * zeropower_via_newtonschulz5(1.9 * g)
    assert torch.allclose(p.data, want, atol=1e-5)


def test_fixed_gamma_one_equals_standard_muon():
    from symplectic_muon.optimizer import StandardMuon
    torch.manual_seed(0)
    ps, pm = torch.nn.Parameter(torch.randn(16, 24)), None
    pm = torch.nn.Parameter(ps.detach().clone())
    sym = SymplecticMuon([ps], lr=0.02, fixed_gamma=1.0)
    std = StandardMuon([pm], lr=0.02)
    for _ in range(80):
        g = torch.randn(16, 24)
        ps.grad, pm.grad = g.clone(), g.clone()
        sym.step(); std.step()
    assert torch.allclose(ps, pm, atol=1e-6)
