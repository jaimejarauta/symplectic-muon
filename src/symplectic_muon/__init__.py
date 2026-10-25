"""Symplectic Muon: Thermostatted Hamiltonian Matrix-Orthogonalized Optimizer."""

from .optimizer import SymplecticMuon, zeropower_via_newtonschulz5
from .spectral import compute_spectral_metrics

__all__ = ["SymplecticMuon", "zeropower_via_newtonschulz5", "compute_spectral_metrics"]

