"""Symplectic Muon: Thermostatted Hamiltonian Matrix-Orthogonalized Optimizer."""

from .nanogpt import GPTConfig, NanoGPT, partition_parameters
from .optimizer import StandardMuon, SymplecticMuon, zeropower_via_newtonschulz5
from .spectral import compute_spectral_metrics

__all__ = [
    "SymplecticMuon",
    "StandardMuon",
    "zeropower_via_newtonschulz5",
    "compute_spectral_metrics",
    "NanoGPT",
    "GPTConfig",
    "partition_parameters",
]
