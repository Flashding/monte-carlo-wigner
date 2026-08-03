"""Monte Carlo Wigner educational simulation utilities."""

from .core import (
    gaussian_wavepacket,
    split_operator_propagate,
    sample_wigner_gaussian,
    gaussian_kde_marginal,
)
from .dynamics import (
    propagate_harmonic,
    propagate_free,
    double_well_potential,
    double_well_force,
    propagate_double_well_heuristic,
)

__all__ = [
    "gaussian_wavepacket",
    "split_operator_propagate",
    "sample_wigner_gaussian",
    "gaussian_kde_marginal",
    "propagate_harmonic",
    "propagate_free",
    "double_well_potential",
    "double_well_force",
    "propagate_double_well_heuristic",
]
