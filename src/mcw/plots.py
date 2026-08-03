\
"""Plotting helpers for side-by-side density comparisons."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np


def compare_densities(x, reference, estimate, *, coordinate="x", title=""):
    """Plot two normalized one-dimensional densities."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, reference, label="Split-step Fourier")
    ax.plot(x, estimate, "--", label="Monte Carlo Wigner (KDE)")
    ax.set_xlabel(coordinate)
    ax.set_ylabel("Normalized density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax
