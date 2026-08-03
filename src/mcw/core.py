\
"""Core wavefunction, sampling, and reconstruction utilities."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray


def gaussian_wavepacket(
    x: ArrayLike,
    x0: float = 0.0,
    p0: float = 0.0,
    sigma: float = 1.0,
    hbar: float = 1.0,
) -> NDArray[np.complex128]:
    """Return a normalized 1D Gaussian wavepacket.

    The convention is

        psi(x) ∝ exp[-(x-x0)^2/(2 sigma^2)] exp[i p0 (x-x0)/hbar].

    Under this convention, the probability-density standard deviation is
    sigma/sqrt(2), and the corresponding Wigner marginal has momentum
    standard deviation hbar/(sqrt(2)*sigma).
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if hbar <= 0:
        raise ValueError("hbar must be positive")

    x_arr = np.asarray(x, dtype=float)
    norm = (1.0 / (np.pi * sigma**2)) ** 0.25
    envelope = np.exp(-((x_arr - x0) ** 2) / (2.0 * sigma**2))
    phase = np.exp(1j * p0 * (x_arr - x0) / hbar)
    return np.asarray(norm * envelope * phase, dtype=np.complex128)


def split_operator_propagate(
    psi0: ArrayLike,
    x: ArrayLike,
    dt: float,
    nsteps: int,
    potential: Callable[[NDArray[np.float64]], ArrayLike],
    *,
    mass: float = 1.0,
    hbar: float = 1.0,
    snapshot_steps: set[int] | None = None,
) -> dict[int, NDArray[np.complex128]]:
    """Propagate a 1D wavefunction with second-order Strang splitting.

    Returns only requested snapshots to avoid storing the full time history.
    Step 0 is always available when requested.
    """
    x_arr = np.asarray(x, dtype=float)
    psi = np.asarray(psi0, dtype=np.complex128).copy()

    if x_arr.ndim != 1 or psi.shape != x_arr.shape:
        raise ValueError("x and psi0 must be one-dimensional arrays of equal length")
    if len(x_arr) < 2 or not np.allclose(np.diff(x_arr), x_arr[1] - x_arr[0]):
        raise ValueError("x must be a uniformly spaced grid")
    if dt <= 0 or nsteps < 0 or mass <= 0 or hbar <= 0:
        raise ValueError("dt, mass, and hbar must be positive; nsteps must be nonnegative")

    requested = snapshot_steps if snapshot_steps is not None else {0, nsteps}
    if not requested.issubset(set(range(nsteps + 1))):
        raise ValueError("snapshot_steps must lie between 0 and nsteps")

    dx = x_arr[1] - x_arr[0]
    k = 2.0 * np.pi * np.fft.fftfreq(x_arr.size, d=dx)
    v = np.asarray(potential(x_arr), dtype=float)
    if v.shape == ():
        v = np.full_like(x_arr, float(v))
    if v.shape != x_arr.shape:
        raise ValueError("potential(x) must return a scalar or an array matching x")

    u_v_half = np.exp(-1j * v * dt / (2.0 * hbar))
    u_t = np.exp(-1j * hbar * k**2 * dt / (2.0 * mass))

    snapshots: dict[int, NDArray[np.complex128]] = {}
    if 0 in requested:
        snapshots[0] = psi.copy()

    for step in range(1, nsteps + 1):
        psi *= u_v_half
        psi = np.fft.ifft(np.fft.fft(psi) * u_t)
        psi *= u_v_half

        if step in requested:
            snapshots[step] = psi.copy()

    return snapshots


def sample_wigner_gaussian(
    n_particles: int,
    *,
    x0: float = 0.0,
    p0: float = 0.0,
    sigma: float = 1.0,
    hbar: float = 1.0,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int8]]:
    """Sample the positive Wigner function of the Gaussian convention above."""
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    if sigma <= 0 or hbar <= 0:
        raise ValueError("sigma and hbar must be positive")

    rng = np.random.default_rng(seed)
    x_std = sigma / np.sqrt(2.0)
    p_std = hbar / (np.sqrt(2.0) * sigma)
    xs = rng.normal(x0, x_std, n_particles)
    ps = rng.normal(p0, p_std, n_particles)
    signs = np.ones(n_particles, dtype=np.int8)
    return xs, ps, signs


def gaussian_kde_marginal(
    positions: ArrayLike,
    weights: ArrayLike,
    grid: ArrayLike,
    *,
    bandwidth: float = 0.25,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Reconstruct a weighted position marginal with a Gaussian kernel.

    Negative values are intentionally retained. Clipping them would alter a
    signed-particle estimator and can hide Monte Carlo or model error.
    """
    x = np.asarray(positions, dtype=float)
    w = np.asarray(weights, dtype=float)
    grid_arr = np.asarray(grid, dtype=float)

    if x.ndim != 1 or w.shape != x.shape:
        raise ValueError("positions and weights must be matching 1D arrays")
    if grid_arr.ndim != 1 or grid_arr.size < 2:
        raise ValueError("grid must be a 1D array with at least two points")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if x.size == 0:
        raise ValueError("cannot reconstruct a density from zero particles")

    rho = np.zeros_like(grid_arr)
    chunk = 2000
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * bandwidth * x.size)
    for start in range(0, x.size, chunk):
        stop = min(start + chunk, x.size)
        offsets = (grid_arr[:, None] - x[None, start:stop]) / bandwidth
        rho += np.exp(-0.5 * offsets**2) @ w[start:stop]
    rho *= prefactor

    if normalize:
        area = np.trapz(rho, grid_arr)
        if np.isclose(area, 0.0):
            raise ValueError("weighted density has near-zero signed area")
        rho /= area
    return rho
