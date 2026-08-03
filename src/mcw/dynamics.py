\
"""Classical and heuristic phase-space propagation routines."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _requested_steps(nsteps: int, snapshot_steps: set[int] | None) -> set[int]:
    requested = snapshot_steps if snapshot_steps is not None else {0, nsteps}
    if not requested.issubset(set(range(nsteps + 1))):
        raise ValueError("snapshot_steps must lie between 0 and nsteps")
    return requested


def propagate_harmonic(
    xs: ArrayLike,
    ps: ArrayLike,
    dt: float,
    nsteps: int,
    *,
    mass: float = 1.0,
    omega: float = 1.0,
    snapshot_steps: set[int] | None = None,
) -> dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Propagate particles in a harmonic potential using velocity Verlet."""
    x = np.asarray(xs, dtype=float).copy()
    p = np.asarray(ps, dtype=float).copy()
    if x.shape != p.shape:
        raise ValueError("xs and ps must have matching shapes")

    requested = _requested_steps(nsteps, snapshot_steps)
    snapshots = {0: (x.copy(), p.copy())} if 0 in requested else {}

    for step in range(1, nsteps + 1):
        p_half = p - 0.5 * dt * mass * omega**2 * x
        x = x + dt * p_half / mass
        p = p_half - 0.5 * dt * mass * omega**2 * x
        if step in requested:
            snapshots[step] = (x.copy(), p.copy())
    return snapshots


def propagate_free(
    xs: ArrayLike,
    ps: ArrayLike,
    distance_step: float,
    nsteps: int,
    *,
    mass: float = 1.0,
    snapshot_steps: set[int] | None = None,
) -> dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Propagate free phase-space particles exactly."""
    x0 = np.asarray(xs, dtype=float)
    p0 = np.asarray(ps, dtype=float)
    if x0.shape != p0.shape:
        raise ValueError("xs and ps must have matching shapes")

    requested = _requested_steps(nsteps, snapshot_steps)
    return {
        step: (x0 + step * distance_step * p0 / mass, p0.copy())
        for step in sorted(requested)
    }


def double_well_potential(x: ArrayLike, *, a: float = 0.01, x0: float = 4.0):
    """Return V(x) = a (x^2 - x0^2)^2."""
    x_arr = np.asarray(x, dtype=float)
    return a * (x_arr**2 - x0**2) ** 2


def double_well_force(x: ArrayLike, *, a: float = 0.01, x0: float = 4.0):
    """Return -dV/dx for the quartic double-well potential."""
    x_arr = np.asarray(x, dtype=float)
    return -4.0 * a * x_arr * (x_arr**2 - x0**2)


def propagate_double_well_heuristic(
    xs: ArrayLike,
    ps: ArrayLike,
    signs: ArrayLike,
    dt: float,
    nsteps: int,
    *,
    mass: float = 1.0,
    a: float = 0.01,
    x0: float = 4.0,
    gamma0: float = 0.02,
    momentum_shift: float = 0.5,
    seed: int | None = None,
    max_particles: int = 250_000,
    snapshot_steps: set[int] | None = None,
) -> dict[int, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int8]]]:
    """Experimental signed-particle branching model.

    This routine is deliberately labeled heuristic: its branching rate and
    momentum shifts are not derived from the exact Wigner kernel. It is useful
    for studying numerical behavior, but not for claiming quantitatively
    correct tunneling dynamics.
    """
    x = np.asarray(xs, dtype=float).copy()
    p = np.asarray(ps, dtype=float).copy()
    s = np.asarray(signs, dtype=np.int8).copy()
    if not (x.shape == p.shape == s.shape):
        raise ValueError("xs, ps, and signs must have matching shapes")

    requested = _requested_steps(nsteps, snapshot_steps)
    snapshots = {0: (x.copy(), p.copy(), s.copy())} if 0 in requested else {}
    rng = np.random.default_rng(seed)

    for step in range(1, nsteps + 1):
        p_half = p + 0.5 * dt * double_well_force(x, a=a, x0=x0)
        x = x + dt * p_half / mass
        p = p_half + 0.5 * dt * double_well_force(x, a=a, x0=x0)

        rate = gamma0 * np.maximum(0.0, np.abs(x) - 0.5 * x0)
        branch = rng.random(x.size) < np.clip(rate * dt, 0.0, 1.0)

        if np.any(branch):
            xb, pb, sb = x[branch], p[branch], s[branch]
            x = np.concatenate((x, xb, xb))
            p = np.concatenate((p, pb + momentum_shift, pb - momentum_shift))
            s = np.concatenate((s, sb, -sb)).astype(np.int8, copy=False)

        if x.size > max_particles:
            raise RuntimeError(
                f"particle population exceeded max_particles={max_particles}; "
                "reduce gamma0, dt, or nsteps"
            )

        if step in requested:
            snapshots[step] = (x.copy(), p.copy(), s.copy())

    return snapshots
