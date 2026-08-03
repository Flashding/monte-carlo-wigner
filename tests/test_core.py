import numpy as np

from mcw import gaussian_wavepacket, sample_wigner_gaussian, propagate_harmonic


def test_wavepacket_is_normalized():
    x = np.linspace(-20.0, 20.0, 20_000)
    psi = gaussian_wavepacket(x, sigma=1.5)
    assert np.isclose(np.trapz(np.abs(psi) ** 2, x), 1.0, atol=1e-6)


def test_wigner_sampling_matches_convention():
    sigma = 1.5
    xs, ps, _ = sample_wigner_gaussian(
        200_000, sigma=sigma, hbar=1.0, seed=123
    )
    assert np.isclose(xs.std(), sigma / np.sqrt(2), rtol=0.02)
    assert np.isclose(ps.std(), 1.0 / (np.sqrt(2) * sigma), rtol=0.02)


def test_harmonic_energy_is_nearly_conserved():
    xs = np.array([1.0, -0.5])
    ps = np.array([0.2, 0.4])
    snapshots = propagate_harmonic(
        xs, ps, 0.001, 10_000, mass=1.0, omega=0.3,
        snapshot_steps={0, 10_000},
    )
    energies = []
    for step in (0, 10_000):
        x, p = snapshots[step]
        energies.append(0.5 * p**2 + 0.5 * 0.3**2 * x**2)
    assert np.allclose(energies[0], energies[1], rtol=1e-5, atol=1e-8)
