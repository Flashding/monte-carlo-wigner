# Monte Carlo Wigner Propagation

An educational Python project comparing Monte Carlo phase-space propagation
with split-step Fourier solutions of one-dimensional quantum and optical wave
equations.

## Demonstrations

1. **Harmonic oscillator** — validation in a quadratic potential
2. **Double well** — experimental signed-particle branching model
3. **Free-space optics** — Gaussian diffraction and phase-space ray propagation

## Important scientific note

The harmonic-oscillator and free-space examples are consistency checks in
systems where Gaussian Wigner evolution is classical.

The double-well branching model is **heuristic**. Its branching rate and
momentum displacement are not derived from the exact Wigner kernel, so it
should not be interpreted as a quantitatively validated tunneling solver.

## Repository structure

```text
monte-carlo-wigner/
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/
│   └── mcw/
│       ├── __init__.py
│       ├── core.py
│       ├── dynamics.py
│       └── plots.py
├── examples/
│   ├── 01_harmonic_oscillator.ipynb
│   ├── 02_double_well.ipynb
│   ├── 03_free_space_optics.ipynb
│   └── original_notebook.ipynb
├── tests/
│   └── test_core.py
└── figures/
```

## Installation

Open Terminal and move into the repository's top-level folder—the folder that
contains `pyproject.toml`:

```bash
cd /path/to/monte-carlo-wigner
python3 -m pip install -e .
```

The `-e` means **editable installation**. Python installs a link to the local
`src/mcw` package, so changes to the source files are immediately available
without reinstalling.

Verify the installation:

```bash
python3 -c "import mcw; print(mcw.__file__)"
```

The printed path should point to this repository's `src/mcw` folder.

## Running the notebooks

After installation:

```bash
jupyter notebook
```

Open one of the notebooks in `examples/` and select the same Python interpreter
used for installation.

## Running the tests

From the repository's top-level folder:

```bash
python3 -m pytest
```

Do not normally run `tests/test_core.py` directly. `pytest` imports the
installed `mcw` package and discovers all test functions automatically.

## Theory summary

The Wigner function \(W(x,p)\) is a phase-space quasi-probability distribution.
Its marginals recover position and momentum probability densities:

\[
\int W(x,p)\,dp = |\psi(x)|^2,
\qquad
\int W(x,p)\,dx = |\tilde{\psi}(p)|^2.
\]

Its dynamics can be written as

\[
\frac{\partial W}{\partial t}
+
\frac{p}{m}\frac{\partial W}{\partial x}
=
\Theta[V]W.
\]

In a particle representation, each sample is described by

\[
(x_i,p_i,s_i),
\qquad
s_i\in\{-1,+1\}.
\]

The classical part follows

\[
\dot{x}=\frac{p}{m},
\qquad
\dot{p}=-\frac{dV}{dx}.
\]

The split-step Fourier reference method applies a half potential step, a full
kinetic step in Fourier space, and another half potential step.
