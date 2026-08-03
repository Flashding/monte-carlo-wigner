# Local Installation Guide

## 1. Open the project folder in VS Code

Open the folder containing:

- `pyproject.toml`
- `src`
- `examples`
- `tests`

Do not open only the `tests` or `examples` subfolder.

## 2. Select a Python interpreter

In VS Code:

1. Press `Command + Shift + P`.
2. Choose `Python: Select Interpreter`.
3. Select the Python installation you want to use.

## 3. Install the package

Open a terminal in the project root and run:

```bash
python3 -m pip install -e .
```

If VS Code shows a different interpreter path, use that full path instead of
`python3`.

## 4. Verify the import

```bash
python3 -c "import mcw; print(mcw.__file__)"
```

## 5. Run the tests

```bash
python3 -m pytest
```

## 6. Use the same interpreter in Jupyter

Open a notebook and select the kernel corresponding to the Python interpreter
used for the installation.

If `import mcw` still fails, compare:

```bash
which python3
python3 -m pip --version
```

The Python and pip paths should refer to the same environment.
