# Tests

The core suite uses synthetic tables and mocked model calls. It runs offline
without checkpoints, germlines or private data; Python network connections are
blocked during test functions. Installing dependencies may require internet access.

From the repository root, in a development environment:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m ruff check src tests tcr.py bootstrap.py
```

The checked-in core fixture, `fixtures/mhcnuggets_2.4.1_model_names.json`, contains
public distribution filenames and SHA-256 values, without model weights or donor data.

## Continuous integration

The default GitHub workflow tests Python 3.9 and 3.12 on Linux. Python 3.12 also
runs the Apple port's CPU-only tokenizer, synthetic conversion and precision checks
without MLX or Torch. It builds and installs a wheel, then runs `check_installed.py`
outside the checkout with isolated Python to verify packaged worker files and
generation of a complete synthetic screening report.

## Apple Metal tests

`esmc-mlx/tests` requires Apple Silicon with Metal access. It tests small randomly
initialized models, tokenizer behavior and synthetic conversion bundles, without
pretrained checkpoints. The Torch conversion case skips if Torch is absent.

The manual **Optional tiny MLX tests** workflow needs a registered self-hosted runner with the labels
`macOS`, `ARM64`, and `mlx-metal`, an updated GitHub Actions runner, and a prepared
Python 3.12+ environment containing the `esmc-mlx` test dependencies. Set the
repository variable `MLX_TEST_PYTHON` to that interpreter's absolute path if it is
not named `python3.12`. This hardware workflow never runs automatically on pull
requests. On a configured Apple machine, the equivalent command is:

```bash
cd esmc-mlx
MLX_ENABLE_TF32=0 /path/to/mlx/python -m pytest -q tests
```

These tests check interfaces, operators and packaging, not biological prediction accuracy.
