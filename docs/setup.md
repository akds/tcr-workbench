# Setup and troubleshooting

For a first installation, follow the [quickstart](quickstart.md). Run `python3 tcr.py doctor` to check an existing installation. Commands below run from the repository folder.

Setup installs the `decodertcr_internal` model package (Python 3.12) and configures DecoderTCR 300M by default. For 600M or 6B, add `--model esmc-600m` or `--model esmc-6b`. Weights are not downloaded per machine. On CPU/GPU, setup defaults to the released HuggingFace weights when no registry is configured; to use the shared internal registry instead — and always for Apple, which has no HuggingFace path yet — provide it when you run setup:

```sh
python3 tcr.py setup --device apple --registry /path/to/decodertcr-registry
# or export DECODERTCR_REGISTRY=/path/to/decodertcr-registry before setup
```

Setup connects the model package to that registry for you; there is no separate `decodertcr configure` step. List releases with `decodertcr models` (add `--all` for archived/blocked). The first scoring run needs source access; later analyses use the locally cached weights. The `esmc-300m`/`esmc-600m`/`esmc-6b` names map to `decodertcr@1.5.0` (V2 sequence convention), a model upgrade from the earlier V0.3 models. The ESM2 variants are not part of this build.

## Existing installations

To reuse existing Workbench settings and model files:

```sh
python3 tcr.py setup --reuse-config /path/to/cpu.json --reuse-config /path/to/apple.json
```

The last configuration becomes the default. The referenced environments must remain in place; weights are resolved from the shared registry.

To select a different release, list what the registry exposes and pick a model size:

```sh
decodertcr models   # active releases; add --all for archived/blocked
```

The `esmc-300m`/`esmc-600m`/`esmc-6b` names map to the corresponding `decodertcr@1.5.0` sizes. See [models and new releases](upgrading.md). To check compatibility and resources without changing your default, use [model preparation](upgrading.md#prepare-a-release-for-use).

## Configuration and other models

Setup saves defaults in `.tcr/runtime.json`; command flags override them. To select another configuration, put `--config` before the workflow:

```sh
python3 tcr.py --config /path/to/custom.json pmhc-score --panel examples/panel.csv --out results/custom
```

If you already have a Python 3.12 environment with `decodertcr_internal` installed, register it:

```sh
python3 tcr.py configure \
  --decoder-dir /path/to/decodertcr-env \
  --python /path/to/decodertcr-env/bin/python \
  --model esmc-300m --device cpu --settings-out custom.json
```

Both `--decoder-dir` (the environment root) and `--python` (its interpreter) are required. CPU/CUDA resolve weights by model id, so do not pass `--checkpoint`; the environment uses released HuggingFace weights, or the shared internal registry it was pointed at (`decodertcr configure --registry <root>` or `DECODERTCR_REGISTRY`).

Linux GPU setup requires a working NVIDIA driver and extra disk space for CUDA dependencies. NVIDIA execution is untested. Apple supports 300M and 600M; 6B is experimental and needs substantial memory. See [models and upgrades](upgrading.md) before changing models.

## Common problems

| Message or symptom | Action |
|---|---|
| Environment missing | Run full `setup --device cpu` or `setup --device apple`; use `setup --core-only` for ordinary reference matching only. |
| `decodertcr_internal` not installed | Rerun full `setup`; it installs `decodertcr_internal==0.5.0`. |
| Registry not configured | `decodertcr configure --registry <root>`, or set `DECODERTCR_REGISTRY`. |
| Weight fetch failed | Check registry access and disk space, then retry; cached artifacts are reused and verified. |
| Release archived/blocked or needs a newer package | Choose an active release with `decodertcr models --all`, or upgrade `decodertcr_internal`. |
| Setup lock exists | Check that no other setup is running. If it stopped, remove `.tcr/setup.lock` and rerun. |
| Metal unavailable | Use an Apple Silicon Mac with working Metal access, or set up CPU execution. |
| Germline download failed | Check access to IMGT and rerun setup for the intended species. |
| Output already exists | Choose a new `--out` directory. |
| Memory check stopped execution | Use a smaller model, reduce `--batch-size` or free memory. See [memory planning](upgrading.md#check-memory-and-rough-runtime). |
| Unresolved result | Read the reason and check the supplied genes, junctions and MHC names. |

For a more thorough installation check, run `python3 tcr.py doctor --deep`. Rerunning setup can repair dependencies without changing result folders; `setup --core-only` preserves existing model settings.

After moving the repository folder, preserve any outputs, checkpoints or data you need, remove the old `.tcr/` directory and run setup again. Windows setup is untested; use WSL/Linux.
