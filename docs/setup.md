# Setup and troubleshooting

For a first installation, follow the [quickstart](quickstart.md). Run `python3 tcr.py doctor` to check an existing installation. Commands below run from the repository folder.

Setup installs DecoderTCR 300M by default. For 600M or 6B, add `--model esmc-600m` or `--model esmc-6b`. Setup needs internet access; prepared analyses use local files.

## Existing installations

To reuse existing Workbench settings and model files:

```sh
python3 tcr.py setup --reuse-config /path/to/cpu.json --reuse-config /path/to/apple.json
```

The last configuration becomes the default. The referenced environments and weights must remain in place.

To reuse a downloaded copy of the released 300M checkpoint while installing the other components:

```sh
python3 tcr.py setup --device apple --checkpoint /path/to/300M.ckpt
```

For newer weights, use setup with `--checkpoint-url` and `--expected-sha256`. See [download a new release](upgrading.md#download-a-new-release). To check a local checkpoint without changing your default, use [model preparation](upgrading.md#prepare-a-local-checkpoint).

## Configuration and other models

Setup saves defaults in `.tcr/runtime.json`; command flags override them. To select another configuration, put `--config` before the workflow:

```sh
python3 tcr.py --config /path/to/custom.json pmhc-score --panel examples/panel.csv --out results/custom
```

If you already installed DecoderTCR separately, register its paths:

```sh
python3 tcr.py configure --decoder-dir /path/to/DecoderTCR \
  --python /path/to/DecoderTCR/.venv/bin/python \
  --model DecoderTCR-ESMC_300M --device cpu \
  --checkpoint /path/to/300M.ckpt --settings-out custom.json
```

Linux GPU setup requires a working NVIDIA driver and extra disk space for CUDA dependencies. NVIDIA execution is untested. Apple supports 300M and 600M; 6B is experimental and needs substantial memory. See [models and upgrades](upgrading.md) before changing models.

## Common problems

| Message or symptom | Action |
|---|---|
| Environment missing | Run full `setup --device cpu` or `setup --device apple`; use `setup --core-only` for ordinary reference matching only. |
| Download failed | Check internet access and disk space, then rerun setup. Completed downloads are reused. |
| Checksum mismatch | Move the named damaged file aside, then rerun setup. |
| Setup lock exists | Check that no other setup is running. If it stopped, remove `.tcr/setup.lock` and rerun. |
| Metal unavailable | Use an Apple Silicon Mac with working Metal access, or set up CPU execution. |
| Germline download failed | Check access to IMGT and rerun setup for the intended species. |
| Output already exists | Choose a new `--out` directory. |
| Memory check stopped execution | Use a smaller model, reduce `--batch-size` or free memory. See [memory planning](upgrading.md#check-memory-and-rough-runtime). |
| Unresolved result | Read the reason and check the supplied genes, junctions and MHC names. |

For a more thorough installation check, run `python3 tcr.py doctor --deep`. Rerunning setup can repair dependencies without changing result folders; `setup --core-only` preserves existing model settings.

After moving the repository folder, preserve any outputs, checkpoints or data you need, remove the old `.tcr/` directory and run setup again. Windows setup is untested; use WSL/Linux.
