# Models and new checkpoints

Use DecoderTCR 300M for a first local analysis. Larger models need more memory; compare their [published benchmarks](../README.md#published-decodertcr-benchmarks) for your task.

| DecoderTCR model | `--model` | Precision |
|---|---|---|
| 300M | `esmc-300m` | FP32; optional approximate FP16 on Apple |
| 600M | `esmc-600m` | FP32 |
| 6B | `esmc-6b` | FP32; inference untested |

These flags select fine-tuned DecoderTCR models. NVIDIA execution is untested, and Apple 6B support is experimental.

For 600M on Apple Silicon:

```sh
python3 tcr.py setup --device apple --model esmc-600m
```

Use `--device cpu` or `--device gpu` for those installations. For 6B:

```sh
python3 tcr.py setup --device apple --model esmc-6b
```

The 6B file is about 25 GB. FP32 weights alone occupy about **23.7 GiB of memory**; loading, Apple preparation and inference need additional memory. Setup checks available memory before downloading. For weights already on disk, use the planning command below before preparation.

## Download a new release

Use the download URL and SHA-256 supplied with the release, including GitHub release assets:

```sh
python3 tcr.py setup --device apple --model esmc-300m \
  --checkpoint-url 'HTTPS_DOWNLOAD_URL' --expected-sha256 SHA256
```

Replace both placeholders with the publisher's values. Choose `esmc-600m` or `esmc-6b` for those sizes, and `cpu` or `gpu` for other hardware. Version tags and checkpoint filenames can change; Workbench uses the exact URL you supply.

Setup verifies the checksum, checks model compatibility and prepares Apple weights when needed. It saves the selected model and weights as your default. Run your usual analysis commands afterward. Downloads and prepared weights are cached separately by checkpoint identity; existing results keep their original model details. Setup without these options uses the pinned V0.3 release.

To install a release you already downloaded, replace `--checkpoint-url` with `--checkpoint /path/to/model.ckpt` and keep `--expected-sha256`.

## Prepare a local checkpoint

After installing the matching runtime:

```sh
python3 tcr.py prepare-model --model esmc-300m --device apple \
  --checkpoint /path/to/new-300M.ckpt
```

This checks compatibility and prepares Apple weights when needed. Preparation also runs automatically before an analysis with new weights. The first run can take several minutes; later runs reuse the prepared files. If the checkpoint provider supplies a checksum, add `--expected-sha256 PROVIDER_HASH`.

Preparation does not change your saved default. Include the same model, device and checkpoint on subsequent commands:

```sh
python3 tcr.py pmhc-score --model esmc-300m --device apple \
  --checkpoint /path/to/new-300M.ckpt --panel examples/panel.csv --out results/new-model
```

Keep the original PyTorch checkpoint after Apple conversion. If using an existing Apple bundle and its original checkpoint cannot be found, provide `--reference-checkpoint /path/to/original.ckpt`. CPU and NVIDIA execution require the original checkpoint.

## Check memory and rough runtime

Estimate resource requirements before loading a large model:

```sh
python3 tcr.py prepare-model --model esmc-6b --device apple \
  --checkpoint /path/to/6B.ckpt --plan
```

`--plan` estimates requirements without running inference or converting weights. Apple preparation also needs memory for the CPU comparison. A model that fits during inference may still be too large to prepare on that machine.

If a memory check stops the run, free memory, reduce `--batch-size` or choose a smaller model. `--allow-memory-risk` overrides that stop and can cause memory exhaustion or heavy swapping.

After successful preparation, you can request a rough timing estimate:

```sh
python3 tcr.py prepare-model --plan --sequence-length 700 --estimate-forwards 1000
```

Here, 1000 means model evaluations for distinct contexts, not individual peptides. An estimate is available only when compatible timings have been recorded. It excludes some workflow overhead; actual duration depends on sequence lengths, batching and memory pressure.

## Understand compatibility

New weights must match a supported DecoderTCR architecture. Successful preparation checks that the model can run; it does not establish prediction accuracy. Choose weights trained for the intended species and compare results on representative inputs before adopting a new release.

To use a separately installed DecoderTCR version, provide `--decoder-dir` and `--python` for that installation. See [configuration](setup.md#configuration-and-other-models) to save it in a separate settings file. Existing result folders retain the model details from their original run.
