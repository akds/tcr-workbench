# Models and new releases

Use DecoderTCR 300M for a first local analysis. Larger models need more memory; compare their [published benchmarks](../README.md#published-decodertcr-benchmarks) for your task.

| DecoderTCR model | `--model` | Precision |
|---|---|---|
| 300M | `esmc-300m` | FP32; optional approximate FP16 on Apple |
| 600M | `esmc-600m` | FP32 |
| 6B | `esmc-6b` | FP32; inference untested |

These flags select fine-tuned DecoderTCR models, resolved from the shared registry as `decodertcr@1.0.0` (300M/600M/6B, V2 sequence convention). This is a model upgrade from the earlier V0.3 (V1) models, so scores can differ. NVIDIA execution is untested, and Apple 6B support is experimental. The ESM2 variants are not part of this build.

For 600M on Apple Silicon:

```sh
python3 tcr.py setup --device apple --model esmc-600m
```

Use `--device cpu` or `--device gpu` for those installations. For 6B:

```sh
python3 tcr.py setup --device apple --model esmc-6b
```

The 6B model is large: FP32 weights alone occupy about **23.7 GiB of memory**, and loading, Apple preparation and inference need more. Weights come from the registry and are fetched and verified on first use; the launcher checks available memory before loading. Use the planning command below before a large run.

## Select a registry release

Weights are resolved from the shared registry, not downloaded per machine. Point the model package at the registry once, then list what it exposes:

```sh
decodertcr configure --registry /path/to/decodertcr-registry
decodertcr models          # active releases
decodertcr models --all    # also archived / blocked releases
```

You can instead set `DECODERTCR_REGISTRY` or write `~/.config/decodertcr/config.json`. The `esmc-300m`/`esmc-600m`/`esmc-6b` names map to the corresponding `decodertcr@1.0.0` sizes. A release shown only under `--all` as archived or blocked is not usable as-is; a release requiring a newer package than `decodertcr_internal==0.3.1` needs a package upgrade, not a metadata edit.

Weights are verified when first fetched from the registry, so a partial or wrong artifact is rejected. Cached artifacts are reused by release identity; existing results keep their original model details.

## Prepare a release for use

After installing the matching runtime and configuring the registry:

```sh
python3 tcr.py prepare-model --model esmc-300m --device apple
```

This fetches and verifies the registry weights, checks compatibility and prepares Apple weights (converted from the registry `.safetensors` artifact) when needed. Preparation also runs automatically before an analysis with new weights. The first run can take several minutes; later runs reuse the prepared files.

Preparation does not change your saved default. Include the same model and device on subsequent commands:

```sh
python3 tcr.py pmhc-score --model esmc-300m --device apple \
  --panel examples/panel.csv --out results/new-model
```

The Apple bundle is produced from the registry `.safetensors` artifact. If a prepared Apple bundle's source artifact cannot be resolved, ensure the registry is reachable so it can be re-fetched. CPU and NVIDIA execution use the registry weights directly.

## Check memory and rough runtime

Estimate resource requirements before loading a large model:

```sh
python3 tcr.py prepare-model --model esmc-6b --device apple --plan
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

To use a separately installed `decodertcr_internal` environment, provide `--python` for that installation. See [configuration](setup.md#configuration-and-other-models) to save it in a separate settings file. Existing result folders retain the model details from their original run.
