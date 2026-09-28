---
name: tcr-workbench-setup
description: >-
  Install and verify a local TCR-Workbench environment so a bench scientist can score TCR-pMHC
  data. Use when the user wants to install TCR-Workbench, set it up on their laptop or
  workstation, point it at the shared model registry, choose CPU / Apple Silicon / NVIDIA GPU or a
  model size, or diagnose why setup, the registry, or the model won't run. Triggers: "install
  tcr-workbench", "set it up", "run on my laptop / Mac / GPU", "configure the registry", "which
  model for my hardware", "doctor says it's not set up", "decodertcr not installed / registry not
  configured", environment/import failures. NOT for analysis (see the tcr-workbench skill) or raw
  FASTQ assembly.
---

# TCR-Workbench: install and set up

Goal: get a working local environment that can score TCR–pMHC data, then verify it — with the
least friction for someone who does not live in a terminal. Setup, not analysis, is where these
users get stuck; your job is to pick the right options, install `decodertcr_internal`, point it at
the shared model registry, run the launcher commands, and translate any failure into a fix.

## Scope and prerequisites

This skill covers **agent-ready → scoring-ready**. It assumes the user already has:

- a local machine with **Python 3.9+** (the launcher needs 3.9+; setup then provisions its own
  Python 3.12 in a managed environment — `decodertcr_internal` is Python 3.12 only), and
- the TCR-Workbench checkout on disk (from `git clone` or **Code → Download ZIP** on
  `akds/tcr-workbench`), and
- **read access to the shared model registry** (a configured artifact directory that holds the
  DecoderTCR weights). Setup installs the model package and points it at this registry; it no
  longer downloads a multi-GB checkpoint per machine.

Getting the user to an agent connected to the org's MCP server, getting the checkout onto their
machine, and knowing the registry location are prerequisites **outside** this skill — if any is
missing, point them to the onboarding doc rather than improvising. This skill is optional: an
ordinary user can run the commands below directly.

## Mental model

Everything runs through the repository launcher: **`python3 tcr.py`**. It creates and owns a
managed `.tcr/` environment (its own Python 3.12, model runtime, germlines). The model code and
weights come from the **`decodertcr_internal`** package plus a **shared registry**: setup installs
`decodertcr_internal==0.3.1` and points it at an existing registry root, and weights are
**fetched and verified from that registry on first use** (on `load`/`score`). Three facts that
shape everything:

- **Setup installs the model package and configures the registry; it does not download a per-machine
  checkpoint.** The registry is the single source of weights, shared across machines.
- **`doctor` is read-only** — it checks registry *reachability* without downloading weights, so you
  may run it freely.
- **Don't move the checkout after setup** — saved paths in `.tcr/` break. Pick a permanent
  location first.

Confirm `tcr.py` exists and run commands from its directory. If the skill has been copied away
from the repo, locate the user's checkout instead of guessing a path.

## Step 1 — check the current state first

Always start read-only:

```bash
python3 tcr.py doctor          # is anything already installed? what device/model/precision?
python3 tcr.py doctor --deep   # also check registry reachability and framework/Metal imports
```

`doctor --deep` confirms the registry is reachable but does **not** download weights.

If `doctor` reports a healthy install for the user's intended device/model, **stop** — no setup
needed. Hand off to the tcr-workbench (analysis) skill.

## Step 2 — choose the device

| User's machine | `--device` | Notes |
|---|---|---|
| Mac with Apple M-series chip | `apple` | MLX/Metal. Requires macOS on Apple Silicon (arm64). |
| Linux with a working NVIDIA driver | `gpu` | CUDA. Automatic GPU setup targets Linux only. |
| Anything else — Intel Mac, Windows, no GPU, unsure | `cpu` | Always works; slower on large models. |
| Reference matching only, no model needed | `--core-only` | Installs the core; **no** model/framework/germline downloads. |

Guardrails the tool enforces (state the fix, don't fight them):

- `--device apple` off Apple Silicon → error; use `--device cpu`.
- `--device gpu` off Linux → error; on a Mac use `--device apple`.

## Step 3 — choose the model

The CLI names below select DecoderTCR variants. Weights live in the shared registry and are fetched
and verified from it on first use, not downloaded per machine at setup.

| Model | Use it when | Never |
|---|---|---|
| `esmc-300m` (default) | Laptops / CPU / Apple; the safe default | — |
| `esmc-600m` | GPU or Apple; stronger on several benchmarks | — |
| `esmc-6b` | Only a large-memory NVIDIA GPU | never on a laptop or CPU |

These `esmc-*` names now map to the released **`decodertcr@1.0.0`** (300M / 600M / 6B) checkpoints,
which use the **V2** sequence convention. This is a model upgrade: scores and benchmarks can differ
from the earlier V0.3 (V1) models. Don't present old numbers as current.

The ESM2 model aliases (`esm2-650m` / `esm2-3b`) are **out of scope in this build** — do not offer
them. List what the registry actually exposes with `decodertcr models` (see Step 4). Default is
`esmc-300m`.

## Step 4 — run setup, then point at the registry

Setup installs dependencies and `decodertcr_internal==0.3.1` into the managed environment. It does
not fetch a per-machine checkpoint, so there is no large download to gate here.

```bash
# Apple Silicon Mac, default 300M model:
python3 tcr.py setup --device apple

# CPU, default 300M model:
python3 tcr.py setup --device cpu

# Reference matching only, no model runtime:
python3 tcr.py setup --core-only

# Other model size, or mouse germlines:
python3 tcr.py setup --device cpu --model esmc-600m
python3 tcr.py setup --device cpu --species mouse
```

Then point the model package at the existing shared registry (do this once per install; substitute
the real registry root):

```bash
decodertcr configure --registry /path/to/decodertcr-registry
decodertcr models          # list the inventory the registry exposes
decodertcr models --all    # include archived / blocked releases
```

Alternatives to `decodertcr configure`: set the env var `DECODERTCR_REGISTRY=<registry-root>`, or
write `~/.config/decodertcr/config.json`. Any one of these is enough.

Weights are fetched and verified from the registry on **first use** (on `load`/`score`), not at
setup. If that first use stops with an **estimated-memory** message, the model is too big for the
machine — choose a smaller model or better hardware. `--allow-memory-risk` overrides that guard but
can exhaust memory; **do not add it silently** — only with explicit user consent.

## Step 5 — verify

```bash
python3 tcr.py doctor --deep
```

Expect it to report the installed device, model, precision, a reachable registry, and healthy
framework imports. If it passes, setup is done; weights are pulled from the registry on first
scoring.

## Step 6 — first run / hand-off

```bash
python3 tcr.py example --out demo   # writes synthetic inputs and prints the exact next command
```

Follow the printed command, then hand off to the **tcr-workbench** skill for real scoring and
interpretation.

## Registry configuration and access

Weights come from the shared registry, not a per-machine download, so the model package must know
where the registry is and be able to reach it.

1. **Confirm the registry is configured.** Any one of these resolves the registry root:
   - `decodertcr configure --registry /path/to/decodertcr-registry`
   - the env var `DECODERTCR_REGISTRY=/path/to/decodertcr-registry`
   - `~/.config/decodertcr/config.json`

2. **List the inventory** to confirm access and see what is available:

   ```bash
   decodertcr models          # active releases
   decodertcr models --all    # also archived / blocked releases
   ```

   If a release you want appears only under `--all` as archived or blocked, it is not usable as-is —
   choose an active release (`decodertcr@1.0.0`) instead of forcing it.

3. **Version match.** `decodertcr_internal==0.3.1` resolves the released `decodertcr@1.0.0`
   checkpoints. If a release requires a newer package than is installed, `models`/`load` will say
   so; upgrade the package rather than editing metadata.

Weights are verified when first fetched from the registry, so a partial or wrong artifact is
rejected. Scoring runs offline once the artifact is cached locally; only the first fetch needs
registry access.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| "The launcher needs Python 3.9 or newer" | system Python too old | install Python 3.9+; setup then provisions 3.12 itself |
| "Apple setup requires macOS on Apple Silicon (arm64)" | `--device apple` on non-Apple-Silicon | use `--device cpu` |
| "Automatic NVIDIA GPU setup targets Linux" | `--device gpu` off Linux | `--device apple` (Mac) or `--device cpu` |
| "Setup stopped … estimated memory exceeds the available budget" | model too big for RAM/VRAM | smaller model or better hardware; `--allow-memory-risk` only with consent |
| `ModuleNotFoundError: decodertcr_internal` / "decodertcr is not installed" | model package missing from the environment | re-run full `setup --device cpu`/`apple` (installs `decodertcr_internal==0.3.1`) |
| "registry not configured" / no registry root resolved | registry never pointed at | `decodertcr configure --registry <root>`, or set `DECODERTCR_REGISTRY` |
| release shows only under `decodertcr models --all` as archived/blocked | that release is not usable as-is | choose an active release (`decodertcr@1.0.0`) |
| "requires a newer decodertcr" / release too new for package | package older than the release needs | upgrade `decodertcr_internal`; don't edit release metadata |
| "Setup is needed" / `ModuleNotFoundError` at analysis time | not set up (or core-only, no model) | run `setup --device cpu` (or `apple`) |
| `doctor` shows no model after `--core-only` | core-only intentionally skips the model runtime | re-run `setup --device cpu`/`apple`, then configure the registry |

## Caveats to hold to

- **Weights come from the shared registry, fetched and verified on first use; `doctor` only checks
  reachability and never downloads.** Configure the registry before first scoring.
- **Model upgrade:** the `esmc-*` names now resolve `decodertcr@1.0.0` (V2 sequence convention).
  Scores can differ from the earlier V0.3 (V1) models; don't compare across the two.
- **ESM2 aliases are out of scope in this build** — don't offer `esm2-650m` / `esm2-3b`.
- **Mouse still needs `--species mouse`** on setup (installs mouse germlines) and on each analysis.
- **Precision:** FP32 is the default and required for CPU/CUDA and for 600M/6B. Apple
  `--precision float16` is an approximate option for **300M only**; don't apply it elsewhere.
- **Never `--allow-memory-risk` silently** — it can OOM the machine.
- Installing this skill does not authorize uploading the user's biological data, installing
  globally, or moving/publishing the checkout. Respect the authorization already given for the task.
