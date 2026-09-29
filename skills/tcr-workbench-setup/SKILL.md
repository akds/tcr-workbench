---
name: tcr-workbench-setup
description: >-
  Install and verify a local TCR-Workbench environment so a bench scientist can score TCR-pMHC
  data. Use when the user wants to install TCR-Workbench, set it up on their laptop or
  workstation, choose where model weights come from (HuggingFace or the shared registry), choose
  CPU / Apple Silicon / NVIDIA GPU or a model size, or diagnose why setup, the weights, or the model
  won't run. Triggers: "install tcr-workbench", "set it up", "run on my laptop / Mac / GPU",
  "pull weights from HuggingFace", "configure the registry", "which model for my hardware", "doctor
  says it's not set up", "decodertcr not installed / can't reach HuggingFace / registry not
  configured", environment/import/auth failures. NOT for analysis (see the tcr-workbench skill) or
  raw FASTQ assembly.
---

# TCR-Workbench: install and set up

Goal: get a working local environment that can score TCR–pMHC data, then verify it — with the
least friction for someone who does not live in a terminal. Setup, not analysis, is where these
users get stuck; your job is to pick the right options, install `decodertcr_internal`, choose the
weight source, run the launcher commands, and translate any failure into a fix.

## Scope and prerequisites

This skill covers **agent-ready → scoring-ready**. It assumes the user already has:

- a local machine with **Python 3.9+** (the launcher needs 3.9+; setup then provisions its own
  Python 3.12 in a managed environment — `decodertcr_internal` is Python 3.12 only), and
- the TCR-Workbench checkout on disk (from `git clone` or **Code → Download ZIP** on
  `akds/tcr-workbench`), and
- **a way to obtain the weights** — either **internet access to HuggingFace** (the default; a public
  model repo needs no login, a private/gated one needs a token) **or read access to a shared model
  registry** (an internal artifact directory). Setup installs the model package; weights are then
  pulled from whichever source you choose, on first use.

Getting the checkout onto the machine, and (for the registry path) knowing the registry location,
are prerequisites **outside** this skill — if a needed one is missing, point them to the onboarding
doc rather than improvising. This skill is optional: an ordinary user can run the commands below
directly.

## Mental model

Everything runs through the repository launcher: **`python3 tcr.py`**. It creates and owns a
managed `.tcr/` environment (its own Python 3.12, model runtime, germlines). The model *code* is the
**`decodertcr_internal`** package (`0.5.0`); the model *weights* come from one of two sources:

- **HuggingFace (default when no registry is configured).** Setup records the model's published HF
  repository; `from_pretrained` downloads and verifies the weights on first use (on `load`/`score`).
  No HPC mount needed — this is the path a laptop or a fresh sandbox uses.
- **Shared registry (internal).** Setup points the package at an existing registry root; weights are
  fetched and verified from it on first use. Use this when the user has registry access and prefers
  it, or on Apple Silicon (see below).

Facts that shape everything:

- **Setup installs the model package and records the weight source; it does not fetch a checkpoint
  at setup.** Weights download/verify on **first use**, not during `setup`.
- **`doctor` is read-only** — it checks that the chosen source is *reachable* (HF repo resolves, or
  registry resolves) without downloading weights, so you may run it freely.
- **HuggingFace downloads persist inside `.tcr/`** (the launcher sets `HF_HUB_CACHE` under the model
  environment), so an ephemeral sandbox reuses cached weights instead of re-downloading each session.
- **Weight source by device:** HuggingFace runs on **cpu/gpu only**. **Apple Silicon (MLX) uses the
  registry** — HuggingFace conversion for Apple is not yet supported, so an Apple setup needs a
  registry.
- **Don't move the checkout after setup** — saved paths in `.tcr/` break. Pick a permanent location
  first.

Confirm `tcr.py` exists and run commands from its directory. If the skill has been copied away from
the repo, locate the user's checkout instead of guessing a path.

## Step 1 — check the current state first

Always start read-only:

```bash
python3 tcr.py doctor          # is anything already installed? what device/model/source?
python3 tcr.py doctor --deep   # also check weight-source reachability and framework/Metal imports
```

`doctor --deep` confirms the weight source is reachable (HuggingFace repo or registry) but does
**not** download weights. If it reports a healthy install for the user's intended device/model,
**stop** — no setup needed. Hand off to the tcr-workbench (analysis) skill.

## Step 2 — choose the device

| User's machine | `--device` | Notes |
|---|---|---|
| Mac with Apple M-series chip | `apple` | MLX/Metal. Requires macOS on Apple Silicon (arm64). **Registry only.** |
| Linux with a working NVIDIA driver | `gpu` | CUDA. Automatic GPU setup targets Linux only. |
| Anything else — Intel Mac, Windows, no GPU, unsure | `cpu` | Always works; slower on large models. |
| Reference matching only, no model needed | `--core-only` | Installs the core; **no** model/framework/germline install. |

Guardrails the tool enforces (state the fix, don't fight them):

- `--device apple` off Apple Silicon → error; use `--device cpu`.
- `--device gpu` off Linux → error; on a Mac use `--device apple`.
- `--weight-source huggingface` with `--device apple` → error; Apple needs the registry.

## Step 3 — choose the weight source

Most of the time you don't specify this — the default is right:

- **No registry configured → HuggingFace** (cpu/gpu). This is the laptop / fresh-machine default.
  Setup announces the repo it will use.
- **A registry is configured (`--registry` or `DECODERTCR_REGISTRY`) → registry.**
- **Apple Silicon → registry** (HuggingFace isn't supported there yet).

Set it explicitly only when overriding the default:

| Situation | Flag |
|---|---|
| Force released HuggingFace weights (cpu/gpu) | `--weight-source huggingface` |
| Force the internal registry | `--weight-source registry --registry /path/to/registry` |
| Use a non-default / private HF repository | `--weight-source huggingface --hf-repo ORG/REPO` |
| Pin a specific HF revision | `--hf-revision <commit-or-tag>` |

**HuggingFace authentication:** a public model repo needs none. For a **private or gated** repo,
export a read token first — `export HF_TOKEN=hf_…` (or run `hf auth login` once). If the token is
missing, `doctor`/first scoring fails with a clear "not reachable / check HF_TOKEN" message; never
work around it silently.

## Step 4 — choose the model

The CLI names below select DecoderTCR variants. Weights are fetched and verified from the chosen
source on first use, not downloaded per machine at setup.

| Model | Use it when | Never |
|---|---|---|
| `esmc-300m` (default) | Laptops / CPU / Apple; the safe default | — |
| `esmc-600m` | GPU or Apple; stronger on several benchmarks | — |
| `esmc-6b` | Only a large-memory NVIDIA GPU | never on a laptop or CPU |

These `esmc-*` names map to the released **`decodertcr@1.5.0`** (300M / 600M / 6B) checkpoints, which
use the **V2** sequence convention. This is a model upgrade: scores and benchmarks can differ from
the earlier V0.3 (V1) models. Don't present old numbers as current.

Only `esmc-300m` currently has a published HuggingFace repository; `esmc-600m` / `esmc-6b` are
registry-only until their repos are published. If the user asks for 600M/6B on the HuggingFace path,
either supply `--hf-repo` for a repo that has them or use the registry. The ESM2 aliases
(`esm2-650m` / `esm2-3b`) are **out of scope in this build** — do not offer them. Default is
`esmc-300m`.

## Step 5 — run setup

Setup installs dependencies and `decodertcr_internal==0.5.0` into the managed environment, then
records the weight source. There is no large download during setup — weights arrive on first use.

```bash
# Laptop / CPU, default 300M, weights from HuggingFace (the default with no registry):
python3 tcr.py setup --device cpu

# Same, stated explicitly:
python3 tcr.py setup --device cpu --weight-source huggingface

# Private/gated HF repo — authenticate first:
export HF_TOKEN=hf_xxx
python3 tcr.py setup --device cpu --weight-source huggingface

# Internal registry instead (also required for Apple Silicon):
python3 tcr.py setup --device cpu   --registry /path/to/decodertcr-registry
python3 tcr.py setup --device apple --registry /path/to/decodertcr-registry

# Reference matching only, no model runtime:
python3 tcr.py setup --core-only

# Other model size, or mouse germlines:
python3 tcr.py setup --device gpu --model esmc-600m --registry /path/to/decodertcr-registry
python3 tcr.py setup --device cpu --species mouse
```

Weights are fetched and verified on **first use** (on `load`/`score`), not at setup. If that first
use stops with an **estimated-memory** message, the model is too big for the machine — choose a
smaller model or better hardware. `--allow-memory-risk` overrides that guard but can exhaust memory;
**do not add it silently** — only with explicit user consent.

## Step 6 — verify

```bash
python3 tcr.py doctor --deep
```

Expect it to report the installed device, model, precision, and a reachable weight source — either
`HuggingFace source OK: <repo> <commit> …` or a reachable registry — plus healthy framework imports.
If it passes, setup is done; weights are pulled on first scoring.

## Step 7 — first run / hand-off

```bash
python3 tcr.py example --out demo   # writes synthetic inputs and prints the exact next command
```

The first real scoring on the HuggingFace path downloads ~1.3 GB (300M) once, into the `.tcr/`
cache. Follow the printed command, then hand off to the **tcr-workbench** skill for real scoring and
interpretation.

## HuggingFace access and caching

- **Public repo:** no login. **Private/gated repo:** `export HF_TOKEN=hf_…` (a read token with
  access) or `hf auth login` once. The launcher passes the token through and also honours a cached
  login.
- **Where weights live:** the launcher sets `HF_HUB_CACHE` under `.tcr/envs/model/hf-cache`, so
  downloads persist with the project. If the environment is ephemeral (a sandbox that resets),
  keeping `.tcr/` avoids re-downloading; otherwise first scoring re-fetches the weights.
- **Provenance:** each run records `weight_source`, `hf_repo`, the resolved commit and the weights
  hash in the output manifest — use these to confirm which weights produced a result.
- **Pinning:** `--hf-revision` fixes a commit/tag for reproducibility; otherwise the current repo
  revision is resolved and recorded.

## Registry configuration and access (internal / Apple)

When using `--weight-source registry` (or on Apple Silicon), the model package must know where the
registry is and be able to reach it. `setup --registry <root>` does this; alternatives:

- the env var `DECODERTCR_REGISTRY=/path/to/decodertcr-registry`, or
- `decodertcr configure --registry /path/to/decodertcr-registry`, or
- `~/.config/decodertcr/config.json`.

Then list the inventory to confirm access:

```bash
decodertcr models          # active releases
decodertcr models --all    # also archived / blocked releases
```

If a release appears only under `--all` as archived or blocked, it is not usable as-is — choose an
active release (`decodertcr@1.5.0`). `decodertcr_internal==0.5.0` resolves the released
`decodertcr@1.5.0` checkpoints; if a release needs a newer package, upgrade the package rather than
editing metadata. Weights are verified when first fetched, so a partial or wrong artifact is
rejected; scoring runs offline once the artifact is cached locally.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| "The launcher needs Python 3.9 or newer" | system Python too old | install Python 3.9+; setup then provisions 3.12 itself |
| "Apple setup requires macOS on Apple Silicon (arm64)" | `--device apple` on non-Apple-Silicon | use `--device cpu` |
| "Automatic NVIDIA GPU setup targets Linux" | `--device gpu` off Linux | `--device apple` (Mac) or `--device cpu` |
| "HuggingFace weights currently run on cpu or gpu only" | `--weight-source huggingface` with `--device apple` | use the registry on Apple, or `--device cpu` |
| "HuggingFace weights are not reachable … check HF_TOKEN" | no network, wrong repo id, or missing token for a private/gated repo | fix network / repo id; `export HF_TOKEN=hf_…` or `hf auth login` |
| "model … has no HuggingFace source configured" | HF path for a model without a published repo (600M/6B) | supply `--hf-repo ORG/REPO`, or use the registry |
| "Model setup requires --registry …" | `--weight-source registry` (or Apple) with no registry configured | pass `--registry <root>` / set `DECODERTCR_REGISTRY`, or use HuggingFace on cpu/gpu |
| "Setup stopped … estimated memory exceeds the available budget" | model too big for RAM/VRAM | smaller model or better hardware; `--allow-memory-risk` only with consent |
| `ModuleNotFoundError: decodertcr_internal` / "decodertcr is not installed" | model package missing from the environment | re-run full `setup` for the device |
| "decodertcr_internal … is too old; install >= 0.5.0" | HuggingFace path needs `from_pretrained` (0.5.0+) | upgrade the model package |
| release shows only under `decodertcr models --all` as archived/blocked | that release is not usable as-is | choose an active release (`decodertcr@1.5.0`) |
| "Setup is needed" / `ModuleNotFoundError` at analysis time | not set up (or core-only, no model) | run `setup` for the device |
| `doctor` shows no model after `--core-only` | core-only intentionally skips the model runtime | re-run `setup` for the device |

## Caveats to hold to

- **Weights are fetched and verified on first use, never at setup; `doctor` only checks
  reachability.** HuggingFace is the default source (cpu/gpu); the registry is internal and required
  on Apple Silicon.
- **Don't switch weight source, model, device or precision silently.** The default (registry if
  configured, else HuggingFace) is announced; state it, and only override on explicit request.
- **HuggingFace auth:** public repos need none; private/gated need `HF_TOKEN` / `hf auth login` —
  surface the auth error, don't work around it.
- **Model upgrade:** the `esmc-*` names resolve `decodertcr@1.5.0` (V2 sequence convention). Scores
  can differ from the earlier V0.3 (V1) models; don't compare across the two.
- **ESM2 aliases are out of scope in this build** — don't offer `esm2-650m` / `esm2-3b`.
- **Mouse still needs `--species mouse`** on setup (installs mouse germlines) and on each analysis.
- **Precision:** FP32 is the default and required for CPU/CUDA and for 600M/6B. Apple
  `--precision float16` is an approximate option for **300M only**; don't apply it elsewhere.
- **Never `--allow-memory-risk` silently** — it can OOM the machine.
- Installing this skill does not authorize uploading the user's biological data, installing globally,
  or moving/publishing the checkout. Respect the authorization already given for the task.
