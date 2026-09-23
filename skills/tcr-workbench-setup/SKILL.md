---
name: tcr-workbench-setup
description: >-
  Install and verify a local TCR-Workbench environment so a bench scientist can score TCR-pMHC
  data. Use when the user wants to install TCR-Workbench, set it up on their laptop or
  workstation, download model weights, choose CPU / Apple Silicon / NVIDIA GPU or a model size,
  or diagnose why setup, download, or the model won't run. Triggers: "install tcr-workbench",
  "set it up", "run on my laptop / Mac / GPU", "download weights", "which model for my hardware",
  "doctor says it's not set up", "SSL / certificate error on download", environment/import
  failures. NOT for analysis (see the tcr-workbench skill) or raw FASTQ assembly.
---

# TCR-Workbench: install and set up

Goal: get a working local environment that can score TCR–pMHC data, then verify it — with the
least friction for someone who does not live in a terminal. Setup, not analysis, is where these
users get stuck; your job is to pick the right options, gate the large downloads, run the two
launcher commands, and translate any failure into a fix.

## Scope and prerequisites

This skill covers **agent-ready → scoring-ready**. It assumes the user already has:

- a local machine with **Python 3.9+** (the launcher needs 3.9+; setup then provisions its own
  Python 3.12 in a managed environment), and
- the TCR-Workbench checkout on disk (from `git clone` or **Code → Download ZIP** on
  `akds/tcr-workbench`).

Getting the user to an agent connected to the org's MCP server, and getting the checkout onto
their machine, are prerequisites **outside** this skill — if either is missing, point them to the
onboarding doc rather than improvising. This skill is optional: an ordinary user can run the
commands below directly.

## Mental model

Everything runs through the repository launcher: **`python3 tcr.py`**. It creates and owns a
managed `.tcr/` environment (its own Python 3.12, model runtime, germlines, weights). Two facts
that shape everything:

- **Downloads happen only during explicit `setup`.** `doctor` is read-only and never downloads.
  So you may run `doctor` freely, but you must get authorization before running `setup`.
- **Don't move the checkout after setup** — saved paths in `.tcr/` break. Pick a permanent
  location first.

Confirm `tcr.py` exists and run commands from its directory. If the skill has been copied away
from the repo, locate the user's checkout instead of guessing a path.

## Step 1 — check the current state first

Always start read-only:

```bash
python3 tcr.py doctor          # is anything already installed? what device/model/precision?
python3 tcr.py doctor --deep   # also verify checkpoint hashes and framework/Metal imports
```

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

Downloads are large — this is the step to authorize explicitly before running.

| Model | Download | Use it when | Never |
|---|---:|---|---|
| `esmc-300m` (default) | ~4.0 GB | Laptops / CPU / Apple; the safe default | — |
| `esmc-600m` | ~2.3 GB | GPU or Apple; stronger on several benchmarks | — |
| `esmc-6b` | ~25 GB | Only a large-memory NVIDIA GPU | never on a laptop or CPU |

Counter-intuitive but correct: **`esmc-600m` downloads *smaller* than `esmc-300m`** — the 300M
release checkpoint also bundles optimizer state. Don't "correct" this.

Only download what the user will actually use. Default is `esmc-300m`.

## Step 4 — get authorization, then run setup

Tell the user the concrete download size and destination, and confirm before proceeding. Then:

```bash
# Apple Silicon Mac, default 300M model (~4 GB):
python3 tcr.py setup --device apple

# CPU, default 300M model (~4 GB):
python3 tcr.py setup --device cpu

# Reference matching only, no model download:
python3 tcr.py setup --core-only

# Smaller/other model, or mouse germlines:
python3 tcr.py setup --device cpu --model esmc-600m
python3 tcr.py setup --device cpu --species mouse
```

The download is checksum- and size-verified and written atomically (a failed transfer never
replaces a good file). If setup stops with an **estimated-memory** message, the model is too big
for the machine — choose a smaller model or better hardware. `--allow-memory-risk` overrides that
guard but can exhaust memory; **do not add it silently** — only with explicit user consent.

## Step 5 — verify

```bash
python3 tcr.py doctor --deep
```

Expect it to report the installed device, model, precision, and healthy checkpoint hashes /
framework imports. If it passes, setup is done.

## Step 6 — first run / hand-off

```bash
python3 tcr.py example --out demo   # writes synthetic inputs and prints the exact next command
```

Follow the printed command, then hand off to the **tcr-workbench** skill for real scoring and
interpretation.

## Behind a corporate / TLS-inspecting proxy (e.g. CZ Biohub)

On a network that inspects TLS (Umbrella/OpenDNS + Palo Alto), the weights download can fail even
though the network is fine. Exported environment variables **are inherited by setup**, so set the
fix and re-run `setup`:

- **`SSL: CERTIFICATE_VERIFY_FAILED`** — Python's certificate bundle doesn't trust the proxy's
  MITM root CA (even though `curl` does, via the macOS keychain). Point Python at the system trust
  store. Export the macOS trust store once, then re-run setup:

  ```bash
  security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain >  /tmp/macos-ca-bundle.pem
  security find-certificate -a -p /Library/Keychains/System.keychain                        >> /tmp/macos-ca-bundle.pem

  SSL_CERT_FILE=/tmp/macos-ca-bundle.pem \
  REQUESTS_CA_BUNDLE=/tmp/macos-ca-bundle.pem \
    python3 tcr.py setup --device apple
  ```

- **Download stalls / times out** — the launcher's download times out after ~120s of no data. If
  the network uses a proxy, set `HTTPS_PROXY`/`HTTP_PROXY` before setup. For any transfer that
  goes through `huggingface_hub` (e.g. germline tooling, or a future HF-hub weight path), also set
  `HF_HUB_DISABLE_XET=1` — the Xet protocol does not survive the proxy. (The model-weights
  download itself is a plain HTTPS request, so the CA fix above is usually the one that matters.)

Sanity-check the network outside Python first: `curl -sSI -L <weights-url>`. If curl returns
`200`/`302` but the Python download still fails, it's the CA/proxy issue above, not connectivity.
Only download needs the network; scoring runs offline once weights are on disk.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| "The launcher needs Python 3.9 or newer" | system Python too old | install Python 3.9+; setup then provisions 3.12 itself |
| "Apple setup requires macOS on Apple Silicon (arm64)" | `--device apple` on non-Apple-Silicon | use `--device cpu` |
| "Automatic NVIDIA GPU setup targets Linux" | `--device gpu` off Linux | `--device apple` (Mac) or `--device cpu` |
| "Setup stopped … estimated memory exceeds the available budget" | model too big for RAM/VRAM | smaller model or better hardware; `--allow-memory-risk` only with consent |
| `SSL: CERTIFICATE_VERIFY_FAILED` on download | corporate MITM CA not trusted by Python | set `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` to the macOS trust store (see proxy section) |
| "Checksum mismatch" / "Size mismatch" on an existing file | corrupt or partial prior download | move the damaged file aside and retry setup |
| "Setup is needed" / `ModuleNotFoundError` at analysis time | not set up (or core-only, no model) | run `setup --device cpu` (or `apple`) |
| `doctor` shows no weights after `--core-only` | core-only intentionally skips the model | re-run `setup --device cpu`/`apple` to add the model |

## Caveats to hold to

- **Downloads only during explicit `setup`; `doctor` never downloads.** Get authorization first.
- **Precision:** FP32 is the default and required for CPU/CUDA and for 600M/6B. Apple
  `--precision float16` is an approximate option for **300M only**; don't apply it elsewhere.
- **Never `--allow-memory-risk` silently** — it can OOM the machine.
- Installing this skill does not authorize uploading the user's biological data, installing
  globally, or moving/publishing the checkout. Respect the authorization already given for the task.
