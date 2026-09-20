# Working in TCR-Workbench

These instructions apply to this repository. Follow the user's stated task and existing authorization; the file does not grant permission to publish code or share research data.

## Help a user run an analysis

Read `README.md`, the selected command's `--help`, and `docs/input-formats.md`. The optional `skills/tcr-workbench/SKILL.md` gives the fuller biological workflow guidance. Use `python3 tcr.py` from the repository root. Use `doctor` to inspect setup; installation and downloads are explicit `setup` operations. Preserve an explicitly supplied `--config`.

Choose the workflow by the requested quantity:

- `pmhc-score`: DecoderTCR peptide scores in an MHC context, with a top-10 report and scored random-peptide reference.
- `pmhc-profile`: MHC-conditioned amino-acid profile, sequence-logo motif and PSSM for a specified length.
- `tcr-score` / `tcr-profile`: paired-receptor scores for one peptide or a panel, or conditional motif profiles.
- `repertoire-score`: panel screening with cell/chain/receptor audit.
- `screen`: curated exact/similarity reference evidence without model weights.

`screen --embedding-matching` additionally requires DecoderTCR weights and explicit reference species; both searches then restrict reference rows to the requested species. Keep receptor-only embedding neighbors in separate `Experimental` outputs; do not transfer antigen labels or mix cosine distance with CDR3 distance or PLL. Incomplete/ambiguous receptors remain in the audit.

Use a new output directory. Point users to `report.html`, then inspect the complete result tables, unresolved reasons, QC and manifests; HTML is a limited preview. Keep reports offline and escape input-derived text when changing the renderer. Do not invent binding probabilities or summarize `Unresolved` as non-binding. Keep private inputs, outputs and model weights local unless sharing is authorized.

## Preserve biological meaning

- Retain dual-alpha alternatives, missing chains, nonproductive rearrangements and allele ambiguity in the audit. Alternative receptors are not independent cells.
- Use the supplied V/J genes and anchored C-to-F/W amino-acid junctions. Do not repair sequences, invent anchors, select unknown alleles or fabricate cell pairing to make an input scoreable.
- AIRR pairing needs real cell identifiers and `junction_aa`. Shared clonotype labels or reused barcodes do not establish a unique cell across libraries.
- Use the explicit human/mouse species; default is human. Mouse setup installs mouse germlines, and each mouse analysis needs `--species mouse`. Keep `hla` as the compatible MHC field name; never infer species or substitute cross-species genes/chains.
- Class II model contexts require explicit supported alpha/beta MHC pairs. Do not guess a missing partner or substitute a nearby allele.
- Compare model scores within the same checkpoint, species, receptor/MHC context and peptide length. Preserve numerical precision and model identity in provenance.
- DecoderTCR masks the whole peptide simultaneously and averages full-vocabulary log probabilities. The profile normalizes AA20; its logo encodes information content, not signed PSSM or binding probability. Neither scoring nor combining profile columns recovers intra-peptide interactions.
- `Scored` and `ModelHypothesis` are finite model results, not binding labels. Score reports use actually scored uniform-AA20 random draws (default 1,000 per context/length, seed 0). Random peptides are not known nonbinders; never invent missing distributions or calibrated thresholds. `--background-peptides 0` skips that optional comparison.
- `pmhc-score` and `tcr-score` keep PLL and panel rank primary. Optional `--background-mode mhc-profile` samples MHC-only AA20 profiles with replacement at temperature 1, then scores in the query context. No TCR participates in generation. Reference percentiles are descriptive, not binding probabilities; preserve failed profiles without a fallback distribution.
- Public/synthetic examples test software behavior. They are not labeled biological validation data.

## Models and resources

Workflows use DecoderTCR 300M, 600M and 6B checkpoints fine-tuned from ESM-C. The default is DecoderTCR 300M with FP32. Preserve the existing `esmc-300m`, `esmc-600m` and `esmc-6b` CLI names; these select DecoderTCR variants. Do not recommend base-model weights or ESM-2 variants for these workflows. `--model` and `--device` are independent. CPU/CUDA use PyTorch; Apple uses MLX/Metal. Use `prepare-model --plan` before potentially oversized workloads. The launcher prepares unseen checkpoints automatically and refreshes resource checks even for cached models.

Do not silently switch models, devices or precision, or add `--allow-memory-risk`. That override requires an explicit resource-risk choice; it never bypasses architecture, finite-value or parity checks. Keep unknown capacity marked unknown. Respect the separate CPU reference/conversion requirement for Apple preparation.

Apple FP16 is an explicit approximate 300M option. The 6B Apple path is architecture-checked but lacks real-checkpoint inference validation here; NVIDIA execution is also untested here. A successful tiny forward is not universal numerical or biological validation. New architectures need a declared adapter, not inferred attention heads or renamed tensor keys.

## Change the code

Keep `tcr.py` as the small launcher and `bootstrap.py` as explicit setup/diagnostics. Core modules in `src/tcr_workbench/` must remain importable without Torch or MLX. Framework workers belong in `src/tcr_workbench/backends/`; the minimal Apple port is in `esmc-mlx/`.

Preserve strict ingestion/settings/worker contracts, uncertainty statuses, deterministic identities, file hashes and provenance. Prefer bounded streaming or columnar operations for large tables. Do not weaken integrity checks to claim a speedup. Profile representative work before adding a new cache, kernel or dependency. Do not introduce optional architecture frameworks or placeholder flags for deferred features.

For functional changes, run the smallest relevant tests and then the fast core suite. With a development environment installed as described in [tests/README.md](tests/README.md):

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests tcr.py bootstrap.py
```

Core tests are model-free. Do not download checkpoints to run them. Run tiny Metal tests only in an existing compatible MLX environment. When changing model numerics, use the same checkpoint and tokens against the reference, report tolerances and error metrics, then benchmark the retained change. Distinguish executed tests, mock contracts and untested hardware.

For documentation-only edits, check links, command parsing and applicable lightweight examples. Do not rerun large model matrices without a numerical change or a specific unresolved concern.

## Release hygiene

Publish this repository's source, never the parent research workspace. Keep `.tcr/`, environments, checkpoints, private inputs and generated analyses out of version control. `.gitignore` is a convenience, not proof that arbitrary newly added files are safe to publish; inspect the proposed file list.

New Workbench code uses MIT. Retain third-party attribution and separate model/data license terms. Do not claim that the repository license relicenses weights.
