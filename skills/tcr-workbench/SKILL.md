---
name: tcr-workbench
description: Run local TCR-Workbench analyses of processed human or mouse alpha-beta TCR data, peptide–MHC predictions, DecoderTCR scores and peptide profiles, with MHC uncertainty and auditable outputs. Use for workflow selection, setup diagnosis, execution and interpretation; not raw FASTQ assembly or clinical antigen identification.
---

# TCR-Workbench

Use the repository's `python3 tcr.py` launcher. The bundled location is
`skills/tcr-workbench/`; the repository root is two directories above that skill
folder. Confirm `tcr.py` exists and run commands from its directory. If the skill
has been copied elsewhere, locate the user's checkout instead of inventing a path.
This skill is optional; ordinary command-line use does not require an agent.

## Establish the available runtime

- **If the environment is not installed yet, or setup/download fails, use the
  `tcr-workbench-setup` skill** — it owns device/model selection, the download-authorization
  gate, and the install/proxy/TLS failure fixes. Return here once `doctor` reports a healthy
  install. The brief checks below are enough to decide whether that hand-off is needed.
- Read `python3 tcr.py --help`, then the selected command's `--help` for exact
  options. Top-level help works before installation. Do not substitute old executable paths
  or infer flags from examples in another checkout.
- Run `python3 tcr.py doctor` to inspect the local setup. `doctor --deep` checks
  hashes and imports when deeper diagnosis is needed; it does not install assets.
  If it reports the environment is not installed or unhealthy, hand off to the
  `tcr-workbench-setup` skill rather than running `setup` from here.
- The launcher uses its managed `.tcr/` configuration unless the user supplies
  `--config`. Honor that choice. Diagnose missing models, references or incompatible
  environments before starting an expensive run; do not silently switch devices.
- For new local checkpoints, inspect `prepare-model --help`. The launcher runs
  cached compatibility checks before model workflows and converts supported Apple
  architectures when needed. Preparation does not download weights. An existing
  Apple bundle needs its matching original checkpoint for reference parity;
  inspect `--reference-checkpoint`. A failed inventory or numerical check is a
  reason to investigate, not to bypass preparation or relabel the checkpoint.
  Automatic checks are a launcher feature; an independently installed
  `tcr-workbench` command uses explicit `prepare-model`. Technical preparation
  is not validation of biological accuracy.

## Choose the quantity the user actually needs

| Request | Workflow | Interpretation |
|---|---|---|
| Predict peptide–MHC affinity | `mhc-predict` | Optional MHCflurry/MHCnuggets runtime; predicted IC50, not TCR recognition |
| Look up an existing peptide–MHC score library | `mhc` | Exact library keys and declared score metadata; missing keys stay unresolved |
| Score peptides in an HLA context with DecoderTCR | `pmhc-score` | Uncalibrated model PLL, not affinity or binding probability |
| Generate a model peptide profile for HLA | `pmhc-profile` | Conditional amino-acid preferences for one HLA and length |
| Score a specified paired TCR with peptide–MHC | `tcr-score` | DecoderTCR PLL for the supplied receptor/context |
| Generate a TCR-conditioned peptide profile | `tcr-profile` | Conditional marginals/PSSM, not a demonstrated binding motif |
| Screen processed receptors against a supplied panel | `repertoire-score` | Model hypotheses plus retained cell/chain audits |
| Find curated exact/similar receptor evidence | `screen` | Reference matching with source evidence and HLA checks |
| Describe a supplied peptide set | `profile` | Empirical frequencies/PSSM, distinct from model marginals |

Use `validate` to inspect ingestion and QC when input structure is uncertain.
The optional IC50 predictors and precomputed libraries keep their own species
and allele coverage; DecoderTCR's mouse input support does not extend those models.
`decoder-export`, `decoder-run` and `decoder-import` support separated execution;
prefer manifest-verified imports. `evaluate` compares scores with experimental
labels; integration parity and retrospective metrics do not establish unseen
biological accuracy or absence of training overlap.

## Preserve the biological contracts

- Use explicit `--species mouse` for mouse inputs; human is the default. Setup
  with `--species mouse` installs mouse germlines. The bundled mouse MHC reference
  contains exact H-2-Kb, H-2-Db, H-2-IAb and H-2-IAk mature chains; other supported
  names require a versioned `--mhc-reference` file. Inspect the schema in
  `docs/input-formats.md`; never substitute human HLA or human germlines.
  Species support and successful execution do not establish a checkpoint's
  training coverage or predictive accuracy. Future mouse-trained checkpoints
  still need their own identity, compatibility checks and experimental validation.
- Accept processed Cell Ranger V(D)J annotations, AIRR rearrangement TSVs or paired
  receptor tables. Paired inputs use `cell_id,cdr3a,cdr3b`; supply `donor_id` in
  the table or via `--donor-id`. Decoder reconstruction also needs V/J gene calls.
  AIRR uses `locus,junction_aa`; `cdr3_aa` alone is not an anchored junction.
- Preserve conserved C and terminal F/W junction anchors as supplied. Never add
  anchors, repair unknown residues, choose missing genes or discard nonproductive
  chains simply to make a receptor scoreable. Keep QC and reconstruction reasons.
- Dual-alpha alternatives are genuine possibilities, not duplicate cells.
  Alternative pairs must not be summed as independent cells or antigen evidence.
  AIRR records without cell IDs remain unpaired observations; do not fabricate
  cell IDs or cell counts. A missing chain can reflect dropout; a table alone
  does not establish its experimental cause.
- Keep donor and sample/library identities distinct. Reused barcodes across
  libraries are not proof of one cell. Confirm donor mapping before interpreting
  donor HLA compatibility.
- Donor HLA tables use `donor_id,hla`; peptide panels use `peptide,hla`. An absent
  restriction in an incomplete donor allele list is unresolved, not proof of
  incompatibility. Preserve ambiguous alternatives and G/P groups unless a
  suitable versioned mapping resolves them. Do not invent allele punctuation,
  allele-level calls or class II chain pairing.
- Human Decoder class II contexts require explicit supported alpha/beta pairs: DR, DQ
  or DP. DRB alone is not enough for that model context, even if useful in curated
  reference evidence. Exact sequences/weights must exist in the chosen reference;
  do not substitute a nearby allele. Class II binding registers are not inferred.

## Choose execution and precision explicitly

`--device cpu`, `--device gpu` (CUDA, or an explicit `cuda:N`) and
`--device apple` are separate choices. Model size/checkpoint selection is separate
from device selection. Use DecoderTCR 300M, 600M or 6B, fine-tuned from ESM-C;
the existing CLI names are `esmc-300m`, `esmc-600m` and `esmc-6b`. Do not
substitute base-model weights or ESM-2 variants. Apple supports converted
DecoderTCR 300M and 600M bundles on MLX/Metal; CPU/CUDA use the original
DecoderTCR checkpoints. The 6B Apple architecture path is experimental: its exact tensor layout
was checked without allocating the real weights; no development-machine 6B
inference was run. Require successful local same-checkpoint preparation and
resource checks before using it. The 300M and 600M paths have actual local
checkpoint validation. Parameter-storage size is not peak RAM.
CUDA support is not evidence that CUDA was tested on the current machine.

Keep the default FP32 unless the user chooses approximate precision. Apple
`--precision float16` is an explicit approximate 300M option; 600M, 6B and CPU/CUDA require FP32.
Its measured fixture errors are not a guaranteed bound on new inputs. Preserve
precision in provenance and cache identity; never treat a cache from another
precision/model/input as reusable. A verified FP32 bundle stays unchanged during
an in-memory FP16 cast. Check optional predictor and model license terms before
the requested use; installing this skill grants no rights to third-party weights.

Use `prepare-model --plan` for a local metadata-only memory estimate before a
large-model run. Inference and CPU reference/conversion are separate phases;
Apple CPU/GPU share RAM, while CUDA also needs host staging. Estimates are not
OOM guarantees. Unknown capacity stays unknown. A resource block is actionable:
choose a smaller model, reduce batching or free memory. Do not add
`--allow-memory-risk` silently; it is an explicit resource-risk override and
cannot bypass checkpoint/schema/parity checks. Preserve the requested device.
After preparation, `--estimate-forwards` counts distinct model context rows
after cache reuse, not peptide residues; its tiny-fixture extrapolation is not
a calibrated workflow ETA. Elapsed worker heartbeats are not percentages.

## Run and report auditable analyses

Use fresh output paths. Keep existing completed analyses and raw inputs intact;
do not use `--force` merely to bypass a stale-output or hash error. Resolve the
input/configuration mismatch, or explicitly replace outputs only within the
user's authorized scope. Bound panel size, batching and threads using the actual
command's supported options; do not expand a panel into an unrestricted search.

Inspect result tables, QC/skipped-row reasons and manifests, not only the HTML
preview. Report scored and unresolved counts, model/reference versions, device,
precision and output location. Preserve original cell/chain links, source
evidence, file hashes and reconstruction assumptions. Rank Decoder scores only
within the supported same-receptor/HLA/peptide-length context; do not compare
unrelated contexts or mix score scales.

The primary workflows write offline `report.html`. Profile reports contain an
AA20 probability heatmap and an information-content sequence logo; letter heights
are not signed PSSM values. Score reports show up to ten distinct peptides per
receptor/MHC/length group, plus actual model scores for matched random peptides.
The default is 1,000 independent uniform-AA20 draws per group, seed 0;
`--background-peptides 0` disables this comparator. Background draws are not
known nonbinders or a biological proteome, and the histogram is not a calibrated
binding threshold or significance test. Keep the full output folder for sharing.

For `pmhc-score` and `tcr-score`, `--background-mode mhc-profile` instead samples
MHC-only profiles with replacement at temperature 1; TCR chains never enter
generation. `--background-peptides` sets the count (default 1,000). Samples are
scored in the query context. PLL and panel rank remain primary;
`background_percentile` is a descriptive upper-tail comparison (lower is better),
not a binding probability. Do not substitute a background if profiling fails.

`screen --embedding-matching --embedding-top-k 10` adds experimental receptor
neighbors alongside CDR3 evidence. It needs model setup, FP32, paired V/J/CDR3
annotations and explicit reference species; both searches then use only reference
rows of the requested species. Embeddings exclude peptide and MHC;
HLA compatibility is checked separately. Keep `embedding_matches.csv` separate
from ordinary evidence and inspect exclusions in `embedding_audit.csv`. Closer
embeddings do not establish shared specificity. Use only the reference supplied
for that analysis; do not discover or substitute another local catalog.

`tcr-score` accepts either one `--peptide` or a `--panel` with a peptide column.
An optional panel `hla` column must match the supplied `--hla` context. DecoderTCR
masks all peptide positions together and averages the selected amino acids'
full-vocabulary log probabilities. Profiles normalize over AA20 instead. Neither
combining the largest logo letters nor rescoring complete peptides restores
intra-peptide interactions absent from this scoring procedure. Explain this in
plain language and use the report's walkthrough and column guide.

`Scored` and `ModelHypothesis` identify finite, uncalibrated model outputs;
`Candidate` identifies a reference-supported hypothesis. `Unresolved` means insufficient support,
not a negative binding result. Exact CDR3 matches, similarity, clonal expansion,
PLM marginals and predicted affinity do not independently establish TCR antigen
recognition. Multimer negatives are not proof of non-binding. Explain which
claims require experimental validation without inventing confidence values.

Keep research inputs, outputs, environments and weights local unless sharing is
requested. The bundled skill does not authorize downloading assets, uploading
biological data, installing itself globally or publishing a GitHub repository;
respect authorization already supplied for the specific task.
