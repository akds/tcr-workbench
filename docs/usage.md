# Run the three workflows

Run commands from the folder containing `tcr.py`, using a new directory for each `--out`. Examples use public receptor components and synthetic cells, without binding labels.

| Biological question | Inputs | Result to inspect |
|---|---|---|
| Which peptides should I test against this MHC? What residues does the model favor? | Peptide/MHC panel for `pmhc-score`; MHC and peptide length for `pmhc-profile` | Ranked peptides and random comparison, or positional motif and PSSM |
| Which peptides should I test against this receptor–MHC pair? | Paired V/J genes, anchored junctions and MHC; peptide/panel for `tcr-score`, or length for `tcr-profile` | Receptor-conditioned rankings or motif |
| Which candidates should I follow up for these clonotypes? | Processed 10x, AIRR or paired receptors, peptide/MHC panel, and donor typing when available | `repertoire-score` ranks, unresolved reasons and cell/chain audit |

All five model commands and `screen` write an offline `report.html` linked to the complete tables. Profiles include probability heatmaps and sequence logos; peptide scores include top-10 tables and a scored random-peptide comparison. Open the report in a browser and keep the output folder together when sharing.

## Set up once

Follow the [quickstart](quickstart.md), then run `python3 tcr.py doctor`. Commands use the model and device saved during setup. Ordinary reference matching can use `setup --core-only`; model scoring and embedding matching need full setup.

## 1. Score peptides with MHC, or make an HLA-only profile

For scoring, prepare a CSV with `peptide,hla`. For a profile, provide the MHC molecule and peptide length; no peptide list is needed.

```bash
python3 tcr.py pmhc-score --panel examples/panel.csv --out results/pmhc
python3 tcr.py pmhc-profile --hla 'HLA-A*02:01' --length 9 --out results/hla_profile
```

`pmhc-score` reads `peptide,hla` and retains duplicates and unsupported pairs in `results.csv`/`.parquet`. Higher DecoderTCR pseudo-log-likelihood (PLL) ranks a peptide higher within its comparison group; it is not an affinity or binding probability.

`pmhc-profile` writes `profile.csv` with one row per position and 20 amino-acid probabilities, plus `pssm.csv`/`.parquet`. The PSSM is `log2(max(probability, 1e-12) / 0.05)`, relative to a uniform background; the floor applies only to log-odds. The report's logo uses stack height `log2(20) − entropy`, with each letter's height equal to its probability times stack height. Taller stacks indicate more focused preferences. These heights are not signed PSSM values. The profile describes model preferences, not measured binding motifs or inferred class II registers.

Score reports rank up to 10 distinct tested peptides per context/length group. `pmhc-score` and `tcr-score` also score **1,000 random peptides per group, seed 0**, using the same checkpoint and precision. Residues are independent uniform draws from 20 amino acids; repeated draws are retained. Random peptides are not known nonbinders, and this comparison is not a significance test.

```sh
python3 tcr.py pmhc-score --panel examples/panel.csv \
  --background-peptides 2000 --background-seed 42 --out results/pmhc_background
```

Use `--background-peptides 0` to disable the comparison. `background_scores.csv` contains the reference scores; `results.csv` retains every tested input, including duplicates and unresolved rows.

To use an **MHC-only profile reference**, select `--background-mode mhc-profile` on `pmhc-score` or `tcr-score`:

```sh
python3 tcr.py pmhc-score --panel examples/panel.csv \
  --background-mode mhc-profile --background-peptides 1000 \
  --background-seed 0 --out results/pmhc_profile_reference
```

Each position is sampled independently from the MHC-conditioned AA20 probabilities at temperature 1, with replacement. Repeats are retained. TCR chains are excluded from generation; `tcr-score` then scores these peptides with the supplied TCR–MHC pair. If the MHC profile is unavailable, the comparison is omitted and the report gives the reason.

PLL and panel rank remain primary. `background_percentile` is `100 × (1 + count(reference score >= candidate score)) / (N + 1)`, where `background_n` gives the number of finite reference draws in that context and length. Lower is better; 1,000 draws give a minimum of about 0.1%, with limited precision in the tails. A blank percentile means no usable comparison. Neither distribution is a set of verified nonbinders, and its percentile is not a binding probability or calibrated p-value. Keep the reference mode, checkpoint, context and length fixed when comparing these values.

For human class II, supply both chains as an explicit heterodimer:

```bash
python3 tcr.py pmhc-score --panel examples/panel_class_ii.csv --out results/pmhc_class_ii
python3 tcr.py pmhc-profile --hla 'HLA-DRA*01:01/HLA-DRB1*04:01' --length 15 --out results/dr_profile
```

The exact class I A/B/C allele or DR/DQ/DP pair must exist in the sequence reference. Missing partners, ambiguous groups and unavailable alleles remain `Unresolved`, without substitution.

## 2. Score a receptor with peptide–MHC, or make its peptide profile

Provide both chains' V/J genes and anchored amino-acid junctions:

```bash
python3 tcr.py tcr-score \
  --trav TRAV21 --traj TRAJ6 --cdr3a CAVRPGGAGPFFVVF \
  --trbv TRBV7-9 --trbj TRBJ2-7 --cdr3b CASSLGQAYEQYF \
  --hla 'HLA-A*02:01' --peptide GILGFVFTL --out results/tcr_score

python3 tcr.py tcr-profile \
  --trav TRAV21 --traj TRAJ6 --cdr3a CAVRPGGAGPFFVVF \
  --trbv TRBV7-9 --trbj TRBJ2-7 --cdr3b CASSLGQAYEQYF \
  --hla 'HLA-A*02:01' --length 9 --out results/tcr_profile
```

To compare a list of peptides for this same receptor–MHC context:

```sh
python3 tcr.py tcr-score \
  --trav TRAV21 --traj TRAJ6 --cdr3a CAVRPGGAGPFFVVF \
  --trbv TRBV7-9 --trbj TRBJ2-7 --cdr3b CASSLGQAYEQYF \
  --hla 'HLA-A*02:01' --panel data/tcr_peptides.csv --out results/tcr_panel
```

Create `data/tcr_peptides.csv` with a `peptide` column. Any `hla` column must match `--hla` on every row. Use either `--panel` or `--peptide`. Background flags match `pmhc-score`; comparisons remain separate by peptide length.

The software reconstructs full chains from your annotations. Check the selected genes and any reconstruction warnings in the report. Profiles use the same definitions as above, conditioned on both receptor and MHC.

## 3. Screen a repertoire

Supply processed receptors, a `peptide,hla` panel and, when available, donor typing with matching donor IDs.

```bash
python3 tcr.py repertoire-score --input examples/paired.csv --format paired \
  --panel examples/panel.csv --donors examples/donors.csv --out results/repertoire
```

Open `results/repertoire/report.html`. Scores, ranks and reasons are in `results.csv`/`.parquet`; the `cells`, `chains` and `receptors` tables connect scores to the original observations. Ranks are grouped by receptor, HLA and peptide length. The HTML preview shows at most 200 rows; use the tables for complete results.

Processed 10x and AIRR inputs use the same workflow:

```bash
python3 tcr.py repertoire-score --input examples/contigs_10x.csv --format 10x \
  --panel examples/panel.csv --donors examples/donors.csv --out results/repertoire_10x
python3 tcr.py repertoire-score --input examples/rearrangements.airr.tsv --format airr \
  --panel examples/panel.csv --donors examples/donors.csv --out results/repertoire_airr
```

For a single-donor file without `donor_id`, add `--donor-id YOUR_DONOR` and use the same ID in the HLA table. Without `--donors`, the panel defines model context but donor compatibility is unconfirmed. See [input formats](input-formats.md) before combining samples.

## Human and mouse

Human is the default. For mouse, install its germlines and specify `--species mouse` in each analysis:

```sh
python3 tcr.py setup --device apple --species mouse
```

Use `--device cpu` or `--device gpu` for another installation. To add mouse germlines to an existing runtime without redownloading weights, use `python3 tcr.py setup --reuse-config .tcr/runtime-apple.json --species mouse`, substituting your configuration path.

Bundled mouse molecules are `H-2-Kb`, `H-2-Db`, `H-2-IAb` and `H-2-IAk`. Use mouse receptor genes. The `hla` column and `--hla` flag name the MHC molecule in either species; cross-species inputs fail validation.

Replace the checkpoint path in this template with your existing file:

```sh
python3 tcr.py pmhc-profile --species mouse --hla H-2-Kb --length 8 \
  --checkpoint /path/to/mouse-trained.ckpt --out results/mouse_kb_profile
```

The same species/checkpoint flags apply to all five model commands. New mouse-trained checkpoints require a supported architecture; successful preparation does not establish predictive accuracy. Additional molecules need the mouse-only `--mhc-reference data/mouse_mhc.json` override; see [input formats](input-formats.md#human-and-mouse).

## Read scores and profiles

`Scored` or `ModelHypothesis` means a finite model result, not confirmed binding. `Unresolved` means insufficient input or supported context, not non-binding. Screening scores complete, unambiguous single alpha–beta pairs. Missing partners, nonproductive chains and ambiguous pairings remain in the output with reasons.

DecoderTCR masks **all peptide residues at once**, predicts from the MHC or TCR–MHC context, and averages the natural log of the full-vocabulary probabilities assigned to the candidate residues. This is an independent-position compatibility score: neither combining the largest motif letters nor rescoring a complete peptide captures interactions between peptide residues.

Compare scores only within the same checkpoint, species, receptor/MHC context, peptide length and precision mode. Profiles normalize only the 20 canonical amino-acid channels and cannot reconstruct the absolute full-vocabulary score.

`screen` matches curated references by exact or similar CDR3 without model weights. Its `Candidate` label is a hypothesis. The [example guide](../examples/README.md) includes a fabricated reference for testing only.

## Experimental embedding matching

After model setup, `screen --embedding-matching` also retrieves receptor neighbors using DecoderTCR embeddings:

```sh
python3 tcr.py screen --input examples/paired.csv --format paired \
  --reference examples/reference_synthetic.csv --panel examples/panel.csv \
  --donors examples/donors.csv --max-distance 0 \
  --embedding-matching --embedding-top-k 10 --out results/embedding-demo
```

`--max-distance 0` restricts the ordinary evidence table to exact CDR3 matches; omit it to retain the default sequence-distance search too. Supply your own sourced reference for research; the example associations are fabricated.

Embedding matching requires a reference `species` column (`human` or `mouse`) and complete, unambiguous paired `trav,traj,cdr3a,trbv,trbj,cdr3b` annotations. Use `--species` for the query species. The comparison uses reconstructed receptor chains without peptide or MHC input. Cosine distance ranks distinct reference receptors after panel and donor MHC checks; lower is closer. Missing typing remains explicit, and incompatible contexts are excluded.

`evidence.csv` retains ordinary CDR3 matching. `embedding_matches.csv` contains the separate `Experimental` neighbors; `embedding_audit.csv` records excluded and unresolved inputs. Embedding distance has no validated cutoff for shared specificity. Existing model and device flags apply; this mode requires model weights and FP32.

With embedding matching enabled, both searches restrict reference rows to the requested species. Rows with a different or missing species are excluded and recorded in the embedding audit. A zero-neighbor result is not evidence of absent antigen recognition.

## Hardware and saved settings

Commands use saved runtime defaults unless overridden by flags. Select another configuration with `python3 tcr.py --config /path/to/runtime.json ...`. For an existing CPU/CUDA installation, see `python3 tcr.py configure --help`; supply its DecoderTCR directory, interpreter and checkpoint. `--model` and `--device` select model and hardware independently.

| Device | Supported path | Precision |
|---|---|---|
| `cpu` | CPU on Mac or Linux | `float32` |
| `gpu` or `cuda:N` | NVIDIA GPU on Linux; untested | `float32` |
| `apple` | Apple Silicon; 6B is experimental | `float32` default; explicit `float16` for 300M only |

NVIDIA execution and 6B inference are untested. For a different model or checkpoint, follow [models and upgrades](upgrading.md).

For an installed Apple configuration, approximate FP16 is optional:

```bash
python3 tcr.py pmhc-score --panel examples/panel.csv --precision float16 --out results/pmhc_fp16
```

FP16 is available for the five model scoring/profile commands with DecoderTCR 300M on Apple. It can change close rankings. CPU, NVIDIA and experimental embedding matching require FP32.

## Common problems

| Symptom | Action |
|---|---|
| Installation/configuration missing | Run `doctor`, then the appropriate `setup` command. |
| Output directory already exists | Choose a new `--out`; completed analyses are preserved. |
| HLA is Unresolved | Read `reason`; supply colon-delimited typing and both class II chains where known. Do not guess. |
| Receptor cannot be reconstructed | Inspect gene names, anchored junctions and retained reconstruction reasons. |
| AIRR input has no paired scores | Pairing requires true cell IDs; a shared clonotype label does not establish a cell-level alpha–beta pair. |
| Too many panel combinations | Partition the panel, or deliberately increase `--max-pairs` if memory permits. |
| Apple token budget is too small | Increase `--token-budget` for the reported context; the 2048-token model limit cannot be exceeded. |

Large panels increase runtime and memory use. Start with a small panel relevant to the experiment. Pair limits include skipped combinations. Use processed conventional alpha–beta TCR tables; raw FASTQ files are not accepted.
