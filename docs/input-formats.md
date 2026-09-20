# Input formats and biological conventions

Use UTF-8 CSV or TSV with a header. Keep the original preprocessing files and prepare a separate input table. Use the [example files](../examples/README.md) as templates.

## Peptide panel

Required columns: `peptide,hla`. Use the 20 standard amino-acid letters without modification notation, gaps or stop symbols. Scoring accepts lengths 1–50; choose lengths appropriate to the MHC class.

```csv
peptide,hla
GILGFVFTL,HLA-A*02:01
NLVPMVATV,HLA-A*02:01
```

MHC-only scoring retains duplicate rows. Repertoire results preserve their cell/receptor mappings. Invalid columns or peptide characters stop the run; missing or unsupported HLA remains `Unresolved`.

For `tcr-score --panel`, only `peptide` is required: `--hla` fixes the context. Any `hla` column must match that molecule on every row. Use either `--panel` or `--peptide`. Other panel commands require both columns.

## Paired receptors

Required columns: `cell_id,cdr3a,cdr3b`, with `donor_id` in the file or supplied through `--donor-id`. Reconstruction also needs `trav,traj,trbv,trbj` for complete pairs. Supply the observed gene calls; do not fill missing calls by guessing.

```csv
cell_id,donor_id,trav,traj,cdr3a,trbv,trbj,cdr3b
synthetic_cell_1,synthetic_donor,TRAV21,TRAJ6,CAVRPGGAGPFFVVF,TRBV7-9,TRBJ2-7,CASSLGQAYEQYF
```

`cdr3a`/`cdr3b` are amino-acid junctions including the conserved starting C and terminal F or W. Use IMGT gene names and retain experimentally determined allele suffixes such as `*01`. Unsuffixed genes may use reconstruction defaults, recorded with selected genes in the audit.

Missing chains, uncertain sequences and nonproductive rearrangements are retained with QC. Optional productivity flags are `productive_a` and `productive_b`. Single-chain rows remain unpaired.

## Processed 10x Cell Ranger V(D)J annotations

Use a contig annotation CSV, such as `filtered_contig_annotations.csv`, with `--format 10x`. Required columns: `barcode,chain,cdr3`. Retain `contig_id,v_gene,j_gene,productive,high_confidence,is_cell,umis` when available. `TRA` and `TRB` identify alpha/beta chains.

Supply `donor_id` or pass `--donor-id` for a single donor. FASTQ reads are not accepted. Explicit non-cell, low-confidence or nonproductive contigs remain in the chain audit but are excluded from searchable pairing. Missing confidence or productivity remains unknown.

## AIRR rearrangements

Use `--format airr`. Required columns: `locus,junction_aa`. Retain `sequence_id,v_call,j_call,productive,cell_id` when available. Supply `donor_id` or `--donor-id`.

AIRR `junction_aa` includes anchors; `cdr3_aa` is not a substitute. Pairing requires `cell_id`. Without it, rearrangements remain unpaired observations, known cell count is zero and observation count is reported separately.

## Donor HLA typing

Required columns are `donor_id,hla`, with one allele or explicitly reported heterodimer per row:

```csv
donor_id,hla
synthetic_donor,HLA-A*02:01
synthetic_donor,HLA-DRA*01:01/HLA-DRB1*04:01
```

Use donor IDs that exactly match the receptor input. Unreported alleles do not prove incompatibility: typing tables may be incomplete. Before combining samples, prefix reused cell barcodes with sample/library identifiers.

Use colon-delimited HLA names. Legacy `A*0201` is not converted to `A*02:01`. Preserve known higher-resolution fields and expression suffixes. G/P groups and `|`-separated alternatives remain uncertain; no group membership or allele is inferred.

DecoderTCR supports exact class I A/B/C entries and explicit class II pairs present in its pinned sequence reference. Examples of supported pair syntax:

- DR: `HLA-DRA*01:01/HLA-DRB1*04:01`
- DQ: `HLA-DQA1*03:01/HLA-DQB1*02:01`
- DP: `HLA-DPA1*01:03/HLA-DPB1*04:01`

Supply the complete class II pair, not a DRB1 call alone or an inferred combination of DQ chains. Model lookup requires exact supported two-field names; higher-resolution or ambiguous calls remain unresolved. Reference matching can retain broader HLA uncertainty.

## Human and mouse

Choose `--species human` (default) or `--species mouse` for each analysis. Mouse TCR reconstruction requires `setup --species mouse`. Split mixed-species inputs into separate runs and use the corresponding gene calls. Human `HLA-` and mouse `H-2-` names both use the `hla` column and `--hla` flag.

The bundled exact mouse molecules are:

| MHC molecule | Class | Chains supplied by the reference |
|---|---|---|
| `H-2-Kb` | I | Mature heavy chain + mouse beta-2 microglobulin |
| `H-2-Db` | I | Mature heavy chain + mouse beta-2 microglobulin |
| `H-2-IAb` | II | Explicit I-A alpha/beta chains of the b haplotype |
| `H-2-IAk` | II | Explicit I-A alpha/beta chains of the k haplotype |

Mouse class II requires the complete molecule name; partners are not inferred. Cross-species names are rejected. Unsupported molecules need explicit sequences. See the bundled reference and [model notices](model-notices.md) for sources, versions and mature-chain ranges.

### Additional exact mouse MHC sequences

Use `--mhc-reference data/mouse_mhc.json` with the [bundled reference](../src/tcr_workbench/data/mouse_mhc.json) as a template. Retain its schema and supply experimentally supported sequences with sources. The override **replaces the whole reference**, without fallback to bundled entries.

<details>
<summary>Custom reference fields</summary>

The required structure is `schema_version: 1`, `species: "mouse"`, a nonempty `source`, and a `molecules` object keyed by exact canonical H-2 names. Each molecule has `class` (`"I"` or `"II"`), `HLA_a`, `HLA_b`, and its own `source`. For class I these sequences are the mature heavy chain and mouse beta-2 microglobulin; for class II they are alpha and beta. Both chains must contain 50–600 canonical amino acids. Include accessions, sequence versions and residue ranges in `source`. Do not include signal peptides or substitute human beta-2 microglobulin. Analysis does not download missing sequences.

</details>

## Incomplete or ambiguous receptors

Dual-alpha observations remain alternative pairings, not independent cells; do not sum their cell counts. Multiple beta chains can reflect biological or experimental ambiguity. A missing chain alone cannot establish PCR dropout, primer bias or barcode contamination.

Keep the output folder together, including the cell, chain and receptor tables. These retain observations that could not be scored.

## Optional curated reference matching

`screen` requires `reference_id,peptide,source,evidence` and at least one of `cdr3a,cdr3b`. Add known `hla,trav,traj,trbv,trbj`. Reference IDs must be unique; `source` identifies a publication, accession or database release, and `evidence` describes the assay. The shipped reference contains fabricated associations for software testing only.

For optional `--embedding-matching`, also supply `species` (`human` or `mouse`) and all six paired `trav,traj,cdr3a,trbv,trbj,cdr3b` fields. Both searches then restrict references to the requested species; incomplete or ambiguous annotations remain in `embedding_audit.csv`. Embeddings use reconstructed receptor chains only; peptide and MHC annotations are used separately to filter and describe reference evidence. See [embedding matching](usage.md#experimental-embedding-matching).
