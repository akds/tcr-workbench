# Licenses and data sources

Workbench code is [MIT licensed](../LICENSE). Model weights, dependencies and reference data retain their own terms; they are downloaded separately during setup.

## DecoderTCR models and Apple support

The default weights for DecoderTCR 300M, 600M and 6B are pinned to the `biohub/DecoderTCR` V0.3 release. The release metadata declares MIT. Newer and custom checkpoints retain the terms supplied with those weights.

The Apple implementation adapts code or model conventions from MLX-LM, Biohub/esm, DecoderTCR and Meta ESM. Their source revisions and required notices are retained in [THIRD_PARTY_NOTICES.md](../esmc-mlx/THIRD_PARTY_NOTICES.md), the [license directory](../esmc-mlx/licenses/) and [source pins](../esmc-mlx/references/sources.json). Keep these notices when redistributing the code or converted weights.

## TCR reconstruction data

DecoderTCR uses Stitchr and IMGT germline sequences to reconstruct full receptor chains. Stitchr and IMGTgeneDL are MIT-licensed software; downloaded germline records have their own terms. Retain the data source and release information when sharing derived data.

Example receptor components come from the DecoderTCR example. Cell identifiers and reference associations are synthetic. See [examples](../examples/README.md) before using them.

## Mouse MHC reference

The [bundled mouse MHC sequences](../src/tcr_workbench/data/mouse_mhc.json) are adapted from reviewed UniProtKB records by **The UniProt Consortium**, retrieved 2026-09-19, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([UniProt license](https://www.uniprot.org/help/license)). Signal peptides were removed at the annotated boundaries; the remaining mature-chain residues are unchanged.

The reference includes H-2-Kb (P01901 + P01887), H-2-Db (P01899 + P01887), H-2-IAb (P14434 + P14483) and H-2-IAk (P01910 + P06343). Each entry records its accessions, sequence versions, residue ranges and source links. Class I molecules use mouse beta-2-microglobulin; class II entries contain both chains. The I-Ab beta-chain assignment is supported by its [original sequencing paper](https://pubmed.ncbi.nlm.nih.gov/6411350/) and [structure mapping](https://www.rcsb.org/structure/6MKD).

UniProt and germline-data terms apply independently of Workbench's MIT license.
