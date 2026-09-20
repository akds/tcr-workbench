# Example inputs

Use these files as column templates or run them with the [workflow examples](../docs/usage.md). Receptor components come from the public [DecoderTCR example](https://github.com/Biohub/DecoderTCR/blob/3e3d9889d26d79635f674940ddab039c4dc6f9f9/README.md). Cells, donor identifiers and reference associations are synthetic; none of these files provides experimental binding labels.

| File | Contents |
|---|---|
| `panel.csv` | Two nine-residue class I peptides |
| `panel_class_ii.csv` | A 15-residue peptide with explicit DR, DQ and DP contexts |
| `paired.csv` | Three cells: two paired observations and one missing beta chain |
| `contigs_10x.csv` | The same cells as processed 10x-style contigs |
| `rearrangements.airr.tsv` | The same cells as AIRR rearrangements |
| `donors.csv` | MHC typing for the example donor |
| `reference_synthetic.csv` | A fabricated receptor–antigen association |

Check the paired input without running a model:

```sh
python3 tcr.py validate --input examples/paired.csv --format paired --out results/input_check
```

The incomplete cell stays in the output. The two complete observations share a receptor identity.

To try reference matching without model weights:

```sh
python3 tcr.py setup --core-only
python3 tcr.py screen --input examples/paired.csv --format paired \
  --reference examples/reference_synthetic.csv --panel examples/panel.csv \
  --donors examples/donors.csv --out results/reference_demo
```

Open **`results/reference_demo/report.html`**. Replace the fabricated reference with a sourced, curated table for research. [Embedding matching](../docs/usage.md#experimental-embedding-matching) requires full model setup.
