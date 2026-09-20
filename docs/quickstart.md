# Quickstart

Install TCR-Workbench, run an example and open its HTML report. For notebooks or hosted inference, visit [TCR Challenge](https://tcrchallenge.org).

Already installed? Go to [the first profile](#4-generate-your-first-hla-only-peptide-profile), [your own inputs](#6-replace-the-examples-with-your-own-inputs) or the [workflow guide](usage.md).

## 1. Download and open the folder

On [akds/tcr-workbench](https://github.com/akds/tcr-workbench), choose **Code → Download ZIP** and extract it. Choose a permanent location before setup. Open Terminal and enter the folder:

```sh
cd ~/Downloads/tcr-workbench-main
python3 --version
```

Substitute your actual folder path. On macOS, you can type `cd ` and drag the folder into Terminal.

### Check Python

You need **Python 3.9 or newer**. If it is missing, install it from [python.org](https://www.python.org/downloads/) or ask your IT team. Linux also needs Python's `venv` support. Setup manages the analysis environment; you do not need to activate it before running commands.

## 2. Choose one installation

**Apple Silicon Mac:**

```sh
python3 tcr.py setup --device apple
```

**CPU on Mac or Linux:**

```sh
python3 tcr.py setup --device cpu
```

**Linux with an NVIDIA GPU and working driver:**

```sh
python3 tcr.py setup --device gpu
```

Setup needs internet access and downloads **DecoderTCR 300M, about 4 GB**. Allow at least **12 GB of free disk space**, plus space for Apple conversion and caches. Your device choice is saved for later commands. Moving the folder afterward requires setting up its environments again.

For mouse experiments, add `--species mouse` to setup and each analysis. NVIDIA execution and Windows setup are untested; on Windows, use WSL/Linux.

### Optional: try reference matching without downloading a model

```sh
python3 tcr.py setup --core-only
python3 tcr.py screen --input examples/paired.csv --format paired \
  --reference examples/reference_synthetic.csv --panel examples/panel.csv \
  --donors examples/donors.csv --out results/quickstart_reference
```

Open **`results/quickstart_reference/report.html`**. These reference associations are fabricated examples. Model scoring and experimental embedding matching require full setup.

## 3. Check your installation

```sh
python3 tcr.py doctor
```

Follow any reported setup instructions. A core-only installation will report that DecoderTCR is not configured. See [troubleshooting](setup.md#common-problems) if needed.

## 4. Generate your first HLA-only peptide profile

Run from the folder containing `tcr.py`:

```sh
python3 tcr.py pmhc-profile --hla 'HLA-A*02:01' --length 9 \
  --out results/quickstart_hla
```

Keep the quotes around HLA names containing `*`. The first analysis may take longer while the model is prepared.

Open **`results/quickstart_hla/report.html`** in your browser. It shows amino-acid preferences for a nine-residue peptide conditioned on HLA-A*02:01.

| File | Contents |
|---|---|
| `report.html` | Heatmap, sequence logo, interpretation and run details |
| `profile.csv` | Amino-acid probabilities at each peptide position |
| `pssm.csv` | Position-specific log-odds scores |

The logo runs from the peptide N terminus to C terminus. Taller stacks indicate more focused preferences; larger letters indicate higher probabilities. It is a model prediction, not a measured binding motif.

## 5. Score a peptide panel and a small repertoire

```sh
python3 tcr.py pmhc-score --panel examples/panel.csv \
  --out results/quickstart_peptides

python3 tcr.py repertoire-score --input examples/paired.csv --format paired \
  --panel examples/panel.csv --donors examples/donors.csv \
  --out results/quickstart_repertoire
```

Open **`report.html`** in each output folder. The peptide report ranks the two example peptides and compares their scores with random peptides. The repertoire report includes an incomplete receptor as `Unresolved`.

Higher PLL ranks first within the same receptor, MHC and peptide length. Scores are not binding probabilities, and `Unresolved` does not mean nonbinding. The [workflow guide](usage.md) covers single-TCR scoring, profiles and reference-distribution options.

## 6. Replace the examples with your own inputs

Put your files in `data/` and keep the original preprocessing output. Use the [example files](../examples/README.md) as templates and check the [required columns](input-formats.md).

For a single-donor 10x experiment:

```sh
python3 tcr.py repertoire-score --input data/filtered_contig_annotations.csv \
  --format 10x --donor-id donor_01 --panel data/panel.csv \
  --donors data/donors.csv --out results/donor_01_10x
```

Use `donor_01` in the donor table too. AIRR files use `--format airr`; paired tables use `--format paired`. Multiple donors need the correct `donor_id` on each input record. Prefix reused cell barcodes by sample or library before combining files.

### Mouse experiments

Add `--species mouse` to each analysis and use mouse gene calls and MHC names. Choose a checkpoint with suitable mouse training coverage; the flag does not change the model's training. See [mouse setup and examples](usage.md#human-and-mouse).

## Repeat runs and common fixes

Use a **new `--out` folder for every run**. Keep the whole output folder when sharing results so report links work.

| Problem | Next step |
|---|---|
| `tcr.py` cannot be found | Return to the folder containing `tcr.py`. |
| Download fails | Check internet access and disk space, then rerun setup. |
| Memory check stops a run | Close other large applications or use a smaller model. See [memory planning](upgrading.md#check-memory-and-rough-runtime). |
| Apple/Metal is unavailable | Check that you have an M-series Mac, or set up CPU execution. |
| A row is `Unresolved` | Read its reason and check the input annotations. |

For more options, run `python3 tcr.py COMMAND --help`, or see [setup troubleshooting](setup.md).
