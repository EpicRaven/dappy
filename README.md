# DAPPY: Synthetic Data Generator

**By EpicRaven**

DAPPY generates realistic synthetic tabular data using [SDV](https://sdv.dev/) (the Synthetic Data Vault) without needing a GPU or a heavy local setup. It trains the model on a **temporary private Kaggle kernel**, downloads the results, validates them locally, and then deletes the kernel and any temporary dataset it created, even if you hit Ctrl-C.

Point it at a Kaggle link or a local file, answer a few prompts (or pass flags / a config file), and get back synthetic data, a quality score, privacy checks, charts, and an HTML report.

---

## Why use DAPPY?

- **Share data safely.** Work with a synthetic copy of sensitive or proprietary data (customers, students, transactions) instead of the real thing. Built-in PII anonymisation and leakage checks help you confirm nothing was copied over.
- **No local compute needed.** Training runs on Kaggle's free CPU/GPU kernels, so a modest laptop is enough.
- **Augment small or imbalanced datasets.** Generate more rows, or use conditional sampling to produce more of a rare class (e.g. more churned customers).
- **Prototype and demo without real data.** Build dashboards, ML pipelines and tutorials on realistic data that carries no privacy risk.
- **Know if the output is any good.** Every run reports an SDV quality score, per-column shape scores, statistical tests, and a memorisation check, so you are not trusting synthetic data blindly.
- **Reproducible.** Save your settings to a YAML/JSON config and re-run the exact same job later, or repeat your last run in one command.

---

## Features

### Data sources
- Paste a **Kaggle dataset, competition or notebook link**, an `owner/slug` reference, or a **local file/folder path**.
- Supported formats: `csv`, `tsv`, `parquet`, `xlsx/xls`, `json`, `jsonl/ndjson`.
- If a source has several files, pick one (single table) or several (relational multi-table run).

### Synthesizers
| Option | Model | Notes |
|---|---|---|
| `gan` | CTGAN | Deep-learning; good for complex, mixed-type data |
| `tvae` | TVAE | Variational autoencoder alternative |
| `copula` | GaussianCopula | Fast, no epochs needed, a strong baseline |
| `auto` | All three | Trials each on a sample, keeps the best by quality score |
| (multi-table) | HMASynthesizer | Used automatically when you select several tables |

### Data preparation
- **Target column detection:** reports whether your target is regression or classification (or unsupervised if none).
- **Column exclusion:** drop IDs, timestamps, or anything you do not want modelled.
- **Missing-value policy:** `keep` (SDV models NaNs), `impute` (median / most frequent), or `drop`.
- **PII anonymisation:** auto-detects columns such as email, phone, name, address, SSN, credit card and IP, and replaces them with generated values.
- **Row sampling:** fit on a random subset for very large datasets.

### Advanced controls
- **Constraints:** `Inequality`, `ScalarInequality`, `FixedCombinations`, `Unique`.
- **Hyperparameters:** epochs, batch size, learning rates, discriminator steps, embedding size, copula distribution.
- **Conditional sampling:** "give me N rows where column = value".
- **Train/test validation:** fit on 80%, score against the held-out 20%.
- **Multi-table relationships:** define primary and foreign keys, plus a scale factor.
- **GPU / accelerator selection** (default GPU, T4, P100) and a configurable kernel timeout.

### Validation and privacy checks (run locally)
- Overall **SDV quality score** and per-property scores
- **Column-shape warnings** below a threshold you choose (default 0.70)
- **KS test** (numeric) and **chi-square test** (categorical; needs `scipy`)
- **Leakage check:** counts synthetic rows that are verbatim copies of real rows
- **Distance to Closest Record (DCR):** flags possible memorisation
- SDV diagnostic report

### Visuals and reports
- Data overview: preview, types, missing values, numeric/categorical summaries, duplicates, correlations
- Original vs synthetic charts: overlaid distributions, category shares, correlation heatmaps, quality scores
- A **self-contained HTML report** (tables, scores, charts embedded)

### Delivery
- **Upload to Kaggle** as a new dataset or a new version of an existing one, with title, subtitle, description, tags, license and visibility
- **Save locally:** CSV, plus optional Parquet, Excel, HTML report and a zip bundle
- Saves the fitted `synthesizer.pkl` so you can **re-sample later without refitting**

### Ways to run it
- **Interactive CLI** with menus and tab-completion for column names
- **Batch mode** (`--yes`) using flags or a config file, suitable for scripts and CI
- **Streamlit web UI** (`--ui`)
- **Python API** for notebooks and scripts

### Quality of life
- Live progress with elapsed time (with `rich`) and streamed kernel logs
- Session logs in `~/.dappy/logs/`, last-run settings in `~/.dappy/last_run.json`
- Clear failure output: the tail of the kernel log is printed if the remote job fails

---

## Setup

### 1. Requirements
- Python **3.9+**
- A free [Kaggle](https://www.kaggle.com/) account

### 2. Install

From the folder containing `dappy.py` and `pyproject.toml`:

```bash
pip install .            # core: kaggle, pandas, numpy
pip install ".[all]"     # core + charts + pretty output (matplotlib, seaborn, rich)
```

For development, use an editable install:

```bash
pip install -e ".[all]"
```

Available extras:

| Extra | Installs | Gives you |
|---|---|---|
| `charts` | matplotlib, seaborn | Comparison charts and HTML report charts |
| `pretty` | rich | Styled tables, panels and progress spinner |
| `all` | all of the above | Everything |

Optional packages you can add as needed:

```bash
pip install scipy        # chi-square tests (KS falls back to a built-in version)
pip install pyarrow      # read/write Parquet
pip install openpyxl     # read/write Excel
pip install streamlit    # the web UI
pip install pyyaml       # YAML config files
pip install sdv          # only to re-sample locally from a saved model
```

> The `kaggle` CLI must be installed in the **same Python environment** you run DAPPY with. Check with `kaggle --version`. If that fails, add Python's `Scripts` (Windows) or `bin` folder to your PATH.

### 3. Add your Kaggle credentials

Either place your `kaggle.json` (Kaggle → Settings → Create New Token) at:

- Linux/macOS: `~/.kaggle/kaggle.json`
- Windows: `C:\Users\<you>\.kaggle\kaggle.json`

or set environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

---

## Usage

### Interactive

```bash
dappy
```

The guided flow:

1. Paste a Kaggle link or file/folder path.
2. Enter the target column (Enter for none). DAPPY reports the task type.
3. Review the data overview, then choose columns to exclude, missing-value policy, PII handling, synthesizer and row count. Advanced options are optional.
4. The model trains on a temporary private Kaggle kernel, which is deleted afterwards.
5. Review the warnings, statistical tests, leakage and DCR checks, and optional charts.
6. Deliver: upload to Kaggle, or save locally with optional Parquet, Excel, HTML report and zip.

### One-line batch run

```bash
dappy --dataset owner/slug --method gan --rows 10000 --out ./out --yes
```

### Config file (reproducible runs)

```bash
dappy --config dappy.yaml --yes
```

Example `dappy.yaml`:

```yaml
source: owner/slug
target: churn
method: auto
rows: 20000
missing: impute
anonymize: true
exclude: [customer_id, signup_timestamp]
holdout: true
gpu: true
formats: [csv, parquet]
zip: true
out_dir: ./out
constraints:
  - {type: ScalarInequality, column: age, relation: ">=", value: 18}
conditions:
  - {column_values: {churn: "Yes"}, num_rows: 500}
```

Keys mirror the command-line flags. Multi-table settings (`primary_keys`, `relationships`) and Kaggle delivery (`deliver: kaggle`, `kaggle_meta`) are set in the config file. Save your current settings with `--save-config my_run.yaml`.

### Other commands

```bash
dappy --ui                              # Streamlit web UI
dappy --repeat-last                     # re-run your last saved settings
dappy --resample synthesizer.pkl --rows 5000   # sample from a saved model, no Kaggle needed
```

### As a Python module / notebook

```python
import dappy

real, synthetic, quality = dappy.run()          # interactive session
# dappy.run("owner/slug")                       # start from a source
```

Lower-level control is available through `KaggleKernel`, `run_pipeline` and `compare`.

### Command-line flags

| Flag | Purpose |
|---|---|
| `--source` / `--dataset` / `--csv` | Kaggle link, `owner/slug`, or local path |
| `--file` | Table file to use (repeat for multi-table) |
| `--target` | Target column |
| `--exclude` | Comma-separated columns to drop |
| `--missing` | `keep`, `impute`, or `drop` |
| `--train-rows` | Fit on at most N random rows |
| `--anonymize` | Replace detected PII columns |
| `--method` | `gan`, `copula`, `tvae`, or `auto` |
| `--epochs`, `--rows`, `--scale` | Training epochs, output rows, multi-table scale |
| `--holdout` | Fit on 80%, validate on 20% |
| `--gpu`, `--accelerator` | Use a Kaggle GPU (e.g. `NvidiaTeslaT4`) |
| `--timeout` | Kernel timeout in seconds |
| `--out` | Output directory (default `./dappy_output`) |
| `--formats` | Extra outputs: `parquet,xlsx` |
| `--zip`, `--no-report` | Zip bundle on, HTML report off |
| `--min-shape` | Column-shape warning threshold |
| `--config`, `--save-config` | Load / save settings |
| `--yes`, `-y` | Non-interactive, never prompt |
| `--repeat-last` | Re-run last saved settings |
| `--resample` | Sample from a saved `synthesizer.pkl` |
| `--ui` | Open the Streamlit UI |

---

## Output files

By default, results go to `./dappy_output/`:

| File | Description |
|---|---|
| `synthetic.csv` | The synthetic data (`synthetic_<table>.csv` for multi-table) |
| `quality.json` | SDV quality score, property scores, column shapes |
| `diagnostic.json` | SDV diagnostic results |
| `run_meta.json` | Method used, hyperparameters, timings, SDV version |
| `synthesizer.pkl` | The fitted model, for re-sampling without refitting |
| `dappy_report.html` | Self-contained report with tables and charts |
| `charts/` | PNG charts |
| `synthetic.parquet`, `synthetic.xlsx`, `dappy_bundle.zip` | Optional extras |

---

## Good to know

- **Your data passes through Kaggle.** Local files are uploaded as a *private* temporary dataset so the kernel can read them. DAPPY deletes the dataset and kernel afterward, but if cleanup ever fails it prints the URL so you can remove them manually. Do not use DAPPY on data you are not allowed to upload to a third-party service.
- **Synthetic does not automatically mean private.** Use the leakage and DCR checks, exclude or anonymise identifiers, and review the report before sharing.
- **Kaggle limits apply** (kernel runtime, GPU quota). Use `--train-rows`, fewer epochs, or `copula` for quick runs; raise `--timeout` for big jobs.
- **Multi-table runs** skip column exclusion, row sampling and train/test validation, since these could break key relationships.
- The Streamlit UI handles single-table runs; use the CLI or a config file for multi-table.

---

## License

Add your license here.
