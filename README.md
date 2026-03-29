# RITAL NLP Project

This repository contains our coursework for the RITAL NLP project. It includes:

- a **presidents** task: sentence classification between **Chirac** and **Mitterrand**
- a **movies** task: text classification and exploratory notebook work

The most reproducible and best-documented workflow in this repository is the
**presidents pipeline**. If you only want to understand, rerun, or inspect the
main results, start there.

## What This Repository Contains

- [src/rital_nlp_project](src/rital_nlp_project): reusable Python package code
- [spark](spark): runnable training and inference scripts
- [scripts](scripts): preprocessing and report-building utilities
- [docs](docs): short method notes for the main presidents pipelines
- [notebooks](notebooks): exploratory notebooks and report figures
- [Dataset](Dataset): cleaned data, embeddings, and generated outputs

Useful method notes:

- [docs/presidents_simple_bilstm_main.md](docs/presidents_simple_bilstm_main.md)
- [docs/presidents_camembert_fusion_main.md](docs/presidents_camembert_fusion_main.md)

## Quick Start

If you want the fastest way to check that the repository works in a local copy
that already contains the presidents data artifacts:

```bash
uv sync --project envs/py314
uv run --project envs/py314 python scripts/build_presidents_report_assets.py
```

This rebuilds the report-safe presidents figures and tables in
`Dataset/out/presidents_report_assets/`, **provided that** the expected
presidents output folders are already present in `Dataset/out/`.

If your clone does not include those generated outputs, follow the
`Official Presidents Workflow` section below first, then rebuild the report
assets.

If you also want notebook support and the classic preprocessing / gensim tools:

```bash
uv sync --project envs/py313
```

## Requirements

This project uses [`uv`](https://docs.astral.sh/uv/) to manage environments.

You will need:

- `uv`
- Python `3.13` for the `py313` environment
- Python `3.14` for the `py314` environment

If those Python versions are not already available on your machine, `uv` can
install them:

```bash
uv python install 3.13 3.14
```

## Environments

This repository uses two environments on purpose.

### `envs/py314`

This is the **main environment**. Use it for:

- presidents BiLSTM training and prediction
- presidents CamemBERT training and prediction
- fusion
- report asset generation
- most current analysis scripts

Install it with:

```bash
uv sync --project envs/py314
```

No separate `pip install -e .` step is needed: both `uv` environments include
the root package as an editable dependency.

Notes:

- on Linux, the environment is configured to install CUDA-enabled PyTorch
  (`cu130`) when possible
- on macOS, it falls back to the standard PyTorch package
- the CamemBERT pipeline benefits strongly from a GPU, but the BiLSTM pipeline
  is lighter and easier to rerun

### `envs/py313`

Use this environment for:

- classic preprocessing
- notebooks that depend on `spacy`, `gensim`, or `pyLDAvis`
- some exploratory / older notebook work

Install it with:

```bash
uv sync --project envs/py313
```

## Optional Setup For Notebooks

If you want the environments to appear as Jupyter kernels:

```bash
uv run --project envs/py313 python -m ipykernel install --user --name rital-py313 --display-name "RITAL py313"
uv run --project envs/py314 python -m ipykernel install --user --name rital-py314 --display-name "RITAL py314"
```

Recommended notebook kernels:

- `RITAL py314` for `notebooks/presidents_report_figures.ipynb`
- `RITAL py313` for notebooks that rely on `spacy`, `gensim`, or topic-modeling tools

## Data Layout

The main presidents pipeline expects the following inputs:

- `Dataset/clean/presidents_clean_bert.parquet`
- `Dataset/clean/presidents_test_clean_bert.parquet`
- `Dataset/embeddings/presidents_camembert_base_mean.npy`
- `Dataset/embeddings/presidents_test_camembert_base_mean.npy`

In our local working copy, these files are present. However, a **plain git
clone will not include `Dataset/`**, because the repository currently ignores
that directory.

That means there are two different situations:

- if a supervisor receives a full project folder from us, the data artifacts may
  already be present and the main presidents scripts can run immediately
- if a supervisor performs a clean git clone, they will need the raw presidents
  corpora to be placed under `Dataset/raw/` and must rebuild the metadata and
  embeddings first

Generated outputs are written under `Dataset/out/`.

## Rebuilding Presidents Data Artifacts

If `Dataset/clean/` and `Dataset/embeddings/` are missing, rebuild them from
the raw presidents corpora with:

```bash
uv run --project envs/py314 python spark/build_president_sequence_artifacts.py
```

Expected raw inputs:

- `Dataset/raw/presidents/presidents.learn.utf8.txt`
- `Dataset/raw/test/presidents/corpus.tache1.test.utf8`

This script regenerates:

- `Dataset/clean/presidents_clean_bert.parquet`
- `Dataset/clean/presidents_test_clean_bert.parquet`
- `Dataset/embeddings/presidents_camembert_base_mean.npy`
- `Dataset/embeddings/presidents_test_camembert_base_mean.npy`

and writes a build summary to:

- `Dataset/out/presidents_artifacts_build_summary.json`

So the official presidents models **are** reproducible from raw data with the
tracked code in this repository, but the raw corpus files must still be made
available alongside the clone.

## Two Embedding Paths In This Repository

There are two different embedding workflows in the codebase, and they do **not**
serve the same purpose.

### Official presidents embedding path

The official presidents BiLSTM path uses:

- [build_president_sequence_artifacts.py](/home/paulbeglin/projects/RITAL-NLP-project/spark/build_president_sequence_artifacts.py)

This script:

- parses the raw presidents corpora line by line
- rebuilds the structured metadata with `speech_id`, `sentence_id`, `text`,
  `text_raw`, and `label` when available
- computes one **CamemBERT-base mean-pooled sentence embedding** per sentence
- saves the exact `.parquet` and `.npy` files expected by the clean BiLSTM
  pipeline

This is the path to use when reproducing the main presidents results.

### Older generic embedding utility

The repository also contains:

- [embed.py](/home/paulbeglin/projects/RITAL-NLP-project/spark/embed.py)

This is an older, more generic utility script. It differs from the official
presidents builder in several ways:

- it is not presidents-specific
- it works from an already cleaned parquet input such as `movies_test.parquet`
- it produces **chunk-level CLS embeddings**, not one mean-pooled vector per
  presidents sentence
- it writes a parquet file containing embedded rows, rather than the exact
  `presidents_clean_bert.parquet` plus `.npy` pair used by the clean BiLSTM

So `spark/embed.py` is still useful for experiments, but it is **not** the
canonical rebuild path for the presidents BiLSTM submission pipeline.

## Important Evaluation Note

Some local artifacts under `Dataset/out/` were created for internal diagnostics
and should **not** be treated as official evaluation.

In particular, do **not** use anything under:

- `Dataset/out/presidents_archive_matching_dgx_final`

for report figures, report metrics, or normal model evaluation. Those files are
proxy-label artifacts, not standard ground truth.

## Official Presidents Workflow

The commands below use the canonical output folders that our report scripts
expect. The scripts themselves have simpler defaults, but using these explicit
paths makes the workflow easier to follow and keeps the outputs aligned with the
current report assets.

### 1. Train the speech-level BiLSTM

This model uses fixed **CamemBERT-base sentence embeddings** as input, grouped
by `speech_id`, then runs a bidirectional LSTM over each whole speech.

```bash
uv run --project envs/py314 python spark/train_pres_lstm_simple.py \
  --output-dir Dataset/out/presidents_lstm_simple_main_full
```

Main outputs:

- `Dataset/out/presidents_lstm_simple_main_full/checkpoint.pt`
- `Dataset/out/presidents_lstm_simple_main_full/calibration.json`
- `Dataset/out/presidents_lstm_simple_main_full/metrics.json`
- `Dataset/out/presidents_lstm_simple_main_full/oof_predictions.csv`
- `Dataset/out/presidents_lstm_simple_main_full/threshold_sweep.csv`

### 2. Run the BiLSTM on the test set

```bash
uv run --project envs/py314 python spark/predict_pres_lstm_simple.py \
  --checkpoint-path Dataset/out/presidents_lstm_simple_main_full/checkpoint.pt \
  --calibration-path Dataset/out/presidents_lstm_simple_main_full/calibration.json \
  --output-dir Dataset/out/presidents_lstm_simple_main_submission_full
```

Main outputs:

- `Dataset/out/presidents_lstm_simple_main_submission_full/test_predictions_detailed.csv`
- `Dataset/out/presidents_lstm_simple_main_submission_full/submission_probability.csv`
- `Dataset/out/presidents_lstm_simple_main_submission_full/submission_label_calibrated.csv`

### 3. Train the contextual CamemBERT classifier

This model uses `camembert-base` with a small local context window around each
sentence. Unlike the BiLSTM path, it does **not** consume precomputed `.npy`
sentence embeddings; it reads the presidents parquet metadata and tokenizes the
text directly during training and prediction.

```bash
uv run --project envs/py314 python spark/train_pres_camembert_simple.py \
  --output-dir Dataset/out/presidents_camembert_simple_main_full
```

Main outputs:

- `Dataset/out/presidents_camembert_simple_main_full/model/`
- `Dataset/out/presidents_camembert_simple_main_full/model_config.json`
- `Dataset/out/presidents_camembert_simple_main_full/calibration.json`
- `Dataset/out/presidents_camembert_simple_main_full/metrics.json`
- `Dataset/out/presidents_camembert_simple_main_full/oof_predictions.csv`

### 4. Run CamemBERT on the test set

```bash
uv run --project envs/py314 python spark/predict_pres_camembert_simple.py \
  --model-dir Dataset/out/presidents_camembert_simple_main_full/model \
  --config-path Dataset/out/presidents_camembert_simple_main_full/model_config.json \
  --calibration-path Dataset/out/presidents_camembert_simple_main_full/calibration.json \
  --output-dir Dataset/out/presidents_camembert_simple_main_submission_full
```

### 5. Fuse the BiLSTM and CamemBERT predictions

The fusion is a weighted average in logit space. It is calibrated on grouped
out-of-fold predictions.

```bash
uv run --project envs/py314 python spark/fuse_pres_simple_bilstm_camembert.py \
  --bilstm-train-dir Dataset/out/presidents_lstm_simple_main_full \
  --bilstm-submission-dir Dataset/out/presidents_lstm_simple_main_submission_full \
  --camembert-train-dir Dataset/out/presidents_camembert_simple_main_full \
  --camembert-submission-dir Dataset/out/presidents_camembert_simple_main_submission_full \
  --output-dir Dataset/out/presidents_bilstm_camembert_fusion_main_full
```

Main outputs:

- `Dataset/out/presidents_bilstm_camembert_fusion_main_full/alpha_sweep.csv`
- `Dataset/out/presidents_bilstm_camembert_fusion_main_full/calibration.json`
- `Dataset/out/presidents_bilstm_camembert_fusion_main_full/test_predictions_detailed.csv`
- `Dataset/out/presidents_bilstm_camembert_fusion_main_full/submission_probability.csv`
- `Dataset/out/presidents_bilstm_camembert_fusion_main_full/submission_label_calibrated.csv`

### 6. Rebuild the report figures and tables

```bash
uv run --project envs/py314 python scripts/build_presidents_report_assets.py
```

Outputs are written to:

- `Dataset/out/presidents_report_assets/figures`
- `Dataset/out/presidents_report_assets/tables`
- `Dataset/out/presidents_report_assets/metadata`

This script expects the main presidents output folders to exist, and it also
reuses the saved sweep and comparison artifacts under
`Dataset/out/presidents_explainable_push_20260329/` when those are available.

The interactive version of these figures lives in:

- `notebooks/presidents_report_figures.ipynb`

## How To Read The Main Output Files

The presidents scripts all use the same output conventions.

- `metrics.json`: summary metrics and runtime information
- `calibration.json`: chosen threshold and score shift used for calibration
- `oof_predictions.csv`: predictions on held-out training folds
- `test_predictions_detailed.csv`: per-row predictions on the test set
- `submission_probability.csv`: single-column calibrated probability that the
  row belongs to **Mitterrand**
- `submission_label_calibrated.csv`: final single-column submission labels

Probability convention:

- values near `0` mean **Chirac**
- values near `1` mean **Mitterrand**
- thresholding `submission_probability.csv` at `0.5` reproduces
  `submission_label_calibrated.csv`

## Preprocessing And Utilities

### NLTK data

If you rerun preprocessing or older notebooks, you may need the NLTK resources
used in the project:

```bash
uv run --project envs/py313 python scripts/setup_nltk.py
```

### spaCy models

Some notebooks and preprocessing utilities expect these models:

```bash
uv run --project envs/py313 python -m spacy download en_core_web_sm
uv run --project envs/py313 python -m spacy download fr_core_news_md
```

### Cleaning raw datasets

The cleaning script does **not** clean everything automatically; it expects an
explicit dataset choice and paths.

Example:

```bash
uv run --project envs/py313 python scripts/clean_all.py \
  --dataset presidents \
  --pipeline-mode bert \
  --input-path <raw_input_path> \
  --output-path <clean_output_path>
```

Run `--help` to see the available options:

```bash
uv run --project envs/py313 python scripts/clean_all.py --help
```

### FastText download

If you want to experiment with the older FastText-based material, there is a
helper script:

```bash
uv run --project envs/py313 python scripts/download_fasttext.py
```

This is optional and not needed for the main presidents pipeline.

## Other Scripts And Notebooks

The repository also contains exploratory and older modeling material:

- `notebooks/pres_cls.ipynb`
- `notebooks/movies_cls.ipynb`
- `notebooks/eda.ipynb`
- `notebooks/lda.ipynb`
- `spark/finetune_pres.py`
- `spark/finetune_movies.py`
- `spark/eval_pres.py`
- `spark/eval_movies.py`

These can still be useful for exploration, but they are not the clearest entry
point for reproducing the main presidents results.

## Optional MLflow Tracking

Some of the older transformer training scripts log runs to a local `mlruns/`
directory. If you use those scripts and want to inspect the logs:

```bash
uv run --project envs/py314 mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000` in your browser.

## In Short

If you only want the main reproducible path:

1. install `uv`
2. run `uv sync --project envs/py314`
3. train and predict with the presidents scripts in `spark/`
4. rebuild the figures with `scripts/build_presidents_report_assets.py`
5. inspect the outputs in `Dataset/out/`

If you want the notebooks and classic preprocessing tools as well, also install
`envs/py313`.
