# RITAL NLP Project

This project contains a prediction model for RITAL NLP project on president and movie datasets.

## Installation

```bash
uv sync
```

## Usage

```bash
uv run python src/main.py
```

### NLTK data requirements

This project requires the following NLTK packages:

- punkt
- stopwords

To download the required NLTK data, run:

```bash
uv run python scripts/setup_nltk.py
```

### Spacy models requirements

- en_core_web_sm
- fr_core_news_md

Possible problems with Jupyter notebooks and uv, run

```
!{sys.executable} -m spacy download en_core_web_sm
!{sys.executable} -m spacy download fr_core_news_md
```

### Cleaning and preprocessing datasets

To preprocess all datasets and save the cleaned data to Parquet files in the `Dataset/` folder, run:

```bash
uv run -- python3 scripts/clean_all.py
```

This will generate cleaned `.parquet` files for each dataset in the `Dataset/` directory, ready for downstream analysis or modeling.

### Installation

This project utilizes two virtual environments managed by `uv`:

- **Python 3.13 environment**: Dedicated to running `gensim`, which is specifically required for the `notebooks/dim_reduction.ipynb` notebook.
- **Python 3.14 environment**: Serves as the primary environment, featuring `pytorch` with CUDA 13.0 support.

To set up these environments, run:

```
uv sync --project envs/py313
uv sync --project envs/py314
uv pip install -e .
```
