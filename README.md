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
