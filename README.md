# RITAL NLP Project

This repository now provides a runnable pipeline for both supervised tasks in the RITAL project:

1. `presidents`: sentence-level speaker classification (`Mitterrand=-1`, `Chirac=1`)
2. `movies`: movie-review sentiment classification (`negative=0`, `positive=1`)

The local `Dataset/` tree is intentionally gitignored. Raw corpora, cleaned parquet files, embeddings, trained Word2Vec models, benchmark outputs, and hidden-test predictions all live there without being committed.

## Install

Base environment:

```bash
uv sync
uv pip install --python .venv/bin/python -e .
```

Optional extras:

```bash
uv pip install --python .venv/bin/python -e '.[preprocess,word2vec]'
uv pip install --python .venv/bin/python -e '.[embeddings]'
```

Runtime assets:

```bash
env PYTHONPATH=src .venv/bin/python scripts/setup_nltk.py
env PYTHONPATH=src .venv/bin/python scripts/setup_spacy_models.py
```

Notes:

- French preprocessing is already handled. The presidents pipeline uses the French stopword set and now normalizes accented and unaccented forms consistently before removal.
- spaCy is only needed when lemmatization is enabled.
- `torch` and `transformers` are only needed for the `embed` command and embedding experiments.

## Data Layout

Suggested local paths:

```text
Dataset/raw/movies1000/
Dataset/raw/presidents/presidents.learn.utf8.txt
Dataset/raw/test/movies/testSentiment.txt
Dataset/raw/test/presidents/corpus.tache1.test.utf8
```

## Prepare Cleaned Data

Training sets:

```bash
env PYTHONPATH=src .venv/bin/python src/main.py prepare \
  --task movies \
  --split train \
  --input-path Dataset/raw/movies1000 \
  --output-path Dataset/clean/movies_clean \
  --pipeline-mode classic \
  --no-lemmatize

env PYTHONPATH=src .venv/bin/python src/main.py prepare \
  --task presidents \
  --split train \
  --input-path Dataset/raw/presidents/presidents.learn.utf8.txt \
  --output-path Dataset/clean/presidents_clean \
  --pipeline-mode classic \
  --no-lemmatize
```

Hidden test sets:

```bash
env PYTHONPATH=src .venv/bin/python src/main.py prepare \
  --task movies \
  --split test \
  --input-path Dataset/raw/test/movies/testSentiment.txt \
  --output-path Dataset/clean/movies_test_clean \
  --pipeline-mode classic \
  --no-lemmatize

env PYTHONPATH=src .venv/bin/python src/main.py prepare \
  --task presidents \
  --split test \
  --input-path Dataset/raw/test/presidents/corpus.tache1.test.utf8 \
  --output-path Dataset/clean/presidents_test_clean \
  --pipeline-mode classic \
  --no-lemmatize
```

The prepared presidents test parquet keeps `speech_id` and `sentence_id`. The movies test parquet keeps `review_id`.

## Benchmark Experiments

List available experiments:

```bash
env PYTHONPATH=src .venv/bin/python src/main.py list-experiments
```

Classical text benchmarks:

```bash
env PYTHONPATH=src .venv/bin/python src/main.py benchmark \
  --task movies \
  --data-path Dataset/clean/movies_clean.parquet \
  --feature-kind text \
  --output-csv Dataset/out/movies_text_benchmark.csv

env PYTHONPATH=src .venv/bin/python src/main.py benchmark \
  --task presidents \
  --data-path Dataset/clean/presidents_clean.parquet \
  --feature-kind text \
  --output-csv Dataset/out/presidents_text_benchmark.csv
```

Word2Vec:

```bash
env PYTHONPATH=src .venv/bin/python src/main.py train-word2vec \
  --data-path Dataset/clean/movies_clean.parquet \
  --output-path Dataset/models/movies_word2vec.model

env PYTHONPATH=src .venv/bin/python src/main.py benchmark \
  --task movies \
  --data-path Dataset/clean/movies_clean.parquet \
  --word2vec-path Dataset/models/movies_word2vec.model \
  --experiment movies_word2vec_sif_linearsvc \
  --output-csv Dataset/out/movies_word2vec.csv
```

Transformer embeddings:

```bash
env PYTHONPATH=src .venv/bin/python src/main.py embed \
  --data-path Dataset/clean/movies_clean.parquet \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --pooling mean \
  --batch-size 16 \
  --max-length 256 \
  --output-path Dataset/embeddings/movies_minilm.npy

env PYTHONPATH=src .venv/bin/python src/main.py benchmark \
  --task movies \
  --data-path Dataset/clean/movies_clean.parquet \
  --embeddings-path Dataset/embeddings/movies_minilm.npy \
  --feature-kind embeddings \
  --output-csv Dataset/out/movies_embeddings.csv
```

## Predict Hidden Test Sets

Movies labels as plain submission text:

```bash
env PYTHONPATH=src .venv/bin/python src/main.py predict \
  --task movies \
  --experiment movies_tfidf_word12_logreg \
  --train-data-path Dataset/clean/movies_clean.parquet \
  --predict-data-path Dataset/clean/movies_test_clean.parquet \
  --output-path Dataset/out/movies_test_predictions.txt
```

Presidents scores as plain submission text:

```bash
env PYTHONPATH=src .venv/bin/python src/main.py predict \
  --task presidents \
  --experiment presidents_count_word13_nb \
  --train-data-path Dataset/clean/presidents_clean.parquet \
  --predict-data-path Dataset/clean/presidents_test_clean.parquet \
  --output-path Dataset/out/presidents_test_scores.txt
```

Presidents full CSV with metadata and scores:

```bash
env PYTHONPATH=src .venv/bin/python src/main.py predict \
  --task presidents \
  --experiment presidents_count_word13_nb \
  --train-data-path Dataset/clean/presidents_clean.parquet \
  --predict-data-path Dataset/clean/presidents_test_clean.parquet \
  --output-path Dataset/out/presidents_test_predictions.csv \
  --output-kind full
```

## Tests

```bash
env PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
env PYTHONPATH=src .venv/bin/python -m compileall src tests
```
