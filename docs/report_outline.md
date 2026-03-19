# Report Outline

## 1. Problem Setup

- Task 1: presidents sentence classification (`Mitterrand=-1`, `Chirac=1`)
- Task 2: movie-review sentiment classification (`negative=0`, `positive=1`)
- Motivation for evaluating both sparse lexical models and dense embeddings

## 2. Data And Preprocessing

- Describe the raw formats:
  - movies training set: one folder per class
  - movies hidden test: one review per line
  - presidents training set: tagged sentences with `<speech_id:sentence_id:label>`
  - presidents hidden test: tagged sentences with `<speech_id:sentence_id>`
- Explain the two preprocessing modes:
  - `classic`: lowercasing, punctuation cleanup, digit removal, stopword removal, optional stemming/lemmatization
  - `bert`: keep most surface form information, remove only obvious artifacts like URLs/emails/phones
- Mention the French preprocessing detail explicitly:
  - stopwords are French-specific and normalized with `unidecode`, so accented and unaccented forms are treated consistently

## 3. Evaluation Protocol

- Movies:
  - stratified train/test split
  - 5-fold CV for model comparison
- Presidents:
  - contiguous sequential holdout
  - blocked CV instead of random shuffling
  - macro-F1, balanced accuracy, and minority-class recall are the main metrics because the task is imbalanced
- Mention that presidents smoothing is only considered within `speech_id` boundaries

## 4. Baselines And Variants

### Movies

- Count 1-2 grams + MultinomialNB
- TF-IDF word 1-2 grams + LogisticRegression
- TF-IDF char `char_wb` 3-5 grams + LinearSVC
- Hybrid word + char TF-IDF + LogisticRegression
- Word2Vec pooling variants
- Transformer embeddings + linear classifiers

### Presidents

- TF-IDF word 1-2 grams + LogisticRegression
- Count word 1-3 grams + MultinomialNB
- Hybrid word + char TF-IDF + LogisticRegression
- Balanced linear models
- CamemBERT embeddings + LogisticRegression / LinearSVC
- Group-aware speech smoothing on embedding scores

## 5. Main Results

Current validated results from this branch:

| Task | Model | Test macro-F1 | Notes |
|---|---|---:|---|
| Movies | TF-IDF word 1-2 grams + LogisticRegression | 0.8725 | strongest overall classical model |
| Movies | Word2Vec SIF + LinearSVC | 0.6875 | working but clearly worse |
| Movies | MiniLM embeddings + LogisticRegression | 0.6950 | worse than sparse lexical baseline |
| Presidents | Count word 1-3 grams + MultinomialNB | 0.7012 | best classical baseline |
| Presidents | CamemBERT embeddings + LogisticRegression | 0.7359 | best unsmoothed embedding result |
| Presidents | CamemBERT embeddings + speech-wise smoothing (`sigma=0.7`) | 0.7516 | best validated holdout configuration |

## 6. Error Analysis

- Compare minority-class recall across presidents models
- Show representative false positives / false negatives
- For movies, inspect reviews with heavy sarcasm, genre-specific language, or long mixed sentiment
- For presidents, inspect transitions inside speeches and borderline institutional vocabulary

## 7. Final Model Choice

- Movies final submission:
  - `movies_tfidf_word12_logreg`
- Presidents final submission:
  - `presidents_embeddings_logreg` on CamemBERT features
  - speech-wise score smoothing with `sigma=0.7`
- Classical presidents NB kept as fallback because it has no heavy dependency stack

## 8. Reproducibility

- Cleaning: `src/main.py prepare`
- Embedding extraction: `src/main.py embed`
- Benchmarking: `src/main.py benchmark`
- Hidden predictions: `src/main.py predict`
- Mention that local corpora and outputs are stored under gitignored `Dataset/`
