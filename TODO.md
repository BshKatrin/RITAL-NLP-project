# TODO List

1. **Notebooks**
   - `old_pres_base_models.ipynb`: Extract graphics related to smoothing probabilities for the report. The rest of the notebook is not needed.
   - `evals.ipynb`: Evaluate the FastText model (RAM-heavy).
   - `evals.ipynb`: Perform cross-validation on the movies dataset (time-intensive, 3 minutes for one Word2Vec pipeline). Conduct cross-validation on simple setups (vectorizer and model). If the train/test split is representative, avoid performing cross-validation on the entire dataset.
   - `cls_base_models.ipynb`: Attempt to integrate CLS token embeddings into the Transformer model to merge this notebook with `evals.ipynb`.

2. **README.md**
   - Add proper commands to generate cleaned files.

3. **Code for LSTM Neural Network**
   - Implement code for the `presidents` dataset.

4. **Code for Fine-Tuning**
   - Develop code for fine-tuning on the `movies` dataset.

5. **Report (Overleaf)**
   - LSA: Justify the choice of 1000 components (ensure explained variance is at least 80%).
   - Justify the `min_df`, `max_df` parameters. Include different graphics showing word frequencies.
