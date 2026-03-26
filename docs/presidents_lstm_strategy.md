# Presidents LSTM Strategy And Code Walkthrough

This document explains the full president-task pipeline that was added on the `presidents_lstm_sequence_model` branch:

- how the data is represented
- why we moved from sentence classification to speech-level sequence tagging
- how the BiLSTM is trained
- how the sequence smoothing / constrained decoding works
- how the submission files are generated
- how to compare our submission against another model

The goal is to make the codebase explainable, not just runnable.

---

## 1. Problem Framing

The president task is not a standard independent sentence-classification problem.

The important structural observation is:

- sentences are grouped into speeches
- labels are highly correlated inside a speech
- the minority class usually appears as one contiguous segment, not as random isolated sentences

That is why the branch does **not** treat the task as:

- one sentence in
- one label out
- no sequence structure

Instead, the branch treats each speech as a variable-length sequence of sentence embeddings and predicts one label per sentence with a speech-level BiLSTM.

---

## 2. High-Level Strategy

The overall strategy is:

1. Precompute one embedding vector per sentence.
2. Keep `speech_id` and `sentence_id` so sentences stay grouped by speech.
3. Train a BiLSTM over each speech, not over isolated sentences.
4. Use class-weighted loss because the negative class is rarer.
5. Decode predictions with a structural prior that says a speech is either:
   - all one class
   - or contains one contiguous negative span
6. Export both raw probabilities and structure-aware probabilities for submission.
7. Compare submissions by thresholding probabilities at `0.5` and looking at agreement/disagreement by sentence and by speech.

---

## 3. Code Map

The main files are:

- `src/rital_nlp_project/presidents/lstm.py`
- `src/rital_nlp_project/presidents/sequence.py`
- `src/rital_nlp_project/presidents/__init__.py`
- `spark/train_pres_lstm.py`
- `spark/predict_pres_lstm.py`
- `spark/compare_pres_baselines.py`
- `spark/compare_president_submissions.py`

What each one does:

- `lstm.py`: defines the speech-level BiLSTM model.
- `sequence.py`: holds almost all sequence logic, batching, priors, and decoders.
- `train_pres_lstm.py`: trains and evaluates the BiLSTM on a speech-aware split.
- `predict_pres_lstm.py`: loads a trained checkpoint and writes test submission files.
- `compare_pres_baselines.py`: evaluates logistic-regression baselines with sequence-aware decoders.
- `compare_president_submissions.py`: compares any two submission score files row-by-row.

---

## 4. Data Representation

### 4.1 Label order

The internal label order is defined in `src/rital_nlp_project/presidents/sequence.py`:

```python
PRESIDENT_LABEL_ORDER = (1, -1)
```

This means:

- index `0` in model outputs corresponds to label `1`
- index `1` in model outputs corresponds to label `-1`

This is important because the model predicts arrays shaped:

```text
[batch, time, 2]
```

and the meaning of column `0` and column `1` comes from this order.

### 4.2 SpeechSequence

The core container is `SpeechSequence` in `sequence.py`.

Each `SpeechSequence` stores:

- `speech_id`
- `sentence_ids`
- `embeddings`
- `labels` if available

So the atomic unit is **a whole speech**, not a sentence row.

### 4.3 Loading embeddings and metadata

`load_embeddings_with_metadata(...)` ensures:

- embeddings are 2D
- metadata has `speech_id` and `sentence_id`
- labels are present when required
- row counts match
- sentence ids are monotonic within each speech

That last check matters because all later sequence logic assumes sentences stay in speech order.

### 4.4 Building speech sequences

`build_speech_sequences(...)` groups the dataframe by `speech_id` and builds one `SpeechSequence` per speech.

This is the key transition from flat rows to sequential data.

---

## 5. Why This Is Better Than The Old Setup

The old president pipeline effectively behaved like sentence-level classification with optional post-processing.

The current branch fixes several structural issues:

- it does not drop `speech_id`
- it does not split speeches arbitrarily in the middle
- it does not pretend all sentences are independent
- it can use left and right context through the BiLSTM
- it can enforce a speech-level structure during decoding

So this branch is a move from:

- “classify each sentence and maybe smooth a little”

to:

- “model the speech as a sequence, then decode with a structural prior”

---

## 6. Model Architecture

The current model is `BiLSTMSequenceTagger` in `src/rital_nlp_project/presidents/lstm.py`.

### 6.1 Input

Input shape:

```text
[batch, time, input_dim]
```

where:

- `batch` = number of speeches in the batch
- `time` = number of sentences in the speech
- `input_dim` = embedding dimension, typically `768`

### 6.2 Layers

The model is:

1. optional linear projection: `input_dim -> projection_dim`
2. dropout on projected inputs
3. bidirectional LSTM
4. linear classifier to 2 labels

In the full run we used:

- `hidden_dim = 128`
- `projection_dim = 256`
- `num_layers = 1`
- `dropout = 0.2`

### 6.3 Variable-length speeches

The forward pass uses:

- `pack_padded_sequence(...)`
- `pad_packed_sequence(...)`

This avoids wasting recurrent computation on padded tail positions.

---

## 7. Batching And Loss

### 7.1 Collation

`collate_speech_sequences(...)` pads each batch to the longest speech in that batch and returns:

- `inputs`
- `attention_mask`
- `lengths`
- `speech_ids`
- `sentence_ids`
- `labels` when available

### 7.2 Padding label

Padding labels use:

```python
PAD_LABEL_INDEX = -100
```

and training loss ignores that index.

### 7.3 Class weighting

`compute_class_weights(...)` computes inverse-frequency weights from the training speeches.

This matters because label counts are imbalanced. In the labeled training metadata:

- label `1`: `49,890`
- label `-1`: `7,523`

So without weighting, the model would be pulled strongly toward the majority class.

### 7.4 Loss

Training uses:

```python
nn.CrossEntropyLoss(weight=class_weights, ignore_index=PAD_LABEL_INDEX)
```

This gives sentence-level supervision while respecting padded batches.

---

## 8. Training Procedure

Training is handled in `spark/train_pres_lstm.py`.

### 8.1 Split strategy

The split is speech-aware:

- first split speeches into train vs test
- then split the train portion into train vs validation

This is done by `speech_train_test_split(...)`, which tries to preserve total row proportions while keeping whole speeches intact.

### 8.2 Training loop

Each epoch does:

1. forward pass on speech batches
2. cross-entropy loss on sentence labels
3. gradient clipping
4. optimizer step
5. validation under two decoders:
   - `argmax`
   - `single_negative_span`

The selected model checkpoint is the epoch with best validation `f1_macro` under `single_negative_span`.

### 8.3 Optimizer

We use:

- `AdamW`
- learning rate `1e-3`
- weight decay `1e-2`
- gradient clip `1.0`

### 8.4 Outputs

The training script writes:

- `history.jsonl`
- `metrics.json`
- `presidents_lstm.pt`
- `validation_predictions.csv`
- `test_predictions.csv`

The main full run is in:

- `Dataset/out/presidents_lstm_full_mps`

### 8.5 Result summary

On the held-out speech-aware split, the best LSTM test result was:

- `argmax` `f1_macro = 0.8906`
- `single_negative_span` `f1_macro = 0.8935`

Compared with the best speech-aware logistic baseline:

- baseline `single_negative_span` `f1_macro = 0.7539`

So the BiLSTM was materially better on the held-out split.

---

## 9. The Smoothing / Decoding Strategy

This is the most important part to explain well.

### 9.1 Raw probabilities

The model outputs per-sentence probabilities:

```text
P(label=1 | sentence, speech context)
P(label=-1 | sentence, speech context)
```

If we stop there, decoding is simple `argmax` per sentence.

That gives the raw output.

### 9.2 Why raw argmax is not enough

Raw argmax can produce:

- isolated spikes
- very short label flips
- implausible `+ - + - +` patterns inside one speech

But the actual task structure looks more like:

- all sentences same label
- or one contiguous block from the minority class inside a larger speech

### 9.3 Single-span prior

`fit_single_span_prior(...)` learns from labeled speeches:

- how often a speech has no negative span
- how often it has one
- how long negative spans usually are

This produces a `SingleSpanPrior` with:

- `no_span_log_prob`
- `span_log_prob`
- `span_length_log_probs`

### 9.4 Hard constrained decoding

`decode_single_negative_span(...)` searches over:

- the “no negative span” case
- every possible contiguous negative interval `[start:end)`

For each candidate, it computes a score:

- base score if everything stays positive
- plus the evidence gained by flipping the span to negative
- plus prior terms if provided

Then it picks the single best configuration.

So this decoder enforces:

- no negative span
- or exactly one negative span

This is a structural decoder, not just local smoothing.

### 9.5 Posterior probabilities

`single_negative_span_posteriors(...)` goes one step further.

Instead of only picking the best span, it:

1. enumerates all allowed span configurations
2. converts their scores into normalized weights
3. sums the weights of all spans that cover each sentence

That yields a posterior probability per sentence under the single-span model.

This is what powers the “single negative span” probability submission file.

### 9.6 Difference from Gaussian smoothing

The older baseline code also supports Gaussian smoothing in `compare_pres_baselines.py`.

That older smoothing:

- blurs probabilities along the time axis
- does not enforce any discrete speech structure

The new single-span method:

- is discrete and structure-aware
- uses a prior learned from training speeches
- is much stronger than simple Gaussian blur

So when you explain it, say:

> We did not just smooth probabilities numerically. We used a sequence decoder that assumes speeches are globally structured.

---

## 10. Submission Generation

Submission generation is in `spark/predict_pres_lstm.py`.

### 10.1 What it loads

It loads:

- the trained checkpoint
- the full labeled dataset to fit the span prior
- the unlabeled president test embeddings
- the unlabeled president test metadata

### 10.2 What it exports

It writes:

- detailed aligned predictions
- raw probability submission
- inverted raw probability submission
- single-span probability submission
- inverted single-span probability submission
- hard-label submission

Main output directory:

- `Dataset/out/presidents_lstm_submission_checkpoint`

### 10.3 Why inverted copies exist

The repo had a historical convention that mapped label `1` to `C`.
Later we realized the challenge interpretation might expect the opposite probability orientation for submission.

So `predict_pres_lstm.py` now writes both:

- original orientation
- inverted orientation

This keeps the orientation assumption explicit rather than hidden.

---

## 11. What The Two Main Submission Files Mean

### `submission_prob_raw_positive_inverted.csv`

This file is:

- raw sentence-level LSTM score
- no structural decoder
- inverted for the submission mapping we used later

### `submission_prob_single_negative_span_inverted.csv`

This file is:

- sequence-aware posterior score under the single-negative-span model
- also inverted for the same mapping

The difference between them is not just numeric scaling.

The single-span file incorporates:

- a prior on whether a speech contains a minority segment
- a prior on how long that segment should be
- a global search over the speech

### What it changed in practice

Comparing those two LSTM score files directly:

- agreement at threshold `0.5`: `99.27%`
- threshold flips: `199 / 27,162`
- raw predicts the positive president on `4,224` sentences
- single-span predicts the positive president on `4,137`

So the single-span decoder mainly removes isolated local positive spikes.

It is more conservative and more structured.

---

## 12. Comparing Against Another Model

To compare our submission with another score file, use:

- `spark/compare_president_submissions.py`

This script:

1. loads two score files
2. thresholds each one at `0.5`
3. assigns president names
4. aligns them with `speech_id`, `sentence_id`, and text
5. exports:
   - sentence-level comparison CSV
   - disagreement-only CSV
   - speech-level summary CSV
   - 2x2 crosstab
   - agreement heatmap
   - lowest-agreement speeches plot
   - sequence plot for the most divergent speeches

### Comparison with the head model

Using your colleague’s file:

- `Dataset/out/ekatModels/presidents_test_pred_head.csv`

and our structure-aware inverted LSTM submission:

- agreement: `87.85%`
- disagreements: `3,300`
- head model positive count: `1,911`
- LSTM positive count: `4,137`

The main pattern is that the LSTM is much more willing to predict the positive president than the head model.

Outputs are in:

- `Dataset/out/presidents_model_compare`

### Comparison of raw vs single-span inside our LSTM

Outputs are in:

- `Dataset/out/presidents_lstm_raw_vs_span_compare`

This is useful when explaining what the smoothing step actually does.

---

## 13. Commands To Reproduce

Train the BiLSTM:

```bash
uv run --project envs/py314 python spark/train_pres_lstm.py \
  --output-dir Dataset/out/presidents_lstm_full_mps
```

Run the trained checkpoint on the unlabeled test set:

```bash
uv run --project envs/py314 python spark/predict_pres_lstm.py \
  --output-dir Dataset/out/presidents_lstm_submission_checkpoint \
  --device mps
```

Compare another submission against ours:

```bash
uv run --project envs/py314 python spark/compare_president_submissions.py \
  --head-prob-path Dataset/out/ekatModels/presidents_test_pred_head.csv \
  --lstm-prob-path Dataset/out/presidents_lstm_submission_checkpoint/submission_prob_single_negative_span_inverted.csv \
  --output-dir Dataset/out/presidents_model_compare
```

Compare raw vs span-smoothed LSTM scores:

```bash
uv run --project envs/py314 python spark/compare_president_submissions.py \
  --head-prob-path Dataset/out/presidents_lstm_submission_checkpoint/submission_prob_raw_positive_inverted.csv \
  --lstm-prob-path Dataset/out/presidents_lstm_submission_checkpoint/submission_prob_single_negative_span_inverted.csv \
  --output-dir Dataset/out/presidents_lstm_raw_vs_span_compare
```

---

## 14. Short Oral Explanation

If you need to explain the whole approach quickly, use this:

> We represent each speech as a sequence of sentence embeddings instead of classifying sentences independently. A BiLSTM predicts one label per sentence while seeing left and right context. Then we decode the probabilities with a speech-level prior that says a speech is either uniform or contains one contiguous minority segment. This gives structure-aware predictions and removes implausible isolated flips. We export both raw probabilities and structure-aware probabilities for submission, and we compare them against other models by thresholding at 0.5 and analyzing agreement sentence by sentence and speech by speech.

---

## 15. Main Takeaway

The branch does three conceptually important things:

1. **Speech-aware modeling** instead of row-wise modeling.
2. **Sequence-aware decoding** instead of naive local thresholding.
3. **Explicit submission and comparison tooling** so the outputs are inspectable and explainable.

That combination is the full strategy.
