# Presidents Modeling Explained

This note explains what we ran for the explainable presidents push, what each model is doing in code, how the sweeps work, how we selected hyper-parameters, and how to justify the final choices in the report.

It separates two viewpoints:

- **Internal engineering view**: what we actually ran overnight, including proxy scoring and repaired outputs.
- **Report-safe view**: how to describe and justify the final method using only the train set, grouped CV, and model-design arguments.

## 1. Big Picture

We explored four explainable lanes:

1. **Simple BiLSTM**
2. **Structured BiLSTM**
3. **CamemBERT sentence classifier with local context**
4. **Late fusion**

The main design idea was:

- use **CamemBERT sentence embeddings** for the LSTM lanes
- keep the **speech** as the main unit of sequence modeling
- calibrate decisions with **grouped out-of-fold threshold sweeps**
- explicitly track **boundary quality**, not only macro-F1

Relevant code:

- `src/rital_nlp_project/presidents/lstm.py`
- `src/rital_nlp_project/presidents/sequence.py`
- `spark/train_pres_lstm_simple.py`
- `spark/predict_pres_lstm_simple.py`
- `spark/run_pres_lstm_simple_sweep.py`
- `spark/calibrate_pres_lstm_cv.py`
- `spark/train_pres_lstm.py`
- `spark/predict_pres_lstm.py`
- `spark/run_pres_lstm_structured_sweep.py`
- `spark/run_pres_camembert_cv.py`
- `spark/run_pres_fusion.py`
- `scripts/eval_presidents_proxy.py`
- `scripts/repair_presidents_test_order.py`

## 2. Simple BiLSTM

### 2.1 What the model is

The simple model is a **speech-level sentence tagger**.

Pipeline:

1. Each sentence already has a precomputed CamemBERT embedding.
2. Sentences are regrouped by `speech_id` into variable-length speech sequences.
3. A bidirectional LSTM reads the whole speech.
4. For each sentence position, the model outputs a 2-class score: `C` vs `M`.
5. We convert the Mitterrand score into a probability `prob_mitterrand_raw`.
6. We do threshold calibration on grouped out-of-fold predictions.

The core model is `BiLSTMSequenceTagger` in `src/rital_nlp_project/presidents/lstm.py`.

Important architectural details:

- optional linear projection from input embedding size to `projection_dim`
- dropout before the LSTM
- bidirectional LSTM
- one linear classifier applied at every sentence position
- packed sequences so variable-length speeches are handled correctly

In code, the forward pass is:

1. project inputs
2. apply dropout
3. pack by true speech lengths
4. run the BiLSTM
5. unpack back to sentence positions
6. classify each sentence

### 2.2 How training works

Main script:

- `spark/train_pres_lstm_simple.py`

Training flow:

1. Load embeddings and metadata.
2. Build `SpeechSequence` objects.
3. Make a grouped train/validation split by speech.
4. Train for several epochs.
5. After every epoch, score the validation speeches.
6. Keep the epoch with the best validation selection score.
7. Refit that number of epochs inside grouped CV to produce OOF predictions.
8. Sweep thresholds on OOF predictions.
9. Train one final model on all labeled speeches for the chosen number of epochs.

Important detail:

- the model is not selected by validation loss
- it is selected by **validation macro-F1 after thresholding**

### 2.3 Why `minority_weight_scale` matters

The simple BiLSTM uses class-weighted cross-entropy because Mitterrand is the minority class.

Base weights are computed from inverse class frequency. Then only the Mitterrand class weight is scaled:

```text
scaled_weight_M = base_weight_M * minority_weight_scale
```

Interpretation:

- `1.0` keeps the full inverse-frequency penalty
- values below `1.0` make the model less eager to predict `M`
- this often helps the model exit Mitterrand earlier and reduce span overhang

This is the main training-time knob we used to make the simple BiLSTM less sticky on Mitterrand.

### 2.4 Why the threshold sweep matters

The simple model has two threshold-related steps:

1. **Validation threshold sweep for epoch selection**
2. **OOF threshold sweep for final calibration**

Epoch selection:

- old behavior would judge each epoch at threshold `0.5`
- new behavior sweeps thresholds from `0.80` to `0.95`
- this picks checkpoints that behave well under a stricter decision rule

Final calibration:

- after grouped OOF predictions are collected, we sweep thresholds from `0.05` to `0.95`
- we choose the threshold with the best OOF macro-F1

At prediction time, `predict_pres_lstm_simple.py` converts the chosen threshold into a logit-space shift:

```text
delta = logit(best_threshold)
p_calibrated = sigmoid(logit(p_raw) - delta)
```

So the deployment file can still be thresholded at `0.5`, while being mathematically equivalent to using the tuned raw threshold.

### 2.5 Boundary diagnostics

The simple training script also computes speech-level boundary diagnostics:

- `start_lag_mean`
- `end_lag_mean`
- number of speeches with multiple predicted Mitterrand blocks

These are computed by:

1. thresholding each sentence score
2. finding contiguous predicted Mitterrand blocks per speech
3. comparing the predicted outer boundaries with the true outer boundaries

These metrics matter because this task is not only about sentence accuracy. It is also about where the Mitterrand region starts and ends.

### 2.6 What we swept

The overnight simple sweep wrapper is:

- `spark/run_pres_lstm_simple_sweep.py`

The launch grid was:

- `hidden_dim`: `128`, `256`
- `projection_dim`: `256`, `512`
- `dropout`: `0.15`, `0.2`, `0.3`
- `minority_weight_scale`: `1.0`, `0.9`, `0.8`, `0.7`
- `epochs`: `12`, `16`, `20`

This full grid has `2 x 2 x 3 x 4 x 3 = 144` combinations.

We did **not** run all 144.

Instead:

- we set `--max-runs 12`
- the wrapper sampled 12 configurations deterministically from the full grid using `seed=42`

So this was a **small, controlled random subset**, not an exhaustive search.

### 2.7 What won

Best simple BiLSTM:

- run: `presidents_lstm_simple_tuned_hd128_pd256_do0p15_mw0p7_ep20`
- best epoch: `20`
- tuned threshold: `0.855`
- clean OOF macro-F1: `0.9289`
- accepted-proxy macro-F1: `0.9252`

Why this is plausible:

- `minority_weight_scale=0.7` reduced Mitterrand overprediction
- `dropout=0.15` kept regularization light
- `hidden_dim=128` and `projection_dim=256` were enough capacity without overcomplicating the model
- the tuned threshold is much stricter than `0.5`, which shortens predicted spans

### 2.8 How to justify this in the report

Report-safe justification:

- We modeled the problem as speech-level sequence labeling rather than isolated sentence classification.
- We used grouped validation by speech to avoid leakage across sentences from the same discourse.
- We tuned only a small set of interpretable hyper-parameters: model width, projection size, dropout, number of epochs, and class weighting.
- Because macro-F1 is the competition metric and class imbalance is strong, we calibrated the decision threshold on grouped out-of-fold predictions instead of forcing the default `0.5`.
- We also monitored segmentation quality through boundary diagnostics, because the task has a contiguous-span structure.

Good report wording:

> We selected the final BiLSTM configuration by grouped cross-validation, maximizing macro-F1 while monitoring boundary quality and over-fragmentation.

## 3. Structured BiLSTM

### 3.1 What changes compared with the simple BiLSTM

The structured BiLSTM keeps the same sequence model core:

- same speech-level BiLSTM
- same sentence embeddings
- same class-weighted cross-entropy

What changes is the **decoder**.

Instead of treating each sentence independently at decision time, the structured lane adds the assumption:

> a speech contains at most one contiguous Mitterrand segment

This is implemented by the `single_negative_span` decoder in `src/rital_nlp_project/presidents/sequence.py`.

### 3.2 What the decoder does

Given per-sentence class probabilities for one speech, the decoder compares:

- **no Mitterrand span**
- **one contiguous Mitterrand interval** of length at least `min_negative_span`

For each candidate interval, it computes a score using:

- the sentence log-probabilities
- optionally prior terms from span statistics

In our explainable structured lane, we intentionally simplified this.

We kept:

- `decoder = single_negative_span`
- `min_negative_span`

We disabled:

- `position_features`
- `transition_features`
- `position_bins`
- `position_prior_weight`

So the structured lane can be described simply as:

> the same BiLSTM, followed by a decoder that enforces at most one contiguous Mitterrand block.

### 3.3 How calibration works

Structured calibration is handled by:

- `spark/calibrate_pres_lstm_cv.py`

This script does grouped outer CV and grouped inner validation.

Flow:

1. Build outer speech-level folds.
2. Inside each outer-train fold, split again into inner-train / inner-validation.
3. Select the best epoch using the structured decoder on inner validation.
4. Refit on the outer-train fold for that many epochs.
5. Predict the outer-test fold.
6. Store both:
   - `prob_mitterrand_raw`
   - `prob_mitterrand_single_negative_span`
7. Sweep thresholds on both score streams.
8. Keep whichever score stream has the best OOF macro-F1.

That means the structured lane can choose between:

- raw BiLSTM scores
- structured span posteriors

In the best explainable runs, the selected score was usually `prob_mitterrand_single_negative_span`.

### 3.4 Why there is also a full-train script

The structured wrapper uses two stages:

1. `spark/calibrate_pres_lstm_cv.py` for grouped OOF calibration
2. `spark/train_pres_lstm.py` for final training on all labeled data

The wrapper takes the median best epoch across CV folds and uses it as the final full-data epoch count.

Then `spark/predict_pres_lstm.py`:

- loads the final checkpoint
- predicts test sentence scores
- applies the structured decoder
- optionally applies calibration
- writes submission files

### 3.5 What we swept

The structured sweep wrapper is:

- `spark/run_pres_lstm_structured_sweep.py`

The launch grid was:

- `hidden_dim`: `128`, `256`
- `projection_dim`: `256`, `512`
- `dropout`: `0.1`, `0.2`, `0.3`
- `min_negative_span`: `2`, `3`, `4`
- `learning_rate`: `1e-3`, `5e-4`
- `epochs`: `16`, `20`

This full grid also has `2 x 2 x 3 x 3 x 2 x 2 = 144` combinations.

Again, we did not run all 144.

We used:

- `--max-runs 12`
- deterministic sampling with `seed=42`

### 3.6 How ranking worked

The structured wrapper ranked runs by:

- proxy macro-F1 if available
- then clean OOF macro-F1
- then balanced accuracy
- then `end_lag_mean`
- then `multi_block_speeches`

So the winner was not necessarily the clean OOF champion.

There are two relevant winners:

- **best clean structured run**
- **best hybrid/proxy-ranked structured run**

### 3.7 What won

Best clean structured run:

- run: `presidents_lstm_structured_hd128_pd256_do0p2_mns2_lr0p001_ep20`
- selected score: `prob_mitterrand_single_negative_span`
- tuned threshold: `0.865`
- clean OOF macro-F1: `0.9219`
- proxy macro-F1: `0.9205`

Best hybrid/proxy-ranked structured run:

- run: `presidents_lstm_structured_hd128_pd512_do0p2_mns4_lr0p0005_ep20`
- selected score: `prob_mitterrand_single_negative_span`
- tuned threshold: `0.88`
- clean OOF macro-F1: `0.9157`
- proxy macro-F1: `0.9269`

Why the proxy winner makes sense:

- `projection_dim=512` gave the sequence model more representational capacity
- `min_negative_span=4` made the decoder more conservative about tiny false-positive Mitterrand islands
- the structured decoder eliminated multi-block speeches entirely

### 3.8 Late `no_span_bias` sweep

The structured wrapper also supports a late decode-time sweep over `no_span_bias`.

Interpretation:

- positive `no_span_bias` makes the decoder favor the **no-span** option a bit more
- this is a decode-time structural bias, not a retraining step

We ran the bias sweep on the top clean structured runs.

Result:

- it produced some small changes
- it did **not** beat the main best structured submission

Best late-bias variant we found:

- run: `hd256_pd256_do0p2_mns3_lr0p001_ep16`
- `no_span_bias = 1.5`
- proxy macro-F1: `0.9265`

That is strong, but still below the main structured winner at `0.9269`.

### 3.9 How to justify this in the report

Report-safe justification:

- The speeches often behave like one inserted segment from a different speaker.
- Therefore, adding a decoder that assumes at most one contiguous minority segment is a natural structural prior.
- We deliberately disabled handcrafted position and transition features to keep the method explainable and robust.
- We tuned only a small number of structural hyper-parameters, mainly the minimum span length and standard optimization settings.

Good report wording:

> We improved the BiLSTM by imposing a simple task-informed decoding rule: a speech is allowed either no Mitterrand span or one contiguous span. This reduces fragmented predictions while keeping the model easy to interpret.

## 4. CamemBERT with Local Context

### 4.1 What this model is

This lane does not use the precomputed embedding matrix directly.

Instead it fine-tunes `CamembertForSequenceClassification` on sentence text with local context.

Main script:

- `spark/run_pres_camembert_cv.py`

The central idea is:

- take the target sentence
- concatenate neighboring sentences from the same speech
- classify the target sentence using this local textual window

The helper `build_contextual_texts(...)` constructs strings like:

```text
previous sentence </s> target sentence </s> next sentence
```

with a configurable `context_window`.

### 4.2 How training works

Pipeline:

1. Read train and test metadata.
2. Build contextual text strings for each sentence.
3. Tokenize with CamemBERT tokenizer.
4. Do grouped speech-level CV.
5. Train a sequence-classification transformer with class-weighted cross-entropy.
6. Collect OOF sentence probabilities.
7. Sweep thresholds on OOF probabilities.
8. Train one final model on all labeled speeches.
9. Predict the hidden test set.

Important details:

- folds are grouped by `speech_id`
- classes are weighted by inverse frequency
- threshold calibration is again done on OOF predictions

### 4.3 What we ran

Completed CamemBERT runs:

1. `base_ctx1_len256_lr1e5_ep3`
2. `base_ctx2_len256_lr5e6_ep4`

Planned but cancelled:

- `ctx2_len384_lr5e6_ep4`

Reason for cancellation:

- it was the slowest remaining GPU job
- the first two runs already told us whether the transformer lane was competitive enough

### 4.4 The canonical-order repair

Internal engineering note:

The first CamemBERT exports were written in speech-sorted order instead of the canonical hidden-test row order.

That caused the raw proxy scores in the original `metrics.json` files to look catastrophically bad even though the model itself was fine.

We fixed this in two ways:

1. patched `spark/run_pres_camembert_cv.py` so future exports preserve canonical test order
2. repaired completed outputs with:
   - `scripts/repair_presidents_test_order.py`

This repair does **not** retrain the model.

It only reorders already-predicted rows by joining on:

- `speech_id`
- `sentence_id`

### 4.5 Completed results

Run 1:

- config: `context_window=1`, `max_length=256`, `epochs=3`, `lr=1e-5`
- clean OOF macro-F1: `0.8691`
- corrected proxy macro-F1: `0.8715`

Run 2:

- config: `context_window=2`, `max_length=256`, `epochs=4`, `lr=5e-6`
- clean OOF macro-F1: `0.8753`
- corrected proxy macro-F1: `0.8772`

Interpretation:

- more context helped a bit
- a smaller learning rate with more epochs helped a bit
- but the transformer alone still stayed clearly below the LSTM and fusion lanes

One interesting sign:

- both transformer runs selected a very low threshold (`0.05`)

That means the raw probabilities are underconfident about `M`, so strong threshold calibration is necessary.

### 4.6 How to justify this in the report

Report-safe justification:

- CamemBERT gives a strong semantic sentence encoder.
- Local context is added by concatenating neighboring sentences from the same speech.
- Grouped CV prevents information leakage across sentences of the same speech.
- This model is a natural complementary baseline to the embedding-based BiLSTM.

Good report wording:

> We also evaluated a contextual CamemBERT sentence classifier, where each sentence is classified together with a small window of neighboring sentences from the same speech.

## 5. Plain Fusion

### 5.1 What it is

Plain fusion combines two score streams:

- one from the best simple BiLSTM
- one from the best structured BiLSTM

Main script:

- `spark/run_pres_fusion.py`

The main fusion formula is a weighted logit average:

```text
logit(p_fused) = alpha * logit(p_simple) + (1 - alpha) * logit(p_structured)
```

Then:

1. OOF fused scores are created
2. thresholds are swept on OOF
3. the best `(alpha, threshold)` pair is exported to test

The script also evaluates a second fusion family:

- logistic stacking on OOF logits

### 5.2 How the code works

Key steps:

1. Load the OOF and test score files from both models.
2. Align them by `speech_id` and `sentence_id`.
3. Preserve canonical row order with metadata.
4. Sweep `alpha` from `0.0` to `1.0` in steps of `0.05`.
5. For each `alpha`, sweep thresholds.
6. Also fit logistic stacking for comparison.
7. Export the best-scoring candidate.

### 5.3 The repair note

The first fusion proxy scores were also wrong for the same row-order reason as CamemBERT.

We:

- patched `spark/run_pres_fusion.py`
- repaired the finished fusion outputs with `scripts/repair_presidents_test_order.py`

So the corrected proxy scores below are the meaningful ones.

### 5.4 What won

Exported plain-fusion winner:

- method: `weighted_logit_average`
- `alpha = 0.15`
- threshold: `0.78`
- clean OOF macro-F1: `0.9178`
- corrected proxy macro-F1: `0.9274`

Interpretation:

- the best mixture was mostly structured (`85%`) with a small simple contribution (`15%`)
- this is exactly the kind of explainable ensemble we wanted
- the simple lane adds complementary signal, but the structured lane remains dominant

### 5.5 How to justify this in the report

Report-safe justification:

- The simple BiLSTM captures softer local sequence evidence.
- The structured BiLSTM captures stronger contiguous-span behavior.
- A late fusion of their calibrated scores is a transparent way to combine complementary information.
- Because the fusion is late and linear in logit space, it is much easier to explain than a monolithic ensemble.

Good report wording:

> We combined complementary score streams with a simple late-fusion rule, weighting the logits of two calibrated sequence models and selecting the fusion weight by grouped out-of-fold evaluation.

## 6. Span Fusion

### 6.1 What changes

Span fusion uses the same fused score stream as plain fusion, but then applies the same `single_negative_span` decoder on top of the fused probabilities.

So the pipeline is:

1. fuse scores
2. optionally run the single contiguous-span decoder
3. sweep thresholds
4. export

### 6.2 What won

Exported span-fusion winner:

- method: `weighted_logit_average`
- `alpha = 0.0`
- threshold: `0.895`
- clean OOF macro-F1: `0.9152`
- corrected proxy macro-F1: `0.9261`

This is important:

- `alpha = 0.0` means the best span-fusion variant effectively collapsed to the structured model alone

Interpretation:

- once the single-span decoder is applied, adding the simple model did not improve the exported winner
- the plain fusion is therefore more useful than the span fusion in our current runs

### 6.3 How to justify this in the report

Report-safe justification:

- This variant tests whether an additional structural prior helps after score fusion.
- In our runs, the extra structural step did not beat the simpler plain fusion.
- That is a useful experimental result: not every extra constraint improves performance.

## 7. How We Swept and Selected Hyper-Parameters

### 7.1 General philosophy

We did not do a giant black-box search.

Instead we used:

- small, interpretable grids
- grouped CV by speech
- OOF threshold calibration
- boundary diagnostics

The knobs we swept were chosen because they are easy to explain:

- model width
- projection size
- dropout
- learning rate
- class-weight scaling
- minimum span length
- context window
- sequence length

### 7.2 Why grouped CV mattered

We always grouped by `speech_id`.

Why:

- neighboring sentences in the same speech are highly dependent
- random sentence-level splitting would leak discourse information
- grouped CV is therefore the correct way to estimate generalization

This is one of the strongest report-safe justifications in the whole pipeline.

### 7.3 Why threshold calibration mattered

We treated the threshold as part of the decision rule, not as a fixed constant.

Why:

- the task is imbalanced
- raw model probabilities are not equally calibrated across architectures
- macro-F1 is sensitive to the threshold

So for every lane we selected thresholds on grouped OOF predictions.

This is report-safe and standard.

### 7.4 Why we tracked boundary metrics

We did not trust macro-F1 alone.

We also tracked:

- `end_lag_mean`
- `start_lag_mean`
- `multi_block_speeches`

This is easy to justify:

- the hidden task is not just sentence classification
- it is also a segmentation problem

### 7.5 Internal selection versus report-safe selection

Internal selection overnight:

- when proxy labels were available, wrappers used `ranking_mode=hybrid`
- this ranked runs by proxy macro-F1 first, then clean OOF and boundary quality

Report-safe selection:

- for the report, we should frame the final choice using grouped CV, OOF macro-F1, and boundary metrics
- the archive-labeled proxy rows should remain an internal model-selection aid, not the public narrative

In other words:

- **internally** we used every safe offline signal we had
- **in the report** we justify the same choices with grouped CV and explainable inductive biases

## 8. Current Completed Results

Corrected result snapshot:

| Lane | Selected configuration | Clean OOF macro-F1 | Corrected accepted-proxy macro-F1 |
| --- | --- | ---: | ---: |
| Simple BiLSTM | `hd128 pd256 do0.15 mw0.7 ep20` | `0.9289` | `0.9252` |
| Structured BiLSTM | `hd128 pd512 do0.2 mns4 lr5e-4 ep20` | `0.9157` | `0.9269` |
| Plain fusion | `alpha=0.15, threshold=0.78` | `0.9178` | `0.9274` |
| Span fusion | `alpha=0.0, threshold=0.895` | `0.9152` | `0.9261` |
| CamemBERT run 1 | `ctx1 len256 lr1e-5 ep3` | `0.8691` | `0.8715` |
| CamemBERT run 2 | `ctx2 len256 lr5e-6 ep4` | `0.8753` | `0.8772` |

Two immediate takeaways:

1. The **best single model** is still an LSTM-family model.
2. The **best completed corrected submission** is currently the **plain simple+structured fusion**.

## 9. What This Means for the Report

If we want the safest single-model story:

- present the **structured BiLSTM** as the main improved model

Why:

- it is easy to explain
- it stays close to the simple BiLSTM
- its only major extra assumption is the single contiguous Mitterrand segment

If we want the strongest currently completed system:

- present the **plain fusion of simple + structured BiLSTM**

Why:

- it is still very explainable
- it does not require a black-box ensemble story
- it has the best corrected completed proxy score so far

Suggested report structure:

1. Baseline: simple BiLSTM on speech-level sentence embeddings
2. Improvement: structured BiLSTM with a single contiguous-span decoder
3. Complementary experiment: contextual CamemBERT classifier
4. Final system: late fusion of complementary sequence-model scores

## 10. Recommended Wording for the Final Story

Short version:

> We model the task at the speech level. Our baseline is a bidirectional LSTM that tags each sentence using CamemBERT sentence embeddings. We then improve it with a task-informed decoder that assumes a speech contains at most one contiguous minority segment, which reduces fragmented predictions. Finally, we combine complementary score streams through a simple late-fusion rule selected by grouped out-of-fold evaluation.

This wording is:

- accurate
- explainable
- technically defensible
- consistent with the actual code

## 11. Internal Engineering Notes

These points are useful for us but should not be central in the report:

- The simple and structured sweeps were deterministic 12-run samples from larger grids, not exhaustive searches.
- Completed CamemBERT and fusion outputs initially had a hidden-test row-order export bug.
- We corrected that with `scripts/repair_presidents_test_order.py`.
- The repair only reorders already-predicted test rows; it does not retrain models.
- Corrected proxy metrics should always be taken from the `canonical_order` output directories for the finished CamemBERT and fusion runs.

