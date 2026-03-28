# Presidents Simple BiLSTM Explainer

This note is meant to be a practical reference for the "simple" presidents BiLSTM pipeline used in:

- [spark/train_pres_lstm_simple.py](/home/paulbeglin/projects/RITAL-NLP-project/spark/train_pres_lstm_simple.py)
- [spark/predict_pres_lstm_simple.py](/home/paulbeglin/projects/RITAL-NLP-project/spark/predict_pres_lstm_simple.py)
- [spark/run_pres_lstm_simple_sweep.py](/home/paulbeglin/projects/RITAL-NLP-project/spark/run_pres_lstm_simple_sweep.py)

It explains:

- the big picture of the model
- how the model is trained
- what the sweeps are doing
- what `--minority-weight-scale` does
- why the tuned model exits Mitterrand earlier and predicts shorter spans
- how `mean start lag` and `mean end lag` are computed

## Big Picture

The simple presidents pipeline is a speech-level sequence tagging model.

Instead of classifying each sentence independently, it does this:

1. represent each sentence with a CamemBERT embedding
2. regroup those sentence vectors by `speech_id`
3. run a BiLSTM over the whole speech
4. output one probability per sentence: `P(Mitterrand)`
5. learn a better decision threshold on labeled data
6. export calibrated predictions for the test set

The key idea is that a sentence is easier to classify when we know what comes before and after it in the same speech.

## Pipeline Overview

```text
sentence text
  -> CamemBERT embedding
  -> group embeddings by speech_id
  -> one sequence per speech
  -> BiLSTM over the sequence
  -> per-sentence raw P(Mitterrand)
  -> threshold calibration on labeled CV
  -> final labels / submission scores
```

## Data Representation

The low-level sequence objects live in [sequence.py](/home/paulbeglin/projects/RITAL-NLP-project/src/rital_nlp_project/presidents/sequence.py).

- `load_embeddings_with_metadata(...)` loads the sentence embedding matrix and the parquet metadata.
- `build_speech_sequences(...)` groups rows by `speech_id`.
- Each speech becomes a `SpeechSequence` containing:
  - `speech_id`
  - `sentence_ids`
  - `embeddings`
  - `labels` when available

So the model input is not one row at a time. It is one speech at a time.

## Model Architecture

The model itself is in [lstm.py](/home/paulbeglin/projects/RITAL-NLP-project/src/rital_nlp_project/presidents/lstm.py).

At a high level:

1. optionally project the input embeddings to `projection_dim`
2. apply dropout
3. run a bidirectional LSTM
4. apply a linear classifier at every sentence position

That means each sentence gets a prediction that uses:

- left context
- the sentence itself
- right context

This is why the model is better than a plain sentence-only classifier for contiguous speech segments.

## What The Model Predicts

For each sentence, the model outputs 2 logits, then a softmax turns them into probabilities.

In the simple pipeline we keep only:

- `prob_mitterrand_raw`

This is the model's raw estimate that the sentence belongs to Mitterrand.

The raw decision rule is:

```text
if prob_mitterrand_raw >= threshold:
    predict Mitterrand
else:
    predict Chirac
```

That rule is implemented in [train_pres_lstm_simple.py](/home/paulbeglin/projects/RITAL-NLP-project/spark/train_pres_lstm_simple.py#L447).

## What Changed Compared With The Older Simpler Setup

The most important changes were not architectural. They were decision and calibration changes:

1. we stopped selecting epochs only with a fixed `0.5` threshold
2. we swept thresholds on validation and OOF predictions
3. we swept `minority_weight_scale`
4. we chose the final run partly using boundary quality, not just F1

These changes made the output less eager to stay in Mitterrand for too long.

## The Three Sweeps

### 1. Configuration sweep

This is done in [run_pres_lstm_simple_sweep.py](/home/paulbeglin/projects/RITAL-NLP-project/spark/run_pres_lstm_simple_sweep.py#L103).

It launches multiple training runs with different hyperparameters:

- `mw100` -> `minority_weight_scale = 1.00`
- `mw090` -> `minority_weight_scale = 0.90`
- `mw075` -> `minority_weight_scale = 0.75`
- `mw060` -> `minority_weight_scale = 0.60`

Then it compares them using:

- `oof_f1_at_tuned_threshold`
- `end_lag_mean`
- `multi_block_speeches`

This ranking happens in [run_pres_lstm_simple_sweep.py](/home/paulbeglin/projects/RITAL-NLP-project/spark/run_pres_lstm_simple_sweep.py#L211).

So the chosen model is not just the one with the best F1. It is also rewarded for better boundary behavior.

### 2. Validation threshold sweep for epoch selection

This is done in [train_pres_lstm_simple.py](/home/paulbeglin/projects/RITAL-NLP-project/spark/train_pres_lstm_simple.py#L408).

At each epoch:

1. the model predicts on the held-out validation split
2. thresholds are swept over a range such as `0.80 -> 0.95`
3. the best threshold for that epoch is found
4. that best threshold score is used to choose the best epoch

So the checkpoint is chosen under a stricter rule than `0.5`.

### 3. OOF threshold sweep for final calibration

After choosing the epoch, the script trains grouped CV folds, collects out-of-fold predictions for every labeled sentence, and sweeps thresholds again in [train_pres_lstm_simple.py](/home/paulbeglin/projects/RITAL-NLP-project/spark/train_pres_lstm_simple.py#L814).

This gives the final calibrated threshold stored in:

- [calibration.json](/home/paulbeglin/projects/RITAL-NLP-project/Dataset/out/presidents_lstm_simple_tuned_mw090/calibration.json)

For the winning run `mw090`, the learned threshold is:

- raw threshold: `0.895`

That is much stricter than `0.5`.

## What `--minority-weight-scale` Does

This is a training-time parameter.

The base class weights are computed from label frequencies in [sequence.py](/home/paulbeglin/projects/RITAL-NLP-project/src/rital_nlp_project/presidents/sequence.py#L351).

Then [train_pres_lstm_simple.py](/home/paulbeglin/projects/RITAL-NLP-project/spark/train_pres_lstm_simple.py#L380) rescales only the Mitterrand weight:

```text
scaled_weights[NEGATIVE_INDEX] *= minority_weight_scale
```

In this dataset, Mitterrand is the minority class, so it already gets a large inverse-frequency weight. The new parameter lets us soften or keep that emphasis.

Interpretation:

- `1.00` means keep the original minority-class emphasis
- `0.90` means still upweight Mitterrand a lot, but a bit less
- `0.75` and `0.60` make the model progressively more conservative about predicting Mitterrand

For the winning `mw090` run, the recorded weights in [metrics.json](/home/paulbeglin/projects/RITAL-NLP-project/Dataset/out/presidents_lstm_simple_tuned_mw090/metrics.json) are approximately:

- Chirac weight: `0.576`
- Mitterrand base weight: `3.785`
- Mitterrand scaled weight: `3.407`

So the model is still strongly encouraged to care about Mitterrand mistakes, just a little less aggressively than before.

## Why The Tuned Model Exits Mitterrand Earlier

There are two main reasons.

### 1. Lower minority emphasis during training

Reducing `minority_weight_scale` from `1.0` to `0.9` makes the model a bit less eager to call uncertain edge sentences "Mitterrand".

That mainly affects borderline sentences near span boundaries.

### 2. Much stricter decision threshold

The biggest effect comes from threshold calibration.

At `0.5`, any sentence with raw `P(Mitterrand) >= 0.5` is labeled as M.

At `0.895`, only very confident Mitterrand sentences stay labeled as M.

This trims the span edges:

- weak early Mitterrand predictions drop back to Chirac
- weak late Mitterrand predictions also drop back to Chirac

So the predicted Mitterrand block:

- starts later
- ends earlier
- becomes shorter

## What Happened In The Winning `mw090` Run

Using [calibration.json](/home/paulbeglin/projects/RITAL-NLP-project/Dataset/out/presidents_lstm_simple_tuned_mw090/calibration.json):

At threshold `0.5`:

- macro-F1: `0.9017`
- predicted Mitterrand sentences: `9069`
- speeches with predicted Mitterrand block: `432`
- speeches with multiple predicted Mitterrand blocks: `65`
- mean start lag: `-2.3023`
- mean end lag: `3.3307`

At tuned threshold `0.895`:

- macro-F1: `0.9187`
- predicted Mitterrand sentences: `7357`
- speeches with predicted Mitterrand block: `395`
- speeches with multiple predicted Mitterrand blocks: `61`
- mean start lag: `-0.3984`
- mean end lag: `0.4118`

This tells us the tuned version is much less "sticky" in Mitterrand.

In plain language:

- before tuning, the model often entered Mitterrand too early and left too late
- after tuning, it is much closer to the true span boundaries

## How `mean start lag` And `mean end lag` Are Computed

These metrics are computed in [train_pres_lstm_simple.py](/home/paulbeglin/projects/RITAL-NLP-project/spark/train_pres_lstm_simple.py#L467).

For each speech:

1. sort sentences by `speech_id`, `sentence_id`
2. build the true Mitterrand mask:
   - `true_label == -1`
3. build the predicted Mitterrand mask:
   - `prob_mitterrand_raw >= threshold`
4. convert both masks into contiguous blocks using `contiguous_blocks(...)`

Then define:

- `true_start` = first index of the first true Mitterrand block
- `true_end` = last index of the last true Mitterrand block
- `predicted_start` = first index of the first predicted Mitterrand block
- `predicted_end` = last index of the last predicted Mitterrand block

The per-speech lag values are:

```text
start_lag = predicted_start - true_start
end_lag = predicted_end - true_end
```

Then the summary metrics are:

- `mean start lag` = average of all valid `start_lag` values
- `mean end lag` = average of all valid `end_lag` values

Only speeches that have both:

- a true Mitterrand block
- a predicted Mitterrand block

contribute to these lag averages.

### Interpretation

- `start_lag < 0`: prediction starts too early
- `start_lag > 0`: prediction starts too late
- `end_lag < 0`: prediction ends too early
- `end_lag > 0`: prediction ends too late

For `mw090` at the tuned threshold:

- `start_lag_mean = -0.3984`
- `end_lag_mean = 0.4118`

So on average the model:

- starts about 0.4 sentences too early
- ends about 0.4 sentences too late

That is much tighter than the old `0.5` behavior.

## One Important Detail About Multiple Blocks

If a speech has multiple predicted Mitterrand blocks, the lag computation uses the outer boundaries:

- first predicted block start
- last predicted block end

It does not compute a separate lag for every block.

That is why the script also reports:

- `speeches_with_multiple_predicted_mitterrand_blocks`

This metric helps us detect when lag values might look acceptable even though the internal block structure is messy.

## How Prediction Export Works

Prediction export lives in [predict_pres_lstm_simple.py](/home/paulbeglin/projects/RITAL-NLP-project/spark/predict_pres_lstm_simple.py#L169).

The script stores the best raw threshold as a logit-space shift:

```text
calibrated_score = sigmoid(logit(raw_score) - delta)
```

where:

- `delta = logit(best_raw_threshold)`

This does not change the idea. It is just a convenient way to export calibrated probability files while still using a simple `0.5` decision threshold downstream.

So:

- threshold tuning is the important modeling step
- `delta` is the export trick that encodes that tuned threshold

## Deeper View Of Training

The training flow in [train_pres_lstm_simple.py](/home/paulbeglin/projects/RITAL-NLP-project/spark/train_pres_lstm_simple.py#L560) is:

1. split labeled speeches into train and validation
2. train the BiLSTM epoch by epoch
3. after each epoch, predict on validation speeches
4. sweep thresholds on validation predictions
5. keep the epoch with the best validation macro-F1
6. retrain the chosen number of epochs across grouped CV folds
7. collect OOF predictions
8. sweep thresholds again on OOF predictions
9. save the best threshold to `calibration.json`
10. retrain on all labeled speeches
11. run the final model on the unlabeled test set

This gives us:

- a chosen epoch
- a chosen threshold
- grouped OOF diagnostics
- final exportable test predictions

## Why This Simple Model Works Reasonably Well

It is simple, but it keeps the most useful structure:

- strong sentence embeddings from CamemBERT
- speech-level contextual modeling from the BiLSTM
- class-weighted loss
- threshold calibration on grouped labeled data

That combination is often enough to get much better boundary behavior than plain sentence-only classification.

## Short Summary

If you only remember a few things, remember these:

1. The model predicts one label per sentence, but it reads the whole speech.
2. `minority_weight_scale` changes how hard training pushes on Mitterrand errors.
3. The biggest shrinkage in predicted Mitterrand spans comes from using a much higher calibrated threshold than `0.5`.
4. `mean start lag` and `mean end lag` are boundary error metrics measured in sentence positions.
5. The winning `mw090` run was selected because it balanced macro-F1 with cleaner, tighter boundaries.
