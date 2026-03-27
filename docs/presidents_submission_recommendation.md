# Presidents Submission Note

## Recommended File

Current recommended submission:

- [submission_prob_raw_positive_inverted_calibrated.csv](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_lstm_submission_cv_calibrated_full/submission_prob_raw_positive_inverted_calibrated.csv)

Recommended hard-label companion:

- [submission_label_calibrated.csv](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_lstm_submission_cv_calibrated_full/submission_label_calibrated.csv)

Historical fallback if we specifically want the leaderboard-matched variant:

- [delta_4p05/submission_prob_single_negative_span_inverted.csv](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_submission_score_calibration_full19/delta_4p05/submission_prob_single_negative_span_inverted.csv)

## Why This File

Our original full-data LSTM submission was:

- [submission_prob_single_negative_span_inverted.csv](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_lstm_submission_full_data_19ep/submission_prob_single_negative_span_inverted.csv)

That file was too aggressive in predicting `Mitterrand`:

- original full-data submission: `4117` `Mitterrand` rows
- leaderboard best submission (`88.279`): `3267` `Mitterrand` rows

This was the main mismatch. Our model was systematically overpredicting `Mitterrand` relative to the strongest leaderboard files.

The updated recommendation now comes from a **clean labeled-only calibration workflow**:

- grouped 5-fold CV on all labeled speeches
- out-of-fold probability collection
- threshold tuning for macro-F1 on those out-of-fold scores
- export on test with that fixed calibration

This avoids choosing `delta` from leaderboard matching.

## What `delta` Means

We kept the model fixed and calibrated the final submission scores.

For each score `p`, we applied:

```text
shifted_score = sigmoid(logit(p) - delta)
```

Effect:

- larger `delta` -> lower final `P(Mitterrand)`
- lower `P(Mitterrand)` -> fewer rows crossing the `0.5` threshold
- this makes the submission more conservative

So this is a **score calibration step**, not a retraining step.

## Clean CV Calibration

Calibration artifacts:

- [metrics.json](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_lstm_cv_calibration_full/metrics.json)
- [calibration.json](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_lstm_cv_calibration_full/calibration.json)
- [oof_predictions.csv](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_lstm_cv_calibration_full/oof_predictions.csv)
- [threshold_sweep.csv](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_lstm_cv_calibration_full/threshold_sweep.csv)

Key result from the labeled-only calibration:

- selected score stream: `prob_mitterrand_raw`
- best threshold on out-of-fold predictions: `0.84`
- equivalent score shift: `delta = 1.6582280766`
- out-of-fold macro-F1 after calibration: `0.9224`

The final calibrated export is here:

- [metrics.json](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_lstm_submission_cv_calibrated_full/metrics.json)
- [submission_prob_raw_positive_inverted_calibrated.csv](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_lstm_submission_cv_calibrated_full/submission_prob_raw_positive_inverted_calibrated.csv)

Resulting test-set label counts:

- calibrated submission: `3489` `Mitterrand`, `23673` `Chirac`
- original full-data submission: `4086` `Mitterrand`, `23076` `Chirac`

## Historical Leaderboard-Matched Sweep

Sweep artifacts:

- [summary.json](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_submission_score_calibration_full19/summary.json)
- [score_shift_sweep_summary.csv](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_submission_score_calibration_full19/score_shift_sweep_summary.csv)
- [variant_leaderboard_comparison.csv](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_submission_score_calibration_full19/variant_leaderboard_comparison.csv)

Main candidates:

1. `delta = 4.05`
   - `3408` `Mitterrand` rows
   - `96.1748%` agreement with the `88.279` leaderboard file
   - `1039` disagreements
   - best overall agreement to the top file

2. `delta = 5.15`
   - `3264` `Mitterrand` rows
   - `96.1012%` agreement with the `88.279` leaderboard file
   - `1059` disagreements
   - closest to the top file's `3267` positive rows

## Recommendation

Submit first:

- [submission_prob_raw_positive_inverted_calibrated.csv](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_lstm_submission_cv_calibrated_full/submission_prob_raw_positive_inverted_calibrated.csv)

Keep as backup:

- [delta_4p05/submission_prob_single_negative_span_inverted.csv](/Users/glouno/sourceCode/RITAL-NLP-project/Dataset/out/presidents_submission_score_calibration_full19/delta_4p05/submission_prob_single_negative_span_inverted.csv)

Reason:

- the CV-calibrated file is the cleanest recommendation because it is learned only from labeled training data
- it lands very close to the externally tuned variant in practice
- the `delta_4p05` file remains useful as a competition-specific fallback, but it depends on leaderboard alignment

## Interpretation

The main gain did **not** come from making the span decoder harsher.

It came from **calibrating the final probabilities downward**.

In short:

- model quality was not the only issue
- the final score scale was also off
- score calibration fixed a meaningful part of that mismatch
- the clean version of that idea is: fit the calibration from grouped cross-validation, then freeze it before touching test
