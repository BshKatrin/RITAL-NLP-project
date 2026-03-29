# Explainable Presidents Push

This push keeps four explainable lanes:

1. `spark/run_pres_lstm_simple_sweep.py`
   - simple speech-level BiLSTM
   - grouped OOF threshold calibration
   - optional proxy scoring on accepted archive-labelled test rows

2. `spark/run_pres_lstm_structured_sweep.py`
   - same BiLSTM core
   - same sentence embeddings
   - same grouped OOF calibration
   - plus a single contiguous Mitterrand-span decoder

3. `spark/run_pres_camembert_cv.py`
   - contextual CamemBERT sentence classifier
   - local sentence context via neighboring sentences in the same speech
   - grouped speech-level CV and OOF threshold calibration

4. `spark/run_pres_fusion.py`
   - weighted logit averaging
   - optional logistic stacking
   - optional single-span decoder on fused scores

## Proxy Scoring

Canonical proxy scoring script:

```bash
uv run --project envs/py314 python scripts/eval_presidents_proxy.py \
  --prediction-path Dataset/out/presidents_archive_matching_dgx_final/submission_label_reconstructed.csv \
  --proxy-labels-path Dataset/out/presidents_archive_matching_dgx_final/row_labels_with_evidence.csv \
  --output-dir Dataset/out/presidents_proxy_eval_oracle
```

The scorer uses the exact row order of:

`Dataset/out/presidents_archive_matching_dgx_final/row_labels_with_evidence.csv`

and filters only `review_status == accepted`.

## Lane A: Simple Swept BiLSTM

Example overnight sweep:

```bash
uv run --project envs/py314 python spark/run_pres_lstm_simple_sweep.py \
  --device cuda \
  --hidden-dims 128 256 \
  --projection-dims 256 512 \
  --dropouts 0.15 0.2 0.3 \
  --minority-weight-scales 1.0 0.9 0.8 0.7 \
  --epochs-grid 12 16 20 \
  --max-runs 12 \
  --proxy-labels-path Dataset/out/presidents_archive_matching_dgx_final/row_labels_with_evidence.csv \
  --ranking-mode hybrid
```

## Lane B: Explainable Structured BiLSTM

This wrapper keeps:

- `selection_decoder = single_negative_span`
- `position_features = none`
- `transition_features = false`
- `position_bins = 0`
- `position_prior_weight = 0.0`

Example overnight sweep:

```bash
uv run --project envs/py314 python spark/run_pres_lstm_structured_sweep.py \
  --device cuda \
  --hidden-dims 128 256 \
  --projection-dims 256 512 \
  --dropouts 0.1 0.2 0.3 \
  --min-negative-spans 2 3 4 \
  --learning-rates 1e-3 5e-4 \
  --epochs-grid 16 20 \
  --max-runs 12 \
  --proxy-labels-path Dataset/out/presidents_archive_matching_dgx_final/row_labels_with_evidence.csv \
  --ranking-mode hybrid
```

The structured wrapper also supports a late `no_span_bias` sweep for the top clean runs.

## Lane C: Contextual CamemBERT

Example run:

```bash
uv run --project envs/py314 python spark/run_pres_camembert_cv.py \
  --device cuda \
  --model-name camembert-base \
  --context-window 1 \
  --max-length 256 \
  --epochs 3 \
  --learning-rate 1e-5 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --proxy-labels-path Dataset/out/presidents_archive_matching_dgx_final/row_labels_with_evidence.csv \
  --output-dir Dataset/out/presidents_camembert_base_ctx1_len256_lr1e5_ep3
```

Example large-model retry:

```bash
uv run --project envs/py314 python spark/run_pres_camembert_cv.py \
  --device cuda \
  --model-name almanach/camembert-large \
  --context-window 2 \
  --max-length 384 \
  --epochs 4 \
  --learning-rate 5e-6 \
  --batch-size 4 \
  --eval-batch-size 8 \
  --proxy-labels-path Dataset/out/presidents_archive_matching_dgx_final/row_labels_with_evidence.csv \
  --output-dir Dataset/out/presidents_camembert_large_ctx2_len384_lr5e6_ep4
```

## Lane D: Late Fusion

Fuse two row-aligned score streams:

```bash
uv run --project envs/py314 python spark/run_pres_fusion.py \
  --oof-a-path Dataset/out/presidents_lstm_simple_tuned_mw090/oof_predictions.csv \
  --oof-b-path Dataset/out/presidents_lstm_cv_calibration_full/oof_predictions.csv \
  --test-a-path Dataset/out/presidents_lstm_simple_submission_tuned_mw090/test_predictions_detailed.csv \
  --test-b-path Dataset/out/presidents_lstm_submission_cv_calibrated_full/test_predictions_detailed.csv \
  --oof-a-score-column prob_mitterrand_raw \
  --oof-b-score-column prob_mitterrand_single_negative_span \
  --test-a-score-column prob_mitterrand_raw \
  --test-b-score-column prob_negative_single_negative_span \
  --proxy-labels-path Dataset/out/presidents_archive_matching_dgx_final/row_labels_with_evidence.csv \
  --output-dir Dataset/out/presidents_fusion_simple_structured
```

If you want the structural rule on top of the fusion too:

```bash
uv run --project envs/py314 python spark/run_pres_fusion.py \
  ... \
  --apply-single-span-decoder \
  --min-negative-span 3 \
  --no-span-bias 0.0
```

## Recommended Overnight Order

1. Launch the simple sweep.
2. Launch the structured sweep.
3. Launch 2 to 4 CamemBERT runs with different context/length/LR settings.
4. Fuse the best simple + structured, then the best BiLSTM + best CamemBERT.

## Report-Safe Story

- Simple lane: speech-level BiLSTM over sentence embeddings.
- Structured lane: same BiLSTM, plus the assumption that a speech contains at most one contiguous minority segment.
- CamemBERT lane: sentence classifier with local textual context.
- Fusion lane: simple late combination of semantic and sequential score streams.
