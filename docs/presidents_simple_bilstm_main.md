# Presidents Simple BiLSTM

This is the clean `main`-branch version of the presidents BiLSTM pipeline.

## What It Does

- Loads sentence embeddings and metadata grouped by `speech_id`
- Trains a bidirectional LSTM over whole speeches
- Uses class-weighted loss because Mitterrand is the minority class
- Chooses the training epoch on a grouped validation split
- Learns the final decision threshold on grouped out-of-fold predictions

## Default Training Setup

- `hidden_dim=128`
- `projection_dim=256`
- `num_layers=1`
- `dropout=0.15`
- `batch_size=8`
- `epochs=20`
- `learning_rate=1e-3`
- `weight_decay=1e-2`
- `minority_weight_scale=0.7`
- validation split `0.15`
- OOF calibration folds `5`

## Train

```bash
uv run --project envs/py314 python spark/train_pres_lstm_simple.py
```

Main outputs:

- `Dataset/out/presidents_lstm_simple/checkpoint.pt`
- `Dataset/out/presidents_lstm_simple/calibration.json`
- `Dataset/out/presidents_lstm_simple/metrics.json`
- `Dataset/out/presidents_lstm_simple/oof_predictions.csv`
- `Dataset/out/presidents_lstm_simple/threshold_sweep.csv`

## Predict

```bash
uv run --project envs/py314 python spark/predict_pres_lstm_simple.py
```

Main outputs:

- `Dataset/out/presidents_lstm_simple_submission/test_predictions_detailed.csv`
- `Dataset/out/presidents_lstm_simple_submission/submission_prob_mitterrand_raw.csv`
- `Dataset/out/presidents_lstm_simple_submission/submission_prob_mitterrand_calibrated.csv`
- `Dataset/out/presidents_lstm_simple_submission/submission_probability.csv`
- `Dataset/out/presidents_lstm_simple_submission/submission_label_calibrated.csv`

`submission_probability.csv` is the file intended to be paired with the label
submission. It is the calibrated Mitterrand probability:

- values near `0` mean Chirac
- values near `1` mean Mitterrand
- thresholding it at `0.5` reproduces `submission_label_calibrated.csv`
