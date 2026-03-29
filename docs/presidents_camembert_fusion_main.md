# Presidents CamemBERT And Fusion

This is the clean `main`-branch companion to the simple BiLSTM pipeline.

## CamemBERT

The cleaned CamemBERT path treats each sentence as the prediction target, but it
adds a small amount of local context by concatenating the neighboring sentences
from the same speech.

- model: `camembert-base`
- context window: `2` sentences on each side
- max length: `256`
- epochs: `4`
- learning rate: `5e-6`
- weight decay: `1e-2`
- train batch size: `8`
- eval batch size: `16`
- grouped OOF calibration folds: `5`

Train:

```bash
uv run --project envs/py314 python spark/train_pres_camembert_simple.py
```

Main outputs:

- `Dataset/out/presidents_camembert_simple/model/`
- `Dataset/out/presidents_camembert_simple/model_config.json`
- `Dataset/out/presidents_camembert_simple/calibration.json`
- `Dataset/out/presidents_camembert_simple/metrics.json`
- `Dataset/out/presidents_camembert_simple/oof_predictions.csv`
- `Dataset/out/presidents_camembert_simple/threshold_sweep.csv`

Predict:

```bash
uv run --project envs/py314 python spark/predict_pres_camembert_simple.py
```

Main outputs:

- `Dataset/out/presidents_camembert_simple_submission/test_predictions_detailed.csv`
- `Dataset/out/presidents_camembert_simple_submission/submission_probability.csv`
- `Dataset/out/presidents_camembert_simple_submission/submission_label_calibrated.csv`

`submission_probability.csv` is the calibrated Mitterrand probability:

- values near `0` mean Chirac
- values near `1` mean Mitterrand
- thresholding it at `0.5` reproduces `submission_label_calibrated.csv`

## Weighted Fusion

The cleaned fusion path combines the simple BiLSTM and the cleaned CamemBERT
with a weighted average in logit space:

- BiLSTM contributes speech-level continuity
- CamemBERT contributes local sentence semantics
- the fusion weight `alpha` is chosen on grouped OOF predictions over a small fixed grid
- the final fused threshold is also calibrated on grouped OOF predictions

Run:

```bash
uv run --project envs/py314 python spark/fuse_pres_simple_bilstm_camembert.py
```

Main outputs:

- `Dataset/out/presidents_bilstm_camembert_fusion/alpha_sweep.csv`
- `Dataset/out/presidents_bilstm_camembert_fusion/calibration.json`
- `Dataset/out/presidents_bilstm_camembert_fusion/test_predictions_detailed.csv`
- `Dataset/out/presidents_bilstm_camembert_fusion/submission_probability.csv`
- `Dataset/out/presidents_bilstm_camembert_fusion/submission_label_calibrated.csv`
