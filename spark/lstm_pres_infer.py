import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rital_nlp_project.presidents.lstm import BiLSTMClassifier, build_probability_windows
from spark import SEED, set_global_seed

MODEL_PATH = "Dataset/finetune/model_lstm_pres.pt"
CONFIG_PATH = "Dataset/finetune/model_lstm_pres_config.json"
OUTPUT_PATH = "Dataset/finetune/presidents_test_smoothed_probs_lstm.csv"
TEST_PROBS_PATH = "Dataset/predictions/presidents_test_pred_head.csv"

DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.2


@torch.inference_mode()
def main():
    set_global_seed(SEED)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing trained model: {MODEL_PATH}")
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing model config: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    k = int(config["k"])
    hidden_dim = int(config.get("hidden_dim", DEFAULT_HIDDEN_DIM))
    num_layers = int(config.get("num_layers", DEFAULT_NUM_LAYERS))
    dropout = float(config.get("dropout", DEFAULT_DROPOUT))

    df = pd.read_csv(TEST_PROBS_PATH)
    if "prob_1" not in df.columns:
        raise ValueError(f"CSV must contain a 'prob_1' column: {TEST_PROBS_PATH}")

    base_probs = df["prob_1"].to_numpy(dtype=np.float32)
    windows = build_probability_windows(base_probs, k=k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(
        input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    window_tensor = torch.tensor(windows, dtype=torch.float32, device=device)
    smoothed_probs = model.predict_proba(window_tensor).cpu().numpy()

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "base_prob_1": base_probs,
            "smoothed_prob_1": smoothed_probs,
            "pred_smoothed": (smoothed_probs >= 0.5).astype(int),
        }
    ).to_csv(OUTPUT_PATH, index=False)

    print(f"Input probabilities: {TEST_PROBS_PATH}")
    print(f"Saved smoothed test probabilities to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
