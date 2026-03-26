import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rital_nlp_project.presidents.lstm import BiLSTMClassifier

MIN_SEQUENCE_LENGTH = 3


def build_probability_windows(probabilities, k):
	window_size = 2 * k + 1
	if window_size < MIN_SEQUENCE_LENGTH:
		raise ValueError(
			f"Window size must be >= {MIN_SEQUENCE_LENGTH}. Got {window_size} (k={k})."
		)

	n = len(probabilities)
	offsets = np.arange(-k, k + 1)
	windows = np.empty((n, window_size), dtype=np.float32)

	for i in range(n):
		indices = np.clip(i + offsets, 0, n - 1)
		windows[i] = probabilities[indices]

	return windows


def resolve_test_probs_path():
	primary = "Dataset/presidents_test_pred_head.csv"
	fallback = "Dataset/finetune/presidents_test_pred_head.csv"

	if os.path.exists(primary):
		return primary
	if os.path.exists(fallback):
		return fallback
	raise FileNotFoundError(
		"Could not find test probabilities CSV in either "
		f"'{primary}' or '{fallback}'"
	)


@torch.inference_mode()
def main():
	model_path = "Dataset/finetune/model_lstm_pres.pt"
	config_path = "Dataset/finetune/model_lstm_pres_config.json"
	output_path = "Dataset/finetune/presidents_test_smoothed_probs_lstm.csv"

	test_probs_path = resolve_test_probs_path()

	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Missing trained model: {model_path}")
	if not os.path.exists(config_path):
		raise FileNotFoundError(f"Missing model config: {config_path}")

	with open(config_path, "r", encoding="utf-8") as f:
		config = json.load(f)

	k = int(config["k"])
	hidden_dim = int(config.get("hidden_dim", 64))
	num_layers = int(config.get("num_layers", 1))
	dropout = float(config.get("dropout", 0.2))

	df = pd.read_csv(test_probs_path)
	if "prob_1" not in df.columns:
		raise ValueError(f"CSV must contain a 'prob_1' column: {test_probs_path}")

	base_probs = df["prob_1"].to_numpy(dtype=np.float32)
	windows = build_probability_windows(base_probs, k=k)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = BiLSTMClassifier(
		input_dim=1,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		dropout=dropout,
	).to(device)

	state_dict = torch.load(model_path, map_location=device)
	model.load_state_dict(state_dict)
	model.eval()

	window_tensor = torch.tensor(windows, dtype=torch.float32, device=device)
	smoothed_probs = model.predict_proba(window_tensor).cpu().numpy()

	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(
		{
			"base_prob_1": base_probs,
			"smoothed_prob_1": smoothed_probs,
			"pred_smoothed": (smoothed_probs >= 0.5).astype(int),
		}
	).to_csv(output_path, index=False)

	print(f"Input probabilities: {test_probs_path}")
	print(f"Saved smoothed test probabilities to {output_path}")


if __name__ == "__main__":
	main()
