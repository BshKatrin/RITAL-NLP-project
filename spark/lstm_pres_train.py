import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rital_nlp_project.common.models.finetune_utils import prepare_train_test
from rital_nlp_project.common.utils import load_clean_data
from rital_nlp_project.presidents.lstm import BiLSTMClassifier


class ProbabilityWindowDataset(Dataset):
	def __init__(self, windows, labels):
		self.windows = torch.tensor(windows, dtype=torch.float32)
		self.labels = torch.tensor(labels, dtype=torch.float32)

	def __len__(self):
		return len(self.windows)

	def __getitem__(self, idx):
		return self.windows[idx], self.labels[idx]


def build_probability_windows(probabilities, k):
	n = len(probabilities)
	offsets = np.arange(-k, k + 1)
	windows = np.empty((n, 2 * k + 1), dtype=np.float32)

	for i in range(n):
		indices = np.clip(i + offsets, 0, n - 1)
		windows[i] = probabilities[indices]

	return windows


def weighted_bce_with_logits(logits, labels, class_weights):
	weights = torch.where(labels > 0.5, class_weights[1], class_weights[0])
	return nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weights)


def train_epoch(model, dataloader, optimizer, criterion, device):
	model.train()
	total_loss = 0.0

	for windows, labels in dataloader:
		windows = windows.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		logits = model(windows)
		loss = criterion(logits, labels)
		loss.backward()
		optimizer.step()

		total_loss += loss.item() * windows.size(0)

	return total_loss / len(dataloader.dataset)


@torch.inference_mode()
def eval_epoch(model, dataloader, criterion, device):
	model.eval()
	total_loss = 0.0
	correct = 0
	total = 0

	for windows, labels in dataloader:
		windows = windows.to(device)
		labels = labels.to(device)

		logits = model(windows)
		loss = criterion(logits, labels)

		probs = torch.sigmoid(logits)
		preds = (probs >= 0.5).float()

		total_loss += loss.item() * windows.size(0)
		correct += (preds == labels).sum().item()
		total += labels.numel()

	return total_loss / len(dataloader.dataset), correct / max(total, 1)


def main():
	data_path = "Dataset/clean/presidents_clean_bert.parquet"
	base_probs_path = "Dataset/predictions/presidents_train_pred_head.csv"
    # base_probs_path = "Dataset/finetune/presidents_train_pred_head.csv"
    # base_probs_path = "Dataset/finetune/pred_test_log_reg.csv"

	output_model_path = "Dataset/finetune/model_lstm_pres.pt"
	output_probs_path = "Dataset/finetune/presidents_train_smoothed_probs_lstm.csv"
	output_config_path = "Dataset/finetune/model_lstm_pres_config.json"

	k = 2
	epochs = 30
	batch_size = 128
	learning_rate = 1e-4
	hidden_dim = 64
	num_layers = 2
	dropout = 0.2

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	dataset = prepare_train_test(load_clean_data, data_path, split_fn=None)
	dataset = dataset.map(
		lambda batch: {"label": [0 if label == 1 else 1 for label in batch["label"]]},
		batched=True,
	)
	labels = np.asarray(dataset["label"], dtype=np.float32)

	probs_df = pd.read_csv(base_probs_path)

	base_probs = probs_df["prob_1"].to_numpy(dtype=np.float32)
	windows = build_probability_windows(base_probs, k=k)

	train_loader = DataLoader(ProbabilityWindowDataset(windows, labels), batch_size=batch_size, shuffle=True)

	label_counts = np.bincount(labels.astype(np.int64), minlength=2)
	safe_counts = np.where(label_counts == 0, 1, label_counts)
	class_weights = torch.tensor(
		len(labels) / (len(safe_counts) * safe_counts),
		dtype=torch.float32,
		device=device,
	)
	print("Train label counts:", label_counts.tolist())
	print("LSTM class weights:", class_weights.tolist())

	model = BiLSTMClassifier(
		input_dim=1,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		dropout=dropout,
	).to(device)

	criterion = lambda logits, labels: weighted_bce_with_logits(logits, labels, class_weights)
	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

	for epoch in range(1, epochs + 1):
		train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
		print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f}")

	model.eval()
	all_windows = torch.tensor(windows, dtype=torch.float32, device=device)
	smoothed_probs = model.predict_proba(all_windows).detach().cpu().numpy()

	Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
	torch.save(model.state_dict(), output_model_path)

	pd.DataFrame(
		{
			"base_prob_1": base_probs,
			"lstm_prob_1": smoothed_probs,
			"label": labels.astype(int),
		}
	).to_csv(output_probs_path, index=False)

	config = {
		"k": k,
		"window_size": 2 * k + 1,
		"hidden_dim": hidden_dim,
		"num_layers": num_layers,
		"dropout": dropout,
		"epochs": epochs,
		"batch_size": batch_size,
		"learning_rate": learning_rate,
		"data_path": data_path,
		"base_probs_path": base_probs_path,
		"trained_on_full_dataset": True,
	}
	with open(output_config_path, "w", encoding="utf-8") as f:
		json.dump(config, f, indent=2)

	print(f"Saved model to {output_model_path}")
	print(f"Saved smoothed probabilities to {output_probs_path}")


if __name__ == "__main__":
	main()
