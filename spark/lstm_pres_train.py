import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from rital_nlp_project.common.models.finetune_utils import prepare_train_test, calc_weights
from rital_nlp_project.common.utils import load_clean_data
from rital_nlp_project.presidents.lstm import BiLSTMClassifier, build_probability_windows, weighted_bce_with_logits


DATA_PATH = "Dataset/clean/presidents_clean_bert.parquet"
BASE_PROBS_PATH = "Dataset/presidents_train_pred_head.csv"

OUTPUT_MODEL_PATH = "Dataset/finetune/model_lstm_pres_log_reg.pt"
OUTPUT_PROBS_PATH = "Dataset/finetune/presidents_train_smoothed_probs_lstm.csv"
OUTPUT_CONFIG_PATH = "Dataset/finetune/model_lstm_pres_config.json"

K = 3
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
WEIGHT_DECAY = 0.01
NUM_LABELS = 2


class ProbabilityWindowDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = prepare_train_test(load_clean_data, DATA_PATH, split_fn=None)
    dataset = dataset.map(
        lambda batch: {"label": [0 if label == 1 else 1 for label in batch["label"]]},
        batched=True,
    )
    labels = np.asarray(dataset["label"], dtype=np.float32)

    probs_df = pd.read_csv(BASE_PROBS_PATH)

    base_probs = probs_df["prob_1"].to_numpy(dtype=np.float32)
    windows = build_probability_windows(base_probs, k=K)

    train_loader = DataLoader(ProbabilityWindowDataset(windows, labels), batch_size=BATCH_SIZE, shuffle=True)

    class_weights = calc_weights(dataset)
    # label_counts = np.bincount(labels.astype(np.int64), minlength=NUM_LABELS)
    # safe_counts = np.where(label_counts == 0, 1, label_counts)
    # class_weights = torch.tensor(
    #     len(labels) / (len(safe_counts) * safe_counts),
    #     dtype=torch.float32,
    #     device=device,
    # )
    # print("Train label counts:", label_counts.tolist())
    # print("LSTM class weights:", class_weights.tolist())

    model = BiLSTMClassifier(
        input_dim=1,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    def criterion(logits, labels): return weighted_bce_with_logits(logits, labels, class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f}")

    model.eval()
    all_windows = torch.tensor(windows, dtype=torch.float32, device=device)
    smoothed_probs = model.predict_proba(all_windows).detach().cpu().numpy()

    Path(OUTPUT_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)

    print(base_probs.shape, smoothed_probs.shape, labels.shape)
    pd.DataFrame(
        {
            "base_prob_1": base_probs,
            "lstm_prob_1": smoothed_probs,
            "label": labels.astype(int),
        }
    ).to_csv(OUTPUT_PROBS_PATH, index=False)

    config = {
        "k": K,
        "window_size": 2 * K + 1,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "data_path": DATA_PATH,
        "base_probs_path": BASE_PROBS_PATH,
        "trained_on_full_dataset": True,
    }
    with open(OUTPUT_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved model to {OUTPUT_MODEL_PATH}")
    print(f"Saved smoothed probabilities to {OUTPUT_PROBS_PATH}")


if __name__ == "__main__":
    main()
