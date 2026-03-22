from torch.utils.data import DataLoader
import torch
# from transformers import BigBirdTokenizer, BigBirdForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from rital_nlp_project.movies.models_utils import split_fn
from rital_nlp_project.movies.finetune_utils import prepare_train_test, tokenize_fn, compute_metrics


model = "Dataset/finetune/model"
tokenizer = "Dataset/finetune/tokenizer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(
    model,
    num_labels=2,
    #ignore_mismatched_sizes=True
).to(device)

model.eval()

_, dataset_test = prepare_train_test("Dataset/clean/movies_clean_bert.parquet", split_fn)
dataset_test = dataset_test.map(
    lambda batch: tokenize_fn(tokenizer, batch),
    batched=True,
)
dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

test_loader = DataLoader(dataset_test, batch_size=16)

logits = []
labels = []

model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        batch_logits = outputs.logits

        logits.append(batch_logits.detach().cpu().numpy())
        labels.append(batch_labels.detach().cpu().numpy())

    logits = np.concatenate(logits, axis=0)
    labels = np.concatenate(labels, axis=0)

print(compute_metrics((logits, labels)))