from datasets import Dataset, ClassLabel, Features, Value
from torch.nn.functional import softmax
from torch import tensor
import numpy as np

from rital_nlp_project.common.utils import load_clean_data
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def tokenize_fn(tokenizer, example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

def prepare_train_test(dataset_path, split_fn):
    X, y = load_clean_data(dataset_path)
    X_train, X_test, y_train, y_test = split_fn(X, y)
    features = Features({
        "text": Value("string"),
        "label": ClassLabel(num_classes=2)
    })

    train_dataset = Dataset.from_dict(
        {"text": X_train, "label": y_train},
        features=features
    )

    test_dataset = Dataset.from_dict(
        {"text": X_test, "label": y_test},
        features=features
    )
    return train_dataset, test_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probas = softmax(tensor(logits), dim=-1)[:, 1].numpy()
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probas)
    }