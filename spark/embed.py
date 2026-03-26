import numpy as np
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from rital_nlp_project.common.utils import load_clean_data

CONFIG = {
    "google-bert/bert-base-uncased": {
        "max_length": 512,
        "stride": 128,
    },
    "camembert/camembert-base": {
        "max_length": 512,
        "stride": None,
    },
    "camembert/camembert-large": {
        "max_length": 512,
        "stride": None,
    }
}
MODEL_NAME_OR_PATH = "google-bert/bert-base-uncased"
TOKENIZER_NAME_OR_PATH = "google-bert/bert-base-uncased"

DATA_PATH = "Dataset/clean/movies_test.parquet"
DATASET_NAME = Path(DATA_PATH).stem
MAX_LENGTH = CONFIG[MODEL_NAME_OR_PATH]["max_length"]
STRIDE = CONFIG[MODEL_NAME_OR_PATH]["stride"]

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME_OR_PATH)
model = AutoModel.from_pretrained(MODEL_NAME_OR_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

texts, labels = load_clean_data(DATA_PATH)

embeddings = []
labels_ext = []
example_ids_ext = []
chunk_ids = []
tokenizer_kwargs = {
    "return_tensors": "pt",
    "truncation": True,
    "padding": "max_length",
    "return_overflowing_tokens": True,
}
if MAX_LENGTH is not None:
    tokenizer_kwargs["max_length"] = MAX_LENGTH
if STRIDE is not None:
    tokenizer_kwargs["stride"] = STRIDE

for example_id, text in enumerate(texts):
    print(example_id)
    label = labels[example_id] if labels is not None else None
    
    tokens = tokenizer(text, **tokenizer_kwargs)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Shape: (num_chunks, hidden_dim)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    num_chunks = cls_embeddings.shape[0]
    embeddings.extend(cls_embeddings)
    labels_ext.extend([label] * num_chunks)
    example_ids_ext.extend([example_id] * num_chunks)
    chunk_ids.extend(range(num_chunks))

embeddings = np.asarray(embeddings)
labels_ext = np.asarray(labels_ext)
example_ids_ext = np.asarray(example_ids_ext)
chunk_ids = np.asarray(chunk_ids)
metadata = pd.DataFrame(
    {
        "example_id": example_ids_ext,
        "chunk_id": chunk_ids,
        "label": labels_ext,
        "embedding": embeddings.tolist(),
    }
)

model_name = MODEL_NAME_OR_PATH.split("/")[-1]
metadata.to_parquet(
    f"Dataset/embeddings/{DATASET_NAME}_metadata_{model_name}.parquet",
    index=False,
)
