import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rital_nlp_project.common.utils import load_clean_data

from rital_nlp_project.common.utils import load_clean_data
from rital_nlp_project.presidents.models_utils import split_fn
from rital_nlp_project.common.models.finetune_utils import concat_neighbours, prepare_train_test, tokenize_head


print("CUDA available:", torch.cuda.is_available())
print("sm", torch.cuda.get_device_capability())
print(list(torch.cuda.get_arch_list()))

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version (PyTorch):", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())


train_dataset, test_dataset = prepare_train_test(load_clean_data, "Dataset/clean/presidents_clean_bert.parquet", split_fn)
model = "almanach/camembert-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(
    model,
    num_labels=2, 
).to(device)

sep_token = tokenizer.sep_token
sep_token_id = tokenizer.sep_token_id

print("SEP token:", sep_token)
print("SEP token id:", sep_token_id)


train_dataset = train_dataset.map(
    lambda example, idx: concat_neighbours(example, idx, train_dataset, tokenizer),
    with_indices=True
)
print(train_dataset[0])

train_dataset = train_dataset.map(
    lambda batch : tokenize_head(tokenizer, batch),
    batched=True,
    remove_columns=train_dataset.column_names,
)

print(train_dataset[0])