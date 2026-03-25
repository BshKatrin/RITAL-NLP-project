from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch
from torch.optim import AdamW

import os
import mlflow


from rital_nlp_project.common.utils import load_clean_data
from rital_nlp_project.presidents.models_utils import split_fn
from rital_nlp_project.common.models.finetune_utils import prepare_train_test, compute_metrics, tokenize_head, concat_neighbours

train_dataset = prepare_train_test(load_clean_data, "Dataset/clean/presidents_clean_bert.parquet", split_fn=None)

print(len(train_dataset))

train_dataset = train_dataset.map(
    lambda batch: {"label": [0 if label == 1 else 1 for label in batch["label"]]},
    batched=True,
)

# test_dataset = test_dataset.map(
#     lambda batch: {"label": [0 if label == 1 else 1 for label in batch["label"]]},
#     batched=True,
# )

model = "almanach/camembert-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(
    model,
    num_labels=2, 
).to(device)
tokenize_fn = tokenize_head

sep_token = tokenizer.sep_token
sep_token_id = tokenizer.sep_token_id

print("SEP token:", sep_token)
print("SEP token id:", sep_token_id)

cls_token = tokenizer.cls_token
cls_token_id = tokenizer.cls_token_id

print("CLS token:", cls_token)
print("CLS token id:", cls_token_id)


#model.config.hidden_dropout_prob = 0.2 # for small dataset
#model.config.attention_probs_dropout_prob = 0.2

# Train only classification head
# for parameter in model.parameters():
#     parameter.requires_grad = False

# for parameter in model.classifier.parameters():
#     parameter.requires_grad = True

trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_parameters = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_parameters}/{all_parameters}")

train_dataset = train_dataset.map(
    lambda example, idx: concat_neighbours(example, idx, train_dataset, tokenizer),
    with_indices=True
)
train_dataset = train_dataset.remove_columns(["text"])
train_dataset = train_dataset.rename_column("text_with_context", "text")

print(train_dataset[0])
print(train_dataset[1])
print(train_dataset[-1])

# test_dataset = test_dataset.map(
#     lambda example, idx: concat_neighbours(example, idx, test_dataset, tokenizer),
#     with_indices=True
# )

train_dataset = train_dataset.map(
    lambda batch : tokenize_fn(tokenizer, batch),
    batched=True,
    #remove_columns=train_dataset.column_names,
)
print(train_dataset[0])

# test_dataset = test_dataset.map(
#     lambda batch : tokenize_fn(tokenizer, batch),
#     batched=True,
#     #remove_columns=test_dataset.column_names,
# )


train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
#test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("pres-camembert-finetune")

# Finetune
training_args = TrainingArguments(
    output_dir="Dataset/finetune/results_camembert",
    report_to=["mlflow"],
    run_name="model-pres",
    eval_strategy="no",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epoch",
    learning_rate=5e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    #weight_decay=0.01,
    load_best_model_at_end=False,
    max_grad_norm=0.5,
    warmup_ratio=0.1,
)


no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=test_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
    optimizers=(AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-6), None), 
)

trainer.train()
trainer.save_model("Dataset/finetune/model_camembert")
tokenizer.save_pretrained("Dataset/finetune/tokenizer_camembert")
mlflow.end_run()