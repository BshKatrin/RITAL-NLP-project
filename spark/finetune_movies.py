from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch
from torch.optim import AdamW

import os
import mlflow


from rital_nlp_project.common.utils import load_clean_data
from rital_nlp_project.presidents.models_utils import split_fn
from rital_nlp_project.common.models.finetune_utils import prepare_train_test, compute_metrics, tokenize_head, tokenize_head_tail

train_dataset, test_dataset = prepare_train_test(load_clean_data, "Dataset/clean/movies_clean_bert.parquet", split_fn)
model = "google-bert/bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(
    model,
    num_labels=2, 
).to(device)
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

# Tokenization (get first 256 tokens & last 254 tokens, 2 tokens reserved for CLS & SEP)


tokenize_fn = tokenize_head
train_dataset = train_dataset.map(
    lambda batch : tokenize_fn(tokenizer, batch),
    batched=True,
    remove_columns=train_dataset.column_names,
)

test_dataset = test_dataset.map(
    lambda batch : tokenize_fn(tokenizer, batch),
    batched=True,
    remove_columns=test_dataset.column_names,
)

columns = ["input_ids", "attention_mask"]

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("movies-mbert-finetune")

# Finetune
training_args = TrainingArguments(
    output_dir="Dataset/finetune/results_bert",
    report_to=["mlflow"],
    run_name="model-movies",
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epoch",
    learning_rate=5e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    #weight_decay=0.01,
    metric_for_best_model="eval_loss",
    greater_is_better=False, 
    load_best_model_at_end=True,
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
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
    optimizers=(AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-6), None), 
)

trainer.train()
trainer.save_model("Dataset/finetune/model_bert")
tokenizer.save_pretrained("Dataset/finetune/tokenizer_bert")
mlflow.end_run()