from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import mlflow

from rital_nlp_project.movies.models_config import bert_model
from rital_nlp_project.movies.models_utils import split_fn
from rital_nlp_project.movies.finetune_utils import prepare_train_test, tokenize_fn, compute_metrics

train_dataset, test_dataset = prepare_train_test("Dataset/clean/movies_clean_bert.parquet", split_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)

# Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    num_labels=2, 
    ignore_mismatched_sizes=True
).to(device)

# Train only classification head
# for parameter in model.parameters():
#     parameter.requires_grad = False

# for parameter in model.classifier.parameters():
#     parameter.requires_grad = True

trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_parameters = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_parameters}/{all_parameters}")

# Tokenization
train_dataset = train_dataset.map(
    lambda batch: tokenize_fn(tokenizer, batch),
    batched=True,
    remove_columns=["text"],
)
test_dataset = test_dataset.map(
    lambda batch: tokenize_fn(tokenizer, batch),
    batched=True,
    remove_columns=["text"],
)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("movies-bigbird-finetune")

# Finetune
training_args = TrainingArguments(
    output_dir="Dataset/finetune/results",
    report_to=["mlflow"],
    run_name="model-movies",
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("Dataset/finetune/model")
tokenizer.save_pretrained("Dataset/finetune/tokenizer")
mlflow.end_run()
