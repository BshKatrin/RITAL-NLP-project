from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch
from torch.optim import AdamW

import os
import mlflow


from rital_nlp_project.common.utils import load_clean_data
from rital_nlp_project.common.models.finetune_utils import prepare_train_test, compute_metrics, tokenize_head
from rital_nlp_project.movies.models_utils import split_fn
from spark import SEED, set_global_seed

DATASET_PATH = "Dataset/clean/movies_clean_bert.parquet"
BASE_MODEL = "google-bert/bert-base-uncased"
NUM_LABELS = 2

MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT = "movies-mbert-finetune"
RUN_NAME = "model-pres"

OUTPUT_DIR = "Dataset/finetune/results_bert_head"
MODEL_SAVE_DIR = "Dataset/finetune/model_bert_head"
TOKENIZER_SAVE_DIR = "Dataset/finetune/tokenizer_bert_head"
TOKENIZER_FN = tokenize_head

LEARNING_RATE = 5e-6
ADAM_EPS = 1e-6
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 3
MAX_GRAD_NORM = 0.5
WARMUP_RATIO = 0.1
LOGGING_STEPS = 50
DROPOUT_PROB = 0.2
NO_DECAY_PARAMS = ["bias", "LayerNorm.weight", "layernorm.weight"]


def main() -> None:
    set_global_seed(SEED)
    train_dataset, test_dataset = prepare_train_test(load_clean_data, DATASET_PATH, split_fn=split_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
    ).to(device)

    # for small dataset
    model.config.hidden_dropout_prob = DROPOUT_PROB
    model.config.attention_probs_dropout_prob = DROPOUT_PROB

    # Train only classification head
    # for parameter in model.parameters():
    #     parameter.requires_grad = False
    #
    # for parameter in model.classifier.parameters():
    #     parameter.requires_grad = True

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_parameters = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_parameters}/{all_parameters}")

    train_dataset = train_dataset.map(
        lambda batch: TOKENIZER_FN(tokenizer, batch),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    test_dataset = test_dataset.map(
        lambda batch: TOKENIZER_FN(tokenizer, batch),
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        report_to=["mlflow"],
        run_name=RUN_NAME,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        load_best_model_at_end=True,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
        seed=SEED,
        data_seed=SEED,
    )

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in NO_DECAY_PARAMS)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in NO_DECAY_PARAMS)],
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
        optimizers=(AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=ADAM_EPS), None),
    )

    trainer.train()
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(TOKENIZER_SAVE_DIR)
    mlflow.end_run()


if __name__ == "__main__":
    main()
