from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch
import numpy as np
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

import os
import mlflow


from rital_nlp_project.common.utils import load_clean_data
from rital_nlp_project.common.models.finetune_utils import prepare_train_test, tokenize_head, calc_weights
from rital_nlp_project.presidents.models_utils import split_fn

DATASET_PATH = "Dataset/clean/presidents_clean_bert.parquet"
BASE_MODEL = "almanach/camembert-large"

MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT = "pres-camembert-finetune"
RUN_NAME = "model-pres"

OUTPUT_DIR = "Dataset/finetune/results_camembert"
MODEL_SAVE_DIR = "Dataset/finetune/model_camembert_head"
TOKENIZER_SAVE_DIR = "Dataset/finetune/tokenizer_camembert_head"

LEARNING_RATE = 5e-6
ADAM_EPS = 1e-6
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 3
MAX_GRAD_NORM = 0.5
WARMUP_RATIO = 0.1
LOGGING_STEPS = 50
NO_DECAY_PARAMS = ["bias", "LayerNorm.weight", "layernorm.weight"]


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            class_weights = self.class_weights.to(logits.device)
            loss_fn = CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = CrossEntropyLoss()

        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    train_dataset = prepare_train_test(load_clean_data, DATASET_PATH, split_fn=None)

    train_dataset = train_dataset.map(
        lambda batch: {"label": [0 if label == 1 else 1 for label in batch["label"]]},
        batched=True,
    )

    # test_dataset = test_dataset.map(
    #     lambda batch: {"label": [0 if label == 1 else 1 for label in batch["label"]]},
    #     batched=True,
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
    ).to(device)

    # Train only classification head
    # for parameter in model.parameters():
    #     parameter.requires_grad = False
    #
    # for parameter in model.classifier.parameters():
    #     parameter.requires_grad = True

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_parameters = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_parameters}/{all_parameters}")

    print(train_dataset[0])
    print(train_dataset[1])
    print(train_dataset[-1])

    # test_dataset = test_dataset.map(
    #     lambda example, idx: concat_neighbours(example, idx, test_dataset, tokenizer),
    #     with_indices=True
    # )

    train_dataset = train_dataset.map(
        lambda batch: tokenize_head(tokenizer, batch),
        batched=True,
        # remove_columns=train_dataset.column_names,
    )

    # test_dataset = test_dataset.map(
    #     lambda batch: tokenize_head(tokenizer, batch),
    #     batched=True,
    #     # remove_columns=test_dataset.column_names,
    # )

    class_weights = calc_weights(train_dataset)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    # test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        report_to=["mlflow"],
        run_name=RUN_NAME,
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        load_best_model_at_end=False,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
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

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=test_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
        optimizers=(AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=ADAM_EPS), None),
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(TOKENIZER_SAVE_DIR)
    mlflow.end_run()


if __name__ == "__main__":
    main()
