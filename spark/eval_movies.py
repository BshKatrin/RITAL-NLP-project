from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import numpy as np
import pandas as pd

from rital_nlp_project.common.utils import load_clean_data
from rital_nlp_project.common.models.finetune_utils import prepare_train_test, tokenize_head


MODEL_PATH = "Dataset/finetune/model_bert_head"
TOKENIZER_PATH = "Dataset/finetune/tokenizer_bert_head"
DATASET_PATH = "Dataset/clean/movies_test.parquet"
OUTPUT_PATH = "Dataset/predictions/movies_test_pred_head.csv"
NUM_LABELS = 2
BATCH_SIZE = 128


@torch.inference_mode()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=NUM_LABELS).to(device)
    model.eval()

    dataset_test = prepare_train_test(load_clean_data, DATASET_PATH, split_fn=None)
    dataset_test = dataset_test.map(
        lambda batch: tokenize_head(tokenizer, batch),
        batched=True,
    )
    dataset_test = dataset_test.remove_columns(["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, collate_fn=data_collator)
    probabilities = []

    print("Total", len(test_loader))
    for i, batch in enumerate(test_loader):
        print(i)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        batch_logits = outputs.logits
        batch_probs = torch.softmax(batch_logits, dim=-1)
        probabilities.append(batch_probs.detach().cpu().numpy())

    probabilities = np.concatenate(probabilities, axis=0)

    predictions = pd.DataFrame({"prob_1": probabilities[:, 1]})
    predictions.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
