from torch.utils.data import DataLoader
import torch
# from transformers import BigBirdTokenizer, BigBirdForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import numpy as np
import pandas as pd

from rital_nlp_project.common.utils import load_clean_data
from rital_nlp_project.movies.models_utils import split_fn
from rital_nlp_project.movies.finetune_utils import prepare_train_test

model = "Dataset/finetune/model_bert"
tokenizer = "Dataset/finetune/tokenizer_bert"
# model = "dfurman/deberta-v3-base-imdb"
# tokenizer = "dfurman/deberta-v3-base-imdb"
#model = "microsoft/deberta-v3-base"
#tokenizer = "microsoft/deberta-v3-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(tokenizer,
                                          truncation=True,
                                          max_length=512,
                                          stride=128,
                                          return_overflowing_tokens=True,
                                          return_offsets_mapping=True)

def tokenize(batch):
    # Tokenizer with 
    results = {"input_ids": [], "attention_mask": []}
    
    for text in batch["text"]:
        tokens = tokenizer(
            text,
            truncation=False,
            add_special_tokens=False,  # added manually
        )
        input_ids = tokens["input_ids"]
        
        if len(input_ids) <= 510:  # 510 = 512 - 2 special tokens
            final_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
        else:
            head = input_ids[:255]
            tail = input_ids[-254:] 
            final_ids = [tokenizer.cls_token_id] + head + [tokenizer.sep_token_id] + tail + [tokenizer.sep_token_id]
        
        final_mask = [1] * len(final_ids)
        
        results["input_ids"].append(final_ids)
        results["attention_mask"].append(final_mask)
    
    return results

model = AutoModelForSequenceClassification.from_pretrained(
    model,
    num_labels=2,
).to(device)

model.eval()

#_, dataset_test = prepare_train_test("Dataset/clean/movies_test.parquet", split_fn)
dataset_test = prepare_train_test(load_clean_data, "Dataset/clean/movies_test.parquet", split_fn=None)
dataset_test = dataset_test.map(
    tokenize,
    batched=True,
    remove_columns=dataset_test.column_names,
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
test_loader = DataLoader(dataset_test, batch_size=64, collate_fn=data_collator)
example_ids = []
#labels = []
pred_labels = []
probabilities = []

model.eval()
print("Total", len(test_loader))
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        print(i)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # batch_example_ids = batch["example_id"]
        # batch_labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        batch_logits = outputs.logits
        batch_probs = torch.softmax(batch_logits, dim=-1)
        batch_preds = torch.argmax(batch_logits, dim=-1)

        # example_ids.append(batch_example_ids.detach().cpu().numpy())
        # labels.append(batch_labels.detach().cpu().numpy())
        pred_labels.append(batch_preds.detach().cpu().numpy())
        probabilities.append(batch_probs.detach().cpu().numpy())

    # example_ids = np.concatenate(example_ids, axis=0)
    # labels = np.concatenate(labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    probabilities = np.concatenate(probabilities, axis=0)

predictions = pd.DataFrame(
    {
        # "example_id": example_ids,
        # "true_label": labels,
        "pred_label": pred_labels,
        "prob_0": probabilities[:, 0],
        "prob_1": probabilities[:, 1],
    }
)

predictions.to_csv("Dataset/finetune/eval_512_test_pred_2.csv", index=False)