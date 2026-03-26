from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import numpy as np
import pandas as pd

from rital_nlp_project.common.utils import load_clean_data
from rital_nlp_project.common.models.finetune_utils import prepare_train_test, tokenize_head_tail, tokenize_head


model = "Dataset/finetune/model_bert_head"
tokenizer = "Dataset/finetune/tokenizer_bert_head"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=2).to(device)
tokenize_fn = tokenize_head

model.eval()

dataset_test = prepare_train_test(load_clean_data, "Dataset/clean/movies_test.parquet", split_fn=None)

dataset_test = dataset_test.map(
	lambda batch: tokenize_fn(tokenizer, batch),
	batched=True,
)

dataset_test = dataset_test.remove_columns(["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
test_loader = DataLoader(dataset_test, batch_size=128, collate_fn=data_collator)
probabilities = []

print("Total", len(test_loader))
with torch.no_grad():
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
predictions.to_csv("Dataset/predictions/movies_test_pred_head.csv", index=False)
