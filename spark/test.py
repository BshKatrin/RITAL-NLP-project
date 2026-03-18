import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from src.rital_nlp_project.common.utils import load_clean_data

print("CUDA available:", torch.cuda.is_available())
print("sm", torch.cuda.get_device_capability())
print(list(torch.cuda.get_arch_list()))

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version (PyTorch):", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())

tokenizer = AutoTokenizer.from_pretrained("camembert/camembert-large")
model = AutoModel.from_pretrained("camembert/camembert-large")
# tokenizer = AutoTokenizer.from_pretrained("camembert-base")
# model = AutoModel.from_pretrained("camembert-base")

device = torch.device("cuda")
model.to(device)
model.eval()

texts, _ = load_clean_data("Dataset/presidents_clean_bert.parquet")
embeddings = []

for i, text in enumerate(texts):
    tokens = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens)

    # Get CLS embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].detach().squeeze().cpu().numpy()
    embeddings.append(cls_embedding)


embeddings = np.array(embeddings)
np.save("Dataset/pres_embeddings_large.npy", embeddings)
