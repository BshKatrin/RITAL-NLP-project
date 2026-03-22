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
