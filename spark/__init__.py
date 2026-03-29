import os
import random

import numpy as np
import torch


SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        # Keep training/inference deterministic where kernels support it.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
