import os
import random
from pathlib import Path

import numpy as np


def seed_everything(seed: int = 42):
    """Seeds basic parameters for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_kaggle_paths():
    is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""
    data_dir = "data"
    conf_dir = "conf"

    if is_kaggle:
        # Find data_dir reliably
        for p in Path("/kaggle/input").rglob("ch2026_metrics_train.csv"):
            data_dir = str(p.parent)
            break

        # Find conf_dir reliably
        for p in Path("/kaggle/input").rglob("config.yaml"):
            conf_dir = str(p.parent)
            break

    return is_kaggle, data_dir, conf_dir
