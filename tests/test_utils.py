import os
import random

import numpy as np

from src.utils.common import seed_everything


def test_seed_everything():
    seed = 42
    seed_everything(seed)

    assert random.randint(0, 100) == random.Random(seed).randint(0, 100)
    assert os.environ.get("PYTHONHASHSEED") == str(seed)

    np_rand_1 = np.random.rand()
    np.random.seed(seed)
    np_rand_2 = np.random.rand()
    assert np_rand_1 == np_rand_2

    # Check torch logic if available
    try:
        import torch

        assert torch.initial_seed() == seed
    except ImportError:
        pass


def test_seed_everything_no_torch(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)

    seed = 42
    # Should run without error and hit the except block
    seed_everything(seed)
