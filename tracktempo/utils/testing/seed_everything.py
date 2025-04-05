"""
seed_everything.py

Utility to set random seeds for reproducibility across numpy, torch, and random.
"""

import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    """
    Set the seed across all libraries for reproducibility.

    Parameters:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"✅ Seed set: {seed}")
