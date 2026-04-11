import torch
import random
import numpy as np

def set_seeds(seed: int) -> int:
    """Set deterministic seeds for Torch, NumPy, and Python random.

    Args:
        seed: Base integer seed.

    Returns:
        Offset seed value intended for subsequent environment seeding.
    """
    torch.manual_seed(seed + 135)
    np.random.seed(seed + 235)
    random.seed(seed + 335)

    return seed + 435
