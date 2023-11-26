import torch
import random
import numpy as np

def set_seeds(seed: int) -> int:
    torch.manual_seed(seed + 135)
    np.random.seed(seed + 235)
    random.seed(seed + 335)

    return seed + 435
