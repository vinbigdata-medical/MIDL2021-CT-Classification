import torch
import numpy as np
import random

def setup_determinism(seed:int):
    """Setup random seed so that result is reproducible

    Args:
        seed (int): Seed number to use
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # avoid non-determinstic algo
    torch.backends.cudnn.benchmark = False
    random.seed(seed)