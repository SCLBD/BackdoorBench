import sys, logging
sys.path.append('../')

import random
import numpy as np
import torch
import torchvision.transforms as transforms

def fix_random(
        random_seed : int = 0
) -> None:

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False