# idea: fix the randomness.

import sys, logging
sys.path.append('../../')

import random
import numpy as np
import torch
import torchvision.transforms as transforms

def fix_random(
        random_seed: int = 0
) -> None:
    '''
    use to fix randomness in the script, but if you do not want to replicate experiments, then remove this can speed up your code
    :param random_seed:
    :return: None
    '''
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False