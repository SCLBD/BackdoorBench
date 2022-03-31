#This script is for Sig attack callable transform

'''
This code is based on https://github.com/bboylyg/NAD

The original license:
License CC BY-NC

The update include:
    1. change to callable object
    2. change the way of trigger generation, use the original formulation.

# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
'''

from typing import Union
import torch
import numpy as np


class sigTriggerAttack(object):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    def __init__(self,
                 delta : Union[int, float, complex, np.number, torch.Tensor] = 40,
                 f : Union[int, float, complex, np.number, torch.Tensor] =6
                 ) -> None:

        self.delta = delta
        self.f = f

    def __call__(self, img, target = None, image_serial_id = None):
        return self.sigTrigger(img)


    def sigTrigger(self, img):

        img = np.float32(img)
        pattern = np.zeros_like(img)
        m = pattern.shape[1]
        for i in range(int(img.shape[0])):
              for j in range(int(img.shape[1])):
                    pattern[i, j] = self.delta * np.sin(2 * np.pi * j * self.f / m)

        img = np.uint32(img) + pattern
        img = np.uint8(np.clip(img, 0, 255))

        return img


