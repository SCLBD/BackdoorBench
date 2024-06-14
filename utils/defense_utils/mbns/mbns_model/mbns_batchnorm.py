# This code is based on:
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d
# only perturbing weights

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class BatchNorm2d_MBNS(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features)
        self.batch_var = 0
        self.batch_mean = 0
        self.collect_stats = False
        
    def forward(self, x):
        output = super().forward(x)
        if self.collect_stats:
            self.batch_var = x.var((0,2,3))
            self.batch_mean = x.mean((0,2,3))
        return output
