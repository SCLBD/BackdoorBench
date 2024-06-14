# This code is based on:
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d
# only perturbing weights

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class BatchNorm2d_DDE(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features)
        self.batch_feats = None
        self.collect_feats = False

    def forward(self, x):
        if self.collect_feats:
            self.batch_feats = x.reshape(x.shape[0], x.shape[1], -1).max(-1)[0].permute(1, 0).reshape(x.shape[1], -1)
        output = super().forward(x)
        return output
