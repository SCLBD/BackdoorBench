# This code is based on:
# https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
# only perturbing weights

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch import Tensor, Size
from typing import Union, List, Tuple

_shape_t = Union[int, List[int], Size]

class LayerNorm_DDE(nn.LayerNorm):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None):
        super().__init__(normalized_shape = normalized_shape, eps = eps, elementwise_affine = elementwise_affine,
                 device = device, dtype = dtype)
        self.batch_feats = None
        self.collect_feats = False

    def forward(self, x):
        if self.collect_feats:
            self.batch_feats = x.reshape(x.shape[0], x.shape[-1], -1).max(-1)[0].permute(1, 0).reshape(x.shape[-1], -1)
        output = super().forward(x)
        return output

class LayerNorm2D_DDE(nn.LayerNorm):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None):
        super().__init__(normalized_shape = normalized_shape, eps = eps, elementwise_affine = elementwise_affine,
                 device = device, dtype = dtype)
        self.batch_feats = None
        self.collect_feats = False

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        if self.collect_feats:
            self.batch_feats = x.reshape(x.shape[0], x.shape[-1], -1).max(-1)[0].permute(1, 0).reshape(x.shape[-1], -1)
        output = super().forward(x)
        output = output.permute(0, 3, 1, 2)
        return output

