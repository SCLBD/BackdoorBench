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

class LayerNorm_MBNS(nn.LayerNorm):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None):
        super().__init__(normalized_shape = normalized_shape, eps = eps, elementwise_affine = elementwise_affine,
                 device = device, dtype = dtype)
        self.batch_var_clean = 0
        self.batch_mean_clean = 0
        self.batch_var_bd = 0
        self.batch_mean_bd = 0
        self.collect_stats = False
        self.collect_stats_clean = False
        self.collect_stats_bd = False

    def forward(self, x):
        if self.collect_stats:
            if self.collect_stats_clean:
                feats = x.reshape(x.shape[-1],-1)
                self.batch_var_clean = feats.var(-1)
                self.batch_mean_clean = feats.mean(-1)
            elif self.collect_stats_bd:
                feats = x.reshape(x.shape[-1],-1)
                self.batch_var_bd = feats.var(-1)
                self.batch_mean_bd = feats.mean(-1)
        output = super().forward(x)
        return output

class LayerNorm2D_MBNS(nn.LayerNorm):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None):
        super().__init__(normalized_shape = normalized_shape, eps = eps, elementwise_affine = elementwise_affine,
                 device = device, dtype = dtype)
        self.batch_var_clean = 0
        self.batch_mean_clean = 0
        self.batch_var_bd = 0
        self.batch_mean_bd = 0
        self.collect_stats = False
        self.collect_stats_clean = False
        self.collect_stats_bd = False

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        if self.collect_stats:
            if self.collect_stats_clean:
                feats = x.reshape(x.shape[-1],-1)
                self.batch_var_clean = feats.var(-1)
                self.batch_mean_clean = feats.mean(-1)
            elif self.collect_stats_bd:
                feats = x.reshape(x.shape[-1],-1)
                self.batch_var_bd = feats.var(-1)
                self.batch_mean_bd = feats.mean(-1)
        output = super().forward(x)
        output = output.permute(0, 3, 1, 2)
        return output

