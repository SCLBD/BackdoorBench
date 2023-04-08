# This code is based on:
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d
# only perturbing weights

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class NoisyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NoisyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.neuron_mask = Parameter(torch.Tensor(num_features))
        self.neuron_noise = Parameter(torch.Tensor(num_features))
        self.neuron_noise_bias = Parameter(torch.Tensor(num_features))
        init.ones_(self.neuron_mask)
        init.zeros_(self.neuron_noise)
        init.zeros_(self.neuron_noise_bias)
        self.is_perturbed = False

    def reset(self, rand_init=False, eps=0.0):
        if rand_init:
            init.uniform_(self.neuron_noise, a=-eps, b=eps)
            init.uniform_(self.neuron_noise_bias, a=-eps, b=eps)
        else:
            init.zeros_(self.neuron_noise)
            init.zeros_(self.neuron_noise_bias)

    def include_noise(self):
        self.is_perturbed = True

    def exclude_noise(self):
        self.is_perturbed = False

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        if self.is_perturbed:
            coeff_weight = self.neuron_mask + self.neuron_noise
            coeff_bias = 1.0 + self.neuron_noise_bias
        else:
            coeff_weight = self.neuron_mask
            coeff_bias = 1.0
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight * coeff_weight, self.bias * coeff_bias,
            bn_training, exponential_average_factor, self.eps)


class NoisyBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(NoisyBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.neuron_mask_fc = Parameter(torch.Tensor(num_features))
        self.neuron_noise_fc = Parameter(torch.Tensor(num_features))
        self.neuron_noise_bias_fc = Parameter(torch.Tensor(num_features))
        init.ones_(self.neuron_mask_fc)
        init.zeros_(self.neuron_noise_fc)
        init.zeros_(self.neuron_noise_bias_fc)
        self.is_perturbed = False

    def reset(self, rand_init=False, eps=0.0):
        if rand_init:
            init.uniform_(self.neuron_noise_fc, a=-eps, b=eps)
            init.uniform_(self.neuron_noise_bias_fc, a=-eps, b=eps)
        else:
            init.zeros_(self.neuron_noise_fc)
            init.zeros_(self.neuron_noise_bias_fc)

    def include_noise(self):
        self.is_perturbed = True

    def exclude_noise(self):
        self.is_perturbed = False

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        if self.is_perturbed:
            coeff_weight = self.neuron_mask_fc + self.neuron_noise_fc
            coeff_bias = 1.0 + self.neuron_noise_bias_fc
        else:
            coeff_weight = self.neuron_mask_fc
            coeff_bias = 1.0
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight * coeff_weight, self.bias * coeff_bias,
            bn_training, exponential_average_factor, self.eps)
