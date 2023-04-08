import math
import torch
from torch import nn
from torch.nn.functional import relu
import torch.nn.functional as F
import numpy as np
from torch_utils import misc

#执行pixel norm(在第C维进行) [B, C, H, W] -> [B, C, H, W]
class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) 
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1

#定义用于mapping的全连接层权重与偏置
class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5), 
                 use_wscale=False, 
                 lrmul=1.0, 
                 bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        #此处设定的lr_mul似乎只对bias生效
        he_std = gain * in_channels ** (-0.5)  
        if use_wscale: 
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else: #weight * he_std/lrmul * lrmul = weight * he_std
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std) 
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels)) 
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out

class G_mapping(nn.Module): 
    def __init__(self,
                 mapping_fmaps=512,
                 dlatent_size=512,
                 normalize_latents=True,  
                 use_wscale=True,         
                 lrmul=0.01,              
                 gain=2**(0.5)            
                 ):
        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps 
        self.func = nn.Sequential(
            FC(self.mapping_fmaps, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        )

        self.normalize_latents = normalize_latents
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out

class modulated_conv2d(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        fp_size=512, 
        bias_init=None, 
        demodulate=1, 
        fused_modconv=0, 
    ):
        super(modulated_conv2d, self).__init__()
        self.fp_size = fp_size 
        self.in_channels = in_channels
        self.weight_conv = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=torch.contiguous_format)) 
        self.bias_conv = None if not bias_init else torch.nn.Parameter(torch.full([out_channels], np.float32(bias_init))) 

        self.fc = nn.Linear(self.fp_size, self.in_channels)

        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        self.demodulate = demodulate
        self.fused_modconv = fused_modconv

    def forward(self, fingerprint, x):
        
        
        batch_size = x.shape[0] 
        fingerprint = self.fc(fingerprint) 
        
        
        misc.assert_shape(fingerprint, [batch_size, self.in_channels])

        
        if x.dtype == torch.float16 and demodulate:
            self.weight_conv = self.weight_conv * (1 / np.sqrt(self.in_channels * self.k * self.k) / self.weight_conv.norm(float('inf'), dim=[1,2,3], keepdim=True)) 
            fingerprint = fingerprint / fingerprint.norm(float('inf'), dim=1, keepdim=True) 

        
        w = None
        dcoefs = None
        if self.demodulate or self.fused_modconv:
            w = self.weight_conv.unsqueeze(0) 
            w = w * fingerprint.reshape(batch_size, 1, -1, 1, 1) 
        if self.demodulate:
            
            dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() 
        if self.demodulate and self.fused_modconv:
            
            w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) 

        
        if not self.fused_modconv:
            x = x * fingerprint.to(x.dtype).reshape(batch_size, -1, 1, 1) 
            x = F.conv2d(input=x, weight=self.weight_conv.to(x.dtype), stride=self.stride, padding=self.padding)
            if self.demodulate:
                x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1) 
            return x 
        
        
        with misc.suppress_tracer_warnings(): 
            batch_size = int(batch_size)
        x = x.reshape(1, -1, *x.shape[2:]) 
        w = w.reshape(-1, self.in_channels, self.k, self.k) 
        x = F.conv2d(input=x, weight=w.to(x.dtype), stride=self.stride, padding=self.padding, groups=batch_size)
        x = x.reshape(batch_size, -1, *x.shape[2:]) 
        return x