import math
import torch
from torch.nn import functional as F
import copy
import numpy as np

def tensor2img(t):
    t_np = t.detach().cpu().numpy().transpose(1, 2, 0)
    return t_np

def normalization(data):
    _range = np.max(data) - np.min(data)
    return ((data - np.min(data)) / _range)*0.2

#@resource_check
def gauss_smooth(image, sig=6):
    '''
    This is the pre-set low-pass filter discribed in the paper
    '''
    size_denom = 5.
    sigma = sig * size_denom
    kernel_size = sigma
    mgrid = np.arange(kernel_size, dtype=np.float32)
    mean = (kernel_size - 1.) / 2.
    mgrid = mgrid - mean
    mgrid = mgrid * size_denom
    kernel = 1. / (sigma * math.sqrt(2. * math.pi)) * \
             np.exp(-(((mgrid - 0.) / (sigma)) ** 2) * 0.5)
    kernel = kernel / np.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernelx = np.tile(np.reshape(kernel, (1, 1, int(kernel_size), 1)), (3, 1, 1, 1))
    kernely = np.tile(np.reshape(kernel, (1, 1, 1, int(kernel_size))), (3, 1, 1, 1))

    padd0 = int(kernel_size // 2)
    evenorodd = int(1 - kernel_size % 2)

    pad = torch.nn.ConstantPad2d((padd0 - evenorodd, padd0, padd0 - evenorodd, padd0), 0.)
    in_put = torch.from_numpy(np.expand_dims(np.transpose(image[0].astype(np.float32), (2, 0, 1)), axis=0))
    output = pad(in_put)

    weightx = torch.from_numpy(kernelx)
    weighty = torch.from_numpy(kernely)
    conv = F.conv2d
    output = conv(output, weightx, groups=3)
    output = conv(output, weighty, groups=3)
    output = tensor2img(output[0])

    return np.expand_dims(output,axis=0)

def smooth_clip(x, v, smoothing, max_iters=200):

    test_x = copy.deepcopy(x)
    v_i = copy.deepcopy(v)
    iter_i = 0
    n = 1.

    while n > 0 and iter_i < max_iters:
        result_img = test_x + v_i

        overshoot = ((result_img - 1.) >= 0)
        belowshoot = ((result_img - 0.) <= 0)

        ov_max = (result_img - 1.)* 0.1
        bl_max = (result_img - 0.)* 0.1 * -1.

        ov_max = np.maximum(ov_max.max(), 0.01)
        bl_max = np.maximum(bl_max.max(), 0.01)

        overshoot = smoothing(overshoot)
        belowshoot = smoothing(belowshoot)

        maxx_ov = np.max(overshoot) + 1e-12
        maxx_bl = np.max(belowshoot) + 1e-12

        overshoot = overshoot / maxx_ov
        belowshoot = belowshoot / maxx_bl

        v_i = v_i - overshoot * ov_max + belowshoot * bl_max
        result_img = test_x + v_i

        overshoot = ((result_img - 1.) >= 0)
        belowshoot = ((result_img - 0.) <= 0)

        n_ov = overshoot.sum()
        n_bl = belowshoot.sum()
        n = n_ov + n_bl
        iter_i += 1

    return v_i
