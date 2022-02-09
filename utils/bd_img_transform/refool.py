'''
thanks to

@inproceedings{Liu2020Refool,
	title={Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks},
	author={Yunfei Liu, Xingjun Ma, James Bailey, and Feng Lu},
	booktitle={ECCV},
	year={2020}
}

site: https://github.com/DreamtaleCore/Refool

in this script, the refoolOutOfFocusAttack (out of focus)
  refoolGhostEffectAttack (ghost effect),
  and refoolMixStrategyAttack (mix of above two)
 are the callable object you need to perform refool attack

the blend_images function is the original function,
which may do one of above two attack method depending on the probability.
'''

from typing import Tuple

import cv2
import random
import numpy as np
import scipy.stats as st
from functools import partial
from skimage.measure import compare_ssim

ssim_func = partial(compare_ssim, multichannel=True)

def npFloatImgUint8ImgSwitch(
        img : np.ndarray,
       ):
    if img.dtype == np.dtype('uint8'):
        return np.float32(img) / 255.
    elif img.dtype in [
        np.dtype('float16'),
        np.dtype('float32'),
        np.dtype('float64'),
    ]:
        return np.uint8(img * 255)

def input_resize(
        img_with_desire_shape : np.ndarray,
        img_need_to_resize : np.ndarray,
        max_size : int,  # the max(height, width) of output.
        ):

    assert img_with_desire_shape.dtype == np.dtype('uint8')
    assert img_need_to_resize.dtype == np.dtype('uint8')

    t = npFloatImgUint8ImgSwitch(img_with_desire_shape) 
    r = npFloatImgUint8ImgSwitch(img_need_to_resize) 

    h, w, _ = t.shape
    # convert t.shape to max_image_size's limitation
    scale_ratio = float(max(h, w)) / float(max_size)
    w, h = (max_size, int(round(h / scale_ratio))) if w > h \
        else (int(round(w / scale_ratio)), max_size)
    t = cv2.resize(t, (w, h), cv2.INTER_CUBIC)
    r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)

    t = npFloatImgUint8ImgSwitch(t)
    r = npFloatImgUint8ImgSwitch(r)

    return t, r # resized img_with_desire_shape and resized img_need_to_resize


class refoolOutOfFocusAttack(object):

    '''
    img_r: img use to blend with original img
    alpha_t=-1.: 1 - intensity number (ratio) of blend , when negative, pick from 1- U[0.05,0.45]
    sigma=-1: sigma in 2-d gaussian
    max_image_size=560: the max(height, width) of output.
    '''

    def __init__(self,
                 img_r : np.ndarray,
                 alpha_t = -1.,
                 sigma  = -1,
                 max_image_size : int = 560,
                 ):
        self.img_r = img_r
        
        if alpha_t < 0:
            alpha_t = 1. - random.uniform(0.05, 0.45)
        self.alpha_t = alpha_t 
        
        if sigma < 0:
            sigma = random.uniform(1, 5)
        self.sigma = sigma

        self.max_image_size = max_image_size

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):

        t, r = input_resize(img, self.img_r, self.max_image_size)

        t, r = npFloatImgUint8ImgSwitch(t), npFloatImgUint8ImgSwitch(r)
        
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        sz = int(2 * np.ceil(2 * self.sigma) + 1)
        r_blur = cv2.GaussianBlur(r, (sz, sz), self.sigma, self.sigma, 0)
        blend = r_blur + t

        # get the reflection layers' proper range
        att = 1.08 + np.random.random() / 10.0
        for i in range(3):
            maski = blend[:, :, i] > 1
            mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
        r_blur[r_blur >= 1] = 1
        r_blur[r_blur <= 0] = 0

        def gen_kernel(kern_len=100, nsig=1):
            """Returns a 2D Gaussian kernel array."""
            interval = (2 * nsig + 1.) / kern_len
            x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
            # get normal distribution
            kern1d = np.diff(st.norm.cdf(x))
            kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
            kernel = kernel_raw / kernel_raw.sum()
            kernel = kernel / kernel.max()
            return kernel

        h, w = r_blur.shape[0: 2]
        new_w = np.random.randint(0, self.max_image_size - w - 10) if w < self.max_image_size - 10 else 0
        new_h = np.random.randint(0, self.max_image_size - h - 10) if h < self.max_image_size - 10 else 0

        g_mask = gen_kernel(self.max_image_size, 3)
        g_mask = np.dstack((g_mask, g_mask, g_mask))
        alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - self.alpha_t / 2.)

        r_blur_mask = np.multiply(r_blur, alpha_r)
        blur_r = min(1., 4 * (1 - self.alpha_t)) * r_blur_mask
        blend = r_blur_mask + t * self.alpha_t

        transmission_layer = np.power(t * self.alpha_t, 1 / 2.2)
        r_blur_mask = np.power(blur_r, 1 / 2.2)
        blend = np.power(blend, 1 / 2.2)
        blend[blend >= 1] = 1
        blend[blend <= 0] = 0

        self.blended = npFloatImgUint8ImgSwitch(blend)
        self.reflection_layer = npFloatImgUint8ImgSwitch(r_blur_mask)
        self.transmission_layer = npFloatImgUint8ImgSwitch(transmission_layer)

        return self.blended, self.transmission_layer, self.reflection_layer
        

class refoolGhostEffectAttack(object):

    '''
    img_r: img use to blend with original img
    max_image_size=560: the max(height, width) of output.
    alpha_t=-1.: 1 - intensity number (ratio) of blend , when negative, pick from 1- U[0.05,0.45]
    offset=(0, 0):  padding of to img_r in ghost mode
    ghost_alpha=-1: alpha in ghost mode
    '''

    def __init__(self,
                 img_r,
                 max_image_size : int = 560,
                 alpha_t = -1.,
                 offset : Tuple[int,int] = (0,0),
                 ghost_alpha = -1.,
                 ):
        self.img_r = img_r

        self.max_image_size = max_image_size

        if alpha_t < 0:
            alpha_t = 1. - random.uniform(0.05, 0.45)
        self.alpha_t = alpha_t

        if offset[0] == 0 and offset[1] == 0:
            offset = (random.randint(3, 8), random.randint(3, 8))
        self.offset = offset

        if ghost_alpha < 0:
            ghost_alpha_switch = 1 if random.random() > 0.5 else 0
            ghost_alpha = abs(ghost_alpha_switch - random.uniform(0.15, 0.5))
        self.ghost_alpha = ghost_alpha

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):

        t, r = input_resize(img, self.img_r, self.max_image_size)

        h, w, _ = t.shape

        t, r = npFloatImgUint8ImgSwitch(t), npFloatImgUint8ImgSwitch(r)

        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        # generate the blended image with ghost effect

        r_1 = np.lib.pad(r, ((0, self.offset[0]), (0, self.offset[1]), (0, 0)),
                         'constant', constant_values=0)
        r_2 = np.lib.pad(r, ((self.offset[0], 0), (self.offset[1], 0), (0, 0)),
                         'constant', constant_values=(0, 0))


        ghost_r = r_1 * self.ghost_alpha + r_2 * (1 - self.ghost_alpha)
        ghost_r = cv2.resize(ghost_r[self.offset[0]: -self.offset[0], self.offset[1]: -self.offset[1], :], (w, h))
        reflection_mask = ghost_r * (1 - self.alpha_t)

        blended = reflection_mask + t * self.alpha_t

        transmission_layer = np.power(t * self.alpha_t, 1 / 2.2)

        ghost_r = np.power(reflection_mask, 1 / 2.2)
        ghost_r[ghost_r > 1.] = 1.
        ghost_r[ghost_r < 0.] = 0.

        blended = np.power(blended, 1 / 2.2)
        blended[blended > 1.] = 1.
        blended[blended < 0.] = 0.

        ghost_r = np.power(ghost_r, 1 / 2.2)
        ghost_r[blended > 1.] = 1.
        ghost_r[blended < 0.] = 0.

        self.reflection_layer = npFloatImgUint8ImgSwitch(ghost_r)
        self.blended = npFloatImgUint8ImgSwitch(blended)
        self.transmission_layer = npFloatImgUint8ImgSwitch(transmission_layer)

        return self.blended, self.transmission_layer, self.reflection_layer
        # blended, transmission_layer, reflection_layer

class refoolMixStrategyAttack(object):

    def __init__(self,
                 img_r_seq,
                 max_image_size=560,
                 ghost_rate=0.49,
                 alpha_t=-1.,
                 offset=(0, 0),
                 sigma=-1,
                 ghost_alpha=-1.
                 ):

        '''

        :param img_r_seq: a sequence of np.ndarray image, may draw one in equal probability

        other param plz see two attack above
        '''

        self.img_r_seq = img_r_seq
        self.max_image_size = max_image_size
        self.ghost_rate = ghost_rate
        self.alpha_t = alpha_t
        self.offset = offset
        self.sigma = sigma
        self.ghost_alpha = ghost_alpha

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):

        for img_r in random.sample(self.img_r_seq,len(self.img_r_seq)):

            if random.randint(0, 100) < self.ghost_rate * 100:
                self.refoolGhostEffectAttack = refoolGhostEffectAttack(
                    img_r,
                    self.max_image_size,
                    self.alpha_t,
                    self.offset,
                    self.ghost_alpha,
                )
                blended, transmission_layer, reflection_layer = self.refoolGhostEffectAttack(img)
            else:
                self.refoolOutOfFocusAttack = refoolOutOfFocusAttack(
                    img_r,
                    self.alpha_t,
                    self.sigma,
                    self.max_image_size,
                )
                blended, transmission_layer, reflection_layer =  self.refoolOutOfFocusAttack(img)

            img_in, img_tr, img_rf = blended, transmission_layer, reflection_layer
            # find a image with reflections with transmission as the primary layer
            if np.mean(img_rf) > np.mean(img_in - img_rf) * 0.8:
                continue
            elif img_in.max() < 0.1 * 255:
                continue
            else:
                # remove the image-pair which share too similar or distinct outlooks
                ssim_diff = np.mean(ssim_func(img_in, img_tr))
                if ssim_diff < 0.70 or ssim_diff > 0.85:
                    continue
                else:
                    break

        return blended


# def blend_images(img_t,
#                  img_r,
#                  max_image_size=560,
#                  ghost_rate=0.49,
#                  alpha_t=-1.,
#                  offset=(0, 0),
#                  sigma=-1,
#                  ghost_alpha=-1.):
#
#     """
#     img_t: original img
#     img_r: img use to blend with original img
#     max_image_size=560: the max(height, width) of output.
#     ghost_rate=0.49: fraction of poison samples that being ghost mode
#     alpha_t=-1.: 1 - intensity number (ratio) of blend , when negative, pick from 1- U[0.05,0.45]
#     offset=(0: 0):  padding of to img_r in ghost mode
#     sigma=-1: sigma in 2-d gaussian
#     ghost_alpha=-1: alpha in ghost mode
#
#     Blend transmit layer and reflection layer together (include blurred & ghosted reflection layer) and
#     return the blended image and precessed reflection image
#     """
#     t = np.float32(img_t) / 255.
#     r = np.float32(img_r) / 255.
#     h, w, _ = t.shape
#     # convert t.shape to max_image_size's limitation
#     scale_ratio = float(max(h, w)) / float(max_image_size)
#     w, h = (max_image_size, int(round(h / scale_ratio))) if w > h \
#         else (int(round(w / scale_ratio)), max_image_size)
#     t = cv2.resize(t, (w, h), cv2.INTER_CUBIC)
#     r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)
#
#     if alpha_t < 0:
#         alpha_t = 1. - random.uniform(0.05, 0.45)
#
#     if random.randint(0, 100) < ghost_rate * 100:
#         #ghost mode
#         t = np.power(t, 2.2)
#         r = np.power(r, 2.2)
#
#         # generate the blended image with ghost effect
#         if offset[0] == 0 and offset[1] == 0:
#             offset = (random.randint(3, 8), random.randint(3, 8))
#         r_1 = np.lib.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)),
#                          'constant', constant_values=0)
#         r_2 = np.lib.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)),
#                          'constant', constant_values=(0, 0))
#         if ghost_alpha < 0:
#             ghost_alpha_switch = 1 if random.random() > 0.5 else 0
#             ghost_alpha = abs(ghost_alpha_switch - random.uniform(0.15, 0.5))
#
#         ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
#         ghost_r = cv2.resize(ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :], (w, h))
#         reflection_mask = ghost_r * (1 - alpha_t)
#
#         blended = reflection_mask + t * alpha_t
#
#         transmission_layer = np.power(t * alpha_t, 1 / 2.2)
#
#         ghost_r = np.power(reflection_mask, 1 / 2.2)
#         ghost_r[ghost_r > 1.] = 1.
#         ghost_r[ghost_r < 0.] = 0.
#
#         blended = np.power(blended, 1 / 2.2)
#         blended[blended > 1.] = 1.
#         blended[blended < 0.] = 0.
#
#         ghost_r = np.power(ghost_r, 1 / 2.2)
#         ghost_r[blended > 1.] = 1.
#         ghost_r[blended < 0.] = 0.
#
#         reflection_layer = np.uint8(ghost_r * 255)
#         blended = np.uint8(blended * 255)
#         transmission_layer = np.uint8(transmission_layer * 255)
#     else:
#         # generate the blended image with focal blur
#         if sigma < 0:
#             sigma = random.uniform(1, 5)
#
#         t = np.power(t, 2.2)
#         r = np.power(r, 2.2)
#
#         sz = int(2 * np.ceil(2 * sigma) + 1)
#         r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
#         blend = r_blur + t
#
#         # get the reflection layers' proper range
#         att = 1.08 + np.random.random() / 10.0
#         for i in range(3):
#             maski = blend[:, :, i] > 1
#             mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
#             r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
#         r_blur[r_blur >= 1] = 1
#         r_blur[r_blur <= 0] = 0
#
#         def gen_kernel(kern_len=100, nsig=1):
#             """Returns a 2D Gaussian kernel array."""
#             interval = (2 * nsig + 1.) / kern_len
#             x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
#             # get normal distribution
#             kern1d = np.diff(st.norm.cdf(x))
#             kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
#             kernel = kernel_raw / kernel_raw.sum()
#             kernel = kernel / kernel.max()
#             return kernel
#
#         h, w = r_blur.shape[0: 2]
#         new_w = np.random.randint(0, max_image_size - w - 10) if w < max_image_size - 10 else 0
#         new_h = np.random.randint(0, max_image_size - h - 10) if h < max_image_size - 10 else 0
#
#         g_mask = gen_kernel(max_image_size, 3)
#         g_mask = np.dstack((g_mask, g_mask, g_mask))
#         alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - alpha_t / 2.)
#
#         r_blur_mask = np.multiply(r_blur, alpha_r)
#         blur_r = min(1., 4 * (1 - alpha_t)) * r_blur_mask
#         blend = r_blur_mask + t * alpha_t
#
#         transmission_layer = np.power(t * alpha_t, 1 / 2.2)
#         r_blur_mask = np.power(blur_r, 1 / 2.2)
#         blend = np.power(blend, 1 / 2.2)
#         blend[blend >= 1] = 1
#         blend[blend <= 0] = 0
#
#         blended = np.uint8(blend * 255)
#         reflection_layer = np.uint8(r_blur_mask * 255)
#         transmission_layer = np.uint8(transmission_layer * 255)
#
#     return blended, transmission_layer, reflection_layer

if __name__ == '__main__':
    img_original = npFloatImgUint8ImgSwitch(np.zeros((32,32,3)))
    import imageio
    img_r = (imageio.imread('/Users/chenhongrui/Documents/Screen Shot 2021-12-30 at 8.09.07 PM.png')[:,:,[0,1,2]])
    at1 = refoolOutOfFocusAttack(img_r)
    at2 = refoolGhostEffectAttack(img_r)
    import matplotlib.pyplot as plt

    plt.imshow(at1(img_original)[0])
    plt.show()
    plt.imshow(at2(img_original)[0])
    plt.show()
    plt.imshow(at1.reflection_layer)
    plt.show()
    plt.imshow(at2.reflection_layer)
    plt.show()