import numpy as np
import cv2

def RGB2YUV(x_rgb):
    W, H, C = x_rgb.shape
    x_yuv = np.zeros(x_rgb.shape, dtype=np.float)
    img = cv2.cvtColor(x_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    x_yuv = img
    return x_yuv

def YUV2RGB(x_yuv):
    W, H, C = x_yuv.shape
    x_rgb = np.zeros(x_yuv.shape, dtype=np.float)
    img = cv2.cvtColor(x_yuv.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    x_rgb = img
    return x_rgb


def DCT(x_train, window_size):
    W, H, C = x_train.shape
    x_dct = np.zeros((W, H, C), dtype=np.float)
    for ch in range(C):
        for w in range(0, W, window_size):
            for h in range(0, H, window_size):
                sub_dct = cv2.dct(x_train[w:w+window_size, h:h+window_size, ch].astype(np.float))
                x_dct[w:w+window_size, h:h+window_size, ch] = sub_dct
    return x_dct


def IDCT(x_train, window_size):
    W, H, C = x_train.shape
    x_idct = np.zeros(x_train.shape, dtype=np.float)
    for ch in range(C):
        for w in range(0, W, window_size):
            for h in range(0, H, window_size):
                sub_idct = cv2.idct(x_train[w:w+window_size, h:h+window_size, ch].astype(np.float))
                x_idct[w:w+window_size, h:h+window_size, ch] = sub_idct
    return x_idct

def poison_frequency(img_d, yuv_flag, window_size, pos_list, magnitude):
    W, H, C = img_d.shape
    if yuv_flag:
        x_train = RGB2YUV(img_d)

    x_train = DCT(x_train, window_size)

    for ch in range(C):
        for w in range(0, W, window_size):
            for h in range(0, H, window_size):
                for pos in pos_list:
                    x_train[w + pos][h + pos][ch] += magnitude

    x_train = IDCT(x_train, window_size)

    if yuv_flag:
        x_train = YUV2RGB(x_train)
    return x_train


class FtTrojanAttack(object):
    def __init__(self, yuv_flag, window_size, pos_list, magnitude):
        self.yuv_flag = yuv_flag
        self.window_size = window_size
        self.pos_list = pos_list
        self.magnitude = magnitude

    def __call__(self, img_d):
        return self.add_trigger(img_d)

    def add_trigger(self, img_d):
        return poison_frequency(img_d, self.yuv_flag, self.window_size, self.pos_list, self.magnitude)