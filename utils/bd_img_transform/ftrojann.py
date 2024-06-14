

from typing import Sequence
import logging
import numpy as np
import cv2

def RGB2YUV(x_rgb):
    x_yuv = np.zeros(x_rgb.shape, dtype=np.float)
    for i in range(x_rgb.shape[0]):
        img = cv2.cvtColor(x_rgb[i].astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        x_yuv[i] = img
    return x_yuv

def YUV2RGB(x_yuv):
    x_rgb = np.zeros(x_yuv.shape, dtype=np.float)
    for i in range(x_yuv.shape[0]):
        img = cv2.cvtColor(x_yuv[i].astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        x_rgb[i] = img
    return x_rgb


def DCT(x_train, window_size):
    # x_train: (idx, w, h, ch)
    x_dct = np.zeros((x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2]), dtype=np.float)
    x_train = np.transpose(x_train, (0, 3, 1, 2))

    for i in range(x_train.shape[0]):
        for ch in range(x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    sub_dct = cv2.dct(x_train[i][ch][w:w+window_size, h:h+window_size].astype(np.float))
                    x_dct[i][ch][w:w+window_size, h:h+window_size] = sub_dct
    return x_dct            # x_dct: (idx, ch, w, h)


def IDCT(x_train, window_size):
    # x_train: (idx, ch, w, h)
    x_idct = np.zeros(x_train.shape, dtype=np.float)

    for i in range(x_train.shape[0]):
        for ch in range(0, x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    sub_idct = cv2.idct(x_train[i][ch][w:w+window_size, h:h+window_size].astype(np.float))
                    x_idct[i][ch][w:w+window_size, h:h+window_size] = sub_idct
    x_idct = np.transpose(x_idct, (0, 2, 3, 1))
    return x_idct

class ftrojann_version(object):
    # idea : in this attack, this transform just replace the image by the image_serial_id, the real transform does not happen here

    def __init__(self, YUV, channel_list, window_size, magnitude, pos_list) -> None:
        logging.debug(
            "in SSBA_attack_replace_version, the real transform does not happen here, input img, target must be NONE, only image_serial_id used"
        )
        self.YUV = YUV
        self.channel_list = channel_list
        self.window_size = window_size
        self.magnitude = magnitude
        self.pos_list = pos_list
    def __call__(self, img: None) -> np.ndarray:
        # add a new axis to img, from w,h,c to idx,w,h,c
        img = np.expand_dims(img, axis=0)
        if self.YUV:
            # transfer to YUV channel
            img = RGB2YUV(img)

        # transfer to frequency domain
        img = DCT(img, self.window_size)  # (idx, ch, w, h)

        # plug trigger frequency
        for i in range(img.shape[0]):
            for ch in self.channel_list:
                for w in range(0, img.shape[2], self.window_size):
                    for h in range(0, img.shape[3], self.window_size):
                        for pos in self.pos_list:
                            img[i][ch][w + pos[0]][h + pos[1]] += self.magnitude

        img = IDCT(img, self.window_size)  # (idx, w, h, ch)

        if self.YUV:
            img = YUV2RGB(img)
        # remove the first axis, from idx,w,h,c to w,h,c
        img = np.squeeze(img, axis=0)
        return img




