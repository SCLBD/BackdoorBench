import numpy as np
import cv2
from albumentations.augmentations import functional


class OverstretchTransorform(object):
    def __init__(self):
        self.random()

    def random(self):
        self.flag = np.random.random() < 0.5
        self.hflag = np.random.random() < 0.5
        self.ratio = np.random.uniform(0.1, 0.9)

    def run(self, img, status=False):
        if status:
            self.random()

        if self.flag:
            if self.hflag:
                h, w = img.shape
                ah = int(self.ratio * h)
                newimg = np.zeros((h + ah, w), np.float)
                newimg[ah//2:h+ah//2, :] = img[:,:]
                newimg = cv2.resize(newimg, (w, h))
            else:
                h, w = img.shape
                aw = int(self.ratio * w)
                newimg = np.zeros((h, w+aw), np.float)
                newimg[:, aw//2:w+aw//2] = img[:,:]
                newimg = cv2.resize(newimg, (w, h))
        else:
            newimg = img

        return newimg




class FlipTransorform(object):
    def __init__(self):
        self.random()

    def random(self):
        self.vflag = np.random.random() < 0.5
        self.hflag = np.random.random() < 0.5

    def run(self, img, status=False):
        if status:
            self.random()
        if self.hflag:
            img = functional.hflip(img)
        if self.vflag:
            img = functional.vflip(img)

        return img


class AffineTransorform(object):
    def __init__(self):
        self.random()

    def random(self):
        self.Affine_angle = np.random.uniform(-10, 10)
        self.Affine_scale = np.random.uniform(1 - 0.2, 1 + 0.2)
        self.Affine_dx = np.random.uniform(-0.0625, 0.0625)
        self.Affine_dy = np.random.uniform(-0.0625, 0.0625)
        self.flag = np.random.random() < 0.5

    def run(self, img, status=False):
        if status:
            self.random()
        if self.flag:
            img = functional.shift_scale_rotate(img, self.Affine_angle, self.Affine_scale, self.Affine_dx, self.Affine_dy,interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT)

        return img


class GridDistorbTransorform(object):
    def __init__(self):

        self.random()


    def random(self):
        self.GridDistorb_num_steps = 5
        self.GridDistorb_distort_limit = (-0.3, 0.3)
        self.GridDistorb_stepsx = [1 + np.random.uniform(self.GridDistorb_distort_limit[0], self.GridDistorb_distort_limit[1]) for i in
                              range(self.GridDistorb_num_steps + 1)]
        self.GridDistorb_stepsy = [1 + np.random.uniform(self.GridDistorb_distort_limit[0], self.GridDistorb_distort_limit[1]) for i in
                              range(self.GridDistorb_num_steps + 1)]
        self.flag = np.random.random() < 0.5

    def run(self, img, status=False):
        if status:
            self.random()
        if self.flag:
            img = functional.grid_distortion(img, self.GridDistorb_num_steps, self.GridDistorb_stepsx, self.GridDistorb_stepsy, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT)

        return img

class ElasticDistortionTransorform(object):
    def __init__(self):

        self.random()

    def random(self):
        self.ElasticDistortion_alpha = 1
        self.ElasticDistortion_sigma = 50
        self.ElasticDistortion_alpha_affine = 50
        self.ElasticDistortion_interpolation = cv2.INTER_LINEAR
        self.ElasticDistortion_random_state = np.random.randint(0, 10000)
        self.flag = np.random.random() < 0.5

    def run(self, img, status=False):
        if status:
            self.random()
        if self.flag:
            img=functional.elastic_transform_fast(img, self.ElasticDistortion_alpha, self.ElasticDistortion_sigma, self.ElasticDistortion_alpha_affine, self.ElasticDistortion_interpolation,cv2.BORDER_CONSTANT, np.random.RandomState(self.ElasticDistortion_random_state))

        return img



def MultiTransform(*imgs):

    # 初始化随机图像空间变换器
    # Overstretch = OverstretchTransorform()
    Flip = FlipTransorform()
    Affine = AffineTransorform()
    GridDistorb = GridDistorbTransorform()
    ElasticDistortion = ElasticDistortionTransorform()

    newimg = []
    for img in imgs:
        # img = Overstretch.run(img)
        img = Flip.run(img)
        img = Affine.run(img)
        img = GridDistorb.run(img)
        img = ElasticDistortion.run(img)
        newimg.append(img)

    return newimg
