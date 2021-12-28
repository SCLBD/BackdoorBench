import sys, logging
sys.path.append('../')

import numpy as np

import torch
from torchvision.transforms import transforms

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def data_transforms_cifar10(train_add_gaussian_noise = False, test_add_rotation_and_crop = False):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    # CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    CIFAR_STD = [0.2023, 0.1994, 0.2010] # this is from the paper
    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # train_transform.transforms.append(Cutout(16))
    logging.info('without！ cutout')

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    if train_add_gaussian_noise:
        train_transform.transforms.append(AddGaussianNoise(0, 0.05))

    if test_add_rotation_and_crop:
        valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(degrees=(-30,30)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def constructUnnormalize(
        CIFAR_MEAN =  [0.49139968, 0.48215827, 0.44653124],
        CIFAR_STD = [0.2023, 0.1994, 0.2010]
):
    CIFAR_MEAN = np.array(CIFAR_MEAN)
    CIFAR_STD = np.array(CIFAR_STD)
    return transforms.Normalize(-CIFAR_MEAN/CIFAR_STD, 1/CIFAR_STD)