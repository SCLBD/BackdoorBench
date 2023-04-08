# Modified from https://github.com/bboylyg/NAD/blob/main/data_loader.py

import os
import csv
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import sys
from matplotlib import image as mlt
import cv2
import logging

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# from utils.bd_dataset import prepro_cls_DatasetBD


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class TransformThree:
    def __init__(self, transform1, transform2, transform3):
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        out3 = self.transform3(inp)
        return out1, out2, out3


class Dataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        flag = self.dataset[index][2]

        if self.transform:
            image = self.transform(image)
        # print(type(image), image.shape)
        return image, label, flag

    def __len__(self):
        return self.dataLen

def normalization(opt, inputs):
    output = inputs.clone()
    if opt.dataset == "cifar10":
        f = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    elif opt.dataset == "mnist":
        f = transforms.Normalize([0.5], [0.5])
    elif opt.dataset == 'tiny':
        f = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
        # pass
        return output
    elif opt.dataset == 'imagenet':
        f = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    elif opt.dataset == "cifar100":
        f = transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    else:
        raise Exception("Invalid Dataset")
    for i in range(inputs.shape[0]):
        output[i] = f(inputs[i])
    return output


def get_transform_st(opt, train=True):
    ### transform1 ###
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    transforms_list.append(transforms.ToTensor())
    transforms1 = transforms.Compose(transforms_list)

    if train == False:
        return transforms1

    ### transform2 ###
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if train:
        if opt.dataset == 'cifar10' or opt.dataset == 'gtsrb':
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
            transforms_list.append(transforms.RandomHorizontalFlip())
        elif opt.dataset == 'cifar100':
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
            transforms_list.append(transforms.RandomHorizontalFlip())
            transforms_list.append(transforms.RandomRotation(15))
        elif opt.dataset == "imagenet":
            transforms_list.append(transforms.RandomRotation(20))
            transforms_list.append(transforms.RandomHorizontalFlip(0.5))
        elif opt.dataset == "tiny":
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=8))
            transforms_list.append(transforms.RandomHorizontalFlip())
    transforms_list.append(transforms.ToTensor())
    transforms2 = transforms.Compose(transforms_list)

    ### transform3 ###
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if opt.trans1 == 'rotate':
        transforms_list.append(transforms.RandomRotation(180))
    elif opt.trans1 == 'affine':
        transforms_list.append(transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)))
    elif opt.trans1 == 'flip':
        transforms_list.append(transforms.RandomHorizontalFlip(p=1.0))
    elif opt.trans1 == 'crop':
        transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
    elif opt.trans1 == 'blur':
        transforms_list.append(transforms.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0)))
    elif opt.trans1 == 'erase':
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.RandomErasing(p=1.0, scale=(0.2, 0.3), ratio=(0.5, 1.0), value='random'))
        transforms_list.append(transforms.ToPILImage())

    if opt.trans2 == 'rotate':
        transforms_list.append(transforms.RandomRotation(180))
        transforms_list.append(transforms.ToTensor())
    elif opt.trans2 == 'affine':
        transforms_list.append(transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)))
        transforms_list.append(transforms.ToTensor())
    elif opt.trans2 == 'flip':
        transforms_list.append(transforms.RandomHorizontalFlip(p=1.0))
        transforms_list.append(transforms.ToTensor())
    elif opt.trans2 == 'crop':
        transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
        transforms_list.append(transforms.ToTensor())
    elif opt.trans2 == 'blur':
        transforms_list.append(transforms.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0)))
        transforms_list.append(transforms.ToTensor())
    elif opt.trans2 == 'erase':
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.RandomErasing(p=1.0, scale=(0.2, 0.3), ratio=(0.5, 1.0), value='random'))
    elif opt.trans2 == 'none':
        transforms_list.append(transforms.ToTensor())

    transforms3 = transforms.Compose(transforms_list)

    return transforms1, transforms2, transforms3

# def get_sd_dataloader(args,result):
#     x = result['bd_train']['x']
#     y = result['bd_train']['y']
#     data_bd_train = list(zip(x,y))

#     ### train_dataset and train_dataloader
#     transform1, transform2, transform3 = get_transform_st(args, train=True)

#     poisoned_train = prepro_cls_DatasetBD(
#         full_dataset_without_transform=data_bd_train,
#         poison_idx=np.zeros(len(data_bd_train)),  # one-hot to determine which image may take bd_transform
#         bd_image_pre_transform=None,
#         bd_label_pre_transform=None,
#         ori_image_transform_in_loading=TransformThree(transform1, transform2, transform3),
#         ori_label_transform_in_loading=None,
#         add_details_in_preprocess=True,
#     )

#     bd_trainloader = torch.utils.data.DataLoader(poisoned_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
#     ### test_dataset and test_dataloader
#     transform = get_transform_st(args, train=False)
#     x = result['bd_test']['x']
#     y = result['bd_test']['y']
#     data_bd_test = list(zip(x,y))

#     data_bd_testset = prepro_cls_DatasetBD(
#         full_dataset_without_transform=data_bd_test,
#         poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
#         bd_image_pre_transform=None,
#         bd_label_pre_transform=None,
#         ori_image_transform_in_loading=transform,
#         ori_label_transform_in_loading=None,
#         add_details_in_preprocess=True,
#     )
#     bd_testloader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

#     transform = get_transform_st(args, train=False)
#     x = result['clean_test']['x']
#     y = result['clean_test']['y']
#     data_clean_test = list(zip(x,y))
#     data_clean_testset = prepro_cls_DatasetBD(
#         full_dataset_without_transform=data_clean_test,
#         poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
#         bd_image_pre_transform=None,
#         bd_label_pre_transform=None,
#         ori_image_transform_in_loading=transform,
#         ori_label_transform_in_loading=None,
#         add_details_in_preprocess=True,
#     )
#     clean_testloader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

#     # return bd_trainloader, bd_testloader, clean_testloader
#     return bd_trainloader

def get_st_train_loader(opt, module='sscl'):
    transforms_list = [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=opt.input_height, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ]

    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == "mnist":
        mean = [0.5,]
        std = [0.5,]
    elif opt.dataset == 'tiny':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
    elif opt.dataset == 'imagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
    elif opt.dataset == 'gtsrb':
        mean = None
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    if mean != None:
        normalize = transforms.Normalize(mean=mean, std=std)
        transforms_list.append(normalize)
    
    train_transform = transforms.Compose(transforms_list)

    folder_path = folder_path = f'{opt.save_path}data_produce'
    data_path_clean = os.path.join(folder_path, 'clean_samples.npy')
    data_path_poison = os.path.join(folder_path, 'poison_samples.npy')
    data_path_suspicious = os.path.join(folder_path, 'suspicious_samples.npy')
    if opt.debug:
        # data_path_poison = os.path.join(folder_path, 'suspicious_samples.npy')
        opt.batch_size = 5

    clean_data = np.load(data_path_clean, allow_pickle=True)
    poison_data = np.load(data_path_poison, allow_pickle=True)
    suspicious_data = np.load(data_path_suspicious, allow_pickle=True)
    logging.info(f'Num of clean, poison and suspicious: {clean_data.shape[0]}, {poison_data.shape[0]}, {suspicious_data.shape[0]}')
    all_data = np.concatenate((clean_data, poison_data, suspicious_data), axis=0)
    if module == 'mixed_ce':
        train_dataset = Dataset_npy(full_dataset=all_data, transform=train_transform)
    elif module == 'sscl':
        train_dataset = Dataset_npy(full_dataset=all_data, transform=TwoCropTransform(train_transform))
    else:
        raise ValueError('module not specified')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)

    return train_loader

# def get_st_val_loader(opt):
#     # construct data loader
#     if opt.dataset == 'cifar10':
#         mean = (0.4914, 0.4822, 0.4465)
#         std = (0.2023, 0.1994, 0.2010)
#     elif opt.dataset == 'cifar100':
#         mean = (0.5071, 0.4867, 0.4408)
#         std = (0.2675, 0.2565, 0.2761)
#     elif opt.dataset == "mnist":
#         mean = [0.5]
#         std = [0.5]
#     elif opt.dataset == 'tiny':
#         mean = (0.4802, 0.4481, 0.3975)
#         std = (0.2302, 0.2265, 0.2262)
#     elif opt.dataset == 'imagenet':
#         mean = (0.4802, 0.4481, 0.3975)
#         std = (0.2302, 0.2265, 0.2262)
#     elif opt.dataset == 'gtsrb':
#         mean = None
#     elif opt.dataset == 'path':
#         mean = eval(opt.mean)
#         std = eval(opt.std)
#     else:
#         raise ValueError('dataset not supported: {}'.format(opt.dataset))
#     transforms_list = [transforms.ToTensor(),]
#     if mean != None:
#         normalize = transforms.Normalize(mean=mean, std=std)
#         transforms_list.append(normalize)

#     val_transform = transforms.Compose(transforms_list)
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=256, shuffle=False,
#         num_workers=8, pin_memory=True)

#     return val_loader