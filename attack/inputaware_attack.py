'''
This file is modified based on the following source:

link : https://github.com/VinAIResearch/input-aware-backdoor-attack-release
The original license is placed at the end of this file.

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. during training the backdoor attack generalization to lower poison ratio (generalize_to_lower_pratio)
    5. calculate part of ASR
    6. save process

basic sturcture for main:
    1. config args, save_path, fix random seed
    2. set the device, model, criterion, optimizer, training schedule.
    3. set the clean train data and clean test data
    4. clean train 25 epochs
    5. training with backdoor modification simultaneously
    6. save attack result
'''

import sys, yaml, os, time

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

from pprint import pformat
import shutil
import argparse
from utils.log_assist import get_git_info
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape, get_dataset_normalization, get_dataset_denormalization
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.save_path_generate import generate_save_folder
from utils.save_load_attack import summary_dict
from utils.trainer_cls import Metric_Aggregator, ModelTrainerCLS
from utils.bd_dataset import prepro_cls_DatasetBD, xy_iter
from utils.save_load_attack import save_attack_result, sample_pil_imgs

from copy import deepcopy
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()

agg = Metric_Aggregator()

import csv
import logging
import os

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate

import torch.nn.functional as F
import torchvision
from torchvision import transforms

import torch
from torch import nn


class Conv2dBlock(nn.Module):
    def __init__(self, in_c, out_c, ker_size=(3, 3), stride=1, padding=1, batch_norm=True, relu=True):
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_c, out_c, ker_size, stride, padding)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.05, affine=True)
        if relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, ker_size=(2, 2), stride=2, dilation=(1, 1), ceil_mode=False, p=0.0):
        super(DownSampleBlock, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=ker_size, stride=stride, dilation=dilation, ceil_mode=ceil_mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, scale_factor=(2, 2), mode="bilinear", p=0.0):
        super(UpSampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x



class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


# ---------------------------- Generators ----------------------------#


class Generator(nn.Sequential):
    def __init__(self, opt, out_channels=None):
        super(Generator, self).__init__()
        if opt.dataset == "mnist":
            channel_init = 16
            steps = 2
        else:
            channel_init = 32
            steps = 3

        channel_current = opt.input_channel
        channel_next = channel_init
        for step in range(steps):
            self.add_module("convblock_down_{}".format(2 * step), Conv2dBlock(channel_current, channel_next))
            self.add_module("convblock_down_{}".format(2 * step + 1), Conv2dBlock(channel_next, channel_next))
            self.add_module("downsample_{}".format(step), DownSampleBlock())
            if step < steps - 1:
                channel_current = channel_next
                channel_next *= 2

        self.add_module("convblock_middle", Conv2dBlock(channel_next, channel_next))

        channel_current = channel_next
        channel_next = channel_current // 2
        for step in range(steps):
            self.add_module("upsample_{}".format(step), UpSampleBlock())
            self.add_module("convblock_up_{}".format(2 * step), Conv2dBlock(channel_current, channel_current))
            if step == steps - 1:
                self.add_module(
                    "convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next, relu=False)
                )
            else:
                self.add_module("convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next))
            channel_current = channel_next
            channel_next = channel_next // 2
            if step == steps - 2:
                if out_channels is None:
                    channel_next = opt.input_channel
                else:
                    channel_next = out_channels

        self._EPSILON = 1e-7
        self._normalizer = self._get_normalize(opt)
        self._denormalizer = self._get_denormalize(opt)

    def _get_denormalize(self, opt):
        denormalizer = Denormalize(opt, get_dataset_normalization(opt.dataset).mean,
                                   get_dataset_normalization(opt.dataset).std)
        return denormalizer

    def _get_normalize(self, opt):
        normalizer = Normalize(opt, get_dataset_normalization(opt.dataset).mean,
                                   get_dataset_normalization(opt.dataset).std)
        return normalizer

    def forward(self, x):
        for module in self.children():
            x = module(x)
        x = nn.Tanh()(x) / (2 + self._EPSILON) + 0.5
        return x

    def normalize_pattern(self, x):
        if self._normalizer:
            x = self._normalizer(x)
        return x

    def denormalize_pattern(self, x):
        if self._denormalizer:
            x = self._denormalizer(x)
        return x

    def threshold(self, x):
        return nn.Tanh()(x * 20 - 10) / (2 + self._EPSILON) + 0.5


# ---------------------------- Classifiers ----------------------------#


class NetC_MNIST(nn.Module):
    def __init__(self):
        super(NetC_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), 1, 0)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(0.1)

        self.maxpool4 = nn.MaxPool2d((2, 2))
        self.conv5 = nn.Conv2d(32, 64, (5, 5), 1, 0)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(0.1)

        self.maxpool5 = nn.MaxPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.1)
        self.linear9 = nn.Linear(512, 10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


term_width = int(60)

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()

class ColorDepthShrinking(object):
    def __init__(self, c=3):
        self.t = 1 << int(8 - c)

    def __call__(self, img):
        im = np.asarray(img)
        im = (im / self.t).astype("uint8") * self.t
        img = Image.fromarray(im.astype("uint8"))
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(t={})".format(self.t)


class Smoothing(object):
    def __init__(self, k=3):
        self.k = k

    def __call__(self, img):
        im = np.asarray(img)
        im = cv2.GaussianBlur(im, (self.k, self.k), 0)
        img = Image.fromarray(im.astype("uint8"))
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(k={})".format(self.k)


def get_transform(opt, train=True, c=0, k=0):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if train:
        transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
        if opt.dataset != "mnist":
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))
        if opt.dataset == "cifar10":
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if c > 0:
        transforms_list.append(ColorDepthShrinking(c))
    if k > 0:
        transforms_list.append(Smoothing(k))

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(get_dataset_normalization(opt.dataset))
    return transforms.Compose(transforms_list)

class Args:
    pass

def get_dataloader(opt, train=True, c=0, k=0):

    args = Args()
    args.dataset = opt.dataset
    args.dataset_path = opt.dataset_path
    args.img_size = (opt.input_height, opt.input_width, opt.input_channel)

    train_dataset_without_transform, \
    train_img_transform, \
    train_label_transfrom, \
    test_dataset_without_transform, \
    test_img_transform, \
    test_label_transform = dataset_and_transform_generate(args=opt)

    if train:
        dataset = train_dataset_without_transform
        train_transform = get_transform(opt, train, c=c, k=k)
        dataset = prepro_cls_DatasetBD(
            full_dataset_without_transform=dataset,
            poison_idx=np.zeros(len(dataset)),
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=train_transform,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
    else:
        dataset = test_dataset_without_transform
        test_transform = get_transform(opt, train, c=c, k=k)
        dataset = prepro_cls_DatasetBD(
            full_dataset_without_transform=dataset,
            poison_idx=np.zeros(len(dataset)),
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=test_transform,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=True
    )
    return dataloader


def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def create_bd(inputs, targets, netG, netM, opt, train_or_test):
    if train_or_test == 'train':
        bd_targets = create_targets_bd(targets, opt)
        if inputs.__len__() == 0:  # for case that no sample should be poisoned
            return inputs, bd_targets, inputs.detach().clone(), inputs.detach().clone()
        patterns = netG(inputs)
        patterns = netG.normalize_pattern(patterns)

        masks_output = netM.threshold(netM(inputs))
        bd_inputs = inputs + (patterns - inputs) * masks_output
        return bd_inputs, bd_targets, patterns, masks_output
    if train_or_test == 'test':
        bd_targets = create_targets_bd(targets, opt)

        position_changed = (bd_targets - targets != 0) # no matter all2all or all2one, we want location changed to tell whether the bd is effective

        inputs, bd_targets = inputs[position_changed], bd_targets[position_changed]

        if inputs.__len__() == 0:  # for case that no sample should be poisoned
            return inputs, bd_targets, inputs.detach().clone(), inputs.detach().clone()
        patterns = netG(inputs)
        patterns = netG.normalize_pattern(patterns)

        masks_output = netM.threshold(netM(inputs))
        bd_inputs = inputs + (patterns - inputs) * masks_output
        return bd_inputs, bd_targets, patterns, masks_output, position_changed, targets



def create_cross(inputs1, inputs2, netG, netM, opt):
    if inputs1.__len__() == 0: # for case that no sample should be poisoned
        return inputs2.detach().clone(), inputs2, inputs2.detach().clone()
    patterns2 = netG(inputs2)
    patterns2 = netG.normalize_pattern(patterns2)
    masks_output = netM.threshold(netM(inputs2))
    inputs_cross = inputs1 + (patterns2 - inputs1) * masks_output
    return inputs_cross, patterns2, masks_output

def generalize_to_lower_pratio(pratio, bs):

    if pratio * bs >= 1:
        # the normal case that each batch can have at least one poison sample
        return pratio * bs
    else:
        # then randomly return number of poison sample
        if np.random.uniform(0,1) < pratio * bs: # eg. pratio = 1/1280, then 1/10 of batch(bs=128) should contains one sample
            return 1
        else:
            return 0

logging.warning('In train, if ratio of bd/cross/clean being zero, plz checkout the TOTAL number of bd/cross/clean !!!\n\
We set the ratio being 0 if TOTAL number of bd/cross/clean is 0 (otherwise 0/0 happens)')

def train_step(
    netC, netG, netM, optimizerC, optimizerG, schedulerC, schedulerG, train_dl1, train_dl2, epoch, opt
):
    netC.train()
    netG.train()
    logging.info(" Training:")
    total = 0
    total_cross = 0
    total_bd = 0
    total_clean = 0

    total_correct_clean = 0
    total_cross_correct = 0
    total_bd_correct = 0

    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    criterion_div = nn.MSELoss(reduction="none")
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):
        optimizerC.zero_grad()

        inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
        inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)

        bs = inputs1.shape[0]
        num_bd = int(generalize_to_lower_pratio(opt.pratio, bs)) #int(opt.pratio * bs)
        num_cross = num_bd

        inputs_bd, targets_bd, patterns1, masks1 = create_bd(inputs1[:num_bd], targets1[:num_bd], netG, netM, opt, 'train')
        inputs_cross, patterns2, masks2 = create_cross(
            inputs1[num_bd : num_bd + num_cross], inputs2[num_bd : num_bd + num_cross], netG, netM, opt
        )

        total_inputs = torch.cat((inputs_bd, inputs_cross, inputs1[num_bd + num_cross :]), 0)
        total_targets = torch.cat((targets_bd, targets1[num_bd:]), 0)

        preds = netC(total_inputs)
        loss_ce = criterion(preds, total_targets)

        # Calculating diversity loss
        distance_images = criterion_div(inputs1[:num_bd], inputs2[num_bd : num_bd + num_bd])
        distance_images = torch.mean(distance_images, dim=(1, 2, 3))
        distance_images = torch.sqrt(distance_images)

        distance_patterns = criterion_div(patterns1, patterns2)
        distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
        distance_patterns = torch.sqrt(distance_patterns)

        loss_div = distance_images / (distance_patterns + opt.EPSILON)
        loss_div = torch.mean(loss_div) * opt.lambda_div

        total_loss = loss_ce + loss_div
        total_loss.backward()
        optimizerC.step()
        optimizerG.step()

        total += bs
        total_bd += num_bd
        total_cross += num_cross
        total_clean += bs - num_bd - num_cross

        total_correct_clean += torch.sum(
            torch.argmax(preds[num_bd + num_cross :], dim=1) == total_targets[num_bd + num_cross :]
        )
        total_cross_correct += (torch.sum(
            torch.argmax(preds[num_bd : num_bd + num_cross], dim=1) == total_targets[num_bd : num_bd + num_cross]
        )) if num_cross > 0 else 0
        total_bd_correct += (torch.sum(torch.argmax(preds[:num_bd], dim=1) == targets_bd)) if num_bd > 0 else 0
        total_loss += loss_ce.detach() * bs
        avg_loss = total_loss / total

        acc_clean = total_correct_clean  / total_clean
        acc_bd = (total_bd_correct  / total_bd) if total_bd > 0 else 0
        acc_cross = (total_cross_correct  / total_cross) if total_cross > 0 else 0
        infor_string = "CE loss: {:.4f} - Accuracy: {:.3f} | BD Accuracy: {:.3f} | Cross Accuracy: {:3f}".format(
            avg_loss, acc_clean, acc_bd, acc_cross
        )
        progress_bar(batch_idx, len(train_dl1), infor_string)

        # Saving images for debugging

        if batch_idx == len(train_dl1) - 2 and num_bd > 0:
            dir_temps = os.path.join(opt.temps, opt.dataset)
            if not os.path.exists(dir_temps):
                os.makedirs(dir_temps)
            images = netG.denormalize_pattern(torch.cat((inputs1[:num_bd], patterns1, inputs_bd), dim=2))
            file_name = "{}_{}_images.png".format(opt.dataset, opt.attack_mode)
            file_path = os.path.join(dir_temps, file_name)
            torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)

    if opt.C_lr_scheduler == "ReduceLROnPlateau":
        schedulerC.step(loss_ce)
    else:
        schedulerC.step()
    schedulerG.step()

    #agg
    # logging.info(f'End train epoch {epoch} : acc_clean : {acc_clean}, acc_bd : {acc_bd}, acc_cross : {acc_cross} ')
    agg({
        'train_epoch_num':float(epoch),
        'train_acc_clean':float(acc_clean),
        'train_acc_bd':float(acc_bd),
        'train_acc_cross':float(acc_cross),
    })

    return


def eval(
    netC,
    netG,
    netM,
    optimizerC,
    optimizerG,
    schedulerC,
    schedulerG,
    test_dl1,
    test_dl2,
    epoch,
    opt,
):
    netC.eval()
    netG.eval()
    logging.info(" Eval:")


    # set shuffle here = False, since other place need randomness to generate cross sample.
    test_dl1 = torch.utils.data.DataLoader(test_dl1.dataset, batch_size=opt.batchsize,
                                                      num_workers=opt.num_workers,
                                                      shuffle=False)

    transforms_reversible = transforms.Compose(
        list(
            filter(
                lambda x: isinstance(x, (transforms.Normalize, transforms.Resize, transforms.ToTensor)),
                deepcopy(test_dl1.dataset.ori_image_transform_in_loading.transforms)
            )
        )
    )
    # get denormalizer
    for trans_t in deepcopy(test_dl1.dataset.ori_image_transform_in_loading.transforms):
        if isinstance(trans_t, transforms.Normalize):
            denormalizer = get_dataset_denormalization(trans_t)
            logging.info(f"{denormalizer}")

    # Notice that due to the fact that we need random sequence to get cross samples
    # So we set the reversible_test_dl2 with shuffle = True
    reversible_test_ds1 = deepcopy(test_dl1.dataset)
    reversible_test_ds1.ori_image_transform_in_loading = transforms_reversible
    reversible_test_dl1 = torch.utils.data.DataLoader(reversible_test_ds1, batch_size=opt.batchsize, num_workers=opt.num_workers,
                                                     shuffle=False)

    reversible_test_ds2 = deepcopy(test_dl1.dataset)
    reversible_test_ds2.ori_image_transform_in_loading = transforms_reversible
    reversible_test_dl2 = torch.utils.data.DataLoader(reversible_test_ds2, batch_size=opt.batchsize,
                                                      num_workers=opt.num_workers,
                                                      shuffle=True)

    x_poison, y_poison = [], []
    x_cross, y_cross = [], []

    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(reversible_test_dl1)), reversible_test_dl1, reversible_test_dl2):
        with torch.no_grad():
            inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
            inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)
            bs = inputs1.shape[0]

            inputs_bd, targets_bd, _, _,  position_changed, targets = create_bd(inputs1, targets1, netG, netM, opt, 'test')

            inputs_cross, _, _ = create_cross(inputs1, inputs2, netG, netM, opt)

            x_poison += ([to_pil(denormalizer(t_img)) for (t_img) in inputs_bd.detach().clone().cpu()])
            y_poison += targets_bd.detach().clone().cpu().tolist()

            x_cross += ([to_pil(denormalizer(t_img)) for t_img in inputs_cross.detach().clone().cpu()])
            y_cross += (targets.detach().clone().cpu().tolist())

    poison_test_ds = xy_iter(x_poison, y_poison, deepcopy(test_dl1.dataset.ori_image_transform_in_loading))
    poison_test_dl = torch.utils.data.DataLoader(poison_test_ds, batch_size=opt.batchsize, num_workers=opt.num_workers,
                                                 shuffle=False)

    cross_test_ds = xy_iter(x_cross, y_cross, deepcopy(test_dl1.dataset.ori_image_transform_in_loading))
    cross_test_dl = torch.utils.data.DataLoader(cross_test_ds, batch_size=opt.batchsize, num_workers=opt.num_workers,
                                                shuffle=False)

    trainer = ModelTrainerCLS(netC)
    trainer.criterion = torch.nn.CrossEntropyLoss()

    if epoch == 1 or epoch % ((opt.epochs//10) + 1) == (opt.epochs//10):
        sample_pil_imgs(test_dl1.dataset.data, f"{opt.save_path}/test_dl_{epoch}_samples")
        sample_pil_imgs(poison_test_dl.dataset.data, f"{opt.save_path}/poison_test_dl_{epoch}_samples")
        sample_pil_imgs(cross_test_dl.dataset.data, f"{opt.save_path}/cross_test_dl_{epoch}_samples")

    clean_test_metric = trainer.test(
        test_dl1, device=opt.device
    )
    avg_acc_clean = clean_test_metric['test_correct'] / clean_test_metric['test_total']

    poison_test_metric = trainer.test(
        poison_test_dl, device=opt.device
    )
    avg_acc_bd = poison_test_metric['test_correct'] / poison_test_metric['test_total']

    cross_test_metric = trainer.test(
        cross_test_dl, device=opt.device
    )
    avg_acc_cross = cross_test_metric['test_correct'] / cross_test_metric['test_total']

    logging.info(
            f"epoch:{epoch}, acc_clean:{avg_acc_clean},  acc_bd:{avg_acc_bd},  acc_cross:{avg_acc_cross}"
        )

    logging.info(" Saving!!")

    state_dict = {
        "netC": netC.state_dict(),
        "netG": netG.state_dict(),
        "netM": netM.state_dict(),
        "optimizerC": optimizerC.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "schedulerC": schedulerC.state_dict(),
        "schedulerG": schedulerG.state_dict(),
        "test_avg_acc_clean": avg_acc_clean,
        "test_avg_acc_bd": avg_acc_bd,
        "test_avg_acc_cross": avg_acc_cross,
        "epoch": epoch,
        "opt": opt,
    }

    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    torch.save(state_dict, ckpt_path)
    return avg_acc_clean,avg_acc_bd,avg_acc_cross, epoch


# -------------------------------------------------------------------------------------
def train_mask_step(netM, optimizerM, schedulerM, train_dl1, train_dl2, epoch, opt):
    netM.train()
    logging.info(" Training:")
    total = 0

    total_loss = 0
    criterion_div = nn.MSELoss(reduction="none")
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):
        optimizerM.zero_grad()

        inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
        inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)

        bs = inputs1.shape[0]
        masks1 = netM(inputs1)
        masks1, masks2 = netM.threshold(netM(inputs1)), netM.threshold(netM(inputs2))

        # Calculating diversity loss
        distance_images = criterion_div(inputs1, inputs2)
        distance_images = torch.mean(distance_images, dim=(1, 2, 3))
        distance_images = torch.sqrt(distance_images)

        distance_patterns = criterion_div(masks1, masks2)
        distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
        distance_patterns = torch.sqrt(distance_patterns)

        loss_div = distance_images / (distance_patterns + opt.EPSILON)
        loss_div = torch.mean(loss_div) * opt.lambda_div

        loss_norm = torch.mean(F.relu(masks1 - opt.mask_density))

        total_loss = opt.lambda_norm * loss_norm + opt.lambda_div * loss_div
        total_loss.backward()
        optimizerM.step()
        infor_string = "Mask loss: {:.4f} - Norm: {:.3f} | Diversity: {:.3f}".format(total_loss, loss_norm, loss_div)
        progress_bar(batch_idx, len(train_dl1), infor_string)

        # Saving images for debugging
        if batch_idx == len(train_dl1) - 2:
            dir_temps = os.path.join(opt.temps, opt.dataset, "masks")
            if not os.path.exists(dir_temps):
                os.makedirs(dir_temps)
            path_masks = os.path.join(dir_temps, "{}_{}_masks.png".format(opt.dataset, opt.attack_mode))
            torchvision.utils.save_image(masks1, path_masks, pad_value=1)



    schedulerM.step()


def eval_mask(netM, optimizerM, schedulerM, test_dl1, test_dl2, epoch, opt):
    netM.eval()
    logging.info(" Eval:")
    total = 0.0

    criterion_div = nn.MSELoss(reduction="none")
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(test_dl1)), test_dl1, test_dl2):
        with torch.no_grad():
            inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
            inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)
            bs = inputs1.shape[0]
            masks1, masks2 = netM.threshold(netM(inputs1)), netM.threshold(netM(inputs2))

            # Calculating diversity loss
            distance_images = criterion_div(inputs1, inputs2)
            distance_images = torch.mean(distance_images, dim=(1, 2, 3))
            distance_images = torch.sqrt(distance_images)

            distance_patterns = criterion_div(masks1, masks2)
            distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
            distance_patterns = torch.sqrt(distance_patterns)

            loss_div = distance_images / (distance_patterns + opt.EPSILON)
            loss_div = torch.mean(loss_div) * opt.lambda_div

            loss_norm = torch.mean(F.relu(masks1 - opt.mask_density))

            infor_string = "Norm: {:.3f} | Diversity: {:.3f}".format(loss_norm, loss_div)
            progress_bar(batch_idx, len(test_dl1), infor_string)

    state_dict = {
        "netM": netM.state_dict(),
        "optimizerM": optimizerM.state_dict(),
        "schedulerM": schedulerM.state_dict(),
        "epoch": epoch,
        "opt": opt,
    }
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, "mask")
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    torch.save(state_dict, ckpt_path)
    return epoch


# -------------------------------------------------------------------------------------


def train(opt):
    ### 3. set the device, model, criterion, optimizer, training schedule.
    logging.info('use generate_cls_model() ')
    netC = generate_cls_model(opt.model, opt.num_classes,image_size=opt.img_size[0],)
    if torch.cuda.device_count() > 1 and opt.device == 'cuda':
        logging.info("device='cuda', default use all device")
        netC = torch.nn.DataParallel(netC)
    netC.to(opt.device)
    logging.warning(f'actually model use = {opt.model}')
    netG = Generator(opt)
    if torch.cuda.device_count() > 1 and opt.device == 'cuda':
        logging.info("device='cuda', default use all device")
        netG = torch.nn.DataParallel(netG)
    netG.to(opt.device)
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)
    optimizerG = torch.optim.Adam(netG.parameters(), opt.lr_G, betas=(0.5, 0.9))

    if opt.C_lr_scheduler == "ReduceLROnPlateau":
        schedulerC = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerC)
    else:
        schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, opt.schedulerG_milestones, opt.schedulerG_lambda)

    netM = Generator(opt, out_channels=1).to(opt.device)
    if torch.cuda.device_count() > 1 and opt.device == 'cuda':
        logging.info("device='cuda', default use all device")
        netM = torch.nn.DataParallel(netM)
    optimizerM = torch.optim.Adam(netM.parameters(), opt.lr_M, betas=(0.5, 0.9))
    schedulerM = torch.optim.lr_scheduler.MultiStepLR(optimizerM, opt.schedulerM_milestones, opt.schedulerM_lambda)

    # For tensorboard
    log_dir = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, "log_dir")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Continue training ?
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    if opt.continue_training and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path)
        netC.load_state_dict(state_dict["netC"])
        netG.load_state_dict(state_dict["netG"])
        netM.load_state_dict(state_dict["netM"])
        epoch = state_dict["epoch"] + 1
        optimizerC.load_state_dict(state_dict["optimizerC"])
        optimizerG.load_state_dict(state_dict["optimizerG"])
        schedulerC.load_state_dict(state_dict["schedulerC"])
        schedulerG.load_state_dict(state_dict["schedulerG"])

        opt = state_dict["opt"]
        logging.info("Continue training")
    else:
        # Prepare mask

        epoch = 1

        # Reset tensorboard
        shutil.rmtree(log_dir)
        os.makedirs(log_dir)
        logging.info("Training from scratch")

    ### 3. set the clean train data and clean test data
    train_dl1 = get_dataloader(opt, train=True)
    train_dl2 = get_dataloader(opt, train=True)
    test_dl1 = get_dataloader(opt, train=False)
    test_dl2 = get_dataloader(opt, train=False)

    logging.info(pformat(opt.__dict__)) #set here since the opt change at beginning of this function
    try:
        logging.info(pformat(get_git_info()))
    except:
        logging.info('Getting git info fails.')
    ### 4. clean train 25 epochs
    if epoch == 1:
        netM.train()
        for i in range(opt.clean_train_epochs):
            logging.info(
                "Epoch {} - {} - {} | mask_density: {} - lambda_div: {}  - lambda_norm: {}:".format(
                    epoch, opt.dataset, opt.attack_mode, opt.mask_density, opt.lambda_div, opt.lambda_norm
                )
            )
            train_mask_step(netM, optimizerM, schedulerM, train_dl1, train_dl2, epoch, opt)
            epoch = eval_mask(netM, optimizerM, schedulerM, test_dl1, test_dl2, epoch, opt)
            epoch += 1
    netM.eval()
    netM.requires_grad_(False)

    ### 5. training with backdoor modification simultaneously
    for i in range(opt.epochs):
        logging.info(
            "Epoch {} - {} - {} | mask_density: {} - lambda_div: {}:".format(
                epoch, opt.dataset, opt.attack_mode, opt.mask_density, opt.lambda_div
            )
        )

        train_step(
            netC,
            netG,
            netM,
            optimizerC,
            optimizerG,
            schedulerC,
            schedulerG,
            train_dl1,
            train_dl2,
            epoch,
            opt,
        )

        test_avg_acc_clean, test_avg_acc_bd, test_avg_acc_cross, epoch=eval(
            netC,
            netG,
            netM,
            optimizerC,
            optimizerG,
            schedulerC,
            schedulerG,
            test_dl1,
            test_dl2,
            epoch,
            opt,
        )
        agg({
            "test_avg_acc_clean":float(test_avg_acc_clean),
            "test_avg_acc_bd":float(test_avg_acc_bd),
            "test_avg_acc_cross":float(test_avg_acc_cross),
            "test_epoch_num":float(epoch),
        })

        epoch += 1
        if epoch > opt.epochs:
            break

    ###6. save attack result

    train_dl1.dataset.ori_image_transform_in_loading = transforms.Compose(list(filter(lambda x: isinstance(x, (transforms.Normalize, transforms.Resize, transforms.ToTensor)),
                                         train_dl1.dataset.ori_image_transform_in_loading.transforms)))
    train_dl2.dataset.ori_image_transform_in_loading = transforms.Compose(list(filter(lambda x: isinstance(x, (transforms.Normalize, transforms.Resize, transforms.ToTensor)),
                                         train_dl2.dataset.ori_image_transform_in_loading.transforms)))
    for trans_t in train_dl1.dataset.ori_image_transform_in_loading.transforms:
        if isinstance(trans_t, transforms.Normalize):
            denormalizer = get_dataset_denormalization(trans_t)
            logging.info(f"{denormalizer}")


    train_dl1 = torch.utils.data.DataLoader(
        train_dl1.dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=False)
    train_dl2 = torch.utils.data.DataLoader(
        train_dl2.dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=True) # true since only the first one decide the order.
    one_hot_original_index = []
    original_targets = []
    bd_input = []
    bd_targets = []
    netC.eval()
    netC.to(opt.device)
    netG.eval()
    netG.to(opt.device)
    netM.eval()
    netM.to(opt.device)
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):

        optimizerC.zero_grad()

        inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
        inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)

        bs = inputs1.shape[0]
        num_bd = int(generalize_to_lower_pratio(opt.pratio, bs)) #int(opt.pratio * bs)
        num_cross = num_bd

        inputs_bd, targets_bd, patterns1, masks1 = create_bd(inputs1[:num_bd], targets1[:num_bd], netG, netM, opt, 'train')
        inputs_cross, patterns2, masks2 = create_cross(
            inputs1[num_bd : num_bd + num_cross], inputs2[num_bd : num_bd + num_cross], netG, netM, opt
        )
        if num_bd > 0:
            inputs_bd = torch.cat([denormalizer(img)[None, ...] for img in inputs_bd])
        if num_cross > 0:
            inputs_cross = torch.cat([denormalizer(img)[None, ...] for img in inputs_cross])

        inputs_bd_cpu, inputs_cross_cpu = inputs_bd.detach().clone().cpu(), inputs_cross.detach().clone().cpu()
        targets_bd_cpu, targets1_cpu =  targets_bd.detach().clone().cpu(), targets1.detach().clone().cpu()

        one_hot = np.zeros(bs)
        one_hot[:(num_bd + num_cross)] = 1
        one_hot_original_index.append(one_hot)
        original_targets += ((targets1.detach().clone().cpu())[: (num_bd + num_cross)]).tolist()
        bd_input.append(torch.cat([inputs_bd_cpu, inputs_cross_cpu], dim=0))
        bd_targets.append(torch.cat([targets_bd_cpu, targets1_cpu[num_bd: (num_bd + num_cross)]], dim=0))

    bd_train_x = [to_pil(t_img) for t_img in torch.cat(bd_input, dim=0).float().cpu()]
    bd_train_y = torch.cat(bd_targets, dim=0).long().cpu().numpy()
    train_poison_indicator = np.concatenate(one_hot_original_index)
    bd_train_original_index = np.where(train_poison_indicator == 1)[
        0] if train_poison_indicator is not None else None

    bd_train_for_save = prepro_cls_DatasetBD(
        full_dataset_without_transform=list(zip(bd_train_x, bd_train_y)),
        poison_idx=np.ones_like(bd_train_y),
        add_details_in_preprocess=True,
        clean_image_pre_transform=None,
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        end_pre_process=None,
        ori_image_transform_in_loading=None,
        ori_label_transform_in_loading=None,
    )
    bd_train_for_save.original_targets = np.array(original_targets)
    bd_train_for_save.original_index = np.array(bd_train_original_index)
    bd_train_for_save.dataset = None

    print(
        # bd_train_for_save.data,
        bd_train_for_save.poison_indicator,
        bd_train_for_save.original_index,
        bd_train_for_save.original_targets,
        bd_train_for_save.targets,
    )

    test_dl1.dataset.ori_image_transform_in_loading = transforms.Compose(list(filter(lambda x: isinstance(x, (transforms.Normalize, transforms.Resize, transforms.ToTensor)),
                                         test_dl1.dataset.ori_image_transform_in_loading.transforms)))
    test_dl2.dataset.ori_image_transform_in_loading = transforms.Compose(list(filter(lambda x: isinstance(x, (transforms.Normalize, transforms.Resize, transforms.ToTensor)),
                                         test_dl2.dataset.ori_image_transform_in_loading.transforms)))
    for trans_t in test_dl1.dataset.ori_image_transform_in_loading.transforms:
        if isinstance(trans_t, transforms.Normalize):
            denormalizer = get_dataset_denormalization(trans_t)
            logging.info(f"{denormalizer}")

    test_dl1 = torch.utils.data.DataLoader(
        test_dl1.dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=False)
    test_dl2 = torch.utils.data.DataLoader(
        test_dl2.dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=True)

    test_bd_input = []
    test_bd_targets = []
    test_bd_poison_indicator = []
    test_bd_origianl_targets = []
    netC.eval()
    netC.to(opt.device)
    netG.eval()
    netG.to(opt.device)
    netM.eval()
    netM.to(opt.device)
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(test_dl1)), test_dl1, test_dl2):
        with torch.no_grad():
            inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
            inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)
            bs = inputs1.shape[0]

            inputs_bd, targets_bd, _, _,  position_changed, targets = create_bd(inputs1, targets1, netG, netM, opt, 'test')
            inputs_bd = torch.cat([denormalizer(img)[None, ...]  for img in inputs_bd])

            # inputs_cross, _, _ = create_cross(inputs1, inputs2, netG, netM, opt)

            inputs_bd_cpu = inputs_bd.detach().clone().cpu()
            targets_bd_cpu = targets_bd.detach().clone().cpu()
            targets_cpu = targets.detach().clone().cpu()

            test_bd_input.append(inputs_bd_cpu)
            test_bd_targets.append(targets_bd_cpu)

            test_bd_poison_indicator.append(position_changed)
            test_bd_origianl_targets.append(targets_cpu)

    bd_test_x = [to_pil(t_img) for t_img in torch.cat(test_bd_input, dim=0).float().cpu()]
    bd_test_y = torch.cat(test_bd_targets, dim=0).long().cpu().numpy()
    test_bd_origianl_index = np.where(torch.cat(test_bd_poison_indicator, dim=0).long().cpu().numpy())[0]
    test_bd_origianl_targets = torch.cat(test_bd_origianl_targets, dim=0).long().cpu()
    test_bd_origianl_targets = test_bd_origianl_targets[test_bd_origianl_index]

    bd_test_for_save = prepro_cls_DatasetBD(
        full_dataset_without_transform=list(zip(bd_test_x, bd_test_y)),
        poison_idx=np.ones_like(bd_test_y),
        add_details_in_preprocess=True,
        clean_image_pre_transform=None,
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        end_pre_process=None,
        ori_image_transform_in_loading=None,
        ori_label_transform_in_loading=None,
    )
    bd_test_for_save.dataset = None
    bd_test_for_save.original_index = np.array(test_bd_origianl_index)
    bd_test_for_save.original_targets = np.array(test_bd_origianl_targets)

    print(
        # bd_test_for_save.data,
        bd_test_for_save.poison_indicator,
        bd_test_for_save.original_index,
        bd_test_for_save.original_targets,
        bd_test_for_save.targets,
    )

    save_attack_result(
        model_name=opt.model,
        num_classes=opt.num_classes,
        model=netC.cpu().state_dict(),
        data_path=opt.dataset_path,
        img_size=(opt.input_height, opt.input_width, opt.input_channel),
        clean_data=opt.dataset,
        bd_train=bd_train_for_save,
        bd_test=bd_test_for_save,
        save_path=f'{opt.save_path}',
    )

    # final_save_dict = {
    #         'model_name': opt.model,
    #         'num_classes': opt.num_classes,
    #         'model': netC.cpu().state_dict(),
    #
    #         'data_path': opt.dataset_path,
    #         'img_size': (opt.input_height, opt.input_width, opt.input_channel),
    #
    #         'clean_data': opt.dataset,
    #
    #         'bd_train': ({
    #             'x': bd_train_x,
    #             'y': bd_train_y,
    #             'original_index': bd_train_original_index ,
    #         }),
    #
    #         'bd_test': {
    #             'x': bd_test_x,
    #             'y': bd_test_y,
    #             'original_index': test_bd_origianl_index,
    #             'original_targets': test_bd_origianl_targets,
    #         },
    #     }
    #
    # logging.info(f"save dict summary : {summary_dict(final_save_dict)}")
    # torch.save(
    #     final_save_dict,
    #
    #     f'{opt.save_path}/attack_result.pt',
    # )

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--yaml_path', type=str, default='../config/attack/inputaware/default.yaml',
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--model', type=str, help='Only use when model is not given in original code !!!')
    parser.add_argument('--save_folder_name', type=str,
                        help='(Optional) should be time str + given unique identification str')
    parser.add_argument("--continue_training", action="store_true")

    parser.add_argument("--dataset_path", type=str, )#default="data/")
    parser.add_argument("--checkpoints", type=str, )#default="./record/inputAwareAttack/checkpoints/")
    parser.add_argument("--temps", type=str, )#default="./record/inputAwareAttack/temps")
    parser.add_argument("--save_path", type=str, )#default="./record/inputAwareAttack/")
    parser.add_argument("--device", type=str, )#default="cuda")

    parser.add_argument("--dataset", type=str, )#default="cifar10")


    parser.add_argument("--batchsize", type=int, )#default=128)
    parser.add_argument("--lr_G", type=float, )#default=1e-2)
    parser.add_argument("--lr_C", type=float, )#default=1e-2)
    parser.add_argument("--lr_M", type=float, )#default=1e-2)

    parser.add_argument('--C_lr_scheduler', type = str)

    parser.add_argument("--schedulerG_milestones", type=list, )#default=[200, 300, 400, 500])
    parser.add_argument("--schedulerC_milestones", type=list, )#default=[100, 200, 300, 400])
    parser.add_argument("--schedulerM_milestones", type=list, )#default=[10, 20])
    parser.add_argument("--schedulerG_lambda", type=float, )#default=0.1)
    parser.add_argument("--schedulerC_lambda", type=float, )#default=0.1)
    parser.add_argument("--schedulerM_lambda", type=float, )#default=0.1)
    parser.add_argument("--epochs", type=int, )#default=100)
    parser.add_argument("--lambda_div", type=float, )#default=1)
    parser.add_argument("--lambda_norm", type=float, )#default=100)
    parser.add_argument("--num_workers", type=float, )#default=4)

    parser.add_argument("--target_label", type=int, )#default=0)
    parser.add_argument("--attack_mode", type=str, )#default="all2one", help="all2one or all2all")
    parser.add_argument("--pratio", type=float, )#default=0.1)
    # parser.add_argument("--p_cross", type=float, )#default=0.1)
    parser.add_argument("--mask_density", type=float, )#default=0.032)
    parser.add_argument("--EPSILON", type=float, )#default=1e-7)
    parser.add_argument('--clean_train_epochs',type =int)

    parser.add_argument("--random_rotation", type=int, )#default=10)
    parser.add_argument("--random_crop", type=int, )#default=5)
    parser.add_argument("--random_seed", type=int, )#default=0)

    return parser



def main():
    ### 1. config args, save_path, fix random seed
    opt = get_arguments().parse_args()

    with open(opt.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)
    defaults.update({k: v for k, v in opt.__dict__.items() if v is not None})
    opt.__dict__ = defaults

    opt.dataset_path = opt.dataset_path

    opt.terminal_info = sys.argv

    opt.num_classes = get_num_classes(opt.dataset)
    opt.input_height, opt.input_width, opt.input_channel = get_input_shape(opt.dataset)
    opt.img_size = (opt.input_height, opt.input_width, opt.input_channel)
    opt.dataset_path = f"{opt.dataset_path}/{opt.dataset}"

    if 'save_folder_name' not in opt:
        save_path = generate_save_folder(
            run_info='inputaware',
            given_load_file_path=None,
            all_record_folder_path='../record',
        )
    else:
        save_path = '../record/' + opt.save_folder_name
        os.mkdir(save_path)

    opt.save_path = save_path

    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    # logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")

    fileHandler = logging.FileHandler(save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)


    fix_random(int(opt.random_seed))

    train(opt)

    torch.save(opt.__dict__, save_path + '/info.pickle')

    agg.to_dataframe().to_csv(f"{save_path}/attack_df.csv")
    agg.summary().to_csv(f"{save_path}/attack_df_summary.csv")


if __name__ == "__main__":
    main()

'''
original license:

MIT License

Copyright (c) 2021 VinAI Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''