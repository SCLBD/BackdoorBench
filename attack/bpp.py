'''
BppAttack: Stealthy and Efficient Trojan Attacks Against Deep Neural Networks via Image Quantization and Contrastive Adversarial Learning
this script is for bpp attack
github link : https://github.com/RU-System-Software-and-Security/BppAttack

@InProceedings{Wang_2022_CVPR,
    author    = {Wang, Zhenting and Zhai, Juan and Ma, Shiqing},
    title     = {BppAttack: Stealthy and Efficient Trojan Attacks Against Deep Neural Networks via Image Quantization and Contrastive Adversarial Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {15074-15084}
}

basic sturcture for main:
1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the device, model, criterion, optimizer, training schedule.
4. set the backdoor image processing, Image quantization, Dithering,
5. training with backdoor modification simultaneously, which include Contrastive Adversarial Training
6. save attack result



license from the original code:

MIT License

Copyright (c) 2022 RUSSS

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

import sys, os, logging
import os
import sys

sys.path = ["./"] + sys.path

import time
import argparse
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor

to_pil = ToPILImage()
to_tensor = ToTensor()
from torch.utils.data import DataLoader

import numpy as np
import torch
import torchvision.transforms as transforms

import random
from numba import jit
from numba.types import float64, int64

from utils.aggregate_block.dataset_and_transform_generate import get_dataset_normalization, get_dataset_denormalization
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.trainer_cls import Metric_Aggregator
from utils.save_load_attack import save_attack_result
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from attack.badnet import add_common_attack_args, BadNet
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from utils.trainer_cls import all_acc, given_dataloader_test, general_plot_for_epoch


def generalize_to_lower_pratio(pratio, bs):
    if pratio * bs >= 1:
        # the normal case that each batch can have at least one poison sample
        return pratio * bs
    else:
        # then randomly return number of poison sample
        if np.random.uniform(0,
                             1) < pratio * bs:  # eg. pratio = 1/1280, then 1/10 of batch(bs=128) should contains one sample
            return 1
        else:
            return 0


def back_to_np_4d(inputs, args):
    if args.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif args.dataset == "cifar100":
        expected_values = [0.5071, 0.4867, 0.4408]
        variance = [0.2675, 0.2565, 0.2761]
    elif args.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5]
    elif args.dataset in ["gtsrb", "celeba"]:
        expected_values = [0, 0, 0]
        variance = [1, 1, 1]
    elif args.dataset == "imagenet":
        expected_values = [0.485, 0.456, 0.406]
        variance = [0.229, 0.224, 0.225]
    elif args.dataset == "tiny":
        expected_values = [0.4802, 0.4481, 0.3975]
        variance = [0.2302, 0.2265, 0.2262]
    inputs_clone = inputs.clone()

    if args.dataset == "mnist":
        inputs_clone[:, :, :, :] = inputs_clone[:, :, :, :] * variance[0] + expected_values[0]
    else:
        for channel in range(3):
            inputs_clone[:, channel, :, :] = inputs_clone[:, channel, :, :] * variance[channel] + expected_values[
                channel]

    return inputs_clone * 255


def np_4d_to_tensor(inputs, args):
    if args.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif args.dataset == "cifar100":
        expected_values = [0.5071, 0.4867, 0.4408]
        variance = [0.2675, 0.2565, 0.2761]
    elif args.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5]
    elif args.dataset in ["gtsrb", "celeba"]:
        expected_values = [0, 0, 0]
        variance = [1, 1, 1]
    elif args.dataset == "imagenet":
        expected_values = [0.485, 0.456, 0.406]
        variance = [0.229, 0.224, 0.225]
    elif args.dataset == "tiny":
        expected_values = [0.4802, 0.4481, 0.3975]
        variance = [0.2302, 0.2265, 0.2262]
    inputs_clone = inputs.clone().div(255.0)

    if args.dataset == "mnist":
        inputs_clone[:, :, :, :] = (inputs_clone[:, :, :, :] - expected_values[0]).div(variance[0])
    else:
        for channel in range(3):
            inputs_clone[:, channel, :, :] = (inputs_clone[:, channel, :, :] - expected_values[channel]).div(
                variance[channel])
    return inputs_clone


@jit(float64[:](float64[:], int64, float64[:]), nopython=True)
def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)


@jit(nopython=True)
def floydDitherspeed(image, squeeze_num):
    channel, h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[:, y, x]
            temp = np.empty_like(old).astype(np.float64)
            new = rnd1(old / 255.0 * (squeeze_num - 1), 0, temp) / (squeeze_num - 1) * 255
            error = old - new
            image[:, y, x] = new
            if x + 1 < w:
                image[:, y, x + 1] += error * 0.4375
            if (y + 1 < h) and (x + 1 < w):
                image[:, y + 1, x + 1] += error * 0.0625
            if y + 1 < h:
                image[:, y + 1, x] += error * 0.3125
            if (x - 1 >= 0) and (y + 1 < h):
                image[:, y + 1, x - 1] += error * 0.1875
    return image


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(torch.nn.Module):
    def __init__(self, args):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            transforms.RandomCrop((args.input_height, args.input_width), padding=args.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(transforms.RandomRotation(args.random_rotation),
                                             p=0.5)  # 50% random rotation
        if args.dataset == "cifar10":
            self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class Denormalize:
    def __init__(self, args, expected_values, variance):
        self.n_channels = args.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


class Denormalize:
    def __init__(self, args, expected_values, variance):
        self.n_channels = args.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


class Denormalizer:
    def __init__(self, args):
        self.denormalizer = self._get_denormalizer(args)

    def _get_denormalizer(self, args):
        denormalizer = Denormalize(args, get_dataset_normalization(args.dataset).mean,
                                   get_dataset_normalization(args.dataset).std)
        return denormalizer

    def __call__(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x


class Bpp(BadNet):

    def __init__(self):
        super(Bpp, self).__init__()

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/bpp/default.yaml',
                            help='path for yaml file provide additional default attributes')

        parser.add_argument("--neg_ratio", type=float, )  # default=0.2)  
        parser.add_argument("--random_rotation", type=int, )  # default=10)
        parser.add_argument("--random_crop", type=int, )  # default=5)

        parser.add_argument("--squeeze_num", type=int, )  # default=8
        parser.add_argument("--dithering", type=bool, )  # default=False

        return parser

    def stage1_non_training_data_prepare(self):
        logging.info("stage1 start")

        assert "args" in self.__dict__
        args = self.args

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform, \
        clean_train_dataset_with_transform, \
        clean_train_dataset_targets, \
        clean_test_dataset_with_transform, \
        clean_test_dataset_targets \
            = self.benign_prepare()

        logging.info("Be careful, here must replace the regular train tranform with test transform.")
        # you can find in the original code that get_transform function has pretensor_transform=False always.
        clean_train_dataset_with_transform.wrap_img_transform = test_img_transform

        clean_train_dataloader = DataLoader(clean_train_dataset_with_transform, pin_memory=args.pin_memory,
                                            batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        clean_train_dataloader_shuffled = DataLoader(clean_train_dataset_with_transform, pin_memory=args.pin_memory,
                                                     batch_size=args.batch_size, num_workers=args.num_workers,
                                                     shuffle=True)

        clean_test_dataloader = DataLoader(clean_test_dataset_with_transform, pin_memory=args.pin_memory,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers, shuffle=False)
        self.stage1_results = clean_train_dataset_with_transform, \
                              clean_train_dataloader, \
                              clean_train_dataloader_shuffled, \
                              clean_test_dataset_with_transform, \
                              clean_test_dataloader

    def stage2_training(self):
        logging.info(f"stage2 start")
        assert 'args' in self.__dict__
        args = self.args
        agg = Metric_Aggregator()

        clean_train_dataset_with_transform, \
        clean_train_dataloader, \
        clean_train_dataloader_shuffled, \
        clean_test_dataset_with_transform, \
        clean_test_dataloader = self.stage1_results

        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device

            ) if torch.cuda.is_available() else "cpu"
        )

        netC = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        ).to(self.device, non_blocking=args.non_blocking)

        if "," in args.device:
            netC = torch.nn.DataParallel(
                netC,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )

        optimizerC, schedulerC = argparser_opt_scheduler(netC, args=args)

        logging.info("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        best_cross_acc = 0.0
        epoch_current = 0

        # filter out transformation that not reversible
        transforms_reversible = transforms.Compose(
            list(
                filter(
                    lambda x: isinstance(x, (transforms.Normalize, transforms.Resize, transforms.ToTensor)),
                    (clean_test_dataset_with_transform.wrap_img_transform.transforms)
                )
            )
        )
        # get denormalizer
        for trans_t in (clean_test_dataset_with_transform.wrap_img_transform.transforms):
            if isinstance(trans_t, transforms.Normalize):
                denormalizer = get_dataset_denormalization(trans_t)
                logging.info(f"{denormalizer}")



        # ---------------------------
        self.clean_train_dataset = prepro_cls_DatasetBD_v2(
            clean_train_dataset_with_transform, save_folder_path=f"{args.save_path}/clean_train_dataset"
        )
        self.bd_train_dataset = prepro_cls_DatasetBD_v2(
            clean_train_dataset_with_transform, save_folder_path=f"{args.save_path}/bd_train_dataset_Save"
        )
        self.cross_train_dataset = prepro_cls_DatasetBD_v2(
            clean_train_dataset_with_transform, save_folder_path=f"{args.save_path}/cross_train_dataset"
        )
        self.bd_train_dataset_save = prepro_cls_DatasetBD_v2(
            clean_train_dataset_with_transform,
            save_folder_path=f"{args.save_path}/bd_train_dataset"
        )
        for batch_idx, (inputs, targets) in enumerate(clean_train_dataloader):
            with torch.no_grad():

                inputs, targets = inputs.to(self.device, non_blocking=args.non_blocking), targets.to(self.device,
                                                                                                     non_blocking=args.non_blocking)
                # bs = inputs.shape[0]
                bs = args.batch_size
                inputs_bd = torch.round(denormalizer(inputs) * 255)
                inputs = denormalizer(inputs)
                # save clean
                for idx_in_batch, t_img in enumerate(inputs.detach().clone().cpu()):
                    self.clean_train_dataset.set_one_bd_sample(
                        selected_index=int(batch_idx * bs + idx_in_batch),
                        # manually calculate the original index, since we do not shuffle the dataloader
                        img=(t_img),
                        bd_label=int(targets[idx_in_batch]),
                        label=int(targets[idx_in_batch]),
                    )


                if args.dithering:
                    for i in range(inputs_bd.shape[0]):
                        inputs_bd[i, :, :, :] = torch.round(torch.from_numpy(
                            floydDitherspeed(inputs_bd[i].detach().cpu().numpy(), float(args.squeeze_num))).to(
                            args.device))
                else:
                    inputs_bd = torch.round(inputs_bd / 255.0 * (args.squeeze_num - 1)) / (args.squeeze_num - 1) * 255

                inputs_bd = inputs_bd.div(255.0)

                if args.attack_label_trans == "all2one":
                    targets_bd = torch.ones_like(targets) * args.attack_target
                if args.attack_label_trans == "all2all":
                    targets_bd = torch.remainder(targets + 1, args.num_classes)

                targets = targets.detach().clone().cpu()
                y_poison_batch = targets_bd.detach().clone().cpu().tolist()
                for idx_in_batch, t_img in enumerate(inputs_bd.detach().clone().cpu()):
                    self.bd_train_dataset.set_one_bd_sample(
                        selected_index=int(batch_idx * bs + idx_in_batch),
                        # manually calculate the original index, since we do not shuffle the dataloader
                        img=(t_img),
                        bd_label=int(y_poison_batch[idx_in_batch]),
                        label=int(targets[idx_in_batch]),
                    )



        reversible_test_dataset = (clean_test_dataset_with_transform)

        reversible_test_dataset.wrap_img_transform = transforms_reversible

        reversible_test_dataloader = DataLoader(reversible_test_dataset, batch_size=args.batch_size,
                                                                 pin_memory=args.pin_memory,
                                                                 num_workers=args.num_workers, shuffle=False)

        self.clean_test_dataset = prepro_cls_DatasetBD_v2(
            clean_test_dataset_with_transform, save_folder_path=f"{args.save_path}/clean_test_dataset"
        )
        self.bd_test_dataset = prepro_cls_DatasetBD_v2(
            clean_test_dataset_with_transform, save_folder_path=f"{args.save_path}/bd_test_all_dataset"
        )
        self.bd_test_r_dataset = prepro_cls_DatasetBD_v2(
            clean_test_dataset_with_transform, save_folder_path=f"{args.save_path}/bd_test_dataset"
        )
        self.cross_test_dataset = prepro_cls_DatasetBD_v2(
            clean_test_dataset_with_transform, save_folder_path=f"{args.save_path}/cross_test_dataset"
        )
        for batch_idx, (inputs, targets) in enumerate(reversible_test_dataloader):
            with torch.no_grad():
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                bs = inputs.shape[0]
                inputs_bd = torch.round(denormalizer(inputs) * 255)
                inputs = denormalizer(inputs)
                # save clean
                for idx_in_batch, t_img in enumerate(inputs.detach().clone().cpu()):
                    self.clean_test_dataset.set_one_bd_sample(
                        selected_index=int(batch_idx * int(args.batch_size) + idx_in_batch),
                        # manually calculate the original index, since we do not shuffle the dataloader
                        img=(t_img),
                        bd_label=int(targets[idx_in_batch]),
                        label=int(targets[idx_in_batch]),
                    )

                # Evaluate Backdoor
                if args.dithering:
                    for i in range(inputs_bd.shape[0]):
                        inputs_bd[i, :, :, :] = torch.round(torch.from_numpy(
                            floydDitherspeed(inputs_bd[i].detach().cpu().numpy(), float(args.squeeze_num))).to(
                            self.device))

                else:
                    inputs_bd = torch.round(inputs_bd / 255.0 * (args.squeeze_num - 1)) / (args.squeeze_num - 1) * 255

                inputs_bd = inputs_bd.div(255.0)

                if args.attack_label_trans == "all2one":
                    targets_bd = torch.ones_like(targets) * args.attack_target
                    position_changed = (
                            args.attack_target != targets)  # since if label does not change, then cannot tell if the poison is effective or not.
                    targets_bd_r = (torch.ones_like(targets) * args.attack_target)[position_changed]
                    inputs_bd_r = inputs_bd[position_changed]
                if args.attack_label_trans == "all2all":
                    targets_bd = torch.remainder(targets + 1, args.num_classes)
                    targets_bd_r = torch.remainder(targets + 1, args.num_classes)
                    inputs_bd_r = inputs_bd
                    position_changed = torch.ones_like(targets)

                targets = targets.detach().clone().cpu()
                y_poison_batch = targets_bd.detach().clone().cpu().tolist()
                for idx_in_batch, t_img in enumerate(inputs_bd.detach().clone().cpu()):
                    self.bd_test_dataset.set_one_bd_sample(
                        selected_index=int(batch_idx * int(args.batch_size) + idx_in_batch),
                        # manually calculate the original index, since we do not shuffle the dataloader
                        img=(t_img),
                        bd_label=int(y_poison_batch[idx_in_batch]),
                        label=int(targets[idx_in_batch]),
                    )
                y_poison_batch_r = targets_bd_r.detach().clone().cpu().tolist()
                for idx_in_batch, t_img in enumerate(inputs_bd_r.detach().clone().cpu()):
                    self.bd_test_r_dataset.set_one_bd_sample(
                        selected_index=int(batch_idx * int(args.batch_size) + torch.where(position_changed.detach().clone().cpu())[0][
                            idx_in_batch]),
                        # manually calculate the original index, since we do not shuffle the dataloader
                        img=(t_img),
                        bd_label=int(y_poison_batch_r[idx_in_batch]),
                        label=int(targets[torch.where(position_changed.detach().clone().cpu())[0][idx_in_batch]]),
                    )

        for batch_idx, (inputs, targets) in enumerate(reversible_test_dataloader):
            with torch.no_grad():
                inputs = inputs.to(self.device)
                bs = inputs.shape[0]
                t_nom = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
                # Evaluate cross
                if args.neg_ratio:
                    index_list = list(np.arange(len(clean_test_dataset_with_transform)))
                    residual_index = random.sample(index_list, bs)

                    inputs_negative = torch.zeros_like(inputs)
                    inputs_negative1 = torch.zeros_like(inputs)
                    inputs_d = torch.round(denormalizer(inputs) * 255)
                    for i in range(bs):
                        inputs_negative[i] = inputs_d[i] + (
                                    to_tensor(self.clean_test_dataset[residual_index[i]][0]) * 255).to(self.device) - (
                                                         to_tensor(
                                                             self.bd_test_dataset[residual_index[i]][0]) * 255).to(
                            self.device)

                    inputs_negative = inputs_negative.div(255.0)
                    for idx_in_batch, t_img in enumerate(inputs_negative):
                        self.cross_test_dataset.set_one_bd_sample(
                            selected_index=int(batch_idx * int(args.batch_size) + idx_in_batch),
                            # manually calculate the original index, since we do not shuffle the dataloader
                            img=(t_img),
                            bd_label=int(targets[idx_in_batch]),
                            label=int(targets[idx_in_batch]),
                        )



        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            self.bd_test_dataset,
            clean_test_dataset_with_transform.wrap_img_transform,
        )

        bd_test_dataloader = DataLoader(bd_test_dataset_with_transform,
                                        pin_memory=args.pin_memory,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        shuffle=False)

        bd_test_r_dataset_with_transform = dataset_wrapper_with_transform(
            self.bd_test_r_dataset,
            clean_test_dataset_with_transform.wrap_img_transform,
        )
        self.bd_test_r_dataset.subset(
            np.where(self.bd_test_r_dataset.poison_indicator == 1)[0].tolist()
        )
        bd_test_r_dataloader = DataLoader(bd_test_r_dataset_with_transform,
                                          pin_memory=args.pin_memory,
                                          batch_size=args.batch_size,
                                          num_workers=args.num_workers,
                                          shuffle=False)

        if args.neg_ratio:
            cross_test_dataset_with_transform = dataset_wrapper_with_transform(
                self.cross_test_dataset,
                clean_test_dataset_with_transform.wrap_img_transform,
            )
            cross_test_dataloader = DataLoader(cross_test_dataset_with_transform,
                                               pin_memory=args.pin_memory,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=False)

        else:
            cross_test_dataloader = None

        test_dataloaders = (clean_test_dataloader, bd_test_dataloader, cross_test_dataloader, bd_test_r_dataloader)

        train_loss_list = []
        train_mix_acc_list = []
        train_clean_acc_list = []
        train_asr_list = []
        train_ra_list = []
        train_cross_acc_only_list = []

        clean_test_loss_list = []
        bd_test_loss_list = []
        cross_test_loss_list = []
        ra_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []
        test_cross_acc_list = []

        for epoch in range(epoch_current, args.epochs):
            logging.info("Epoch {}:".format(epoch + 1))

            train_epoch_loss_avg_over_batch, \
            train_mix_acc, \
            train_clean_acc, \
            train_asr, \
            train_ra, \
            train_cross_acc = self.train_step(
                netC,
                optimizerC,
                schedulerC,
                clean_train_dataloader_shuffled,
                args)

            clean_test_loss_avg_over_batch, \
            bd_test_loss_avg_over_batch, \
            cross_test_loss_avg_over_batch, \
            ra_test_loss_avg_over_batch, \
            test_acc, \
            test_asr, \
            test_ra, \
            test_cross_acc \
                = self.eval_step(
                netC,
                clean_test_dataset_with_transform,
                clean_test_dataloader,
                bd_test_r_dataloader,
                cross_test_dataloader,
                args,
            )

            agg({
                "epoch": epoch,

                "train_epoch_loss_avg_over_batch": train_epoch_loss_avg_over_batch,
                "train_acc": train_mix_acc,
                "train_acc_clean_only": train_clean_acc,
                "train_asr_bd_only": train_asr,
                "train_ra_bd_only": train_ra,
                "train_cross_acc_only": train_cross_acc,

                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "cross_test_loss_avg_over_batch": cross_test_loss_avg_over_batch,
                "ra_test_loss_avg_over_batch": ra_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
                "test_cross_acc": test_cross_acc,
            })

            train_loss_list.append(train_epoch_loss_avg_over_batch)
            train_mix_acc_list.append(train_mix_acc)
            train_clean_acc_list.append(train_clean_acc)
            train_asr_list.append(train_asr)
            train_ra_list.append(train_ra)
            train_cross_acc_only_list.append(train_cross_acc)

            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            cross_test_loss_list.append(cross_test_loss_avg_over_batch)
            ra_test_loss_list.append(ra_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)
            test_cross_acc_list.append(test_cross_acc)

            general_plot_for_epoch(
                {
                    "Train Acc": train_mix_acc_list,
                    "Train Acc (clean sample only)": train_clean_acc_list,
                    "Train ASR": train_asr_list,
                    "Train RA": train_ra_list,
                    "Train Cross Acc": train_cross_acc_only_list,
                    "Test C-Acc": test_acc_list,
                    "Test ASR": test_asr_list,
                    "Test RA": test_ra_list,
                    "Test Cross Acc": test_cross_acc_list,
                },
                save_path=f"{args.save_path}/acc_like_metric_plots.png",
                ylabel="percentage",
            )

            general_plot_for_epoch(
                {
                    "Train Loss": train_loss_list,
                    "Test Clean Loss": clean_test_loss_list,
                    "Test Backdoor Loss": bd_test_loss_list,
                    "Test Cross Loss": cross_test_loss_list,
                    "Test RA Loss": ra_test_loss_list,
                },
                save_path=f"{args.save_path}/loss_metric_plots.png",
                ylabel="percentage",
            )

            agg.to_dataframe().to_csv(f"{args.save_path}/attack_df.csv")

            if args.frequency_save != 0 and epoch % args.frequency_save == args.frequency_save - 1:
                state_dict = {
                    "netC": netC.state_dict(),
                    "schedulerC": schedulerC.state_dict(),
                    "optimizerC": optimizerC.state_dict(),
                    "epoch_current": epoch,
                }
                torch.save(state_dict, args.save_path + "/state_dict.pt")

        agg.summary().to_csv(f"{args.save_path}/attack_df_summary.csv")


        netC.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(clean_train_dataloader):
                inputs, targets = inputs.to(self.device, non_blocking=args.non_blocking), targets.to(self.device,
                                                                                                    non_blocking=args.non_blocking)
                bs = inputs.shape[0]

                # Create backdoor data
                num_bd = int(generalize_to_lower_pratio(args.pratio, bs))
                num_neg = int(bs * args.neg_ratio)

                if num_bd != 0 and num_neg != 0:
                    inputs_bd = back_to_np_4d(inputs[:num_bd], args)
                    if args.dithering:
                        for i in range(inputs_bd.shape[0]):
                            inputs_bd[i, :, :, :] = torch.round(torch.from_numpy(
                                floydDitherspeed(inputs_bd[i].detach().cpu().numpy(), float(args.squeeze_num))).to(
                                args.device))
                    else:
                        inputs_bd = torch.round(inputs_bd / 255.0 * (args.squeeze_num - 1)) / (args.squeeze_num - 1) * 255

                    inputs_bd = np_4d_to_tensor(inputs_bd, args)

                    if args.attack_label_trans == "all2one":
                        targets_bd = torch.ones_like(targets[:num_bd]) * args.attack_target
                    if args.attack_label_trans == "all2all":
                        targets_bd = torch.remainder(targets[:num_bd] + 1, args.num_classes)

                    train_dataset_num = len(clean_train_dataloader.dataset)
                    if args.dataset == "celeba":
                        index_list = list(np.arange(train_dataset_num))
                        residual_index = random.sample(index_list, bs)
                    else:
                        index_list = list(np.arange(train_dataset_num * 5))
                        residual_index = random.sample(index_list, bs)
                        residual_index = [x % train_dataset_num for x in random.sample(list(index_list), bs)]

                    inputs_negative = torch.zeros_like(inputs[num_bd: (num_bd + num_neg)])
                    inputs_d = torch.round(back_to_np_4d(inputs, args))
                    for i in range(num_neg):
                        inputs_negative[i] = inputs_d[num_bd + i] + (
                                    to_tensor(self.bd_train_dataset[residual_index[i]][0]) * 255).to(self.device).to(
                            self.device) - (to_tensor(self.clean_train_dataset[residual_index[i]][0]) * 255).to(self.device)

                    inputs_negative = torch.clamp(inputs_negative, 0, 255)
                    inputs_negative = np_4d_to_tensor(inputs_negative, args)

                    total_inputs = torch.cat([inputs_bd, inputs_negative, inputs[(num_bd + num_neg):]], dim=0)
                    total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)

                    input_changed = torch.cat([inputs_bd, inputs_negative, ], dim=0).detach().clone().cpu()
                    input_changed = denormalizer(  # since we normalized once, we need to denormalize it back.
                        input_changed
                    ).detach().clone().cpu()
                    target_changed = torch.cat([targets_bd, targets[num_bd: (num_bd + num_neg)], ],
                                            dim=0).detach().clone().cpu()

                elif (num_bd > 0 and num_neg == 0):
                    inputs_bd = back_to_np_4d(inputs[:num_bd], args)
                    if args.dithering:
                        for i in range(inputs_bd.shape[0]):
                            inputs_bd[i, :, :, :] = torch.round(torch.from_numpy(
                                floydDitherspeed(inputs_bd[i].detach().cpu().numpy(), float(args.squeeze_num))).to(
                                args.device))
                    else:
                        inputs_bd = torch.round(inputs_bd / 255.0 * (args.squeeze_num - 1)) / (args.squeeze_num - 1) * 255

                    inputs_bd = np_4d_to_tensor(inputs_bd, args)

                    if args.attack_label_trans == "all2one":
                        targets_bd = torch.ones_like(targets[:num_bd]) * args.attack_target
                    if args.attack_label_trans == "all2all":
                        targets_bd = torch.remainder(targets[:num_bd] + 1, args.num_classes)

                    total_inputs = torch.cat([inputs_bd, inputs[num_bd:]], dim=0)
                    total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)

                    input_changed = inputs_bd.detach().clone().cpu()
                    input_changed = denormalizer(  # since we normalized once, we need to denormalize it back.
                        input_changed
                    ).detach().clone().cpu()
                    target_changed = targets_bd.detach().clone().cpu()
                    

                elif (num_bd == 0 and num_neg > 0):
                    train_dataset_num = len(clean_train_dataloader.dataset)
                    if args.dataset == "celeba":
                        index_list = list(np.arange(train_dataset_num))
                        residual_index = random.sample(index_list, bs)
                    else:
                        index_list = list(np.arange(train_dataset_num * 5))
                        residual_index = random.sample(index_list, bs)
                        residual_index = [x % train_dataset_num for x in random.sample(list(index_list), bs)]

                    inputs_negative = torch.zeros_like(inputs[num_bd: (num_bd + num_neg)])
                    inputs_d = torch.round(back_to_np_4d(inputs, args))
                    for i in range(num_neg):
                        inputs_negative[i] = inputs_d[num_bd + i] + (
                                    to_tensor(self.bd_train_dataset[residual_index[i]][0]) * 255).to(self.device).to(
                            self.device) - (to_tensor(self.clean_train_dataset[residual_index[i]][0]) * 255).to(self.device)

                    inputs_negative = torch.clamp(inputs_negative, 0, 255)
                    inputs_negative = np_4d_to_tensor(inputs_negative, args)

                    total_inputs = inputs
                    total_targets = targets

                    input_changed = inputs_negative.detach().clone().cpu()
                    input_changed = denormalizer(  # since we normalized once, we need to denormalize it back.
                        input_changed
                    ).detach().clone().cpu()
                    target_changed = targets[num_bd: (num_bd + num_neg)].detach().clone().cpu()
                    
                else:
                    continue

                
                # save container
                for idx_in_batch, t_img in enumerate(
                        input_changed
                ):
                    # here we know it starts from 0 and they are consecutive
                    self.bd_train_dataset_save.set_one_bd_sample(
                        selected_index=int(batch_idx * int(args.batch_size) + idx_in_batch),
                        img=(t_img),
                        bd_label=int(target_changed[idx_in_batch]),
                        label=int(targets[idx_in_batch]),
                    )


        save_attack_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=netC.cpu().state_dict(),
            data_path=args.dataset_path,
            img_size=args.img_size,
            clean_data=args.dataset,
            bd_train=self.bd_train_dataset_save,
            bd_test=self.bd_test_r_dataset,
            save_path=args.save_path,
        )
        print("done")

    def train_step(self, netC, optimizerC, schedulerC, clean_train_dataloader, args):
        logging.info(" Train:")
        netC.train()
        rate_bd = args.pratio
        squeeze_num = args.squeeze_num

        criterion_CE = torch.nn.CrossEntropyLoss()

        transforms = PostTensorTransform(args).to(args.device)
        total_time = 0

        batch_loss_list = []
        batch_predict_list = []
        batch_label_list = []
        batch_poison_indicator_list = []
        batch_original_targets_list = []

        for batch_idx, (inputs, targets) in enumerate(clean_train_dataloader):
            optimizerC.zero_grad()

            inputs, targets = inputs.to(self.device, non_blocking=args.non_blocking), targets.to(self.device,
                                                                                                 non_blocking=args.non_blocking)
            bs = inputs.shape[0]

            # Create backdoor data
            num_bd = int(generalize_to_lower_pratio(rate_bd, bs))
            num_neg = int(bs * args.neg_ratio)

            if num_bd != 0 and num_neg != 0:
                inputs_bd = back_to_np_4d(inputs[:num_bd], args)
                if args.dithering:
                    for i in range(inputs_bd.shape[0]):
                        inputs_bd[i, :, :, :] = torch.round(torch.from_numpy(
                            floydDitherspeed(inputs_bd[i].detach().cpu().numpy(), float(squeeze_num))).to(
                            args.device))
                else:
                    inputs_bd = torch.round(inputs_bd / 255.0 * (squeeze_num - 1)) / (squeeze_num - 1) * 255

                inputs_bd = np_4d_to_tensor(inputs_bd, args)

                if args.attack_label_trans == "all2one":
                    targets_bd = torch.ones_like(targets[:num_bd]) * args.attack_target
                if args.attack_label_trans == "all2all":
                    targets_bd = torch.remainder(targets[:num_bd] + 1, args.num_classes)

                train_dataset_num = len(clean_train_dataloader.dataset)
                if args.dataset == "celeba":
                    index_list = list(np.arange(train_dataset_num))
                    residual_index = random.sample(index_list, bs)
                else:
                    index_list = list(np.arange(train_dataset_num * 5))
                    residual_index = random.sample(index_list, bs)
                    residual_index = [x % train_dataset_num for x in random.sample(list(index_list), bs)]

                inputs_negative = torch.zeros_like(inputs[num_bd: (num_bd + num_neg)])
                inputs_d = back_to_np_4d(inputs, args)
                for i in range(num_neg):
                    inputs_negative[i] = inputs_d[num_bd + i] + (
                                to_tensor(self.bd_train_dataset[residual_index[i]][0]) * 255).to(self.device).to(
                        self.device) - (to_tensor(self.clean_train_dataset[residual_index[i]][0]) * 255).to(self.device)

                inputs_negative = torch.clamp(inputs_negative, 0, 255)
                inputs_negative = np_4d_to_tensor(inputs_negative, args)

                total_inputs = torch.cat([inputs_bd, inputs_negative, inputs[(num_bd + num_neg):]], dim=0)
                total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)

            elif (num_bd > 0 and num_neg == 0):
                inputs_bd = back_to_np_4d(inputs[:num_bd], args)
                if args.dithering:
                    for i in range(inputs_bd.shape[0]):
                        inputs_bd[i, :, :, :] = torch.round(torch.from_numpy(
                            floydDitherspeed(inputs_bd[i].detach().cpu().numpy(), float(args.squeeze_num))).to(
                            args.device))
                else:
                    inputs_bd = torch.round(inputs_bd / 255.0 * (squeeze_num - 1)) / (squeeze_num - 1) * 255

                inputs_bd = np_4d_to_tensor(inputs_bd, args)

                if args.attack_label_trans == "all2one":
                    targets_bd = torch.ones_like(targets[:num_bd]) * args.attack_target
                if args.attack_label_trans == "all2all":
                    targets_bd = torch.remainder(targets[:num_bd] + 1, args.num_classes)

                total_inputs = torch.cat([inputs_bd, inputs[num_bd:]], dim=0)
                total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)

            elif (num_bd == 0):
                total_inputs = inputs
                total_targets = targets

            total_inputs = transforms(total_inputs)
            start = time.time()
            total_preds = netC(total_inputs)
            total_time += time.time() - start

            loss_ce = criterion_CE(total_preds, total_targets)

            loss = loss_ce
            loss.backward()

            optimizerC.step()

            batch_loss_list.append(loss.item())
            batch_predict_list.append(torch.max(total_preds, -1)[1].detach().clone().cpu())
            batch_label_list.append(total_targets.detach().clone().cpu())

            poison_indicator = torch.zeros(bs)
            poison_indicator[:num_bd] = 1  # all others are cross/clean samples cannot conut up to train acc
            poison_indicator[num_bd:num_neg + num_bd] = 2  # indicate for the cross terms

            batch_poison_indicator_list.append(poison_indicator)
            batch_original_targets_list.append(targets.detach().clone().cpu())

        schedulerC.step()

        train_epoch_loss_avg_over_batch, \
        train_epoch_predict_list, \
        train_epoch_label_list, \
        train_epoch_poison_indicator_list, \
        train_epoch_original_targets_list = sum(batch_loss_list) / len(batch_loss_list), \
                                            torch.cat(batch_predict_list), \
                                            torch.cat(batch_label_list), \
                                            torch.cat(batch_poison_indicator_list), \
                                            torch.cat(batch_original_targets_list)

        train_mix_acc = all_acc(train_epoch_predict_list, train_epoch_label_list)

        train_bd_idx = torch.where(train_epoch_poison_indicator_list == 1)[0]
        train_cross_idx = torch.where(train_epoch_poison_indicator_list == 2)[0]
        train_clean_idx = torch.where(train_epoch_poison_indicator_list == 0)[0]
        train_clean_acc = all_acc(
            train_epoch_predict_list[train_clean_idx],
            train_epoch_label_list[train_clean_idx],
        )
        if num_bd:
            train_asr = all_acc(
                train_epoch_predict_list[train_bd_idx],
                train_epoch_label_list[train_bd_idx],
            )
        else:
            train_asr = 0
        if num_neg:
            train_cross_acc = all_acc(
                train_epoch_predict_list[train_cross_idx],
                train_epoch_label_list[train_cross_idx],
            )
        else:
            train_cross_acc = 0
        if num_bd:
            train_ra = all_acc(
                train_epoch_predict_list[train_bd_idx],
                train_epoch_original_targets_list[train_bd_idx],
            )
        else:
            train_ra = 0

        return train_epoch_loss_avg_over_batch, \
               train_mix_acc, \
               train_clean_acc, \
               train_asr, \
               train_ra, \
               train_cross_acc

    def eval_step(
            self,
            netC,
            clean_test_dataset_with_transform,
            clean_test_dataloader,
            bd_test_r_dataloader,
            cross_test_dataloader,
            args,

    ):


        # -----------------------

        clean_metrics, clean_epoch_predict_list, clean_epoch_label_list = given_dataloader_test(
            netC,
            clean_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.device,
            verbose=0,
        )

        clean_test_loss_avg_over_batch = clean_metrics['test_loss_avg_over_batch']
        test_acc = clean_metrics['test_acc']
        bd_metrics, bd_epoch_predict_list, bd_epoch_label_list = given_dataloader_test(
            netC,
            bd_test_r_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.device,
            verbose=0,
        )
        bd_test_loss_avg_over_batch = bd_metrics['test_loss_avg_over_batch']
        test_asr = bd_metrics['test_acc']

        self.bd_test_r_dataset.getitem_all_switch = True  # change to return the original label instead
        ra_test_dataset_with_transform = dataset_wrapper_with_transform(
            self.bd_test_r_dataset,
            clean_test_dataset_with_transform.wrap_img_transform,
        )
        ra_test_dataloader = DataLoader(ra_test_dataset_with_transform,
                                        pin_memory=args.pin_memory,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        shuffle=False)
        ra_metrics, ra_epoch_predict_list, ra_epoch_label_list = given_dataloader_test(
            netC,
            ra_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.device,
            verbose=0,
        )
        ra_test_loss_avg_over_batch = ra_metrics['test_loss_avg_over_batch']
        test_ra = ra_metrics['test_acc']
        self.bd_test_r_dataset.getitem_all_switch = False  # switch back

        cross_metrics, cross_epoch_predict_list, cross_epoch_label_list = given_dataloader_test(
            netC,
            cross_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.device,
            verbose=0,
        )
        cross_test_loss_avg_over_batch = cross_metrics['test_loss_avg_over_batch']
        test_cross_acc = cross_metrics['test_acc']

        return clean_test_loss_avg_over_batch, \
               bd_test_loss_avg_over_batch, \
               cross_test_loss_avg_over_batch, \
               ra_test_loss_avg_over_batch, \
               test_acc, \
               test_asr, \
               test_ra, \
               test_cross_acc


if __name__ == '__main__':
    attack = Bpp()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()
