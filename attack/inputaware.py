'''
Input-aware dynamic backdoor attack
This file is modified based on the following source:
link : https://github.com/VinAIResearch/input-aware-backdoor-attack-release

@article{nguyen2020input,
  title={Input-aware dynamic backdoor attack},
  author={Nguyen, Tuan Anh and Tran, Anh},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={3454--3464},
  year={2020}
}

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



Note that since this attack rely on batch-wise modification of the input data,
when this method encounters lower poison ratio, the original implementation
will fail (poison ratio < 1 / batch size), we add a function named generalize_to_lower_pratio
to generalize the attack to lower the poison ratio. The basic idea is to calculate the theoretical
the number of poison samples each batch should have, then randomly select batches to do poisoning.
This change may result in instability and a higher variance in final
results' metrics, but it is a necessary change to make the attack workable in a low poison ratio.
Please be careful when you use this attack in a low poison ratio, and interpret the results with
caution.
'''

import argparse
import logging
import os
import sys
import time
import torch

sys.path = ["./"] + sys.path

import numpy as np
from copy import deepcopy
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from attack.badnet import BadNet, add_common_attack_args
from torchvision import transforms

from utils.aggregate_block.dataset_and_transform_generate import get_dataset_normalization, get_dataset_denormalization
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.trainer_cls import Metric_Aggregator
from utils.save_load_attack import save_attack_result
from utils.trainer_cls import all_acc, given_dataloader_test, general_plot_for_epoch
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform

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
    def __init__(self, args, expected_values, variance):
        self.n_channels = args.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
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


class InputAwareGenerator(nn.Sequential):
    def __init__(self, args, out_channels=None):
        super(InputAwareGenerator, self).__init__()
        self.args = args
        if self.args.dataset == "mnist":
            channel_init = 16
            steps = 2
        else:
            channel_init = 32
            steps = 3

        channel_current = self.args.input_channel
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
                    channel_next = self.args.input_channel
                else:
                    channel_next = out_channels

        self._EPSILON = 1e-7
        self._normalizer = self._get_normalize(self.args)
        self._denormalizer = self._get_denormalize(self.args)
        self.tanh = nn.Tanh()

    def _get_denormalize(self, args):
        denormalizer = Denormalize(args, get_dataset_normalization(self.args.dataset).mean,
                                   get_dataset_normalization(self.args.dataset).std)
        return denormalizer

    def _get_normalize(self, args):
        normalizer = Normalize(args, get_dataset_normalization(self.args.dataset).mean,
                               get_dataset_normalization(self.args.dataset).std)
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


class Threshold(nn.Module):
    def __init__(self):
        super(Threshold, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x * 20 - 10) / (2 + 1e-7) + 0.5


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


class InputAware(BadNet):

    def __init__(self):
        super(InputAware, self).__init__()

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

        parser = add_common_attack_args(parser)

        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/inputaware/default.yaml',
                            help='path for yaml file provide additional default attributes')

        parser.add_argument("--lr_G", type=float, )  # default=1e-2)
        parser.add_argument("--lr_C", type=float, )  # default=1e-2)
        parser.add_argument("--lr_M", type=float, )  # default=1e-2)
        parser.add_argument('--C_lr_scheduler', type=str)
        parser.add_argument("--schedulerG_milestones", type=list, )  # default=[200, 300, 400, 500])
        parser.add_argument("--schedulerC_milestones", type=list, )  # default=[100, 200, 300, 400])
        parser.add_argument("--schedulerM_milestones", type=list, )  # default=[10, 20])
        parser.add_argument("--schedulerG_lambda", type=float, )  # default=0.1)
        parser.add_argument("--schedulerC_lambda", type=float, )  # default=0.1)
        parser.add_argument("--schedulerM_lambda", type=float, )  # default=0.1)
        parser.add_argument("--lambda_div", type=float, )  # default=1)
        parser.add_argument("--lambda_norm", type=float, )  # default=100)
        parser.add_argument("--mask_density", type=float, )  # default=0.032)
        parser.add_argument("--EPSILON", type=float, )  # default=1e-7)
        parser.add_argument('--clean_train_epochs', type=int)
        parser.add_argument("--random_rotation", type=int, )  # default=10)
        parser.add_argument("--random_crop", type=int, )  # default=5)
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

        clean_train_dataloader1 = DataLoader(clean_train_dataset_with_transform, pin_memory=args.pin_memory,
                                             batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        clean_train_dataloader2 = DataLoader(clean_train_dataset_with_transform, pin_memory=args.pin_memory,
                                             batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        clean_test_dataloader1 = DataLoader(clean_test_dataset_with_transform, pin_memory=args.pin_memory,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers, shuffle=True)
        clean_test_dataloader2 = DataLoader(clean_test_dataset_with_transform, pin_memory=args.pin_memory,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers, shuffle=True)

        self.stage1_results = clean_train_dataset_with_transform, clean_train_dataloader1, \
                              clean_train_dataloader2, \
                              clean_test_dataset_with_transform, \
                              clean_test_dataloader1, \
                              clean_test_dataloader2

    def stage2_training(self):
        # since we need the network to do poison, 
        #  we can only put prepare of bd dataset to stage2 with training process.

        logging.info(f"stage2 start")
        assert 'args' in self.__dict__
        args = self.args
        agg = Metric_Aggregator()

        clean_train_dataset_with_transform, \
        clean_train_dataloader1, \
        clean_train_dataloader2, \
        clean_test_dataset_with_transform, \
        clean_test_dataloader1, \
        clean_test_dataloader2 = self.stage1_results

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
        netG = InputAwareGenerator(args).to(self.device, non_blocking=args.non_blocking)
        netM = InputAwareGenerator(args, out_channels=1).to(self.device, non_blocking=args.non_blocking)
        self.threshold = Threshold().to(self.device, non_blocking=args.non_blocking)

        if "," in args.device:
            netC = torch.nn.DataParallel(
                netC,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            netG = torch.nn.DataParallel(
                netG,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            netM = torch.nn.DataParallel(
                netM,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.threshold = torch.nn.DataParallel(
                self.threshold,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )

        optimizerC = torch.optim.SGD(netC.parameters(), args.lr_C, momentum=0.9, weight_decay=5e-4)
        optimizerG = torch.optim.Adam(netG.parameters(), args.lr_G, betas=(0.5, 0.9))
        optimizerM = torch.optim.Adam(netM.parameters(), args.lr_M, betas=(0.5, 0.9))

        if args.C_lr_scheduler == "ReduceLROnPlateau":
            schedulerC = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerC)
        else:
            schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, args.schedulerC_milestones,
                                                              args.schedulerC_lambda)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, args.schedulerG_milestones,
                                                          args.schedulerG_lambda)
        schedulerM = torch.optim.lr_scheduler.MultiStepLR(optimizerM, args.schedulerM_milestones,
                                                          args.schedulerM_lambda)

        self.normalizer = Normalize(args, get_dataset_normalization(self.args.dataset).mean,
                                    get_dataset_normalization(self.args.dataset).std)

        epoch = 1

        # first clean_train_epochs epoch clean train
        for i in range(args.clean_train_epochs):
            netM.train()
            logging.info(
                "Epoch {} - {} - {} | mask_density: {} - lambda_div: {}  - lambda_norm: {}:".format(
                    epoch, args.dataset, args.attack_label_trans, args.mask_density, args.lambda_div, args.lambda_norm
                )
            )
            self.train_mask_step(
                netM, optimizerM, schedulerM, clean_train_dataloader1, clean_train_dataloader2, args
            )
            epoch = self.eval_mask(netM, optimizerM, schedulerM, clean_test_dataloader1, clean_test_dataloader2, epoch,
                                   args)

            if args.frequency_save != 0 and epoch % args.frequency_save == args.frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')

                state_dict = {
                    "netM": netM.state_dict(),
                    "optimizerM": optimizerM.state_dict(),
                    "schedulerM": schedulerM.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }

                torch.save(state_dict, args.save_path + "/mask_state_dict.pt")

            epoch += 1
        netM.eval()
        netM.requires_grad_(False)

        # real train (attack)

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

        for i in range(args.epochs):

            logging.info(
                "Epoch {} - {} - {} | mask_density: {} - lambda_div: {}:".format(
                    epoch, args.dataset, args.attack_label_trans, args.mask_density, args.lambda_div
                )
            )

            train_epoch_loss_avg_over_batch, \
            train_mix_acc, \
            train_clean_acc, \
            train_asr, \
            train_ra, \
            train_cross_acc = self.train_step(
                netC,
                netG,
                netM,
                optimizerC,
                optimizerG,
                schedulerC,
                schedulerG,
                clean_train_dataloader1,
                clean_train_dataloader2,
                args,
            )

            clean_test_loss_avg_over_batch, \
            bd_test_loss_avg_over_batch, \
            cross_test_loss_avg_over_batch, \
            ra_test_loss_avg_over_batch, \
            test_acc, \
            test_asr, \
            test_ra, \
            test_cross_acc = self.eval_step(
                netC,
                netG,
                netM,
                optimizerC,
                optimizerG,
                schedulerC,
                schedulerG,
                clean_test_dataset_with_transform,
                epoch,
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

            if args.frequency_save != 0 and epoch % args.frequency_save == args.frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                state_dict = {
                    "netC": netC.state_dict(),
                    "netG": netG.state_dict(),
                    "netM": netM.state_dict(),
                    "optimizerC": optimizerC.state_dict(),
                    "optimizerG": optimizerG.state_dict(),
                    "schedulerC": schedulerC.state_dict(),
                    "schedulerG": schedulerG.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }

                torch.save(state_dict, args.save_path + "/netCGM.pt")

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

            epoch += 1
            if epoch > args.epochs:
                break

            agg.to_dataframe().to_csv(f"{args.save_path}/attack_df.csv")

        agg.summary().to_csv(f"{args.save_path}/attack_df_summary.csv")

        ### save the poison train data for inputaware

        bd_train_dataset = prepro_cls_DatasetBD_v2(
            clean_train_dataset_with_transform.wrapped_dataset,
            save_folder_path=f"{args.save_path}/bd_train_dataset"
        )

        transforms_reversible = transforms.Compose(
            list(
                filter(
                    lambda x: isinstance(x, (transforms.Normalize, transforms.Resize, transforms.ToTensor)),
                    deepcopy(clean_train_dataset_with_transform.wrap_img_transform.transforms)
                )
            )
        )
        clean_train_dataset_with_transform.wrap_img_transform = transforms_reversible  # change it to reversiable

        # get denormalizer
        for trans_t in deepcopy(transforms_reversible.transforms):
            if isinstance(trans_t, transforms.Normalize):
                denormalizer = get_dataset_denormalization(trans_t)
                logging.info(f"{denormalizer}")

        clean_train_dataloader_without_shuffle = torch.utils.data.DataLoader(clean_train_dataset_with_transform,
                                                                             batch_size=args.batch_size,
                                                                             pin_memory=args.pin_memory,
                                                                             num_workers=args.num_workers,
                                                                             shuffle=False)
        clean_train_dataloader2.dataset.wrap_img_transform = transforms_reversible
        # change it to reversiable, notice that this dataloader is shuffled, since we need img2 is different from img1 (whose dataloader is not shuffled)
        netC.eval()
        netG.eval()
        with torch.no_grad():
            for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(
                    range(len(clean_train_dataloader_without_shuffle)),
                    clean_train_dataloader_without_shuffle,
                    clean_train_dataloader2):
                inputs1, targets1 = inputs1.to(self.device, non_blocking=args.non_blocking), targets1.to(self.device,
                                                                                                         non_blocking=args.non_blocking)
                inputs2, targets2 = inputs2.to(self.device, non_blocking=args.non_blocking), targets2.to(self.device,
                                                                                                         non_blocking=args.non_blocking)

                num_bd = int(generalize_to_lower_pratio(args.pratio, inputs1.shape[0]))  # int(args.pratio * bs)
                num_cross = num_bd

                inputs_bd, targets_bd, patterns1, masks1 = self.create_bd(inputs1[:num_bd], targets1[:num_bd], netG,
                                                                          netM, args,
                                                                          'train')
                inputs_cross, patterns2, masks2 = self.create_cross(
                    inputs1[num_bd: num_bd + num_cross], inputs2[num_bd: num_bd + num_cross], netG, netM, args,
                )

                input_changed = torch.cat([inputs_bd, inputs_cross, ], dim=0)
                # input_changed = p_transforms(input_changed) # this is non-reversible part, should not be included

                input_changed = denormalizer(  # since we normalized once, we need to denormalize it back.
                    input_changed
                ).detach().clone().cpu()
                target_changed = torch.cat([targets_bd, targets1[num_bd: (num_bd + num_cross)], ],
                                           dim=0).detach().clone().cpu()

                # save to the container
                for idx_in_batch, t_img in enumerate(
                        input_changed
                ):
                    # here we know it starts from 0 and they are consecutive
                    bd_train_dataset.set_one_bd_sample(
                        selected_index=int(batch_idx * int(args.batch_size) + idx_in_batch),
                        img=(t_img),
                        bd_label=int(target_changed[idx_in_batch]),
                        label=int(targets1[idx_in_batch]),
                    )

        save_attack_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=netC.cpu().state_dict(),
            data_path=args.dataset_path,
            img_size=args.img_size,
            clean_data=args.dataset,
            bd_train=bd_train_dataset,
            bd_test=self.bd_test_dataset,
            save_path=args.save_path,
        )

    def train_mask_step(self, netM, optimizerM, schedulerM, train_dataloader1, train_dataloader2, args):
        netM.train()
        total = 0

        total_loss = 0
        criterion_div = nn.MSELoss(reduction="none")
        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dataloader1)), train_dataloader1,
                                                                       train_dataloader2):
            optimizerM.zero_grad()

            inputs1, targets1 = inputs1.to(self.device, non_blocking=args.non_blocking), targets1.to(self.device,
                                                                                                     non_blocking=args.non_blocking)
            inputs2, targets2 = inputs2.to(self.device, non_blocking=args.non_blocking), targets2.to(self.device,
                                                                                                     non_blocking=args.non_blocking)

            # bs = inputs1.shape[0]
            masks1 = netM(inputs1)
            masks1, masks2 = self.threshold(netM(inputs1)), self.threshold(netM(inputs2))

            # Calculating diversity loss
            distance_images = criterion_div(inputs1, inputs2)
            distance_images = torch.mean(distance_images, dim=(1, 2, 3))
            distance_images = torch.sqrt(distance_images)

            distance_patterns = criterion_div(masks1, masks2)
            distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
            distance_patterns = torch.sqrt(distance_patterns)

            loss_div = distance_images / (distance_patterns + args.EPSILON)
            loss_div = torch.mean(loss_div) * args.lambda_div

            loss_norm = torch.mean(F.relu(masks1 - args.mask_density))

            total_loss = args.lambda_norm * loss_norm + args.lambda_div * loss_div
            total_loss.backward()
            optimizerM.step()
            infor_string = "Mask loss: {:.4f} - Norm: {:.3f} | Diversity: {:.3f}".format(total_loss, loss_norm,
                                                                                         loss_div)
            progress_bar(batch_idx, len(train_dataloader1), infor_string)

        schedulerM.step()

    def eval_mask(self, netM, optimizerM, schedulerM, test_dataloader1, test_dataloader2, epoch, args):
        netM.eval()
        logging.info(" Eval:")
        total = 0.0

        criterion_div = nn.MSELoss(reduction="none")
        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(test_dataloader1)), test_dataloader1,
                                                                       test_dataloader2):
            with torch.no_grad():
                inputs1, targets1 = inputs1.to(self.device, non_blocking=args.non_blocking), targets1.to(self.device,
                                                                                                         non_blocking=args.non_blocking)
                inputs2, targets2 = inputs2.to(self.device, non_blocking=args.non_blocking), targets2.to(self.device,
                                                                                                         non_blocking=args.non_blocking)
                # bs = inputs1.shape[0]
                masks1, masks2 = self.threshold(netM(inputs1)), self.threshold(netM(inputs2))

                # Calculating diversity loss
                distance_images = criterion_div(inputs1, inputs2)
                distance_images = torch.mean(distance_images, dim=(1, 2, 3))
                distance_images = torch.sqrt(distance_images)

                distance_patterns = criterion_div(masks1, masks2)
                distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
                distance_patterns = torch.sqrt(distance_patterns)

                loss_div = distance_images / (distance_patterns + args.EPSILON)
                loss_div = torch.mean(loss_div) * args.lambda_div

                loss_norm = torch.mean(F.relu(masks1 - args.mask_density))

                infor_string = "Norm: {:.3f} | Diversity: {:.3f}".format(loss_norm, loss_div)
                progress_bar(batch_idx, len(test_dataloader1), infor_string)

        return epoch

    def create_targets_bd(self, targets, args):
        if args.attack_label_trans == "all2one":
            bd_targets = torch.ones_like(targets) * args.attack_target
        elif args.attack_label_trans == "all2all":
            bd_targets = torch.tensor([(label + 1) % args.num_classes for label in targets])
        else:
            raise Exception("{} attack mode is not implemented".format(args.attack_label_trans))
        return bd_targets.to(self.device, non_blocking=args.non_blocking)

    def create_bd(self, inputs, targets, netG, netM, args, train_or_test):
        if train_or_test == 'train':
            bd_targets = self.create_targets_bd(targets, args)
            if inputs.__len__() == 0:  # for case that no sample should be poisoned
                return inputs, bd_targets, inputs.detach().clone(), inputs.detach().clone()
            patterns = netG(inputs)
            patterns = self.normalizer(patterns)

            masks_output = self.threshold(netM(inputs))
            bd_inputs = inputs + (patterns - inputs) * masks_output
            return bd_inputs, bd_targets, patterns, masks_output
        if train_or_test == 'test':
            bd_targets = self.create_targets_bd(targets, args)

            position_changed = (
                    bd_targets - targets != 0)  # no matter all2all or all2one, we want location changed to tell whether the bd is effective

            inputs, bd_targets = inputs[position_changed], bd_targets[position_changed]

            if inputs.__len__() == 0:  # for case that no sample should be poisoned
                return torch.tensor([]), torch.tensor([]), None, None, position_changed, targets
            patterns = netG(inputs)
            patterns = self.normalizer(patterns)

            masks_output = self.threshold(netM(inputs))
            bd_inputs = inputs + (patterns - inputs) * masks_output
            return bd_inputs, bd_targets, patterns, masks_output, position_changed, targets

    def create_cross(self, inputs1, inputs2, netG, netM, args):
        if inputs1.__len__() == 0:  # for case that no sample should be poisoned
            return inputs2.detach().clone(), inputs2, inputs2.detach().clone()
        patterns2 = netG(inputs2)
        patterns2 = self.normalizer(patterns2)
        masks_output = self.threshold(netM(inputs2))
        inputs_cross = inputs1 + (patterns2 - inputs1) * masks_output
        return inputs_cross, patterns2, masks_output

    def train_step(self,
                   netC, netG, netM, optimizerC, optimizerG, schedulerC, schedulerG, train_dataloader1,
                   train_dataloader2, args,
                   ):
        netC.train()
        netG.train()
        logging.info(" Training:")

        criterion = nn.CrossEntropyLoss()
        criterion_div = nn.MSELoss(reduction="none")

        batch_loss_list = []
        batch_predict_list = []
        batch_label_list = []
        batch_poison_indicator_list = []
        batch_original_targets_list = []

        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dataloader1)), train_dataloader1,
                                                                       train_dataloader2):
            optimizerC.zero_grad()

            inputs1, targets1 = inputs1.to(self.device, non_blocking=args.non_blocking), targets1.to(self.device,
                                                                                                     non_blocking=args.non_blocking)
            inputs2, targets2 = inputs2.to(self.device, non_blocking=args.non_blocking), targets2.to(self.device,
                                                                                                     non_blocking=args.non_blocking)

            # bs = int(args.batch_size)
            num_bd = int(generalize_to_lower_pratio(args.pratio, inputs1.shape[0]))  # int(args.pratio * bs)
            num_cross = num_bd

            inputs_bd, targets_bd, patterns1, masks1 = self.create_bd(inputs1[:num_bd], targets1[:num_bd], netG, netM,
                                                                      args,
                                                                      'train')
            inputs_cross, patterns2, masks2 = self.create_cross(
                inputs1[num_bd: num_bd + num_cross], inputs2[num_bd: num_bd + num_cross], netG, netM, args,
            )

            total_inputs = torch.cat((inputs_bd, inputs_cross, inputs1[num_bd + num_cross:]), 0)
            total_targets = torch.cat((targets_bd, targets1[num_bd:]), 0)

            preds = netC(total_inputs)
            loss_ce = criterion(preds, total_targets)

            # Calculating diversity loss
            distance_images = criterion_div(inputs1[:num_bd], inputs2[num_bd: num_bd + num_bd])
            distance_images = torch.mean(distance_images, dim=(1, 2, 3))
            distance_images = torch.sqrt(distance_images)

            distance_patterns = criterion_div(patterns1, patterns2)
            distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
            distance_patterns = torch.sqrt(distance_patterns)

            loss_div = distance_images / (distance_patterns + args.EPSILON)
            loss_div = torch.mean(loss_div) * args.lambda_div

            total_loss = loss_ce + loss_div
            total_loss.backward()
            optimizerC.step()
            optimizerG.step()

            batch_loss_list.append(total_loss.item())
            batch_predict_list.append(torch.max(preds, -1)[1].detach().clone().cpu())
            batch_label_list.append(total_targets.detach().clone().cpu())

            poison_indicator = torch.zeros(inputs1.shape[0])
            poison_indicator[:num_bd] = 1  # all others are cross/clean samples cannot conut up to train acc
            poison_indicator[num_bd:num_cross + num_bd] = 2  # indicate for the cross terms

            batch_poison_indicator_list.append(poison_indicator)
            batch_original_targets_list.append(targets1.detach().clone().cpu())

        if args.C_lr_scheduler == "ReduceLROnPlateau":
            schedulerC.step(loss_ce)
        else:
            schedulerC.step()
        schedulerG.step()

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
        train_asr = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_label_list[train_bd_idx],
        )
        train_cross_acc = all_acc(
            train_epoch_predict_list[train_cross_idx],
            train_epoch_label_list[train_cross_idx],
        )
        train_ra = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_original_targets_list[train_bd_idx],
        )

        return train_epoch_loss_avg_over_batch, \
               train_mix_acc, \
               train_clean_acc, \
               train_asr, \
               train_ra, \
               train_cross_acc

    def eval_step(
            self,
            netC,
            netG,
            netM,
            optimizerC,
            optimizerG,
            schedulerC,
            schedulerG,
            clean_test_dataset_with_transform,
            epoch,
            args,
    ):
        netC.eval()
        netG.eval()

        # set shuffle here = False, since other place need randomness to generate cross sample.
        transforms_reversible = transforms.Compose(
            list(
                filter(
                    lambda x: isinstance(x, (transforms.Normalize, transforms.Resize, transforms.ToTensor)),
                    deepcopy(clean_test_dataset_with_transform.wrap_img_transform.transforms)
                )
            )
        )
        # get denormalizer
        for trans_t in deepcopy(clean_test_dataset_with_transform.wrap_img_transform.transforms):
            if isinstance(trans_t, transforms.Normalize):
                denormalizer = get_dataset_denormalization(trans_t)
                logging.info(f"{denormalizer}")

        # Notice that due to the fact that we need random sequence to get cross samples
        # So we set the reversible_test_dataloader2 with shuffle = True
        reversible_test_dataset1 = (clean_test_dataset_with_transform)
        reversible_test_dataset1.wrap_img_transform = transforms_reversible
        reversible_test_dataloader1 = DataLoader(reversible_test_dataset1, batch_size=args.batch_size,
                                                 pin_memory=args.pin_memory,
                                                 num_workers=args.num_workers,
                                                 shuffle=False)

        reversible_test_dataset2 = (clean_test_dataset_with_transform)
        reversible_test_dataset2.wrap_img_transform = transforms_reversible
        reversible_test_dataloader2 = DataLoader(reversible_test_dataset2, batch_size=args.batch_size,
                                                 pin_memory=args.pin_memory,
                                                 num_workers=args.num_workers,
                                                 shuffle=True)

        self.bd_test_dataset = prepro_cls_DatasetBD_v2(
            clean_test_dataset_with_transform.wrapped_dataset, save_folder_path=f"{args.save_path}/bd_test_dataset"
        )
        self.cross_test_dataset = prepro_cls_DatasetBD_v2(
            clean_test_dataset_with_transform.wrapped_dataset, save_folder_path=f"{args.save_path}/cross_test_dataset"
        )

        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(reversible_test_dataloader1)),
                                                                       reversible_test_dataloader1,
                                                                       reversible_test_dataloader2):
            with torch.no_grad():
                inputs1, targets1 = inputs1.to(self.device, non_blocking=args.non_blocking), targets1.to(self.device,
                                                                                                         non_blocking=args.non_blocking)
                inputs2, targets2 = inputs2.to(self.device, non_blocking=args.non_blocking), targets2.to(self.device,
                                                                                                         non_blocking=args.non_blocking)

                inputs_bd, targets_bd, _, _, position_changed, targets = self.create_bd(inputs1, targets1, netG, netM,
                                                                                        args,
                                                                                        'test')
                inputs_cross, _, _ = self.create_cross(inputs1, inputs2, netG, netM, args)

                targets1 = targets1.detach().clone().cpu()
                y_poison_batch = targets_bd.detach().clone().cpu().tolist()
                for idx_in_batch, t_img in enumerate(inputs_bd.detach().clone().cpu()):
                    self.bd_test_dataset.set_one_bd_sample(
                        selected_index=int(
                            batch_idx * int(args.batch_size) + torch.where(position_changed.detach().clone().cpu())[0][
                                idx_in_batch]),
                        # manually calculate the original index, since we do not shuffle the dataloader
                        img=denormalizer(t_img),
                        bd_label=int(y_poison_batch[idx_in_batch]),
                        label=int(targets1[torch.where(position_changed.detach().clone().cpu())[0][idx_in_batch]]),
                    )

                for idx_in_batch, t_img in enumerate(inputs_cross.detach().clone().cpu()):
                    self.cross_test_dataset.set_one_bd_sample(
                        selected_index=int(batch_idx * int(args.batch_size) + idx_in_batch),
                        # manually calculate the original index, since we do not shuffle the dataloader
                        img=denormalizer(t_img),
                        bd_label=int(targets1[idx_in_batch]),
                        label=int(targets1[idx_in_batch]),
                    )

        self.bd_test_dataset.subset(
            np.where(self.bd_test_dataset.poison_indicator == 1)[0].tolist()
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

        cross_test_dataset_with_transform = dataset_wrapper_with_transform(
            self.cross_test_dataset,
            clean_test_dataset_with_transform.wrap_img_transform,
        )
        cross_test_dataloader = DataLoader(cross_test_dataset_with_transform,
                                           pin_memory=args.pin_memory,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           shuffle=False)

        clean_test_dataloader = DataLoader(
            clean_test_dataset_with_transform,
            pin_memory=args.pin_memory,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False
        )

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
            bd_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.device,
            verbose=0,
        )
        bd_test_loss_avg_over_batch = bd_metrics['test_loss_avg_over_batch']
        test_asr = bd_metrics['test_acc']

        self.bd_test_dataset.getitem_all_switch = True  # change to return the original label instead
        ra_test_dataset_with_transform = dataset_wrapper_with_transform(
            self.bd_test_dataset,
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
        self.bd_test_dataset.getitem_all_switch = False  # switch back

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
    attack = InputAware()
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
