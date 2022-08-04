# MIT License

# Copyright (c) 2021 VinAI Research

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


'''
This file is modified based on the following source:
link : https://github.com/VinAIResearch/input-aware-backdoor-attack-release/tree/master/defenses
The defense method is called nc.

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. implement finetune operation according to nc paper
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. nc defense:
        a. initialize the model and trigger
        b. train triggers according to different target labels
        c. Determine whether the trained reverse trigger is a real backdoor trigger
            If it is a real backdoor trigger:
            d. select samples as clean samples and unlearning samples, finetune the origin model
    4. test the result and get ASR, ACC, RA 
'''
import logging
import random
import time

from calendar import c
import cv2

import torch
import logging
import argparse
import sys
import os
from torch import Tensor, nn

sys.path.append('../')
sys.path.append(os.getcwd())
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from matplotlib import image as mlt

import numpy as np
from PIL import Image
import torchvision
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import *
from utils.save_load_attack import load_attack_result
sys.path.append(os.getcwd())
import yaml
from pprint import pprint, pformat
# def create_dir(path_dir):
#     list_subdir = path_dir.strip(".").split("/")
#     list_subdir.remove("")
#     base_dir = "./"
#     for subdir in list_subdir:
#         base_dir = os.path.join(base_dir, subdir)
#         try:
#             os.mkdir(base_dir)
#         except:
#             pass

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

class RegressionModel(nn.Module):
    def __init__(self, opt, init_mask, init_pattern,result):
        self._EPSILON = opt.EPSILON
        super(RegressionModel, self).__init__()
        self.mask_tanh = nn.Parameter(torch.tensor(init_mask))
        self.pattern_tanh = nn.Parameter(torch.tensor(init_pattern))
        self.result = result
        self.classifier = self._get_classifier(opt)
        self.normalizer = self._get_normalize(opt)
        self.denormalizer = self._get_denormalize(opt)

        
    def forward(self, x):
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
        if self.normalizer:
          pattern = self.normalizer(self.get_raw_pattern())
        x = (1 - mask) * x + mask * pattern
        return self.classifier(x)

    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        return mask / (2 + self._EPSILON) + 0.5

    def get_raw_pattern(self):
        pattern = nn.Tanh()(self.pattern_tanh)
        return pattern / (2 + self._EPSILON) + 0.5

    def _get_classifier(self, opt):
       
        classifier = generate_cls_model(args.model,args.num_classes)
        classifier.load_state_dict(self.result['model'])
        classifier.to(args.device)
        
        for param in classifier.parameters():
            param.requires_grad = False
        classifier.eval()
        return classifier.to(opt.device)

    def _get_denormalize(self, opt):
        if opt.dataset == "cifar10" or opt.dataset == "cifar100":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            denormalizer = None
        elif opt.dataset == 'tiny':
            denormalizer = Denormalize(opt, [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, opt):
        if opt.dataset == "cifar10" or opt.dataset == "cifar100":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            normalizer = None
        elif opt.dataset == 'tiny':
            normalizer = Normalize(opt, [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        else:
            raise Exception("Invalid dataset")
        return normalizer


class Recorder:
    def __init__(self, opt):
        super().__init__()

        # Best optimization results
        self.mask_best = None
        self.pattern_best = None
        self.reg_best = float("inf")

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.reg_best

        # Cost
        self.cost = opt.init_cost
        self.cost_multiplier_up = opt.cost_multiplier
        self.cost_multiplier_down = opt.cost_multiplier ** 1.5

    def reset_state(self, opt):
        self.cost = opt.init_cost
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        print("Initialize cost to {:f}".format(self.cost))

    def save_result_to_dir(self, opt):
        # result_dir = os.path.join(opt.result, opt.dataset)
        # if not os.path.exists(result_dir):
        #     os.makedirs(result_dir)
        # # result_dir = os.path.join(result_dir, opt.attack_mode)
        # result_dir = os.path.join(result_dir, opt.trigger_type)
        # if not os.path.exists(result_dir):
        #     os.makedirs(result_dir)
        # result_dir = os.path.join(result_dir, str(opt.target_label))
        # if not os.path.exists(result_dir):
        #     os.makedirs(result_dir)

        result_dir = (os.getcwd() + f'{opt.log}')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # result_dir = os.path.join(result_dir, opt.attack_mode)
        # result_dir = os.path.join(os.getcwd(),opt.save_path,  opt.trigger_type)
        # if not os.path.exists(result_dir):
        #     os.makedirs(result_dir)
        result_dir = os.path.join(result_dir, str(opt.target_label))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        pattern_best = self.pattern_best
        mask_best = self.mask_best
        trigger = pattern_best * mask_best

        path_mask = os.path.join(result_dir, "mask.png")
        path_pattern = os.path.join(result_dir, "pattern.png")
        path_trigger = os.path.join(result_dir, "trigger.png")

        torchvision.utils.save_image(mask_best, path_mask, normalize=True)
        torchvision.utils.save_image(pattern_best, path_pattern, normalize=True)
        torchvision.utils.save_image(trigger, path_trigger, normalize=True)


def train(opt, result, init_mask, init_pattern):

    tran = get_transform(opt.dataset, *([opt.input_height,opt.input_width]) , train = True)
    x = result['clean_train']['x']
    y = result['clean_train']['y']
    data_all_length = len(y)
    args.ratio = args.cleaning_ratio
    ran_idx = choose_index(args, data_all_length) 
    data_set = list(zip([x[ii] for ii in ran_idx],[y[ii] for ii in ran_idx]))
    data_bd_trainset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    trainloader = torch.utils.data.DataLoader(data_bd_trainset, batch_size=opt.batch_size, num_workers=opt.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    # test_dataloader = get_dataloader(opt, train=False)
    # trainloader = get_dataloader(opt, True)

    # Build regression model
    regression_model = RegressionModel(opt, init_mask, init_pattern, result).to(opt.device)

    # Set optimizer
    optimizerR = torch.optim.Adam(regression_model.parameters(), lr=opt.mask_lr, betas=(0.5, 0.9))

    # Set recorder (for recording best result)
    recorder = Recorder(opt)

    for epoch in range(opt.nc_epoch):
        # early_stop = train_step(regression_model, optimizerR, test_dataloader, recorder, epoch, opt)
        early_stop = train_step(regression_model, optimizerR, trainloader, recorder, epoch, opt)
        if early_stop:
            break

    # Save result to dir
    recorder.save_result_to_dir(opt)

    return recorder, opt


def train_step(regression_model, optimizerR, dataloader, recorder, epoch, opt):
    # print("Epoch {} - Label: {} | {} - {}:".format(epoch, opt.target_label, opt.dataset, opt.attack_mode))
    # Set losses
    cross_entropy = nn.CrossEntropyLoss()
    total_pred = 0
    true_pred = 0

    # Record loss for all mini-batches
    loss_ce_list = []
    loss_reg_list = []
    loss_list = []
    loss_acc_list = []

    # Set inner early stop flag
    inner_early_stop_flag = False
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # Forwarding and update model
        optimizerR.zero_grad()

        inputs = inputs.to(opt.device)
        sample_num = inputs.shape[0]
        total_pred += sample_num
        target_labels = torch.ones((sample_num), dtype=torch.int64).to(opt.device) * opt.target_label
        predictions = regression_model(inputs)

        loss_ce = cross_entropy(predictions, target_labels)
        loss_reg = torch.norm(regression_model.get_raw_mask(), opt.use_norm)
        total_loss = loss_ce + recorder.cost * loss_reg
        total_loss.backward()
        optimizerR.step()

        # Record minibatch information to list
        minibatch_accuracy = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() * 100.0 / sample_num
        loss_ce_list.append(loss_ce.detach())
        loss_reg_list.append(loss_reg.detach())
        loss_list.append(total_loss.detach())
        loss_acc_list.append(minibatch_accuracy)

        true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()
        # progress_bar(batch_idx, len(dataloader))

    loss_ce_list = torch.stack(loss_ce_list)
    loss_reg_list = torch.stack(loss_reg_list)
    loss_list = torch.stack(loss_list)
    loss_acc_list = torch.stack(loss_acc_list)

    avg_loss_ce = torch.mean(loss_ce_list)
    avg_loss_reg = torch.mean(loss_reg_list)
    avg_loss = torch.mean(loss_list)
    avg_loss_acc = torch.mean(loss_acc_list)

    # Check to save best mask or not
    if avg_loss_acc >= opt.atk_succ_threshold and avg_loss_reg < recorder.reg_best:
        recorder.mask_best = regression_model.get_raw_mask().detach()
        recorder.pattern_best = regression_model.get_raw_pattern().detach()
        recorder.reg_best = avg_loss_reg
        recorder.save_result_to_dir(opt)
        # print(" Updated !!!")

    # Show information
    # print(
    #     "  Result: Accuracy: {:.3f} | Cross Entropy Loss: {:.6f} | Reg Loss: {:.6f} | Reg best: {:.6f}".format(
    #         true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_reg, recorder.reg_best
    #     )
    # )

    # Check early stop
    if opt.early_stop:
        if recorder.reg_best < float("inf"):
            if recorder.reg_best >= opt.early_stop_threshold * recorder.early_stop_reg_best:
                recorder.early_stop_counter += 1
            else:
                recorder.early_stop_counter = 0

        recorder.early_stop_reg_best = min(recorder.early_stop_reg_best, recorder.reg_best)

        if (
            recorder.cost_down_flag
            and recorder.cost_up_flag
            and recorder.early_stop_counter >= opt.early_stop_patience
        ):
            print("Early_stop !!!")
            inner_early_stop_flag = True

    if not inner_early_stop_flag:
        # Check cost modification
        if recorder.cost == 0 and avg_loss_acc >= opt.atk_succ_threshold:
            recorder.cost_set_counter += 1
            if recorder.cost_set_counter >= opt.patience:
                recorder.reset_state(opt)
        else:
            recorder.cost_set_counter = 0

        if avg_loss_acc >= opt.atk_succ_threshold:
            recorder.cost_up_counter += 1
            recorder.cost_down_counter = 0
        else:
            recorder.cost_up_counter = 0
            recorder.cost_down_counter += 1

        if recorder.cost_up_counter >= opt.patience:
            recorder.cost_up_counter = 0
            print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
            recorder.cost *= recorder.cost_multiplier_up
            recorder.cost_up_flag = True

        elif recorder.cost_down_counter >= opt.patience:
            recorder.cost_down_counter = 0
            print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
            recorder.cost /= recorder.cost_multiplier_down
            recorder.cost_down_flag = True

        # Save the final version
        if recorder.mask_best is None:
            recorder.mask_best = regression_model.get_raw_mask().detach()
            recorder.pattern_best = regression_model.get_raw_pattern().detach()

    del predictions
    torch.cuda.empty_cache()

    return inner_early_stop_flag

def outlier_detection(l1_norm_list, idx_mapping, opt):
    print("-" * 30)
    print("Determining whether model is backdoor")
    consistency_constant = 1.4826
    median = torch.median(l1_norm_list)
    mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median))
    min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad

    print("Median: {}, MAD: {}".format(median, mad))
    print("Anomaly index: {}".format(min_mad))

    if min_mad < 2:
        print("Not a backdoor model")
    else:
        print("This is a backdoor model")

    if opt.to_file:
        # result_path = os.path.join(opt.result, opt.saving_prefix, opt.dataset)
        # output_path = os.path.join(
        #     result_path, "{}_{}_output.txt".format(opt.attack_mode, opt.dataset, opt.attack_mode)
        # )
        output_path = opt.output_path
        with open(output_path, "a+") as f:
            f.write(
                str(median.cpu().numpy()) + ", " + str(mad.cpu().numpy()) + ", " + str(min_mad.cpu().numpy()) + "\n"
            )
            l1_norm_list_to_save = [str(value) for value in l1_norm_list.cpu().numpy()]
            f.write(", ".join(l1_norm_list_to_save) + "\n")

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if torch.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    logging.info(
        "Flagged label list: {}".format(",".join(["{}: {}".format(y_label, l_norm) for y_label, l_norm in flag_list]))
    )

    return flag_list


def get_args():
    #set the basic parameter
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str)
    parser.add_argument('--checkpoint_save', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument("--data_root", type=str)

    parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny') 
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    parser.add_argument("--input_channel", type=int)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--num_workers", type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr') 

    parser.add_argument('--attack', type=str)
    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)
    parser.add_argument('--trigger_type', type=str, help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--random_seed', type=int, help='random seed')
    parser.add_argument('--index', type=str, help='index of clean data')
    parser.add_argument('--result_file', type=str, help='the location of result')
    parser.add_argument('--yaml_path', type=str, default="./config/defense/nc/config.yaml", help='the path of yaml')

    #set the parameter for the ac defense
    parser.add_argument("--mask_lr", type=float)
    parser.add_argument("--init_cost", type=float)
    # parser.add_argument("--bs", type=int)
    parser.add_argument("--atk_succ_threshold", type=float)
    parser.add_argument("--early_stop", type=bool)
    parser.add_argument("--early_stop_threshold", type=float)
    parser.add_argument("--early_stop_patience", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--cost_multiplier", type=float)
    parser.add_argument("--total_label", type=int)
    parser.add_argument("--EPSILON", type=float)
    parser.add_argument("--to_file", type=bool)
    parser.add_argument("--n_times_test", type=int)
    parser.add_argument("--use_norm", type=int)
    # parser.add_argument("--k", type=int)
    parser.add_argument('--ratio', type=float,  help='ratio of training data')
    parser.add_argument('--cleaning_ratio', type=float,  help='ratio of cleaning data')
    parser.add_argument('--unlearning_ratio', type=float, help='ratio of unlearning data')
    parser.add_argument('--nc_epoch', type=int,  help='the epoch for neural cleanse')


    arg = parser.parse_args()

    print(arg)
    return arg

def nc(args,result,config):
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    # logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    if args.log is not None and args.log != '':
        fileHandler = logging.FileHandler(os.getcwd() + args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    else:
        fileHandler = logging.FileHandler(os.getcwd() + './log' + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))

    fix_random(args.random_seed)

    # a. initialize the model and trigger
    result_path = os.getcwd() + f'{args.save_path}/nc/trigger/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    args.output_path = result_path + "{}_output_clean.txt".format(args.dataset)
    if args.to_file:
        with open(args.output_path, "w+") as f:
            f.write("Output for cleanse:  - {}".format(args.dataset) + "\n")

    init_mask = np.ones((1, args.input_height, args.input_width)).astype(np.float32)
    init_pattern = np.ones((args.input_channel, args.input_height, args.input_width)).astype(np.float32)

    flag = 0
    for test in range(args.n_times_test):
        # b. train triggers according to different target labels
        print("Test {}:".format(test))
        logging.info("Test {}:".format(test))
        if args.to_file:
            with open(args.output_path, "a+") as f:
                f.write("-" * 30 + "\n")
                f.write("Test {}:".format(str(test)) + "\n")

        masks = []
        idx_mapping = {}

        for target_label in range(args.num_classes):
            print("----------------- Analyzing label: {} -----------------".format(target_label))
            logging.info("----------------- Analyzing label: {} -----------------".format(target_label))
            args.target_label = target_label
            recorder, args = train(args, result, init_mask, init_pattern)

            mask = recorder.mask_best
            masks.append(mask)
            reg = torch.norm(mask, p=args.use_norm)
            logging.info(f'The regularization of mask for target label {target_label} is {reg}')
            idx_mapping[target_label] = len(masks) - 1

        # c. Determine whether the trained reverse trigger is a real backdoor trigger
        l1_norm_list = torch.stack([torch.norm(m, p=args.use_norm) for m in masks])
        logging.info("{} labels found".format(len(l1_norm_list)))
        logging.info("Norm values: {}".format(l1_norm_list))
        flag_list = outlier_detection(l1_norm_list, idx_mapping, args)
        if len(flag_list) != 0:
            flag = 1

    if flag == 0:
        logging.info('This is not a backdoor model')
        model = generate_cls_model(args.model,args.num_classes)
        model.load_state_dict(result['model'])
        result = {}
        result['model'] = model
        return result  

    # d. select samples as clean samples and unlearning samples, finetune the origin model
    model = generate_cls_model(args.model,args.num_classes)
    model.load_state_dict(result['model'])
    model.to(args.device)
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    x = result['clean_train']['x']
    y = result['clean_train']['y']
    data_all_length = len(y)
    args.ratio = args.cleaning_ratio
    ran_idx = choose_index(args, data_all_length) 
    log_index = os.getcwd() + args.log + 'index.txt'
    np.savetxt(log_index, ran_idx, fmt='%d')
  
    idx_clean = ran_idx[0:int(len(x)*args.cleaning_ratio*(1-args.unlearning_ratio))]
    idx_unlearn = ran_idx[int(len(x)*args.cleaning_ratio*(1-args.unlearning_ratio)):int(len(x)*args.cleaning_ratio)]
    x_new = [x[ii] for ii in idx_clean]
    y_new = [y[ii] for ii in idx_clean]

    for (label,_) in flag_list:
        mask_path = os.getcwd() + f'{args.log}' + '{}/'.format(str(label)) + 'mask.png'
        mask_image = mlt.imread(mask_path)
        mask_image = cv2.resize(mask_image,(args.input_height, args.input_width))
        trigger_path = os.getcwd() + f'{args.log}' + '{}/'.format(str(label)) + 'trigger.png'
        signal_mask = mlt.imread(trigger_path)*255
        signal_mask = cv2.resize(signal_mask,(args.input_height, args.input_width))
        x_unlearn = [x[ii] for ii in idx_unlearn]
        x_unlearn_new = list()
        for img in x_unlearn:
            x_np = np.array(cv2.resize(np.array(img),(args.input_height, args.input_width))) * (1-np.array(mask_image)) + np.array(signal_mask)
            x_np = np.clip(x_np.astype('uint8'), 0, 255)
            x_np_img = Image.fromarray(x_np)
            x_unlearn_new.extend([x_np_img])
        # x_new = torch.cat((x_new,x_unlearn),dim = 0)
        x_new.extend(x_unlearn_new)
        y_new.extend([y[ii] for ii in idx_unlearn])
    data_clean_train = list(zip(x_new,y_new))
    # data_clean_train = torch.utils.data.TensorDataset(x_new,y_new)
    data_set_o = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_train,
        poison_idx=np.zeros(len(data_clean_train)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    trainloader = torch.utils.data.DataLoader(data_set_o, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
  
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer)
    elif args.lr_scheduler ==  'CosineAnnealingLR':
        scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, T_max=100) 
    criterion = torch.nn.CrossEntropyLoss() 
    
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    y = result['bd_test']['y']
    data_bd_test = list(zip(x,y))
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_clean_test = list(zip(x,y))
    data_clean_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_test,
        poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    best_acc = 0
    best_asr = 0
    for j in range(args.epochs):
        batch_loss = []
        for i, (inputs,labels) in enumerate(trainloader):  # type: ignore
            model.train()
            model.to(args.device)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        one_epoch_loss = sum(batch_loss)/len(batch_loss)
        if args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler.step(one_epoch_loss)
        elif args.lr_scheduler ==  'CosineAnnealingLR':
            scheduler.step()
        
        with torch.no_grad():
            model.eval()
            asr_acc = 0
            for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                asr_acc += torch.sum(pre_label == labels)/len(data_bd_test)

            
            clean_acc = 0
            for i, (inputs,labels) in enumerate(data_clean_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                clean_acc += torch.sum(pre_label == labels)/len(data_clean_test)
        
        if not (os.path.exists(os.getcwd() + f'{args.save_path}/nc/ckpt_best/')):
            os.makedirs(os.getcwd() + f'{args.save_path}/nc/ckpt_best/')
        if best_acc < clean_acc:
            best_acc = clean_acc
            best_asr = asr_acc
            torch.save(
            {
                'model_name':args.model,
                'model': model.cpu().state_dict(),
                'asr': asr_acc,
                'acc': clean_acc
            },
            f'./{args.save_path}/nc/ckpt_best/defense_result.pt'
            )
        logging.info(f'Epoch{j}: clean_acc:{clean_acc} asr:{asr_acc} best_acc:{best_acc} best_asr{best_asr}')

    result = {}
    result['model'] = model.to(args.device)
    return result       

if __name__ == '__main__':
    
    ### 1. basic setting: args
    args = get_args()
    with open(args.yaml_path, 'r') as stream: 
        config = yaml.safe_load(stream)
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    
    save_path = '/record/' + args.result_file
    if args.checkpoint_save is None:
        args.checkpoint_save = save_path + '/record/defence/nc/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = save_path + '/saved/nc/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log)  
    args.save_path = save_path

    ### 2. attack result(model, train data, test data)
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt') 
    
    ### 3. nc defense:
    result_defense = nc(args,result,config)

    ### 4. test the result and get ASR, ACC, RC 
    result_defense['model'].eval()
    result_defense['model'].to(args.device)
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    y = result['bd_test']['y']
    data_bd_test = list(zip(x,y))
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    asr_acc = 0
    for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = result_defense['model'](inputs)
        pre_label = torch.max(outputs,dim=1)[1]
        asr_acc += torch.sum(pre_label == labels)
    asr_acc = asr_acc/len(data_bd_test)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_clean_test = list(zip(x,y))
    data_clean_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_test,
        poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    clean_acc = 0
    for i, (inputs,labels) in enumerate(data_clean_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = result_defense['model'](inputs)
        pre_label = torch.max(outputs,dim=1)[1]
        clean_acc += torch.sum(pre_label == labels)
    clean_acc = clean_acc/len(data_clean_test)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    robust_acc = -1
    if 'original_targets' in result['bd_test']:
        y_ori = result['bd_test']['original_targets']
        if y_ori is not None:
            if len(y_ori) != len(x):
                y_idx = result['bd_test']['original_index']
                y = y_ori[y_idx]
            else :
                y = y_ori
            data_bd_test = list(zip(x,y))
            data_bd_testset = prepro_cls_DatasetBD(
                full_dataset_without_transform=data_bd_test,
                poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
                bd_image_pre_transform=None,
                bd_label_pre_transform=None,
                ori_image_transform_in_loading=tran,
                ori_label_transform_in_loading=None,
                add_details_in_preprocess=False,
            )
            data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
        
            robust_acc = 0
            for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = result_defense['model'](inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                robust_acc += torch.sum(pre_label == labels)
            robust_acc = robust_acc/len(data_bd_test)
        
    if not (os.path.exists(os.getcwd() + f'{save_path}/nc/')):
        os.makedirs(os.getcwd() + f'{save_path}/nc/')
    torch.save(
    {
        'model_name':args.model,
        'model': result_defense['model'].cpu().state_dict(),
        'asr': asr_acc,
        'acc': clean_acc,
        'ra': robust_acc
    },
    os.getcwd() + f'/{save_path}/nc/defense_result.pt'
    )
    