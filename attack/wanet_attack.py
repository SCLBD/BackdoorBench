'''
This file is modified based on the following source:

link : https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release
The original license is placed at the end of this file.

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. during training the backdoor attack generalization to lower poison ratio (generalize_to_lower_pratio)
    5. save process

basic sturcture for main:
    1. config args, save_path, fix random seed
    2. set the clean train data and clean test data
    3. set the device, model, criterion, optimizer, training schedule.
    4. set the backdoor warping
    5. training with backdoor modification simultaneously
    6. save attack result
'''


import sys, yaml, os, logging

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import kornia.augmentation as A
import json
import shutil
import argparse
from utils.log_assist import get_git_info
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import random
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pprint import pformat
from copy import deepcopy
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()

from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.save_path_generate import generate_save_folder
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape, get_dataset_normalization, dataset_and_transform_generate, get_dataset_denormalization
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.save_load_attack import summary_dict
from utils.trainer_cls import Metric_Aggregator, ModelTrainerCLS
from utils.bd_dataset import prepro_cls_DatasetBD, xy_iter
from utils.save_load_attack import save_attack_result, sample_pil_imgs
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler

agg = Metric_Aggregator()

class Args:
    pass


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


class Denormalizer:
    def __init__(self, opt):
        self.denormalizer = self._get_denormalizer(opt)

    def _get_denormalizer(self, opt):
        denormalizer = Denormalize(opt, get_dataset_normalization(opt.dataset).mean, get_dataset_normalization(opt.dataset).std)
        return denormalizer

    def __call__(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x


class MNISTBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(MNISTBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ind = None

    def forward(self, x):
        return self.conv1(F.relu(self.bn1(x)))


class NetC_MNIST(nn.Module):
    def __init__(self):
        super(NetC_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 2, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = MNISTBlock(32, 64, 2)
        self.layer3 = MNISTBlock(64, 64, 2)
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.3)
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
        begin_time = time.time()

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

    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def get_transform(opt, train=True, pretensor_transform=False):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if pretensor_transform:
        if train:
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))
            if opt.dataset == "cifar10":
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(get_dataset_normalization(opt.dataset))

    return transforms.Compose(transforms_list)


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
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

def get_dataloader(opt, train=True, pretensor_transform=False):

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
        train_transform = get_transform(opt, train, pretensor_transform)
        # dataset.transform = train_transform
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
        test_transform = get_transform(opt, train, pretensor_transform)
        dataset = prepro_cls_DatasetBD(
            full_dataset_without_transform = dataset,
            poison_idx = np.zeros(len(dataset)),
            bd_image_pre_transform = None,
            bd_label_pre_transform = None,
            ori_image_transform_in_loading = test_transform,
            ori_label_transform_in_loading = None,
            add_details_in_preprocess = False,
        )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=train)

    return dataloader


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--yaml_path', type=str, default='../config/attack/wanet/default.yaml',
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--model', type=str, help='Only use when model is not given in original code !!!')
    parser.add_argument('--save_folder_name', type=str,
                        help='(Optional) should be time str + given unique identification str')

    parser.add_argument('--random_seed', type=int)
    parser.add_argument("--dataset_path", type=str, )  # default="/home/ubuntu/temps/")
    parser.add_argument("--checkpoints", type=str, )  # default="./checkpoints")
    parser.add_argument("--temps", type=str, )  # default="./temps")
    parser.add_argument("--device", type=str, )  # default="cuda")
    parser.add_argument("--continue_training", action="store_true")

    parser.add_argument("--dataset", type=str, )  # default="cifar10")
    parser.add_argument("--attack_mode", type=str, )  # default="all2one")

    parser.add_argument("--bs", type=int, )  # default=128)
    parser.add_argument("--lr", type=float, )  # default=1e-2)
    parser.add_argument("--lr_scheduler", type=str, )
    parser.add_argument('--client_optimizer', type = str)
    parser.add_argument('--sgd_momentum', type = str)
    parser.add_argument('--wd', type=str)
    parser.add_argument("--steplr_milestones", type=list, )  # default=[100, 200, 300, 400])
    parser.add_argument("--steplr_gamma", type=float, )  # default=0.1)
    parser.add_argument("--epochs", type=int, )  # default=1000)
    parser.add_argument("--num_workers", type=float, )  # default=6)

    parser.add_argument("--target_label", type=int, )  # default=0)
    parser.add_argument("--pratio", type=float, )  # default=0.1)
    parser.add_argument("--cross_ratio", type=float, )  # default=2)  # rho_a = pratio, rho_n = pratio * cross_ratio

    parser.add_argument("--random_rotation", type=int, )  # default=10)
    parser.add_argument("--random_crop", type=int, )  # default=5)

    parser.add_argument("--s", type=float, )  # default=0.5)
    parser.add_argument("--k", type=int, )  # default=4)
    parser.add_argument(
        "--grid_rescale", type=float, )  # default=1
    # scale grid values to avoid pixel values going out of [-1, 1]. For example, grid-rescale = 0.98

    return parser


def get_model(opt):

    logging.info('use generate_cls_model() ')
    netC = generate_cls_model(
        opt.model,
        opt.num_classes,
        image_size=opt.img_size[0],
    )
    if torch.cuda.device_count() > 1 and opt.device == 'cuda':
        logging.info("device='cuda', default use all device")
        netC = torch.nn.DataParallel(netC)
    netC.to(opt.device)
    logging.warning(f'actually model use = {opt.model}')

    # args_opt = Args()
    # args_opt.__dict__ = {
    #     "client_optimizer" : opt.client_optimizer,
    #     "lr" : opt.lr,
    #     "lr_scheduler" : opt.lr_scheduler,
    #     "sgd_momentum" : opt.sgd_momentum,
    #     "wd" : opt.wd,
    # }
    optimizerC, schedulerC = argparser_opt_scheduler(netC, args = opt)

    # Optimizer
    # optimizerC = torch.optim.SGD(netC.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)
    # # Scheduler
    # schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC

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
def train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid,  epoch, opt):
    logging.info(" Train:")
    netC.train()
    rate_bd = opt.pratio
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_cross = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    avg_acc_cross = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]

        # Create backdoor data
        num_bd = int(generalize_to_lower_pratio(rate_bd,bs)) #int(bs * rate_bd)
        num_cross = int(num_bd * opt.cross_ratio)
        grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        ins = torch.rand(num_cross, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
        grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / opt.input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
        if opt.attack_mode == "all2all":
            targets_bd = torch.remainder(targets[:num_bd], opt.num_classes)

        inputs_cross = F.grid_sample(inputs[num_bd: (num_bd + num_cross)], grid_temps2, align_corners=True)

        total_inputs = torch.cat([inputs_bd, inputs_cross, inputs[(num_bd + num_cross):]], dim=0)
        total_inputs = transforms(total_inputs)
        total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
        start = time.time()
        total_preds = netC(total_inputs)
        total_time += time.time() - start

        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()

        total_clean += bs - num_bd - num_cross
        total_bd += num_bd
        total_cross += num_cross
        total_clean_correct += torch.sum(
            torch.argmax(total_preds[(num_bd + num_cross):], dim=1) == total_targets[(num_bd + num_cross):]
        )
        total_bd_correct += (torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd) if num_bd > 0 else 0)
        if num_cross:
            total_cross_correct += (torch.sum(
                torch.argmax(total_preds[num_bd: (num_bd + num_cross)], dim=1)
                == total_targets[num_bd: (num_bd + num_cross)]
            ) if num_bd > 0 else 0)
            avg_acc_cross = total_cross_correct  / total_cross if total_cross > 0 else 0

        avg_acc_clean = total_clean_correct  / total_clean if total_clean > 0 else 0
        avg_acc_bd = total_bd_correct  / total_bd if total_bd > 0 else 0

        avg_loss_ce = total_loss_ce / total_sample

        if num_cross:
            progress_bar(
                batch_idx,
                len(train_dl),
                "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross Acc: {:.4f}".format(
                    avg_loss_ce, avg_acc_clean, avg_acc_bd, avg_acc_cross
                ),
            )
        else:
            progress_bar(
                batch_idx,
                len(train_dl),
                "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean, avg_acc_bd),
            )

        # Save image for debugging
        if (num_bd > 0):
            if not os.path.exists(opt.temps):
                os.makedirs(opt.temps)
            path = os.path.join(opt.temps, "backdoor_image.png")
            torchvision.utils.save_image(inputs_bd, path, normalize=True)

        # Image for tensorboard
        if (num_bd > 0):
            residual = inputs_bd - inputs[:num_bd]
            batch_img = torch.cat([inputs[:num_bd], inputs_bd, total_inputs[:num_bd], residual], dim=2)
            batch_img = denormalizer(batch_img)
            batch_img = F.upsample(batch_img, scale_factor=(4, 4))
            grid = torchvision.utils.make_grid(batch_img, normalize=True)

    # for tensorboard

    if isinstance(schedulerC, torch.optim.lr_scheduler.ReduceLROnPlateau):
        schedulerC.step(loss.item())
    else:
        schedulerC.step()

    if num_cross:
        # logging.info(f'End train epoch {epoch} : avg_acc_clean : {avg_acc_clean}, avg_acc_bd : {avg_acc_bd}, avg_acc_cross : {avg_acc_cross} ')
        logging.info(f'End train epoch {epoch}')
        agg(
            {
                'train_avg_acc_clean': float(avg_acc_clean) ,
                'train_avg_acc_bd': float(avg_acc_bd) ,
                'avg_acc_cross': float(avg_acc_cross) ,
            }
        )
    else:
        # logging.info(
        #     f'End train epoch {epoch} : avg_acc_clean : {avg_acc_clean}, avg_acc_bd : {avg_acc_bd}')
        logging.info(f'End train epoch {epoch}')
        agg(
            {
                'train_avg_acc_clean': float(avg_acc_clean) ,
                'train_avg_acc_bd': float(avg_acc_bd) ,
                # 'avg_acc_cross': avg_acc_cross,
            }
        )


def eval(
        netC,
        optimizerC,
        schedulerC,
        test_dls,
        noise_grid,
        identity_grid,
        best_clean_acc,
        best_bd_acc,
        best_cross_acc,
        epoch,
        opt,
):
    (test_dl, poison_test_dl, cross_test_dl) = test_dls
    trainer = ModelTrainerCLS(netC)
    trainer.criterion = torch.nn.CrossEntropyLoss()

    clean_test_metric = trainer.test(
        test_dl, device=opt.device
    )
    acc_clean = clean_test_metric['test_correct'] / clean_test_metric['test_total']

    poison_test_metric = trainer.test(
        poison_test_dl, device=opt.device
    )
    acc_bd = poison_test_metric['test_correct'] / poison_test_metric['test_total']

    if opt.cross_ratio and cross_test_dl is not None:
        cross_test_metric = trainer.test(
            cross_test_dl, device=opt.device
        )
        acc_cross = cross_test_metric['test_correct'] / cross_test_metric['test_total']
        logging.info(f"epoch:{epoch}, acc_clean:{acc_clean}, best_clean_acc:{best_clean_acc}, acc_bd:{acc_bd}, best_bd_acc:{best_bd_acc}, acc_cross:{acc_cross}, best_cross_acc:{best_cross_acc}")
    else:
        logging.info(
            f"epoch:{epoch}, acc_clean:{acc_clean}, best_clean_acc:{best_clean_acc}, acc_bd:{acc_bd}, best_bd_acc:{best_bd_acc}")

    # logging.info(" Eval:")
    # netC.eval()
    #
    # total_sample = 0
    # total_clean_correct = 0
    # total_bd_correct = 0
    # total_cross_correct = 0
    # total_ae_loss = 0
    #
    # criterion_BCE = torch.nn.BCELoss()
    #
    # for batch_idx, (inputs, targets) in enumerate(test_dl):
    #     with torch.no_grad():
    #         inputs, targets = inputs.to(opt.device), targets.to(opt.device)
    #         bs = inputs.shape[0]
    #         total_sample += bs
    #
    #         # Evaluate Clean
    #         preds_clean = netC(inputs)
    #         total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)
    #
    #         # Evaluate Backdoor
    #         grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
    #         grid_temps = torch.clamp(grid_temps, -1, 1)
    #
    #         ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
    #         grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / opt.input_height
    #         grid_temps2 = torch.clamp(grid_temps2, -1, 1)
    #
    #         inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
    #         if opt.attack_mode == "all2one":
    #             position_changed = (
    #                         opt.target_label != targets)  # since if label does not change, then cannot tell if the poison is effective or not.
    #             targets_bd = (torch.ones_like(targets) * opt.target_label)[position_changed]
    #             inputs_bd = inputs_bd[position_changed]
    #         if opt.attack_mode == "all2all":
    #             targets_bd = torch.remainder(targets, opt.num_classes)
    #         preds_bd = netC(inputs_bd)
    #         total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
    #
    #         acc_clean = total_clean_correct  / total_sample
    #         acc_bd = total_bd_correct  / total_sample
    #
    #         # Evaluate cross
    #         if opt.cross_ratio:
    #             inputs_cross = F.grid_sample(inputs, grid_temps2, align_corners=True)
    #             preds_cross = netC(inputs_cross)
    #             total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)
    #
    #             acc_cross = total_cross_correct  / total_sample
    #
    #             info_string = (
    #                 "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f} | Cross: {:.4f}".format(
    #                     acc_clean, best_clean_acc, acc_bd, best_bd_acc, acc_cross, best_cross_acc
    #                 )
    #             )
    #         else:
    #             info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
    #                 acc_clean, best_clean_acc, acc_bd, best_bd_acc
    #             )
    #         progress_bar(batch_idx, len(test_dl), info_string)

    # Save checkpoint
    if acc_clean > best_clean_acc or (acc_clean > best_clean_acc - 0.1 and acc_bd > best_bd_acc):
        logging.info(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        if opt.cross_ratio:
            best_cross_acc = acc_cross
        else:
            best_cross_acc = torch.tensor([0])
        state_dict = {
            "netC": netC.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "best_cross_acc": best_cross_acc,
            "epoch_current": epoch,
            "identity_grid": identity_grid,
            "noise_grid": noise_grid,
        }
        torch.save(state_dict, opt.ckpt_path)
        with open(os.path.join(opt.ckpt_folder, "results.txt"), "w+") as f:
            results_dict = {
                "clean_acc": best_clean_acc,
                "bd_acc": best_bd_acc,
                "cross_acc": best_cross_acc,
            }
            json.dump(results_dict, f, indent=2)

    if opt.cross_ratio:
        return best_clean_acc, best_bd_acc, best_cross_acc, acc_clean, acc_bd, acc_cross
    else:
        return best_clean_acc, best_bd_acc, best_cross_acc, acc_clean, acc_bd, 0

def main():

    ###1. config args, save_path, fix random seed
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
            run_info='wanet',
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

    fileHandler = logging.FileHandler(save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)

    fix_random(int(opt.random_seed))

    ### 2. set the clean train data and clean test data
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    ### 3. set the device, model, criterion, optimizer, training schedule.
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if opt.continue_training:
        if os.path.exists(opt.ckpt_path):
            logging.info("Continue training!!")
            state_dict = torch.load(opt.ckpt_path)
            netC.load_state_dict(state_dict["netC"])
            optimizerC.load_state_dict(state_dict["optimizerC"])
            schedulerC.load_state_dict(state_dict["schedulerC"])
            best_clean_acc = state_dict["best_clean_acc"]
            best_bd_acc = state_dict["best_bd_acc"]
            best_cross_acc = state_dict["best_cross_acc"]
            epoch_current = state_dict["epoch_current"]
            identity_grid = state_dict["identity_grid"]
            noise_grid = state_dict["noise_grid"]
        else:
            logging.info("Pretrained model doesnt exist")
            exit()
    else:
        logging.info("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        best_cross_acc = 0.0
        epoch_current = 0

        ### 4. set the backdoor warping
        ins = torch.rand(1, 2, opt.k, opt.k) * 2 - 1  # generate (1,2,4,4) shape [-1,1] gaussian
        ins = ins / torch.mean(
            torch.abs(ins))  # scale up, increase var, so that mean of positive part and negative be +1 and -1
        noise_grid = (
            F.upsample(ins, size=opt.input_height, mode="bicubic",
                       align_corners=True)  # here upsample and make the dimension match
                .permute(0, 2, 3, 1)
                .to(opt.device)
        )
        array1d = torch.linspace(-1, 1, steps=opt.input_height)
        x, y = torch.meshgrid(array1d,
                              array1d)  # form two mesh grid correspoding to x, y of each position in height * width matrix
        identity_grid = torch.stack((y, x), 2)[None, ...].to(
            opt.device)  # stack x,y like two layer, then add one more dimension at first place. (have torch.Size([1, 32, 32, 2]))

        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)

    logging.info(pformat(opt.__dict__))#set here since the opt change once.
    try:
        logging.info(pformat(get_git_info()))
    except:
        logging.info('Getting git info fails.')

    ### generate the dataloaders for eval

    # filter out transformation that not reversible
    transforms_reversible = transforms.Compose(
        list(
            filter(
                lambda x: isinstance(x, (transforms.Normalize, transforms.Resize, transforms.ToTensor)),
                deepcopy(test_dl.dataset.ori_image_transform_in_loading.transforms)
            )
        )
    )
    # get denormalizer
    for trans_t in deepcopy(test_dl.dataset.ori_image_transform_in_loading.transforms):
        if isinstance(trans_t, transforms.Normalize):
            denormalizer = get_dataset_denormalization(trans_t)
            logging.info(f"{denormalizer}")

    reversible_test_ds = deepcopy(test_dl.dataset)
    reversible_test_ds.ori_image_transform_in_loading = transforms_reversible

    reversible_test_dl = torch.utils.data.DataLoader(reversible_test_ds, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=False)

    x_poison, y_poison = [], []
    if opt.cross_ratio:
        x_cross, y_cross = [], []

    for batch_idx, (inputs, targets) in enumerate(reversible_test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]

            # Evaluate Backdoor
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / opt.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            inputs_bd = denormalizer(F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True))

            if opt.attack_mode == "all2one":
                position_changed = (
                            opt.target_label != targets)  # since if label does not change, then cannot tell if the poison is effective or not.
                targets_bd = (torch.ones_like(targets) * opt.target_label)[position_changed]
                inputs_bd = inputs_bd[position_changed]
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets, opt.num_classes)

            x_poison += ([to_pil(t_img) for t_img in inputs_bd.detach().clone().cpu()])
            y_poison += targets_bd.detach().clone().cpu().tolist()

            # Evaluate cross
            if opt.cross_ratio:
                inputs_cross = denormalizer(F.grid_sample(inputs, grid_temps2, align_corners=True))
                x_cross +=  ([to_pil(t_img) for t_img in inputs_cross.detach().clone().cpu()])
                y_cross += (targets.detach().clone().cpu().tolist())

    poison_test_ds = xy_iter(x_poison, y_poison, deepcopy(test_dl.dataset.ori_image_transform_in_loading))
    poison_test_dl = torch.utils.data.DataLoader(poison_test_ds, batch_size=opt.bs, num_workers=opt.num_workers,
                                                 shuffle=False)
    if opt.cross_ratio:
        cross_test_ds = xy_iter(x_cross, y_cross, deepcopy(test_dl.dataset.ori_image_transform_in_loading))
        cross_test_dl = torch.utils.data.DataLoader(cross_test_ds, batch_size=opt.bs, num_workers=opt.num_workers,
                                                    shuffle=False)
    else:
        cross_test_dl = None

    test_dls = (test_dl, poison_test_dl, cross_test_dl)
    sample_pil_imgs(test_dl.dataset.data, f"{save_path}/test_dl_samples")
    sample_pil_imgs(poison_test_dl.dataset.data, f"{save_path}/poison_test_dl_samples")
    sample_pil_imgs(cross_test_dl.dataset.data, f"{save_path}/cross_test_dl_samples")

    logging.warning(f"acc_cross and best_cross_acc may be 0 if no cross sample are used !!!!")

    ### 5. training with backdoor modification simultaneously
    for epoch in range(epoch_current, opt.epochs):
        logging.info("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid,  epoch, opt)
        best_clean_acc, best_bd_acc, best_cross_acc, acc_clean, acc_bd, acc_cross = eval(
            netC,
            optimizerC,
            schedulerC,
            test_dls,
            noise_grid,
            identity_grid,
            best_clean_acc,
            best_bd_acc,
            best_cross_acc,
            epoch,
            opt,
        )

        agg({
            'test_epoch_num':float(epoch),
            'best_clean_acc': float(best_clean_acc),
            'best_bd_acc': float(best_bd_acc),
            'best_cross_acc': float(best_cross_acc),
            'acc_clean': float(acc_clean),
            'acc_bd': float(acc_bd),
            'acc_cross': float(acc_cross),
        })

    ### 6. save attack result

    train_dl = torch.utils.data.DataLoader(
        train_dl.dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=False)
    train_dl.dataset.ori_image_transform_in_loading = transforms.Compose(
        list(filter(lambda x: isinstance(x, (transforms.Normalize, transforms.Resize, transforms.ToTensor)),
                    train_dl.dataset.ori_image_transform_in_loading.transforms)))
    for trans_t in train_dl.dataset.ori_image_transform_in_loading.transforms:
        if isinstance(trans_t, transforms.Normalize):
            denormalizer = get_dataset_denormalization(trans_t)
            logging.info(f"{denormalizer}")

    one_hot_original_index = []
    bd_input = []
    bd_targets = []
    original_targets = []

    netC.eval()
    netC.to(opt.device)

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]

        # Create backdoor data
        num_bd = int(generalize_to_lower_pratio(opt.pratio,bs)) #int(bs * rate_bd)
        num_cross = int(num_bd * opt.cross_ratio)
        grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        ins = torch.rand(num_cross, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
        grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / opt.input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        inputs_bd = (F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True))
        if num_bd > 0:
            inputs_bd = torch.cat([denormalizer(img)[None,...] for img in inputs_bd])

        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
        if opt.attack_mode == "all2all":
            targets_bd = torch.remainder(targets[:num_bd], opt.num_classes)
        # add indexy
        one_hot = np.zeros(bs)
        one_hot[:(num_bd + num_cross)] = 1
        one_hot_original_index.append(one_hot)

        inputs_cross = F.grid_sample(inputs[num_bd: (num_bd + num_cross)], grid_temps2, align_corners=True)
        if num_cross > 0:
            inputs_cross = torch.cat([denormalizer(img)[None,...] for img in inputs_cross])

        # no transform !
        original_targets += ((targets.detach().clone().cpu())[: (num_bd + num_cross)]).tolist()
        bd_input.append(torch.cat([inputs_bd.detach().clone().cpu(), inputs_cross.detach().clone().cpu()], dim=0))
        bd_targets.append(torch.cat([targets_bd.detach().clone().cpu(), (targets.detach().clone().cpu())[num_bd: (num_bd + num_cross)]], dim=0))

    bd_train_x = [to_pil(t_img) for t_img in torch.cat(bd_input, dim=0).float().cpu()]
    bd_train_y = torch.cat(bd_targets, dim=0).long().cpu().numpy()
    train_poison_indicator = np.concatenate(one_hot_original_index)
    bd_train_original_index = np.where(train_poison_indicator == 1)[
                    0] if train_poison_indicator is not None else None
    logging.warning('Here the bd and cross samples are all saved in attack_result!!!!')

    bd_train_for_save = prepro_cls_DatasetBD(
        full_dataset_without_transform = list(zip(bd_train_x, bd_train_y)),
        poison_idx = np.ones_like(bd_train_y),
        add_details_in_preprocess = True,
        clean_image_pre_transform = None,
        bd_image_pre_transform = None,
        bd_label_pre_transform = None,
        end_pre_process = None,
        ori_image_transform_in_loading = None,
        ori_label_transform_in_loading = None,
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

    test_dl = torch.utils.data.DataLoader(
        test_dl.dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=False)
    test_dl.dataset.ori_image_transform_in_loading = transforms.Compose(
        list(filter(lambda x: isinstance(x, (transforms.Normalize, transforms.Resize, transforms.ToTensor)),
                    test_dl.dataset.ori_image_transform_in_loading.transforms)))
    for trans_t in test_dl.dataset.ori_image_transform_in_loading.transforms:
        if isinstance(trans_t, transforms.Normalize):
            denormalizer = get_dataset_denormalization(trans_t)
            logging.info(f"{denormalizer}")

    test_bd_input = []
    test_bd_targets = []
    test_bd_poison_indicator = []
    test_bd_origianl_targets = []

    netC.eval()
    netC.to(opt.device)

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]

            # Evaluate Backdoor
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / opt.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            inputs_bd = (F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True))
            inputs_bd = torch.cat([denormalizer(img)[None,...] for img in inputs_bd])

            if opt.attack_mode == "all2one":

                position_changed = (opt.target_label != targets) # since if label does not change, then cannot tell if the poison is effective or not.

                test_bd_poison_indicator.append(position_changed)
                test_bd_origianl_targets.append(targets)

                targets_bd = (torch.ones_like(targets) * opt.target_label)[position_changed]
                inputs_bd = inputs_bd[position_changed]

            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets, opt.num_classes)

                test_bd_poison_indicator.append(torch.ones_like(targets))
                test_bd_origianl_targets.append(targets)


            # no transform !
            test_bd_input.append((inputs_bd.detach().clone().cpu()))
            test_bd_targets.append(targets_bd.detach().clone().cpu())

    bd_test_x = [to_pil(t_img) for t_img in torch.cat(test_bd_input, dim=0).float().cpu()]
    bd_test_y = torch.cat(test_bd_targets, dim=0).long().cpu().numpy()
    test_bd_origianl_index = np.where(torch.cat(test_bd_poison_indicator, dim = 0).long().cpu().numpy())[0]
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
        model_name = opt.model,
        num_classes = opt.num_classes,
        model = netC.cpu().state_dict(),
        data_path = opt.dataset_path,
        img_size = (opt.input_height, opt.input_width, opt.input_channel),
        clean_data = opt.dataset,
        bd_train = bd_train_for_save,
        bd_test = bd_test_for_save,
        save_path = f'{save_path}',
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
    #             'original_index': bd_train_original_index,
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
    #
    # torch.save(
    #     final_save_dict,
    #     f'{save_path}/attack_result.pt',
    # )

    torch.save(opt.__dict__, save_path + '/info.pickle')

    agg.to_dataframe().to_csv(f"{save_path}/attack_df.csv")
    agg.summary().to_csv(f"{save_path}/attack_df_summary.csv")

if __name__ == "__main__":
    main()

'''
original license:
                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
software and other kinds of works.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
the GNU General Public License is intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.  We, the Free Software Foundation, use the
GNU General Public License for most of our software; it applies also to
any other work released this way by its authors.  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  To protect your rights, we need to prevent others from denying you
these rights or asking you to surrender the rights.  Therefore, you have
certain responsibilities if you distribute copies of the software, or if
you modify it: responsibilities to respect the freedom of others.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must pass on to the recipients the same
freedoms that you received.  You must make sure that they, too, receive
or can get the source code.  And you must show them these terms so they
know their rights.

  Developers that use the GNU GPL protect your rights with two steps:
(1) assert copyright on the software, and (2) offer you this License
giving you legal permission to copy, distribute and/or modify it.

  For the developers' and authors' protection, the GPL clearly explains
that there is no warranty for this free software.  For both users' and
authors' sake, the GPL requires that modified versions be marked as
changed, so that their problems will not be attributed erroneously to
authors of previous versions.

  Some devices are designed to deny users access to install or run
modified versions of the software inside them, although the manufacturer
can do so.  This is fundamentally incompatible with the aim of
protecting users' freedom to change the software.  The systematic
pattern of such abuse occurs in the area of products for individuals to
use, which is precisely where it is most unacceptable.  Therefore, we
have designed this version of the GPL to prohibit the practice for those
products.  If such problems arise substantially in other domains, we
stand ready to extend this provision to those domains in future versions
of the GPL, as needed to protect the freedom of users.

  Finally, every program is threatened constantly by software patents.
States should not allow patents to restrict development and use of
software on general-purpose computers, but in those that do, we wish to
avoid the special danger that patents applied to a free program could
make it effectively proprietary.  To prevent this, the GPL assures that
patents cannot be used to render the program non-free.

  The precise terms and conditions for copying, distribution and
modification follow.

                       TERMS AND CONDITIONS

  0. Definitions.

  "This License" refers to version 3 of the GNU General Public License.

  "Copyright" also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

  "The Program" refers to any copyrightable work licensed under this
License.  Each licensee is addressed as "you".  "Licensees" and
"recipients" may be individuals or organizations.

  To "modify" a work means to copy from or adapt all or part of the work
in a fashion requiring copyright permission, other than the making of an
exact copy.  The resulting work is called a "modified version" of the
earlier work or a work "based on" the earlier work.

  A "covered work" means either the unmodified Program or a work based
on the Program.

  To "propagate" a work means to do anything with it that, without
permission, would make you directly or secondarily liable for
infringement under applicable copyright law, except executing it on a
computer or modifying a private copy.  Propagation includes copying,
distribution (with or without modification), making available to the
public, and in some countries other activities as well.

  To "convey" a work means any kind of propagation that enables other
parties to make or receive copies.  Mere interaction with a user through
a computer network, with no transfer of a copy, is not conveying.

  An interactive user interface displays "Appropriate Legal Notices"
to the extent that it includes a convenient and prominently visible
feature that (1) displays an appropriate copyright notice, and (2)
tells the user that there is no warranty for the work (except to the
extent that warranties are provided), that licensees may convey the
work under this License, and how to view a copy of this License.  If
the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

  1. Source Code.

  The "source code" for a work means the preferred form of the work
for making modifications to it.  "Object code" means any non-source
form of a work.

  A "Standard Interface" means an interface that either is an official
standard defined by a recognized standards body, or, in the case of
interfaces specified for a particular programming language, one that
is widely used among developers working in that language.

  The "System Libraries" of an executable work include anything, other
than the work as a whole, that (a) is included in the normal form of
packaging a Major Component, but which is not part of that Major
Component, and (b) serves only to enable use of the work with that
Major Component, or to implement a Standard Interface for which an
implementation is available to the public in source code form.  A
"Major Component", in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system
(if any) on which the executable work runs, or a compiler used to
produce the work, or an object code interpreter used to run it.

  The "Corresponding Source" for a work in object code form means all
the source code needed to generate, install, and (for an executable
work) run the object code and to modify the work, including scripts to
control those activities.  However, it does not include the work's
System Libraries, or general-purpose tools or generally available free
programs which are used unmodified in performing those activities but
which are not part of the work.  For example, Corresponding Source
includes interface definition files associated with source files for
the work, and the source code for shared libraries and dynamically
linked subprograms that the work is specifically designed to require,
such as by intimate data communication or control flow between those
subprograms and other parts of the work.

  The Corresponding Source need not include anything that users
can regenerate automatically from other parts of the Corresponding
Source.

  The Corresponding Source for a work in source code form is that
same work.

  2. Basic Permissions.

  All rights granted under this License are granted for the term of
copyright on the Program, and are irrevocable provided the stated
conditions are met.  This License explicitly affirms your unlimited
permission to run the unmodified Program.  The output from running a
covered work is covered by this License only if the output, given its
content, constitutes a covered work.  This License acknowledges your
rights of fair use or other equivalent, as provided by copyright law.

  You may make, run and propagate covered works that you do not
convey, without conditions so long as your license otherwise remains
in force.  You may convey covered works to others for the sole purpose
of having them make modifications exclusively for you, or provide you
with facilities for running those works, provided that you comply with
the terms of this License in conveying all material for which you do
not control copyright.  Those thus making or running the covered works
for you must do so exclusively on your behalf, under your direction
and control, on terms that prohibit them from making any copies of
your copyrighted material outside their relationship with you.

  Conveying under any other circumstances is permitted solely under
the conditions stated below.  Sublicensing is not allowed; section 10
makes it unnecessary.

  3. Protecting Users' Legal Rights From Anti-Circumvention Law.

  No covered work shall be deemed part of an effective technological
measure under any applicable law fulfilling obligations under article
11 of the WIPO copyright treaty adopted on 20 December 1996, or
similar laws prohibiting or restricting circumvention of such
measures.

  When you convey a covered work, you waive any legal power to forbid
circumvention of technological measures to the extent such circumvention
is effected by exercising rights under this License with respect to
the covered work, and you disclaim any intention to limit operation or
modification of the work as a means of enforcing, against the work's
users, your or third parties' legal rights to forbid circumvention of
technological measures.

  4. Conveying Verbatim Copies.

  You may convey verbatim copies of the Program's source code as you
receive it, in any medium, provided that you conspicuously and
appropriately publish on each copy an appropriate copyright notice;
keep intact all notices stating that this License and any
non-permissive terms added in accord with section 7 apply to the code;
keep intact all notices of the absence of any warranty; and give all
recipients a copy of this License along with the Program.

  You may charge any price or no price for each copy that you convey,
and you may offer support or warranty protection for a fee.

  5. Conveying Modified Source Versions.

  You may convey a work based on the Program, or the modifications to
produce it from the Program, in the form of source code under the
terms of section 4, provided that you also meet all of these conditions:

    a) The work must carry prominent notices stating that you modified
    it, and giving a relevant date.

    b) The work must carry prominent notices stating that it is
    released under this License and any conditions added under section
    7.  This requirement modifies the requirement in section 4 to
    "keep intact all notices".

    c) You must license the entire work, as a whole, under this
    License to anyone who comes into possession of a copy.  This
    License will therefore apply, along with any applicable section 7
    additional terms, to the whole of the work, and all its parts,
    regardless of how they are packaged.  This License gives no
    permission to license the work in any other way, but it does not
    invalidate such permission if you have separately received it.

    d) If the work has interactive user interfaces, each must display
    Appropriate Legal Notices; however, if the Program has interactive
    interfaces that do not display Appropriate Legal Notices, your
    work need not make them do so.

  A compilation of a covered work with other separate and independent
works, which are not by their nature extensions of the covered work,
and which are not combined with it such as to form a larger program,
in or on a volume of a storage or distribution medium, is called an
"aggregate" if the compilation and its resulting copyright are not
used to limit the access or legal rights of the compilation's users
beyond what the individual works permit.  Inclusion of a covered work
in an aggregate does not cause this License to apply to the other
parts of the aggregate.

  6. Conveying Non-Source Forms.

  You may convey a covered work in object code form under the terms
of sections 4 and 5, provided that you also convey the
machine-readable Corresponding Source under the terms of this License,
in one of these ways:

    a) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by the
    Corresponding Source fixed on a durable physical medium
    customarily used for software interchange.

    b) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by a
    written offer, valid for at least three years and valid for as
    long as you offer spare parts or customer support for that product
    model, to give anyone who possesses the object code either (1) a
    copy of the Corresponding Source for all the software in the
    product that is covered by this License, on a durable physical
    medium customarily used for software interchange, for a price no
    more than your reasonable cost of physically performing this
    conveying of source, or (2) access to copy the
    Corresponding Source from a network server at no charge.

    c) Convey individual copies of the object code with a copy of the
    written offer to provide the Corresponding Source.  This
    alternative is allowed only occasionally and noncommercially, and
    only if you received the object code with such an offer, in accord
    with subsection 6b.

    d) Convey the object code by offering access from a designated
    place (gratis or for a charge), and offer equivalent access to the
    Corresponding Source in the same way through the same place at no
    further charge.  You need not require recipients to copy the
    Corresponding Source along with the object code.  If the place to
    copy the object code is a network server, the Corresponding Source
    may be on a different server (operated by you or a third party)
    that supports equivalent copying facilities, provided you maintain
    clear directions next to the object code saying where to find the
    Corresponding Source.  Regardless of what server hosts the
    Corresponding Source, you remain obligated to ensure that it is
    available for as long as needed to satisfy these requirements.

    e) Convey the object code using peer-to-peer transmission, provided
    you inform other peers where the object code and Corresponding
    Source of the work are being offered to the general public at no
    charge under subsection 6d.

  A separable portion of the object code, whose source code is excluded
from the Corresponding Source as a System Library, need not be
included in conveying the object code work.

  A "User Product" is either (1) a "consumer product", which means any
tangible personal property which is normally used for personal, family,
or household purposes, or (2) anything designed or sold for incorporation
into a dwelling.  In determining whether a product is a consumer product,
doubtful cases shall be resolved in favor of coverage.  For a particular
product received by a particular user, "normally used" refers to a
typical or common use of that class of product, regardless of the status
of the particular user or of the way in which the particular user
actually uses, or expects or is expected to use, the product.  A product
is a consumer product regardless of whether the product has substantial
commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

  "Installation Information" for a User Product means any methods,
procedures, authorization keys, or other information required to install
and execute modified versions of a covered work in that User Product from
a modified version of its Corresponding Source.  The information must
suffice to ensure that the continued functioning of the modified object
code is in no case prevented or interfered with solely because
modification has been made.

  If you convey an object code work under this section in, or with, or
specifically for use in, a User Product, and the conveying occurs as
part of a transaction in which the right of possession and use of the
User Product is transferred to the recipient in perpetuity or for a
fixed term (regardless of how the transaction is characterized), the
Corresponding Source conveyed under this section must be accompanied
by the Installation Information.  But this requirement does not apply
if neither you nor any third party retains the ability to install
modified object code on the User Product (for example, the work has
been installed in ROM).

  The requirement to provide Installation Information does not include a
requirement to continue to provide support service, warranty, or updates
for a work that has been modified or installed by the recipient, or for
the User Product in which it has been modified or installed.  Access to a
network may be denied when the modification itself materially and
adversely affects the operation of the network or violates the rules and
protocols for communication across the network.

  Corresponding Source conveyed, and Installation Information provided,
in accord with this section must be in a format that is publicly
documented (and with an implementation available to the public in
source code form), and must require no special password or key for
unpacking, reading or copying.

  7. Additional Terms.

  "Additional permissions" are terms that supplement the terms of this
License by making exceptions from one or more of its conditions.
Additional permissions that are applicable to the entire Program shall
be treated as though they were included in this License, to the extent
that they are valid under applicable law.  If additional permissions
apply only to part of the Program, that part may be used separately
under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

  When you convey a copy of a covered work, you may at your option
remove any additional permissions from that copy, or from any part of
it.  (Additional permissions may be written to require their own
removal in certain cases when you modify the work.)  You may place
additional permissions on material, added by you to a covered work,
for which you have or can give appropriate copyright permission.

  Notwithstanding any other provision of this License, for material you
add to a covered work, you may (if authorized by the copyright holders of
that material) supplement the terms of this License with terms:

    a) Disclaiming warranty or limiting liability differently from the
    terms of sections 15 and 16 of this License; or

    b) Requiring preservation of specified reasonable legal notices or
    author attributions in that material or in the Appropriate Legal
    Notices displayed by works containing it; or

    c) Prohibiting misrepresentation of the origin of that material, or
    requiring that modified versions of such material be marked in
    reasonable ways as different from the original version; or

    d) Limiting the use for publicity purposes of names of licensors or
    authors of the material; or

    e) Declining to grant rights under trademark law for use of some
    trade names, trademarks, or service marks; or

    f) Requiring indemnification of licensors and authors of that
    material by anyone who conveys the material (or modified versions of
    it) with contractual assumptions of liability to the recipient, for
    any liability that these contractual assumptions directly impose on
    those licensors and authors.

  All other non-permissive additional terms are considered "further
restrictions" within the meaning of section 10.  If the Program as you
received it, or any part of it, contains a notice stating that it is
governed by this License along with a term that is a further
restriction, you may remove that term.  If a license document contains
a further restriction but permits relicensing or conveying under this
License, you may add to a covered work material governed by the terms
of that license document, provided that the further restriction does
not survive such relicensing or conveying.

  If you add terms to a covered work in accord with this section, you
must place, in the relevant source files, a statement of the
additional terms that apply to those files, or a notice indicating
where to find the applicable terms.

  Additional terms, permissive or non-permissive, may be stated in the
form of a separately written license, or stated as exceptions;
the above requirements apply either way.

  8. Termination.

  You may not propagate or modify a covered work except as expressly
provided under this License.  Any attempt otherwise to propagate or
modify it is void, and will automatically terminate your rights under
this License (including any patent licenses granted under the third
paragraph of section 11).

  However, if you cease all violation of this License, then your
license from a particular copyright holder is reinstated (a)
provisionally, unless and until the copyright holder explicitly and
finally terminates your license, and (b) permanently, if the copyright
holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

  Moreover, your license from a particular copyright holder is
reinstated permanently if the copyright holder notifies you of the
violation by some reasonable means, this is the first time you have
received notice of violation of this License (for any work) from that
copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

  Termination of your rights under this section does not terminate the
licenses of parties who have received copies or rights from you under
this License.  If your rights have been terminated and not permanently
reinstated, you do not qualify to receive new licenses for the same
material under section 10.

  9. Acceptance Not Required for Having Copies.

  You are not required to accept this License in order to receive or
run a copy of the Program.  Ancillary propagation of a covered work
occurring solely as a consequence of using peer-to-peer transmission
to receive a copy likewise does not require acceptance.  However,
nothing other than this License grants you permission to propagate or
modify any covered work.  These actions infringe copyright if you do
not accept this License.  Therefore, by modifying or propagating a
covered work, you indicate your acceptance of this License to do so.

  10. Automatic Licensing of Downstream Recipients.

  Each time you convey a covered work, the recipient automatically
receives a license from the original licensors, to run, modify and
propagate that work, subject to this License.  You are not responsible
for enforcing compliance by third parties with this License.

  An "entity transaction" is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an
organization, or merging organizations.  If propagation of a covered
work results from an entity transaction, each party to that
transaction who receives a copy of the work also receives whatever
licenses to the work the party's predecessor in interest had or could
give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if
the predecessor has it or can get it with reasonable efforts.

  You may not impose any further restrictions on the exercise of the
rights granted or affirmed under this License.  For example, you may
not impose a license fee, royalty, or other charge for exercise of
rights granted under this License, and you may not initiate litigation
(including a cross-claim or counterclaim in a lawsuit) alleging that
any patent claim is infringed by making, using, selling, offering for
sale, or importing the Program or any portion of it.

  11. Patents.

  A "contributor" is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based.  The
work thus licensed is called the contributor's "contributor version".

  A contributor's "essential patent claims" are all patent claims
owned or controlled by the contributor, whether already acquired or
hereafter acquired, that would be infringed by some manner, permitted
by this License, of making, using, or selling its contributor version,
but do not include claims that would be infringed only as a
consequence of further modification of the contributor version.  For
purposes of this definition, "control" includes the right to grant
patent sublicenses in a manner consistent with the requirements of
this License.

  Each contributor grants you a non-exclusive, worldwide, royalty-free
patent license under the contributor's essential patent claims, to
make, use, sell, offer for sale, import and otherwise run, modify and
propagate the contents of its contributor version.

  In the following three paragraphs, a "patent license" is any express
agreement or commitment, however denominated, not to enforce a patent
(such as an express permission to practice a patent or covenant not to
sue for patent infringement).  To "grant" such a patent license to a
party means to make such an agreement or commitment not to enforce a
patent against the party.

  If you convey a covered work, knowingly relying on a patent license,
and the Corresponding Source of the work is not available for anyone
to copy, free of charge and under the terms of this License, through a
publicly available network server or other readily accessible means,
then you must either (1) cause the Corresponding Source to be so
available, or (2) arrange to deprive yourself of the benefit of the
patent license for this particular work, or (3) arrange, in a manner
consistent with the requirements of this License, to extend the patent
license to downstream recipients.  "Knowingly relying" means you have
actual knowledge that, but for the patent license, your conveying the
covered work in a country, or your recipient's use of the covered work
in a country, would infringe one or more identifiable patents in that
country that you have reason to believe are valid.

  If, pursuant to or in connection with a single transaction or
arrangement, you convey, or propagate by procuring conveyance of, a
covered work, and grant a patent license to some of the parties
receiving the covered work authorizing them to use, propagate, modify
or convey a specific copy of the covered work, then the patent license
you grant is automatically extended to all recipients of the covered
work and works based on it.

  A patent license is "discriminatory" if it does not include within
the scope of its coverage, prohibits the exercise of, or is
conditioned on the non-exercise of one or more of the rights that are
specifically granted under this License.  You may not convey a covered
work if you are a party to an arrangement with a third party that is
in the business of distributing software, under which you make payment
to the third party based on the extent of your activity of conveying
the work, and under which the third party grants, to any of the
parties who would receive the covered work from you, a discriminatory
patent license (a) in connection with copies of the covered work
conveyed by you (or copies made from those copies), or (b) primarily
for and in connection with specific products or compilations that
contain the covered work, unless you entered into that arrangement,
or that patent license was granted, prior to 28 March 2007.

  Nothing in this License shall be construed as excluding or limiting
any implied license or other defenses to infringement that may
otherwise be available to you under applicable patent law.

  12. No Surrender of Others' Freedom.

  If conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot convey a
covered work so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you may
not convey it at all.  For example, if you agree to terms that obligate you
to collect a royalty for further conveying from those to whom you convey
the Program, the only way you could satisfy both those terms and this
License would be to refrain entirely from conveying the Program.

  13. Use with the GNU Affero General Public License.

  Notwithstanding any other provision of this License, you have
permission to link or combine any covered work with a work licensed
under version 3 of the GNU Affero General Public License into a single
combined work, and to convey the resulting work.  The terms of this
License will continue to apply to the part which is the covered work,
but the special requirements of the GNU Affero General Public License,
section 13, concerning interaction through a network will apply to the
combination as such.

  14. Revised Versions of this License.

  The Free Software Foundation may publish revised and/or new versions of
the GNU General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

  Each version is given a distinguishing version number.  If the
Program specifies that a certain numbered version of the GNU General
Public License "or any later version" applies to it, you have the
option of following the terms and conditions either of that numbered
version or of any later version published by the Free Software
Foundation.  If the Program does not specify a version number of the
GNU General Public License, you may choose any version ever published
by the Free Software Foundation.

  If the Program specifies that a proxy can decide which future
versions of the GNU General Public License can be used, that proxy's
public statement of acceptance of a version permanently authorizes you
to choose that version for the Program.

  Later license versions may give you additional or different
permissions.  However, no additional obligations are imposed on any
author or copyright holder as a result of your choosing to follow a
later version.

  15. Disclaimer of Warranty.

  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. Limitation of Liability.

  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

  17. Interpretation of Sections 15 and 16.

  If the disclaimer of warranty and limitation of liability provided
above cannot be given local legal effect according to their terms,
reviewing courts shall apply local law that most closely approximates
an absolute waiver of all civil liability in connection with the
Program, unless a warranty or assumption of liability accompanies a
copy of the Program in return for a fee.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
state the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

  If the program does terminal interaction, make it output a short
notice like this when it starts in an interactive mode:

    <program>  Copyright (C) <year>  <name of author>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, your program's commands
might be different; for a GUI interface, you would use an "about box".

  You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU GPL, see
<https://www.gnu.org/licenses/>.

  The GNU General Public License does not permit incorporating your program
into proprietary programs.  If your program is a subroutine library, you
may consider it more useful to permit linking proprietary applications with
the library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.  But first, please read
<https://www.gnu.org/licenses/why-not-lgpl.html>.
'''