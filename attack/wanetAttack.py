'''
from
@inproceedings{
    nguyen2021wanet,
    title={WaNet - Imperceptible Warping-based Backdoor Attack},
    author={Tuan Anh Nguyen and Anh Tuan Tran},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=eEn8KTtJOx}
}
link : https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release
'''


import sys, yaml, os, logging

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import kornia.augmentation as A
import json
import shutil
import argparse

import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import torchvision
import random
import time
import torch
import torch.nn as nn
from torchvision.models import resnet18 as ResNet18
from models.preact_resnet import PreActResNet18
import torchvision.transforms as transforms
from pprint import pformat

from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.save_path_generate import generate_save_folder
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
from utils.aggregate_block.model_trainer_generate import generate_cls_model


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
        if opt.dataset == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
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
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 2, 1)  # 14
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = MNISTBlock(32, 64, 2)  # 7
        self.layer3 = MNISTBlock(64, 64, 2)  # 4
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
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
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


# class GTSRB(data.Dataset):
#     def __init__(self, opt, train, transforms):
#         super(GTSRB, self).__init__()
#         if train:
#             self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
#             self.images, self.labels = self._get_data_train_list()
#         else:
#             self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
#             self.images, self.labels = self._get_data_test_list()
#
#         self.transforms = transforms
#
#     def _get_data_train_list(self):
#         images = []
#         labels = []
#         for c in range(0, 43):
#             prefix = self.data_folder + "/" + format(c, "05d") + "/"
#             gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
#             gtReader = csv.reader(gtFile, delimiter=";")
#             next(gtReader)
#             for row in gtReader:
#                 images.append(prefix + row[0])
#                 labels.append(int(row[7]))
#             gtFile.close()
#         return images, labels
#
#     def _get_data_test_list(self):
#         images = []
#         labels = []
#         prefix = os.path.join(self.data_folder, "GT-final_test.csv")
#         gtFile = open(prefix)
#         gtReader = csv.reader(gtFile, delimiter=";")
#         next(gtReader)
#         for row in gtReader:
#             images.append(self.data_folder + "/" + row[0])
#             labels.append(int(row[7]))
#         return images, labels
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, index):
#         image = Image.open(self.images[index])
#         image = self.transforms(image)
#         label = self.labels[index]
#         return image, label

# class CelebA_attr(data.Dataset):
#     def __init__(self, opt, split, transforms):
#         self.dataset = torchvision.datasets.CelebA(root=opt.data_root, split=split, target_type="attr", download=True)
#         self.list_attributes = [18, 31, 21]
#         self.transforms = transforms
#         self.split = split
#
#     def _convert_attributes(self, bool_attributes):
#         return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, index):
#         input, target = self.dataset[index]
#         input = self.transforms(input)
#         target = self._convert_attributes(target[self.list_attributes])
#         return (input, target)


def get_dataloader(opt, train=True, pretensor_transform=False):
    # transform = get_transform(opt, train, pretensor_transform)
    # if opt.dataset == "gtsrb":
    #     dataset = GTSRB(opt, train, transform)
    # elif opt.dataset == "mnist":
    #     dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    # elif opt.dataset == "cifar10":
    #     dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    # elif opt.dataset == "celeba":
    #     if train:
    #         split = "train"
    #     else:
    #         split = "test"
    #     dataset = CelebA_attr(opt, split, transform)
    # else:
    #     raise Exception("Invalid dataset")

    args = Args()
    args.dataset = opt.dataset
    args.dataset_path = opt.data_root
    args.img_size = (opt.input_height, opt.input_width, opt.input_channel)

    train_dataset_without_transform, \
    train_img_transform, \
    train_label_transfrom, \
    test_dataset_without_transform, \
    test_img_transform, \
    test_label_transform = dataset_and_transform_generate(args=opt)

    if train:
        dataset = train_dataset_without_transform
        try:
            train_transform = get_transform(opt, train, pretensor_transform)
            if train_transform is not None:
                logging.info('WARNING : transform use original transform')
        except:
            train_transform = train_img_transform
        dataset.transform = train_transform
    else:
        dataset = test_dataset_without_transform
        try:
            test_transform = get_transform(opt, train, pretensor_transform)
            if test_transform is not None:
                logging.info('WARNING : transform use original transform')
        except:
            test_transform = test_img_transform
        dataset.transform = test_transform

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
    return dataloader


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--yaml_path', type=str, default='../config/wanetAttack/default.yaml',
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--model_name', type=str, help='Only use when model is not given in original code !!!')
    parser.add_argument('--save_folder_name', type=str,
                        help='(Optional) should be time str + given unique identification str')

    parser.add_argument('--random_seed', type=int)
    parser.add_argument("--data_root", type=str, )  # default="/home/ubuntu/temps/")
    parser.add_argument("--checkpoints", type=str, )  # default="./checkpoints")
    parser.add_argument("--temps", type=str, )  # default="./temps")
    parser.add_argument("--device", type=str, )  # default="cuda")
    parser.add_argument("--continue_training", action="store_true")

    parser.add_argument("--dataset", type=str, )  # default="cifar10")
    parser.add_argument("--attack_mode", type=str, )  # default="all2one")

    parser.add_argument("--bs", type=int, )  # default=128)
    parser.add_argument("--lr_C", type=float, )  # default=1e-2)
    parser.add_argument("--schedulerC_milestones", type=list, )  # default=[100, 200, 300, 400])
    parser.add_argument("--schedulerC_lambda", type=float, )  # default=0.1)
    parser.add_argument("--n_iters", type=int, )  # default=1000)
    parser.add_argument("--num_workers", type=float, )  # default=6)

    parser.add_argument("--target_label", type=int, )  # default=0)
    parser.add_argument("--pc", type=float, )  # default=0.1)
    parser.add_argument("--cross_ratio", type=float, )  # default=2)  # rho_a = pc, rho_n = pc * cross_ratio

    parser.add_argument("--random_rotation", type=int, )  # default=10)
    parser.add_argument("--random_crop", type=int, )  # default=5)

    parser.add_argument("--s", type=float, )  # default=0.5)
    parser.add_argument("--k", type=int, )  # default=4)
    parser.add_argument(
        "--grid_rescale", type=float, )  # default=1
    # scale grid values to avoid pixel values going out of [-1, 1]. For example, grid-rescale = 0.98

    return parser


def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None
    logging.info('WARNING : here model is set by original code !!!')
    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
        opt.model_name = 'preactresnet18'
    elif opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)
        opt.model_name = 'resnet18'
    elif opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)
        opt.model_name = 'netc_mnist'  # TODO add to framework
    else:
        logging.info('use generate_cls_model() ')
        netC = generate_cls_model(opt.model_name, opt.num_classes)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

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
def train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, tf_writer, epoch, opt):
    logging.info(" Train:")
    netC.train()
    rate_bd = opt.pc
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
            avg_acc_cross = total_cross_correct * 100.0 / total_cross if total_cross > 0 else 0

        avg_acc_clean = total_clean_correct * 100.0 / total_clean if total_clean > 0 else 0
        avg_acc_bd = total_bd_correct * 100.0 / total_bd if total_bd > 0 else 0

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
        if not batch_idx % 50 and num_bd > 0:
            if not os.path.exists(opt.temps):
                os.makedirs(opt.temps)
            path = os.path.join(opt.temps, "backdoor_image.png")
            torchvision.utils.save_image(inputs_bd, path, normalize=True)

        # Image for tensorboard
        if batch_idx == len(train_dl) - 2 and num_bd > 0:
            residual = inputs_bd - inputs[:num_bd]
            batch_img = torch.cat([inputs[:num_bd], inputs_bd, total_inputs[:num_bd], residual], dim=2)
            batch_img = denormalizer(batch_img)
            batch_img = F.upsample(batch_img, scale_factor=(4, 4))
            grid = torchvision.utils.make_grid(batch_img, normalize=True)

    # for tensorboard
    if not epoch % 1 and num_bd > 0:
        tf_writer.add_scalars(
            "Clean Accuracy", {"Clean": avg_acc_clean, "Bd": avg_acc_bd, "Cross": avg_acc_cross}, epoch
        )
        tf_writer.add_image("Images", grid, global_step=epoch)

    schedulerC.step()
    if num_cross:
        logging.info(f'End train epoch {epoch} : avg_acc_clean : {avg_acc_clean}, avg_acc_bd : {avg_acc_bd}, avg_acc_cross : {avg_acc_cross} ')
    else:
        logging.info(
            f'End train epoch {epoch} : avg_acc_clean : {avg_acc_clean}, avg_acc_bd : {avg_acc_bd}')


def eval(
        netC,
        optimizerC,
        schedulerC,
        test_dl,
        noise_grid,
        identity_grid,
        best_clean_acc,
        best_bd_acc,
        best_cross_acc,
        tf_writer,
        epoch,
        opt,
):
    logging.info(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    total_ae_loss = 0

    criterion_BCE = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / opt.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets, opt.num_classes)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

            # Evaluate cross
            if opt.cross_ratio:
                inputs_cross = F.grid_sample(inputs, grid_temps2, align_corners=True)
                preds_cross = netC(inputs_cross)
                total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)

                acc_cross = total_cross_correct * 100.0 / total_sample

                info_string = (
                    "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f} | Cross: {:.4f}".format(
                        acc_clean, best_clean_acc, acc_bd, best_bd_acc, acc_cross, best_cross_acc
                    )
                )
            else:
                info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                    acc_clean, best_clean_acc, acc_bd, best_bd_acc
                )
            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    if not epoch % 1:
        tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)

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
                "clean_acc": best_clean_acc.item(),
                "bd_acc": best_bd_acc.item(),
                "cross_acc": best_cross_acc.item(),
            }
            json.dump(results_dict, f, indent=2)

    return best_clean_acc, best_bd_acc, best_cross_acc


def main():
    opt = get_arguments().parse_args()

    with open(opt.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)
    defaults.update({k: v for k, v in opt.__dict__.items() if v is not None})
    opt.__dict__ = defaults

    opt.dataset_path = opt.data_root

    opt.terminal_info = sys.argv

    # if opt.dataset in ["mnist", "cifar10"]:
    #     opt.num_classes = 10
    # elif opt.dataset == "gtsrb":
    #     opt.num_classes = 43
    # elif opt.dataset == "celeba":
    #     opt.num_classes = 8
    # else:
    #     raise Exception("Invalid Dataset")

    opt.num_classes = get_num_classes(opt.dataset)

    # if opt.dataset == "cifar10":
    #     opt.input_height = 32
    #     opt.input_width = 32
    #     opt.input_channel = 3
    # elif opt.dataset == "gtsrb":
    #     opt.input_height = 32
    #     opt.input_width = 32
    #     opt.input_channel = 3
    # elif opt.dataset == "mnist":
    #     opt.input_height = 28
    #     opt.input_width = 28
    #     opt.input_channel = 1
    # elif opt.dataset == "celeba":
    #     opt.input_height = 64
    #     opt.input_width = 64
    #     opt.input_channel = 3
    # else:
    #     raise Exception("Invalid Dataset")

    opt.input_height, opt.input_width, opt.input_channel = get_input_shape(opt.dataset)
    opt.img_size = (opt.input_height, opt.input_width, opt.input_channel)

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
    # logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")

    fileHandler = logging.FileHandler(save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)

    fix_random(int(opt.random_seed))

    # Dataset
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
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
            tf_writer = SummaryWriter(log_dir=opt.log_dir)
        else:
            logging.info("Pretrained model doesnt exist")
            exit()
    else:
        logging.info("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        best_cross_acc = 0.0
        epoch_current = 0

        # Prepare grid
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
        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    logging.info(pformat(opt.__dict__))#set here since the opt change once.

    for epoch in range(epoch_current, opt.n_iters):
        logging.info("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, tf_writer, epoch, opt)
        best_clean_acc, best_bd_acc, best_cross_acc = eval(
            netC,
            optimizerC,
            schedulerC,
            test_dl,
            noise_grid,
            identity_grid,
            best_clean_acc,
            best_bd_acc,
            best_cross_acc,
            tf_writer,
            epoch,
            opt,
        )
        logging.info(f'epoch : {epoch} best_clean_acc : {best_clean_acc}, best_bd_acc : {best_bd_acc}, best_cross_acc : {best_cross_acc}')

    # start saving process

    train_dl = torch.utils.data.DataLoader(
        train_dl.dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=False)

    one_hot_original_index = []
    bd_input = []
    bd_targets = []

    netC.eval()
    netC.to(opt.device)

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]

        # Create backdoor data
        num_bd = int(generalize_to_lower_pratio(opt.pc,bs)) #int(bs * rate_bd)
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
        # add indexy
        one_hot = np.zeros(bs)
        one_hot[:(num_bd + num_cross)] = 1
        one_hot_original_index.append(one_hot)

        inputs_cross = F.grid_sample(inputs[num_bd: (num_bd + num_cross)], grid_temps2, align_corners=True)

        # no transform !
        bd_input.append(torch.cat([inputs_bd, inputs_cross], dim=0))
        bd_targets.append(torch.cat([targets_bd, targets[num_bd: (num_bd + num_cross)]], dim=0))

        # total_inputs = torch.cat([inputs_bd, inputs_cross, inputs[(num_bd + num_cross) :]], dim=0)
        # total_inputs = transforms(total_inputs)
        # total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)

    # logging.warning('Here we drop the cross samples, since this part should never given to defender, in any sense')
    bd_train_x = torch.cat(bd_input, dim=0).float().cpu()
    bd_train_y = torch.cat(bd_targets, dim=0).long().cpu()
    train_poison_indicator = np.concatenate(one_hot_original_index)

    test_dl = torch.utils.data.DataLoader(
        test_dl.dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=False)

    test_bd_input = []
    test_bd_targets = []

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

            inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets, opt.num_classes)

            # no transform !
            test_bd_input.append((inputs_bd))
            test_bd_targets.append(targets_bd)

    bd_test_x = torch.cat(test_bd_input, dim=0).float().cpu()
    bd_test_y = torch.cat(test_bd_targets, dim=0).long().cpu()

    torch.save(
        {
            'model_name': opt.model_name,
            'num_classes': opt.num_classes,
            'model': netC.cpu().state_dict(),

            'data_path': opt.data_root,
            'img_size': (opt.input_height, opt.input_width, opt.input_channel),

            'clean_data': opt.dataset,

            'bd_train': ({
                'x': bd_train_x,
                'y': bd_train_y,
                'original_index': np.where(train_poison_indicator == 1)[
                    0] if train_poison_indicator is not None else None,
            }),

            'bd_test': {
                'x': bd_test_x,
                'y': bd_test_y,
            },
        },

        f'{save_path}/attack_result.pt',
    )

    torch.save(opt.__dict__, save_path + '/info.pickle')

if __name__ == "__main__":
    main()
