import copy
import os
import sys

import torch
import torch.nn as nn
from config import get_arguments


sys.path.insert(0, "../..")
from classifier_models import PreActResNet18
from dataloader import get_dataloader
from networks.models import Generator, NetC_MNIST
from utils import progress_bar


def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def create_bd(netG, netM, inputs, targets, opt):
    bd_targets = create_targets_bd(targets, opt)
    patterns = netG(inputs)
    patterns = netG.normalize_pattern(patterns)

    masks_output = netM.threshold(netM(inputs))
    bd_inputs = inputs + (patterns - inputs) * masks_output
    return bd_inputs, bd_targets


def eval(netC, netG, netM, test_dl, opt):
    print(" Eval:")
    acc_clean = 0.0
    acc_bd = 0.0
    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        total_sample += bs

        # Evaluating clean
        preds_clean = netC(inputs)
        correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets)
        total_correct_clean += correct_clean
        acc_clean = total_correct_clean * 100.0 / total_sample

        # Evaluating backdoor
        inputs_bd, targets_bd = create_bd(netG, netM, inputs, targets, opt)
        preds_bd = netC(inputs_bd)
        correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
        total_correct_bd += correct_bd
        acc_bd = total_correct_bd * 100.0 / total_sample

        progress_bar(batch_idx, len(test_dl), "Acc Clean: {:.3f} | Acc Bd: {:.3f}".format(acc_clean, acc_bd))


def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    if opt.dataset == "mnist" or opt.dataset == "cifar10":
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    else:
        raise Exception("Invalid Dataset")
    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    else:
        raise Exception("Invalid Dataset")

    # Load models and masks
    if opt.dataset == "cifar10":
        netC = PreActResNet18().to(opt.device)
    elif opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=43).to(opt.device)
    elif opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)
    else:
        raise Exception("Invalid dataset")

    path_model = os.path.join(
        opt.checkpoints, opt.dataset, opt.attack_mode, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset)
    )
    state_dict = torch.load(path_model)
    print("load C")
    netC.load_state_dict(state_dict["netC"])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)
    print("load G")
    netG = Generator(opt)
    netG.load_state_dict(state_dict["netG"])
    netG.to(opt.device)
    netG.eval()
    netG.requires_grad_(False)

    netM = Generator(opt, out_channels=1)
    netM.load_state_dict(state_dict["netM"])
    netM.to(opt.device)
    netM.eval()
    netM.requires_grad_(False)

    # Prepare dataloader
    test_dl = get_dataloader(opt, train=False)

    print("Original")
    eval(netC, netG, netM, test_dl, opt)
    print("Smoothing")
    for k in [3, 5]:
        print("k = ", k)
        test_dl2 = get_dataloader(opt, train=False, k=k)
        eval(netC, netG, netM, test_dl2, opt)

    print("Color-depth shrinking")
    for cc in range(3):
        c = cc + 1
        print("c = ", c)
        test_dl2 = get_dataloader(opt, train=False, c=c)
        eval(netC, netG, netM, test_dl2, opt)


if __name__ == "__main__":
    main()
