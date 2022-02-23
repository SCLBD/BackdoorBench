import sys
sys.path.append('.')
import copy
import os

import torch
import torch.nn as nn
import torchvision
from models import PreActResNet18
from config.inputAwareAttack.config import get_arguments
from utils.input_aware_dataloader import get_dataloader
from utils.networks.models import Generator, NetC_MNIST
from utils.input_aware_utils import progress_bar
from utils.aggregate_block.fix_random import fix_random


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


def create_cross(inputs1, inputs2, netG, netM, opt):
    patterns2 = netG(inputs2)
    patterns2 = netG.normalize_pattern(patterns2)
    masks_output = netM.threshold(netM(inputs2))
    inputs_cross = inputs1 + (patterns2 - inputs1) * masks_output
    return inputs_cross, patterns2, masks_output


def eval(netC, netG, netM, test_dl1, test_dl2, opt):
    print(" Eval:")

    n_output_batches = 3
    n_output_images = 3
    acc_clean = 0.0
    acc_bd = 0.0
    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0
    total_correct_cross = 0

    for batch_idx, (inputs, targets), (inputs2, targets2) in zip(range(len(test_dl1)), test_dl1, test_dl2):
        inputs1, targets1 = inputs.to(opt.device), targets.to(opt.device)
        inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)
        bs = inputs1.shape[0]
        total_sample += bs

        # Evaluating clean
        preds_clean = netC(inputs1)
        correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets1)
        total_correct_clean += correct_clean
        acc_clean = total_correct_clean * 100.0 / total_sample

        # Evaluating backdoor
        inputs_bd, targets_bd = create_bd(netG, netM, inputs1, targets1, opt)
        preds_bd = netC(inputs_bd)
        correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
        total_correct_bd += correct_bd
        acc_bd = total_correct_bd * 100.0 / total_sample

        inputs_cross, _, _ = create_cross(inputs1, inputs2, netG, netM, opt)
        preds_cross = netC(inputs_cross)
        correct_cross = torch.sum(torch.argmax(preds_cross, 1) == targets1)
        total_correct_cross += correct_cross
        acc_cross = total_correct_cross * 100.0 / total_sample

        progress_bar(
            batch_idx,
            len(test_dl1),
            "Acc Clean: {:.3f} | Acc Bd: {:.3f} | Acc Cross: {:.3f}".format(acc_clean, acc_bd, acc_cross),
        )

        if batch_idx < n_output_batches:
            subs = []
            for i in range(n_output_images):
                subs.append(inputs_bd[i : (i + 1), :, :, :])
            images = netG.denormalize_pattern(torch.cat(subs, dim=3))
            outpath = "%s_%s_sample_%d.png" % (opt.dataset, opt.attack_mode, batch_idx)
            torchvision.utils.save_image(images, outpath, normalize=True, pad_value=1)


def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    fix_random(int(opt.random_seed))
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
    print("load M")
    netM = Generator(opt, out_channels=1)
    netM.load_state_dict(state_dict["netM"])
    netM.to(opt.device)
    netM.eval()
    netM.requires_grad_(False)

    # Prepare dataloader
    test_dl = get_dataloader(opt, train=False)
    test_dl2 = get_dataloader(opt, train=False)
    eval(netC, netG, netM, test_dl, test_dl2, opt)


if __name__ == "__main__":
    main()
