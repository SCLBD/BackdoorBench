import torch
from torch import Tensor, nn
import torchvision
import torchvision.transforms as transforms
import config
import sys

sys.path.insert(0, "../..")

from classifier_models import *
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import progress_bar
from networks.models import NetC_MNIST, Normalize, Denormalize
from utils.dataloader import get_dataloader

import pdb
sys.path.append('/home/ubuntu/workspaces/chenweixin/PyTorch_CIFAR10/')
from module import CIFAR10Module

class RegressionModel(nn.Module):
    def __init__(self, opt, init_mask, init_pattern):
        self._EPSILON = opt.EPSILON
        super(RegressionModel, self).__init__()
        self.mask_tanh = nn.Parameter(torch.tensor(init_mask))
        self.pattern_tanh = nn.Parameter(torch.tensor(init_pattern))

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
        # if opt.dataset == "mnist":
        #     classifier = NetC_MNIST()
        # elif opt.dataset == "cifar10":
        #     classifier = PreActResNet18()
        # elif opt.dataset == "gtsrb":
        #     classifier = PreActResNet18(num_classes=43)
        # elif opt.dataset == "celeba":
        #     classifier = ResNet18()
        # else:
        #     raise Exception("Invalid Dataset")
        if opt.classifier == 'preactresnet18':
            classifier = PreActResNet18()
            if opt.dataset == "gtsrb":
                classifier = PreActResNet18(num_classes=43)
            if opt.dataset == "cifar100":
                classifier = PreActResNet18(num_classes=100)
        else:
            classifier = CIFAR10Module(opt).model
        # Load pretrained classifie
        # ckpt_path = os.path.join(
        #     opt.checkpoints, opt.dataset, "{}_{}_morph.pth.tar".format(opt.dataset, opt.attack_mode)
        # )

        # state_dict = torch.load(ckpt_path)
        # classifier.load_state_dict(state_dict["netC"])
        checkpoint = torch.load(opt.checkpoint_load)
        classifier.load_state_dict(checkpoint['model'])
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
        result_dir = os.path.join(opt.result, opt.dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # result_dir = os.path.join(result_dir, opt.attack_mode)
        result_dir = os.path.join(result_dir, opt.trigger_type)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
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


def train(opt, init_mask, init_pattern):

    # test_dataloader = get_dataloader(opt, train=False)
    trainloader = get_dataloader(opt, True)

    # Build regression model
    regression_model = RegressionModel(opt, init_mask, init_pattern).to(opt.device)

    # Set optimizer
    optimizerR = torch.optim.Adam(regression_model.parameters(), lr=opt.lr, betas=(0.5, 0.9))

    # Set recorder (for recording best result)
    recorder = Recorder(opt)

    for epoch in range(opt.epoch):
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

    return inner_early_stop_flag


if __name__ == "__main__":
    pass
