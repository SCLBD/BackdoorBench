import logging
import sys, yaml, os, time

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

from pprint import pformat
import shutil
import argparse
import torch.nn as nn
import torch.nn.functional as F
from utils.networks.models import Generator, NetC_MNIST
from models import PreActResNet18
from torch.utils.tensorboard import SummaryWriter
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.save_path_generate import generate_save_folder
from utils.save_load_attack import summary_dict

import csv
import logging
import os

import config
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
# from torch.utils.tensorboard import SummaryWriter
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate

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
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "gtsrb":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label

class Args:
    pass

def get_dataloader(opt, train=True, c=0, k=0):

    args = Args()
    args.dataset = opt.dataset
    args.dataset_path = opt.data_root
    args.img_size = (opt.input_height, opt.input_width, opt.input_channel)

    # transform = get_transform(opt, train, c=c, k=k)
    # if opt.dataset == "gtsrb":
    #     dataset = GTSRB(opt, train, transform)
    # elif opt.dataset == "mnist":
    #     dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    # elif opt.dataset == "cifar10":
    #     dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    # else:
    #     raise Exception("Invalid dataset")

    train_dataset_without_transform, \
    train_img_transform, \
    train_label_transfrom, \
    test_dataset_without_transform, \
    test_img_transform, \
    test_label_transform = dataset_and_transform_generate(args=opt)

    if train:
        dataset = train_dataset_without_transform
        try:
            train_transform = get_transform(opt, train, c=c, k=k)
            if train_transform is not None:
                logging.warning(' transform use original transform')
        except:
            logging.warning(' transform use NON-original transform')
            train_transform = train_img_transform
        dataset.transform = train_transform
    else:
        dataset = test_dataset_without_transform
        try:
            test_transform = get_transform(opt, train, c=c, k=k)
            if test_transform is not None:
                logging.info('WARNING : transform use original transform')
        except:
            logging.warning(' transform use NON-original transform')
            test_transform = test_img_transform
        dataset.transform = test_transform

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=True
    )
    return dataloader


def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all_mask":
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
    netC, netG, netM, optimizerC, optimizerG, schedulerC, schedulerG, train_dl1, train_dl2, epoch, opt, tf_writer
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
    save_bd = 1
    one_hot_original_index = []
    total_inputs_bd = []
    total_targets_bd = []
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):
        optimizerC.zero_grad()

        inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
        inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)

        bs = inputs1.shape[0]
        num_bd = int(generalize_to_lower_pratio(opt.p_attack, bs)) #int(opt.p_attack * bs)
        num_cross = num_bd

        inputs_bd, targets_bd, patterns1, masks1 = create_bd(inputs1[:num_bd], targets1[:num_bd], netG, netM, opt, 'train')
        inputs_cross, patterns2, masks2 = create_cross(
            inputs1[num_bd : num_bd + num_cross], inputs2[num_bd : num_bd + num_cross], netG, netM, opt
        )

        total_inputs = torch.cat((inputs_bd, inputs_cross, inputs1[num_bd + num_cross :]), 0)
        total_targets = torch.cat((targets_bd, targets1[num_bd:]), 0)
        if(epoch==26):

            one_hot = np.zeros(bs)
            one_hot[:(num_bd + num_cross)] = 1

            if(save_bd):
                total_inputs_bd = total_inputs
                total_targets_bd = total_targets
                one_hot_original_index = one_hot
                save_bd = 0
            else:
                total_inputs_bd = torch.cat((total_inputs_bd, total_inputs), 0)
                total_targets_bd = torch.cat((total_targets_bd, total_targets), 0)
                one_hot_original_index = np.concatenate((one_hot_original_index, one_hot), 0)

            logging.info(total_inputs_bd.shape)
            logging.info(total_targets_bd.shape)
            logging.info(one_hot_original_index.shape)

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

        acc_clean = total_correct_clean * 100.0 / total_clean
        acc_bd = (total_bd_correct * 100.0 / total_bd) if total_bd > 0 else 0
        acc_cross = (total_cross_correct * 100.0 / total_cross) if total_cross > 0 else 0
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

    if not epoch % 10:
        # Save figures (tfboard)
        tf_writer.add_scalars(
            "Accuracy/lambda_div_{}/".format(opt.lambda_div),
            {"Clean": acc_clean, "BD": acc_bd, "Cross": acc_cross},
            epoch,
        )

        tf_writer.add_scalars("Loss/lambda_div_{}".format(opt.lambda_div), {"CE": loss_ce, "Div": loss_div}, epoch)

    schedulerC.step()
    schedulerG.step()

    logging.info(f'End train epoch {epoch} : acc_clean : {acc_clean}, acc_bd : {acc_bd}, acc_cross : {acc_cross} ')

    return total_inputs_bd, total_targets_bd, one_hot_original_index


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
    best_acc_clean,
    best_acc_bd,
    best_acc_cross,
    opt,
):
    netC.eval()
    netG.eval()
    logging.info(" Eval:")
    total = 0.0

    total_correct_clean = 0.0
    total_correct_bd = 0.0
    total_correct_cross = 0.0
    save_bd = 1
    total_inputs_bd = []
    total_targets_bd = []
    test_bd_poison_indicator = []
    test_bd_origianl_targets = []
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(test_dl1)), test_dl1, test_dl2):
        with torch.no_grad():
            inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
            inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)
            bs = inputs1.shape[0]

            preds_clean = netC(inputs1)
            correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets1)
            total_correct_clean += correct_clean

            inputs_bd, targets_bd, _, _,  position_changed, targets = create_bd(inputs1, targets1, netG, netM, opt, 'test')
            if(epoch==26):
                if(save_bd):
                    total_inputs_bd = inputs_bd
                    total_targets_bd = targets_bd
                    test_bd_poison_indicator = position_changed
                    test_bd_origianl_targets = targets
                    save_bd = 0
                else:
                    total_inputs_bd = torch.cat((total_inputs_bd, inputs_bd), 0)
                    total_targets_bd = torch.cat((total_targets_bd, targets_bd), 0)
                    test_bd_poison_indicator = torch.cat((test_bd_poison_indicator, position_changed), 0)
                    test_bd_origianl_targets = torch.cat((test_bd_origianl_targets, targets), 0)
                logging.info(total_inputs_bd.shape)
                logging.info(total_targets_bd.shape)
                logging.info(test_bd_poison_indicator.shape)
                logging.info(test_bd_origianl_targets.shape)
            preds_bd = netC(inputs_bd)
            correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            total_correct_bd += correct_bd

            inputs_cross, _, _ = create_cross(inputs1, inputs2, netG, netM, opt)
            preds_cross = netC(inputs_cross)
            correct_cross = torch.sum(torch.argmax(preds_cross, 1) == targets1)
            total_correct_cross += correct_cross

            total += bs
            avg_acc_clean = total_correct_clean * 100.0 / total
            avg_acc_cross = total_correct_cross * 100.0 / total
            avg_acc_bd = total_correct_bd * 100.0 / total

            infor_string = "Clean Accuracy: {:.3f} | Backdoor Accuracy: {:.3f} | Cross Accuracy: {:3f}".format(
                avg_acc_clean, avg_acc_bd, avg_acc_cross
            )
            progress_bar(batch_idx, len(test_dl1), infor_string)

    logging.info(
        " Result: Best Clean Accuracy: {:.3f} - Best Backdoor Accuracy: {:.3f} - Best Cross Accuracy: {:.3f}| Clean Accuracy: {:.3f}".format(
            best_acc_clean, best_acc_bd, best_acc_cross, avg_acc_clean
        )
    )
    logging.info(" Saving!!")
    best_acc_clean = avg_acc_clean
    best_acc_bd = avg_acc_bd
    best_acc_cross = avg_acc_cross
    state_dict = {
        "netC": netC.state_dict(),
        "netG": netG.state_dict(),
        "netM": netM.state_dict(),
        "optimizerC": optimizerC.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "schedulerC": schedulerC.state_dict(),
        "schedulerG": schedulerG.state_dict(),
        "best_acc_clean": best_acc_clean,
        "best_acc_bd": best_acc_bd,
        "best_acc_cross": best_acc_cross,
        "epoch": epoch,
        "opt": opt,
    }
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    torch.save(state_dict, ckpt_path)
    return best_acc_clean, best_acc_bd, best_acc_cross, epoch, total_inputs_bd, total_targets_bd, test_bd_poison_indicator, test_bd_origianl_targets


# -------------------------------------------------------------------------------------
def train_mask_step(netM, optimizerM, schedulerM, train_dl1, train_dl2, epoch, opt, tf_writer):
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

    if not epoch % 10:
        tf_writer.add_scalars(
            "Loss/lambda_norm_{}".format(opt.lambda_norm), {"MaskNorm": loss_norm, "MaskDiv": loss_div}, epoch
        )

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
    # Prepare model related things

    # if opt.dataset == "cifar10":
    #     # netC = PreActResNet18().to(opt.device)
    #     opt.model_name = 'preactresnet18'
    # elif opt.dataset == "gtsrb":
    #     # netC = PreActResNet18(num_classes=43).to(opt.device)
    #     opt.model_name = 'preactresnet18'
    # elif opt.dataset == "mnist":
    #     # netC = NetC_MNIST().to(opt.device)
    #     opt.model_name = 'netc_mnist'  # TODO add to framework
    # else:
    logging.info('use generate_cls_model() ')
    netC = generate_cls_model(opt.model_name, opt.num_classes)
    netC.to(opt.device)
    logging.warning(f'actually model use = {opt.model_name}')

    netG = Generator(opt).to(opt.device)
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)
    optimizerG = torch.optim.Adam(netG.parameters(), opt.lr_G, betas=(0.5, 0.9))
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, opt.schedulerG_milestones, opt.schedulerG_lambda)

    netM = Generator(opt, out_channels=1).to(opt.device)
    optimizerM = torch.optim.Adam(netM.parameters(), opt.lr_M, betas=(0.5, 0.9))
    schedulerM = torch.optim.lr_scheduler.MultiStepLR(optimizerM, opt.schedulerM_milestones, opt.schedulerM_lambda)

    # For tensorboard
    log_dir = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, "log_dir")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tf_writer = SummaryWriter(log_dir=log_dir)

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
        best_acc_clean = state_dict["best_acc_clean"]
        best_acc_bd = state_dict["best_acc_bd"]
        best_acc_cross = state_dict["best_acc_cross"]
        opt = state_dict["opt"]
        logging.info("Continue training")
    else:
        # Prepare mask
        best_acc_clean = 0.0
        best_acc_bd = 0.0
        best_acc_cross = 0.0
        epoch = 1

        # Reset tensorboard
        shutil.rmtree(log_dir)
        os.makedirs(log_dir)
        logging.info("Training from scratch")

    # Prepare dataset
    train_dl1 = get_dataloader(opt, train=True)
    train_dl2 = get_dataloader(opt, train=True)
    test_dl1 = get_dataloader(opt, train=False)
    test_dl2 = get_dataloader(opt, train=False)

    logging.info(pformat(opt.__dict__)) #set here since the opt change at beginning of this function

    if epoch == 1:
        netM.train()
        for i in range(25):
            logging.info(
                "Epoch {} - {} - {} | mask_density: {} - lambda_div: {}  - lambda_norm: {}:".format(
                    epoch, opt.dataset, opt.attack_mode, opt.mask_density, opt.lambda_div, opt.lambda_norm
                )
            )
            train_mask_step(netM, optimizerM, schedulerM, train_dl1, train_dl2, epoch, opt, tf_writer)
            epoch = eval_mask(netM, optimizerM, schedulerM, test_dl1, test_dl2, epoch, opt)
            epoch += 1
    netM.eval()
    netM.requires_grad_(False)

    for i in range(opt.n_iters):
        logging.info(
            "Epoch {} - {} - {} | mask_density: {} - lambda_div: {}:".format(
                epoch, opt.dataset, opt.attack_mode, opt.mask_density, opt.lambda_div
            )
        )
        total_inputs_bd, total_targets_bd, train_poison_indicator = train_step(
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
            tf_writer,
        )
        best_acc_clean, best_acc_bd, best_acc_cross, epoch, test_inputs_bd, test_targets_bd, test_bd_poison_indicator, test_bd_origianl_targets = eval(
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
            best_acc_clean,
            best_acc_bd,
            best_acc_cross,
            opt,
        )
        logging.info(
            f'epoch : {epoch} best_clean_acc : {best_acc_clean}, best_bd_acc : {best_acc_bd}, best_cross_acc : {best_acc_cross}')
        if(epoch == 26): # here > 25 epoch all fine. Since epoch < 25 still have no poison samples
            bd_train_x = total_inputs_bd.float().cpu()
            bd_train_y = total_targets_bd.long().cpu()
            bd_train_poison_indicator = train_poison_indicator
            bd_train_original_index = np.where(bd_train_poison_indicator == 1)[
                    0] if bd_train_poison_indicator is not None else None
            bd_train_x = bd_train_x[bd_train_original_index]
            bd_train_y = bd_train_y[bd_train_original_index]
            bd_test_x = test_inputs_bd.float().cpu()
            bd_test_y = test_targets_bd.long().cpu()
            bd_test_original_index = np.where(test_bd_poison_indicator.long().cpu().numpy())[0]
            bd_test_original_target = test_bd_origianl_targets.long().cpu()
        epoch += 1
        if epoch > opt.n_iters:
            break

    # torch.save(
    #     {
    #         'model_name': opt.model_name,
    #         'model': netC.cpu().state_dict(),
    #
    #         # 'clean_train': {
    #         #     'x' : torch.tensor(train_dl1.dataset.data).float().cpu(),
    #         #     'y' : torch.tensor(train_dl1.dataset.targets).float().cpu(),
    #         # },
    #         #
    #         # 'clean_test' : {
    #         #     'x' : torch.tensor(test_dl1.dataset.data).float().cpu(),
    #         #     'y' : torch.tensor(test_dl1.dataset.targets).float().cpu(),
    #         # },
    #
    #         'bd_train': {
    #             'x' : torch.tensor(bd_train_x).float().cpu(),
    #             'y' : torch.tensor(bd_train_y).float().cpu(),
    #         },
    #
    #         'bd_test': {
    #             'x': torch.tensor(test_inputs_bd).float().cpu(),
    #             'y' : torch.tensor(test_targets_bd).float().cpu(),
    #         },
    #     },

    final_save_dict = {
            'model_name': opt.model_name,
            'num_classes': opt.num_classes,
            'model': netC.cpu().state_dict(),

            'data_path': opt.data_root,
            'img_size': (opt.input_height, opt.input_width, opt.input_channel),

            'clean_data': opt.dataset,

            'bd_train': ({
                'x': bd_train_x,
                'y': bd_train_y,
                'original_index': bd_train_original_index ,
            }),

            'bd_test': {
                'x': bd_test_x,
                'y': bd_test_y,
                'original_index': bd_test_original_index,
                'original_targets': bd_test_original_target,
            },
        }
    logging.info(f"save dict summary : {summary_dict(final_save_dict)}")
    torch.save(
        final_save_dict,

        f'{opt.save_path}/attack_result.pt',
    )

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--yaml_path', type=str, default='../config/inputAwareAttack/default.yaml',
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--model_name', type=str, help='Only use when model is not given in original code !!!')
    parser.add_argument('--save_folder_name', type=str,
                        help='(Optional) should be time str + given unique identification str')
    parser.add_argument("--continue_training", action="store_true")

    parser.add_argument("--data_root", type=str, )#default="data/")
    parser.add_argument("--checkpoints", type=str, )#default="./record/inputAwareAttack/checkpoints/")
    parser.add_argument("--temps", type=str, )#default="./record/inputAwareAttack/temps")
    parser.add_argument("--save_path", type=str, )#default="./record/inputAwareAttack/")
    parser.add_argument("--device", type=str, )#default="cuda")

    parser.add_argument("--dataset", type=str, )#default="cifar10")
    parser.add_argument("--input_height", type=int, )#default=None)
    parser.add_argument("--input_width", type=int, )#default=None)
    parser.add_argument("--input_channel", type=int, )#default=None)
    parser.add_argument("--num_classes", type=int, )#default=10)

    parser.add_argument("--batchsize", type=int, )#default=128)
    parser.add_argument("--lr_G", type=float, )#default=1e-2)
    parser.add_argument("--lr_C", type=float, )#default=1e-2)
    parser.add_argument("--lr_M", type=float, )#default=1e-2)
    parser.add_argument("--schedulerG_milestones", type=list, )#default=[200, 300, 400, 500])
    parser.add_argument("--schedulerC_milestones", type=list, )#default=[100, 200, 300, 400])
    parser.add_argument("--schedulerM_milestones", type=list, )#default=[10, 20])
    parser.add_argument("--schedulerG_lambda", type=float, )#default=0.1)
    parser.add_argument("--schedulerC_lambda", type=float, )#default=0.1)
    parser.add_argument("--schedulerM_lambda", type=float, )#default=0.1)
    parser.add_argument("--n_iters", type=int, )#default=100)
    parser.add_argument("--lambda_div", type=float, )#default=1)
    parser.add_argument("--lambda_norm", type=float, )#default=100)
    parser.add_argument("--num_workers", type=float, )#default=4)

    parser.add_argument("--target_label", type=int, )#default=0)
    parser.add_argument("--attack_mode", type=str, )#default="all2one", help="all2one or all2all")
    parser.add_argument("--p_attack", type=float, )#default=0.1)
    # parser.add_argument("--p_cross", type=float, )#default=0.1)
    parser.add_argument("--mask_density", type=float, )#default=0.032)
    parser.add_argument("--EPSILON", type=float, )#default=1e-7)

    parser.add_argument("--random_rotation", type=int, )#default=10)
    parser.add_argument("--random_crop", type=int, )#default=5)
    parser.add_argument("--random_seed", type=int, )#default=0)

    return parser



def main():
    opt = get_arguments().parse_args()

    with open(opt.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)
    defaults.update({k: v for k, v in opt.__dict__.items() if v is not None})
    opt.__dict__ = defaults

    opt.dataset_path = opt.data_root

    opt.terminal_info = sys.argv



    # if opt.dataset == "mnist" or opt.dataset == "cifar10":
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
    # else:
    #     raise Exception("Invalid Dataset")

    opt.input_height, opt.input_width, opt.input_channel = get_input_shape(opt.dataset)
    opt.img_size = (opt.input_height, opt.input_width, opt.input_channel)

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


if __name__ == "__main__":
    main()
