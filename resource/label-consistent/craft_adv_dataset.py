'''
This is for crafted the
'''

import sys, yaml, os

os.chdir(sys.path[0])
sys.path.append('../../')
os.getcwd()

import argparse
from pprint import  pformat
import numpy as np
import torch
import time
import logging
from tqdm import tqdm

from torchvision.transforms import *
from utils.aggregate_block.save_path_generate import generate_save_folder
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
from utils.bd_dataset_v2 import dataset_wrapper_with_transform
from torch.utils.data import DataLoader
from utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion*4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(num_classes):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes)

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


import torch


class Attack(object):
    r"""
    Base class for all attacks.
    .. note::
        It automatically set device to the device where given model is.
        It temporarily changes the original model's training mode to `test`
        by `.eval()` only during an attack process.
    """

    def __init__(self, name, model, device):
        r"""
        Initializes internal attack state.
        Arguments:
            name (str) : name of an attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        self.training = model.training
        self.device = device

        self._targeted = 1
        self._attack_mode = "original"
        self._return_type = "float"

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def set_attack_mode(self, mode):
        r"""
        Set the attack mode.

        Arguments:
            mode (str) : 'original' (DEFAULT)
                         'targeted' - Use input labels as targeted labels.
                         'least_likely' - Use least likely labels as targeted labels.
        """
        if self._attack_mode is "only_original":
            raise ValueError(
                "Changing attack mode is not supported in this attack method."
            )

        if mode == "original":
            self._attack_mode = "original"
            self._targeted = 1
            self._transform_label = self._get_label
        elif mode == "targeted":
            self._attack_mode = "targeted"
            self._targeted = -1
            self._transform_label = self._get_label
        elif mode == "least_likely":
            self._attack_mode = "least_likely"
            self._targeted = -1
            self._transform_label = self._get_least_likely_label
        else:
            raise ValueError(
                mode
                + " is not a valid mode. [Options : original, targeted, least_likely]"
            )

    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.
        Arguments:
            type (str) : 'float' or 'int'. (DEFAULT : 'float')
        """
        if type == "float":
            self._return_type = "float"
        elif type == "int":
            self._return_type = "int"
        else:
            raise ValueError(type + " is not a valid type. [Options : float, int]")

    def save(self, save_path, data_loader, verbose=True):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.
        Arguments:
            save_path (str) : save_path.
            data_loader (torch.utils.data.DataLoader) : data loader.
            verbose (bool) : True for displaying detailed information. (DEFAULT : True)
        """
        self.model.eval()

        image_list = []
        label_list = []

        correct = 0
        total = 0

        total_batch = len(data_loader)

        for step, (images, labels) in enumerate(data_loader):
            adv_images = self.__call__(images, labels)

            image_list.append(adv_images.cpu())
            label_list.append(labels.cpu())

            if self._return_type == "int":
                adv_images = adv_images.float() / 255

            if verbose:
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

                acc = 100 * float(correct) / total
                print(
                    "- Save Progress : %2.2f %% / Accuracy : %2.2f %%"
                    % ((step + 1) / total_batch * 100, acc),
                    end="\r",
                )

        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)
        torch.save((x, y), save_path)
        print("\n- Save Complete!")

        self._switch_model()

    def _transform_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        """
        return labels

    def _get_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return labels

    def _get_least_likely_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return least likely labels.
        """
        outputs = self.model(images)
        _, labels = torch.min(outputs.data, 1)
        labels = labels.detach_()
        return labels

    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images * 255).type(torch.uint8)

    def _switch_model(self):
        r"""
        Function for changing the training mode of the model.
        """
        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def __str__(self):
        info = self.__dict__.copy()

        del_keys = ["model", "attack"]

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info["attack_mode"] = self._attack_mode
        if info["attack_mode"] == "only_original":
            info["attack_mode"] = "original"

        info["return_type"] = self._return_type

        return (
                self.attack
                + "("
                + ", ".join("{}={}".format(key, val) for key, val in info.items())
                + ")"
        )

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        self._switch_model()

        if self._return_type == "int":
            images = self._to_uint(images)

        return images

class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT : 0.3)
        alpha (float): step size. (DEFALUT : 2/255)
        steps (int): number of steps. (DEFALUT : 40)
        random_start (bool): using random initialization of delta. (DEFAULT : False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps = 8/255, alpha = 1/255, steps=40, random_start=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=0.3, alpha=2 / 255, steps=40, random_start=False, device=None):
        super(PGD, self).__init__("PGD", model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = self._transform_label(images, labels)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1)

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = self._targeted * loss(outputs, labels).to(self.device)

            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])
    parser.add_argument('--device', type = str)
    parser.add_argument('--yaml_path', type=str, default='./default.yaml',
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--lr_scheduler', type=str,
                        help='which lr_scheduler use for optimizer')
    # only all2one can be use for clean-label
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dataset', type=str,
                        help='which dataset to use'
                        )
    parser.add_argument('--dataset_path', type=str)

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--steplr_stepsize', type=int)
    parser.add_argument('--steplr_gamma', type=float)
    parser.add_argument('--sgd_momentum', type=float)
    parser.add_argument('--wd', type=float, help='weight decay of sgd')
    parser.add_argument('--steplr_milestones', type=list)
    parser.add_argument('--client_optimizer', type=int)
    parser.add_argument('--random_seed', type=int,
                        help='random_seed')
    parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')
    parser.add_argument('--model', type=str,
                        help='choose which kind of model')
    parser.add_argument('--save_folder_name', type=str,
                        help='(Optional) should be time str + given unique identification str')
    parser.add_argument('--git_hash', type=str,
                        help='git hash number, in order to find which version of code is used')
    return parser

def main():

    ### 1. config args, save_path, fix random seed
    parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)

    defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

    args.__dict__ = defaults

    args.terminal_info = sys.argv

    args.attack = 'None'

    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"

    logging.info(f"get the training setting for specific dataset")

    ### save path
    if 'save_folder_name' not in args:
        save_path = generate_save_folder(
            run_info=('afterwards' if 'load_path' in args.__dict__ else 'attack') + '_' + args.attack,
            given_load_file_path=args.load_path if 'load_path' in args else None,
            all_record_folder_path='../../record',
        )
    else:
        save_path = '../../record/' + args.save_folder_name
        os.mkdir(save_path)

    args.save_path = save_path

    torch.save(args.__dict__, save_path + '/info.pickle')

    ### set the logger
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
    logging.info(pformat(args.__dict__))

    ### set the random seed
    fix_random(int(args.random_seed))

    ### 2. set the clean train data and clean test data
    train_dataset_without_transform, \
                train_img_transform, \
                train_label_transform, \
    test_dataset_without_transform, \
                test_img_transform, \
                test_label_transform = dataset_and_transform_generate(args)

    train_img_transform = Compose(
        [Resize((args.input_height,args.input_width)), ToTensor()]
    )

    test_img_transform = Compose(
        [Resize((args.input_height,args.input_width)), ToTensor()]
    )

    benign_train_dl = DataLoader(
        dataset_wrapper_with_transform(
            train_dataset_without_transform,
            train_img_transform,
            train_label_transform,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    benign_test_dl = DataLoader(
        dataset_wrapper_with_transform(
            test_dataset_without_transform,
            test_img_transform,
            test_label_transform,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.dataset == 'cifar10':
        net  = ResNet18(num_classes = args.num_classes)
        net.load_state_dict(torch.load('./resnet18_PGD_best_model.pth'))
    elif args.dataset == 'tiny':
        net = PreActResNet18(num_classes=args.num_classes)
        net.load_state_dict(torch.load('./preact_tiny_model.pth'))
    elif args.dataset == 'cifar100':
        net = ResNet18(num_classes = args.num_classes)
        net.load_state_dict(torch.load('./cifar100_resnet18_AT_best_model.pth'))
    elif args.dataset == 'gtsrb':
        net = ResNet18(num_classes = args.num_classes)
        checkpoint = torch.load('./gtsrb_resnet18_AT_best_model.pth')
        from collections import OrderedDict
        try:
            net.load_state_dict(checkpoint)
        except:
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict, False)

    trainer = generate_cls_trainer(
        net,
        args.attack,
        args.amp,
    )

    criterion = argparser_criterion(args)

    optimizer, scheduler = argparser_opt_scheduler(net, args)

    trainer.criterion = criterion

    train_m = trainer.test(
        benign_train_dl, device
    )
    logging.info(train_m)
    logging.info(train_m['test_correct']/train_m['test_total'])

    test_m = trainer.test(
        benign_test_dl, device
    )
    logging.info(test_m)
    logging.info(test_m['test_correct']/test_m['test_total'])

    logging.info('start generate adv dataset for train and test')

    config = {"adv_dataset_dir": "./test/adv_dataset",
        # "adv_model_path": "./model/adv_models/cifar_resnet_e8_a2_s10.pth",
              }

    pgd_config = {
        'eps': 8,
        'alpha': 1.5,
        'steps': 100,
        'max_pixel': 255,
    }
    print("Set PGD attacker: {}.".format(pgd_config))
    max_pixel = pgd_config.pop("max_pixel")
    for k, v in pgd_config.items():
        if k == "eps" or k == "alpha":
            pgd_config[k] = v / max_pixel
    attacker = PGD(net, **pgd_config, device = device)
    attacker.set_return_type("int")

    train_data = benign_train_dl.dataset
    train_loader = benign_train_dl

    perturbed_img = torch.zeros((len(train_data), args.input_height, args.input_width, 3), dtype=torch.uint8)
    target = torch.zeros(len(train_data))
    i = 0
    net.to(device)
    net.eval()
    for (x,y,*other) in tqdm(train_loader):
        # Adversarially perturb image. Note that torchattacks will automatically
        # move `img` and `target` to the gpu where the attacker.model is located.

        img = attacker(x,y)
        perturbed_img[i: i + len(img), :, :, :] = img.permute(0, 2, 3, 1).detach()
        target[i: i + len(y)] = y
        i += img.shape[0]

    if not os.path.exists(config["adv_dataset_dir"]):
        os.makedirs(config["adv_dataset_dir"])
    adv_data_path = os.path.join(
        config["adv_dataset_dir"],f"{args.dataset}_train.npy"
    )
    # np.savez(adv_data_path, data=perturbed_img.numpy(), targets=target.numpy())
    np.save(adv_data_path, perturbed_img.numpy())
    print("Save the train adversarially perturbed dataset to {}".format(adv_data_path))


    ####
    test_data = benign_test_dl.dataset
    test_loader = benign_test_dl

    perturbed_img = torch.zeros((len(test_data), args.input_height, args.input_width, 3), dtype=torch.uint8)
    target = torch.zeros(len(test_data))
    i = 0
    net.to(device)
    net.eval()
    for (x, y, *other) in tqdm(test_loader):
        # Adversarially perturb image. Note that torchattacks will automatically
        # move `img` and `target` to the gpu where the attacker.model is located.

        img = attacker(x, y)
        perturbed_img[i: i + len(img), :, :, :] = img.permute(0, 2, 3, 1).detach()
        target[i: i + len(y)] = y
        i += img.shape[0]
    if not os.path.exists(config["adv_dataset_dir"]):
        os.makedirs(config["adv_dataset_dir"])
    adv_data_path = os.path.join(
        config["adv_dataset_dir"], f"{args.dataset}_test.npy"
    )
    np.save(adv_data_path, perturbed_img.numpy())
    print("Save the test adversarially perturbed dataset to {}".format(adv_data_path))


if __name__ == '__main__':
    main()