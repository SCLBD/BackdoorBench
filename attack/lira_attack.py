'''
This is the code for LIRA attack

github link : https://github.com/sunbelbd/invisible_backdoor_attacks

citation:
@inproceedings{Doan2021lira,
  title     = {LIRA: Learnable, Imperceptible and Robust Backdoor Attacks},
  author    = {Khoa D. Doan and Yingjie Lao and Weijie Zhao and Ping Li},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
  year      = {2021}
}

Please note that
1. The original code was implemented in Paddlepaddle,
    and we replaced all the functionality of Paddlepaddle with the equivalent API of Pytorch.
2. Since this is a training-controllable attack,
    the concepts of poisoning rate and poisoned data may not apply.
    So, this attack remains incompatible with the whole framework for the time being
    (because we require data to be saved during the saving process).
    In the future, we will update this version to make it fully integrated into the framework.

The original LICENSE of the script is put at the bottom of this file.
'''

import sys, os, random, argparse, time, logging

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import yaml
from pprint import pformat

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from torchvision import transforms
from functools import partial

from utils.aggregate_block.save_path_generate import generate_save_folder
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info

class warp_with_transform(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        x,y = self.dataset[index]
        if transforms is not None:
            x = self.transforms(x)
        return x,y

loss_fn = nn.CrossEntropyLoss()

class Autoencoder(nn.Module):
    def __init__(self, channels=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, (4,4), stride=(2,2), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4,4), stride=(2,2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4,4), stride=(2,2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (4,4), stride=(2,2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4,4), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (4,4), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (4,4), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, (4,4), stride=(2,2), padding=(1,1)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, (3, 3), padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class UNet(nn.Module):

    def __init__(self, out_channel):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channel, (1, 1)),
            nn.BatchNorm2d(out_channel),
        )

        self.out_layer = nn.Tanh()

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        x = self.upsample(x)
        x = torch.concat([x, conv3], 1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.concat([x, conv2], 1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.concat([x, conv1], 1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = self.out_layer(out)

        return out




class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x

class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super().__init__()
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
        self.random_crop = transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop)
        self.random_rotation = transforms.RandomRotation(opt.random_rotation)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

def get_dataloader(opt, train=True):

    train_dataset_without_transform, \
    train_img_transform, \
    train_label_transfrom, \
    test_dataset_without_transform, \
    test_img_transform, \
    test_label_transform = dataset_and_transform_generate(
        opt
    )
    if train:
        dataset = train_dataset_without_transform
        transforms_for_data = train_img_transform
    else:
        dataset = test_dataset_without_transform
        transforms_for_data = test_img_transform
    dataset = warp_with_transform(dataset, transforms_for_data)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True
    )
    return dataloader

def all2one_target_transform(x, attack_target=1):
    return torch.ones_like(x) * attack_target

def all2all_target_transform(x, num_classes):
    return (x + 1) % num_classes

def create_models(args):
    """DONE
    """
    if args.attack_model is None or args.attack_model == 'autoencoder':
        atkmodel = Autoencoder(args.input_channel)
        # Copy of attack model
        tgtmodel = Autoencoder(args.input_channel)
    elif args.attack_model == 'unet':
        atkmodel = UNet(args.input_channel)
        # Copy of attack model
        tgtmodel = UNet(args.input_channel)
    else:
        raise Exception(f'Invalid attack_model {args.attack_model}')

    create_net = partial(generate_cls_model,
        model_name = args.clsmodel,
        num_classes = args.num_classes,
        image_size = args.img_size,
    )

    clsmodel = create_net()

    tgtmodel.to(args.device)
    atkmodel.to(args.device)
    clsmodel.to(args.device)

    # Optimizer
    tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=args.lr_atk)

    return atkmodel, tgtmodel, tgtoptimizer, clsmodel, create_net


def stage1_eval(args, atkmodel, scratchmodel, target_transform,
         train_loader, test_loader, epoch, trainepoch, clip_image,
         testoptimizer=None, log_prefix='Internal', epochs_per_test=5):
    # default phase 2 parameters to phase 1
    if args.test_alpha is None:
        args.test_alpha = args.alpha
    if args.test_eps is None:
        args.test_eps = args.eps

    test_loss = 0
    correct = 0

    correct_transform = 0
    test_transform_loss = 0

    atkmodel.eval()
    atkmodel.to(args.device)
    if testoptimizer is None:
        testoptimizer = torch.optim.SGD(scratchmodel.parameters(), lr=args.lr)

    for cepoch in range(trainepoch):
        scratchmodel.train()
        scratchmodel.to(args.device)
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
        for batch_idx, (data, target) in pbar:
            data, target = data.to(args.device), target.to(args.device)
            testoptimizer.zero_grad()
            with torch.no_grad():
                noise = atkmodel(data) * args.test_eps
                atkdata = clip_image(data + noise)

            atkoutput = scratchmodel(atkdata)
            output = scratchmodel(data)

            loss_clean = loss_fn(output, target)
            loss_poison = loss_fn(atkoutput, target_transform(target))

            loss = args.alpha * loss_clean + (1 - args.test_alpha) * loss_poison

            loss.backward()
            testoptimizer.step()

            if batch_idx % 10 == 0 or batch_idx == (len(train_loader) - 1):
                pbar.set_description(
                    'Test [{}-{}] Loss: Clean {:.4f} Poison {:.4f} Total {:.5f}'.format(
                        epoch, cepoch,
                        loss_clean.item(),
                        loss_poison.item(),
                        loss.item()
                    ))

        if cepoch % epochs_per_test == 0 or cepoch == trainepoch - 1:
            scratchmodel.eval()
            scratchmodel.to(args.device)
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(args.device), target.to(args.device)
                    # if len(target.shape) == 1:
                    #     target = target.reshape([data.shape[0], 1])
                    output = scratchmodel(data)
                    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
                    # print(output.shape, target.shape)
                    test_loss += criterion(output, target).item()  # sum up batch loss

                    _, predicted = torch.max(output, -1)
                    correct += predicted.eq(target).sum()

                    noise = atkmodel(data) * args.test_eps
                    atkdata = clip_image(data + noise)
                    atkoutput = scratchmodel(atkdata)
                    # print(atkoutput.shape, target_transform(target).shape)
                    test_transform_loss += criterion(atkoutput, target_transform(target)).item()

                    _, predicted = torch.max(atkoutput, -1)
                    correct_transform += predicted.eq(target_transform(target)).sum()

            test_loss /= len(test_loader.dataset)
            test_transform_loss /= len(test_loader.dataset)

            correct = float(correct) / len(test_loader.dataset)
            correct_transform = float(correct_transform) / len(test_loader.dataset)

            print(
                '\n{}-Test set [{}]: Loss: clean {:.4f} poison {:.4f}, Accuracy: clean {:.2f} poison {:.2f}'.format(
                    log_prefix, cepoch,
                    test_loss, test_transform_loss,
                    correct, correct_transform
                ))

    return correct, correct_transform

def train(args, atkmodel, tgtmodel, clsmodel, tgtoptimizer, clsoptimizer, target_transform,
          train_loader, epoch, clip_image, post_transforms=None):
    clsmodel.train()
    clsmodel.to(args.device)
    atkmodel.eval()
    atkmodel.to(args.device)
    tgtmodel.train()
    tgtmodel.to(args.device)
    losslist = []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
    for batch_idx, (data, target) in pbar:
        data, target = data.to(args.device), target.to(args.device)
        if post_transforms is not None:
            data = post_transforms(data)

        ########################################
        #### Update Transformation Function ####
        ########################################
        noise = tgtmodel(data) * args.eps
        atkdata = clip_image(data + noise)

        # Calculate loss
        atkoutput = clsmodel(atkdata)
        loss_poison = loss_fn(atkoutput, target_transform(target))
        loss1 = loss_poison

        losslist.append(loss1.item())
        clsoptimizer.zero_grad()
        tgtoptimizer.zero_grad()
        loss1.backward()
        tgtoptimizer.step()  # this is the slowest step

        ###############################
        #### Update the classifier ####
        ###############################
        noise = atkmodel(data) * args.eps
        atkdata = clip_image(data + noise)
        output = clsmodel(data)
        atkoutput = clsmodel(atkdata)
        loss_clean = loss_fn(output, target)
        loss_poison = loss_fn(atkoutput, target_transform(target))
        loss2 = loss_clean * args.alpha + (1 - args.alpha) * loss_poison
        clsoptimizer.zero_grad()
        loss2.backward()
        clsoptimizer.step()

        if batch_idx % 10 == 0 or batch_idx == (len(train_loader) - 1):
            pbar.set_description('Train [{}] Loss: clean {:.4f} poison {:.4f} CLS {:.4f} ATK:{:.4f}'.format(
                epoch, loss_clean.item(), loss_poison.item(), loss1.item(), loss2.item()))
    pbar.close()
    atkloss = sum(losslist) / len(losslist)

    return atkloss


def create_paths(args):
    basepath = args.save_path
    checkpoint_path = os.path.join(basepath, 'checkpoint.ckpt')
    bestmodel_path = os.path.join(basepath, 'bestmodel.ckpt')
    return basepath, checkpoint_path, bestmodel_path


def get_target_transform(args):
    """DONE
    """
    if args.mode == 'all2one':
        target_transform = lambda x: all2one_target_transform(x, args.target_label)
    elif args.mode == 'all2all':
        target_transform = lambda x: all2all_target_transform(x, args.num_classes)
    else:
        raise Exception(f'Invalid mode {args.mode}')
    return target_transform


def get_train_test_loaders(args):

    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.num_classes = get_num_classes(args.dataset)

    train_loader = get_dataloader(args, True)
    test_loader = get_dataloader(args, False)

    if args.dataset in ['tiny', 'imagenet']:
        xmin, xmax = -2.1179039478302, 2.640000104904175
        def clip_image(x):
            return torch.clip(x, xmin, xmax)
    elif args.dataset in ['cifar10', 'cifar100', 'mnist']:
        def clip_image(x):
            return x  # no clipping
    elif args.dataset == 'gtsrb':
        def clip_image(x):
            return torch.clip(x, 0.0, 1.0)
    else:
        raise Exception(f'Invalid dataset: {args.dataset}')

    return train_loader, test_loader, clip_image

def stage2_finetune_and_test(args, test_model_path, atkmodel, netC, target_transform, train_loader, test_loader,
               trainepoch, alpha=0.5, optimizerC=None,
               schedulerC=None, log_prefix='Internal', epochs_per_test=1, data_transforms=None, start_epoch=1,
               clip_image=None):
    test_loss = 0
    correct = 0

    clean_accs, poison_accs = [], []

    correct_transform = 0
    test_transform_loss = 0

    best_clean_acc, best_poison_acc = 0, 0

    atkmodel.eval()
    atkmodel.to(args.device)

    if optimizerC is None:
        print('No optimizer, creating default SGD...')
        optimizerC = optim.SGD(netC.parameters(), lr=args.test_lr)
    if schedulerC is None:
        print('No scheduler, creating default 100,200,300,400...')
        schedulerC = optim.lr_scheduler.MultiStepLR(optimizerC, [100, 200, 300, 400], args.test_lr)

    for cepoch in range(start_epoch, trainepoch + 1):
        netC.train()
        netC.to(args.device)
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            data, target = data.to(args.device), target.to(args.device)
            if data_transforms is not None:
                data = data_transforms(data.to(args.device))
            optimizerC.zero_grad()

            output = netC(data)
            loss_clean = loss_fn(output, target)

            if alpha < 1:
                with torch.no_grad():
                    noise = atkmodel(data) * args.test_eps
                    if clip_image is None:
                        atkdata = torch.clip(data + noise, 0, 1)
                    else:
                        atkdata = clip_image(data + noise)
                atkoutput = netC(atkdata)
                loss_poison = loss_fn(atkoutput, target_transform(target))
            else:
                loss_poison = torch.tensor(0.0)

            loss = alpha * loss_clean + (1 - alpha) * loss_poison

            loss.backward()
            optimizerC.step()

            if batch_idx % 10 == 0 or batch_idx == (len(train_loader) - 1):
                pbar.set_description(
                    'Train-{} Loss: Clean {:.5f}  Poison {:.5f}  Total {:.5f} '.format(
                        cepoch,
                        loss_clean.item(),
                        loss_poison.item(),
                        loss.item(),
                    ))
        schedulerC.step()
        if cepoch % epochs_per_test == 0 or cepoch == trainepoch - 1:
            with torch.no_grad():
                for data, target in tqdm(test_loader, desc=f'Evaluation {cepoch}'):
                    data, target = data.to(args.device), target.to(args.device)
                    # if len(target.shape) == 1:
                    #     target = target.reshape([data.shape[0], 1])
                    output = netC(data)
                    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
                    # print(output.shape, target.shape)
                    test_loss += criterion(output, target, ).item()  # sum up batch loss

                    _, predicted = torch.max(output, -1)
                    correct += predicted.eq(target).sum()

                    # correct += torch.metric.accuracy(output, target).item() * len(target)

                    noise = atkmodel(data) * args.test_eps
                    if clip_image is None:
                        atkdata = torch.clip(data + noise, 0, 1)
                    else:
                        atkdata = clip_image(data + noise)
                    atkoutput = netC(atkdata)
                    # print(atkoutput.shape, target_transform(target).shape)
                    test_transform_loss += criterion(
                        atkoutput, target_transform(target)).item()  # sum up batch loss
                    _, predicted = torch.max(atkoutput, -1)
                    correct_transform += predicted.eq(target_transform(target)).sum()

            test_loss /= len(test_loader.dataset)
            test_transform_loss /= len(test_loader.dataset)

            correct = float(correct) / len(test_loader.dataset)
            correct_transform = float(correct_transform) / len(test_loader.dataset)

            clean_accs.append(correct)
            poison_accs.append(correct_transform)

            print('\n{}-Test [{}]: Loss: clean {:.4f} poison {:.4f}, '
                  'Accuracy: clean {:.4f} (best {:.4f}) poison {:.4f} (best {:.4f})'.format(
                log_prefix, cepoch,
                test_loss, test_transform_loss,
                correct, best_clean_acc, correct_transform, best_poison_acc
            ))
            if correct > best_clean_acc or (correct > best_clean_acc - 0.02 and correct_transform > best_poison_acc):
                best_clean_acc = correct
                best_poison_acc = correct_transform

                print(f'Saving current best model in {test_model_path}')
                torch.save({
                    'atkmodel': atkmodel.state_dict(),
                    'netC': netC.state_dict(),
                    'optimizerC': optimizerC.state_dict(),
                    'clean_schedulerC': schedulerC,
                    'best_clean_acc': best_clean_acc,
                    'best_poison_acc': best_poison_acc
                }, test_model_path)

    return clean_accs, poison_accs

def stage1addparse(parser):
    parser.add_argument('--dataset', type=str, )#default='cifar10')
    parser.add_argument('--dataset_path', type=str, )#default='../data')
    parser.add_argument("--random_rotation", type=int, )#default=10)
    parser.add_argument("--random_crop", type=int, )#default=5)
    parser.add_argument("--pretensor_transform",  )#default=False)

    parser.add_argument('--num-workers', type=int, )#default=2, help='dataloader workers')
    parser.add_argument('--batch-size', type=int, )#default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, )#default=1000, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, )#default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--lr-atk', type=float, )#default=0.0001, help='learning rate for attack model')
    parser.add_argument('--seed', type=int, )#default=999, help='random seed (default: 999)')
    parser.add_argument('--save-model',  )#default=False, help='For Saving the current Model')
    parser.add_argument('--train-epoch', type=int, )#default=1, help='training epochs for victim model')

    parser.add_argument('--target_label', type=int, )#default=1)  # only in effect if it's all2one
    parser.add_argument('--eps', type=float, )#default=0.3, help='epsilon for data poisoning')
    parser.add_argument('--alpha', type=float, )#default=0.5)
    parser.add_argument('--clsmodel', type=str, )#default='vgg11')
    parser.add_argument('--attack_model', type=str, )#default='autoencoder')
    parser.add_argument('--mode', type=str, )#default='all2one')
    parser.add_argument('--epochs_per_external_eval', type=int, )#default=50)
    parser.add_argument('--cls_test_epochs', type=int, )#default=20)
    parser.add_argument('--best_threshold', type=float, )#default=0.1)
    parser.add_argument('--verbose', type=int, )#default=1, help='verbosity')
    parser.add_argument('--avoid_clsmodel_reinitialization',
                        )#default=False, help='whether test the poisoned model from scratch')

    parser.add_argument('--test_eps', type=float)#default=None, 
    parser.add_argument('--test_alpha',  type=float)#default=None,
    parser.add_argument('--device', type=str)
    return parser

def stage2addparse(parser):
    parser.add_argument('--test_epochs',  type=int)#default=500,
    parser.add_argument('--test_lr',  type=float)#default=None,
    parser.add_argument('--schedulerC_lambda', type=float)#default=0.05, 
    parser.add_argument('--schedulerC_milestones', )#default='30,60,90,150')
    parser.add_argument('--test_n_size', )#default=10)
    parser.add_argument('--test_optimizer', )#default='sgd')
    parser.add_argument('--test_use_train_best', )#default=False,
    parser.add_argument('--test_use_train_last',  )#default=False,
    parser.add_argument('--use_data_parallel',  )#default=False,
    return parser

def main_stage1(args):


    if args.verbose >= 1:
        print('========== ARGS ==========')
        print(args)

    train_loader, test_loader, clip_image = get_train_test_loaders(args)
    post_transforms = PostTensorTransform(args)

    print('========== DATA ==========')
    print('Loaders: Train {} examples/{} iters, Test {} examples/{} iters'.format(
        len(train_loader.dataset), len(train_loader), len(test_loader.dataset), len(test_loader)))

    atkmodel, tgtmodel, tgtoptimizer, clsmodel, create_net = create_models(args)
    if args.verbose >= 2:
        print('========== MODELS ==========')
        print(atkmodel)
        print(clsmodel)

    target_transform = get_target_transform(args)
    basepath, checkpoint_path, bestmodel_path = create_paths(args)

    print('========== PATHS ==========')
    print(f'Basepath: {basepath}')
    print(f'Checkpoint Model: {checkpoint_path}')
    print(f'Best Model: {bestmodel_path}')

    if os.path.exists(checkpoint_path):
        # Load previously saved models
        checkpoint = torch.load(checkpoint_path)
        print(('Load existing attack model from path {}'.format(checkpoint_path)))
        atkmodel.load_state_dict(checkpoint['atkmodel'], )
        clsmodel.load_state_dict(checkpoint['clsmodel'], )
        trainlosses = checkpoint['trainlosses']
        best_acc_clean = checkpoint['best_acc_clean']
        best_acc_poison = checkpoint['best_acc_poison']
        start_epoch = checkpoint['epoch']
        tgtoptimizer.load_state_dict(checkpoint['tgtoptimizer'])
    else:
        # Create new model
        print(('Create new model from {}'.format(checkpoint_path)))
        best_acc_clean = 0
        best_acc_poison = 0
        trainlosses = []
        start_epoch = 1

    # Initialize the tgtmodel
    atkmodel.to(args.device)
    tgtmodel.load_state_dict(atkmodel.state_dict())

    print('============================')
    print('============================')

    print('BEGIN TRAINING >>>>>>')

    clsoptimizer = torch.optim.SGD(clsmodel.parameters(), lr=args.lr, momentum=0.9)
    for epoch in range(start_epoch, args.epochs + 1):
        for i in range(args.train_epoch):
            print(f'===== EPOCH: {epoch}/{args.epochs + 1} CLS {i + 1}/{args.train_epoch} =====')
            if not args.avoid_clsmodel_reinitialization:
                clsoptimizer = torch.optim.SGD(clsmodel.parameters(), lr=args.lr)
            trainloss = train(args, atkmodel, tgtmodel, clsmodel, tgtoptimizer, clsoptimizer, target_transform,
                              train_loader,
                              epoch, clip_image,
                              post_transforms=post_transforms)
            trainlosses.append(trainloss)
        tgtmodel.to(args.device)
        atkmodel.load_state_dict(tgtmodel.state_dict())
        if args.avoid_clsmodel_reinitialization:
            scratchmodel = create_net()
            clsmodel.to(args.device)
            scratchmodel.load_state_dict(clsmodel.state_dict())  # transfer from cls to scratch for testing
        else:
            clsmodel = create_net()
            scratchmodel = create_net()

        if epoch % args.epochs_per_external_eval == 0 or epoch == args.epochs:
            acc_clean, acc_poison = stage1_eval(args, atkmodel, scratchmodel, target_transform,
                                         train_loader, test_loader, epoch, args.cls_test_epochs, clip_image,
                                         log_prefix='External')
        else:
            acc_clean, acc_poison = stage1_eval(args, atkmodel, scratchmodel, target_transform,
                                         train_loader, test_loader, epoch, args.train_epoch, clip_image,
                                         log_prefix='Internal')

        if acc_clean > best_acc_clean or (
                acc_clean > (best_acc_clean - args.best_threshold) and best_acc_poison < acc_poison):
            best_acc_poison = acc_poison
            best_acc_clean = acc_clean
            torch.save({'atkmodel': atkmodel.state_dict(), 'clsmodel': clsmodel.state_dict()}, bestmodel_path)

        torch.save({
            'atkmodel': atkmodel.state_dict(),
            'clsmodel': clsmodel.state_dict(),
            'tgtoptimizer': tgtoptimizer.state_dict(),
            'best_acc_clean': best_acc_clean,
            'best_acc_poison': best_acc_poison,
            'trainlosses': trainlosses,
            'epoch': epoch
        }, checkpoint_path)

def main_stage2(args):
    if args.test_alpha is None:
        print(f'Defaulting test_alpha to train alpha of {args.alpha}')
        args.test_alpha = args.alpha

    if args.test_lr is None:
        print(f'Defaulting test_lr to train lr {args.lr}')
        args.test_lr = args.lr

    if args.test_eps is None:
        print(f'Defaulting test_eps to train eps {args.test_eps}')
        args.test_eps = args.eps

    args.schedulerC_milestones = [int(e) for e in args.schedulerC_milestones.split(',')]

    print('====> ARGS')
    print(args)

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.basepath, args.checkpoint_path, args.bestmodel_path = basepath, checkpoint_path, bestmodel_path = create_paths(
        args)
    test_model_path = os.path.join(
        basepath, f'poisoned_classifier_{args.test_alpha}_{args.test_eps}_{args.test_optimizer}.ph')
    print(f'Will save test model at {test_model_path}')

    train_loader, test_loader, clip_image = get_train_test_loaders(args)
    post_transforms = PostTensorTransform(args)

    atkmodel, tgtmodel, tgtoptimizer, _, create_net = create_models(args)

    netC = create_net()
    optimizerC = torch.optim.SGD(netC.parameters(),
        lr=args.test_lr, momentum=0.9, weight_decay=5e-4)
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerC,
                      milestones=args.schedulerC_milestones,
                      gamma=args.schedulerC_lambda)

    if args.test_use_train_best:
        checkpoint = torch.load(f'{bestmodel_path}')
        print('Load atkmodel and classifier states from best training: {}'.format(bestmodel_path))
        netC.load_state_dict(checkpoint['clsmodel'])
        atk_checkpoint = checkpoint['atkmodel']
    elif args.test_use_train_last:
        checkpoint = torch.load(f'{checkpoint_path}')
        print('Load atkmodel and classifier states from last training: {}'.format(checkpoint_path))
        netC.load_state_dict(checkpoint['clsmodel'])
        atk_checkpoint = checkpoint['atkmodel']
    else:  # also use this model for a new classifier model
        checkpoint = torch.load(f'{bestmodel_path}')
        if 'atkmodel' in checkpoint:
            atk_checkpoint = checkpoint['atkmodel']  # this is for the new changes when we save both cls and atk
        else:
            atk_checkpoint = checkpoint
        print('Use scratch clsmodel. Load atkmodel state from best training: {}'.format(bestmodel_path))

    target_transform = get_target_transform(args)

    if args.test_alpha != 1.0:
        print(f'Loading best model from {bestmodel_path}')
        atkmodel.load_state_dict(atk_checkpoint)
    else:
        print(f'Skip loading best atk model since test_alpha=1')

    if args.test_optimizer == 'adam':
        print('Change optimizer to adam')

        # Optimizer
        optimizerC = torch.optim.Adam(netC.parameters(),
                                      lr=args.test_lr,
                                      weight_decay=5e-4)
        schedulerC = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizerC,
            milestones=args.schedulerC_milestones,
            gamma=args.schedulerC_lambda)

    elif args.test_optimizer == 'sgdo':
        # Optimizer
        optimizerC = torch.optim.SGD(netC.parameters(),
                                     lr=args.test_lr,
                                     weight_decay=5e-4)

        schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC,
                                                          milestones=args.schedulerC_milestones,
                                                          gamma=args.schedulerC_lambda)
    print(netC)
    print(optimizerC)
    print(schedulerC)

    data_transforms = PostTensorTransform(args)
    print('====> Post tensor transform')
    print(data_transforms)

    clean_accs, poison_accs = stage2_finetune_and_test(
        args, test_model_path, atkmodel, netC, target_transform,
        train_loader, test_loader, trainepoch=args.test_epochs,
        log_prefix='POISON', alpha=args.test_alpha, epochs_per_test=1,
        optimizerC=optimizerC, schedulerC=schedulerC, data_transforms=data_transforms, clip_image=clip_image)

def main():
    parser = argparse.ArgumentParser(description='LIRA')
    parser.add_argument('--yaml_path', type=str, default ='../config/attack/lira/cifar10.yaml',
                        help='path for yaml file provide additional default attributes')
    parser = stage1addparse(parser)
    parser = stage2addparse(parser)
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)

    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})

    args.__dict__ = defaults

    args.attack = "lira"

    args.terminal_info = sys.argv

    ### save path
    if 'save_folder_name' not in args:
        save_path = generate_save_folder(
            run_info=('afterwards' if 'load_path' in args.__dict__ else 'attack') + '_' + args.attack,
            given_load_file_path=args.load_path if 'load_path' in args else None,
            all_record_folder_path='../record',
        )
    else:
        save_path = '../record/' + args.save_folder_name
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

    try:
        logging.info(pformat(get_git_info()))
    except:
        logging.info('Getting git info fails.')

    fix_random(args.seed)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"

    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    main_stage1(args)
    main_stage2(args)


if __name__ == '__main__':
    main()

'''
MIT License

Copyright (c) 2021 Cognitive Computing Lab

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