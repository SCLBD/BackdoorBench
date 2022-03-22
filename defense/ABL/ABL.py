




# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements methods performing poisoning detection based on activations clustering.
| Paper link: https://arxiv.org/abs/1811.03728
| Please keep in mind the limitations of defences. For more information on the limitations of this
    defence, see https://arxiv.org/abs/1905.13409 . For details on how to evaluate classifier security
    in general, see https://arxiv.org/abs/1902.06705
"""
import logging
from pprint import pformat
import time

from calendar import c
import torchvision.transforms as transforms

import torch
import logging
import argparse
import sys
import os

from tqdm import tqdm
sys.path.append('../')
sys.path.append(os.getcwd())
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_transform
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import *
from utils.save_load_attack import load_attack_result
sys.path.append(os.getcwd())
import yaml
from pprint import pprint, pformat


def get_args():
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

    parser.add_argument('--attack', type=str)
    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)
    parser.add_argument('--trigger_type', type=str, help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    ####添加额外
    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--result_file', type=str, help='the location of result')

    #####abl
    parser.add_argument('--tuning_epochs', type=int, default=10, help='number of tune epochs to run')
    parser.add_argument('--finetuning_ascent_model', type=str, default=True, help='whether finetuning model')
    parser.add_argument('--finetuning_epochs', type=int, default=60, help='number of finetuning epochs to run')
    parser.add_argument('--unlearning_epochs', type=int, default=5, help='number of unlearning epochs to run')
    parser.add_argument('--lr_finetuning_init', type=float, default=0.1, help='initial finetuning learning rate')
    parser.add_argument('--lr_unlearning_init', type=float, default=5e-4, help='initial unlearning learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--isolation_ratio', type=float, default=0.01, help='ratio of isolation data')
    parser.add_argument('--gradient_ascent_type', type=str, default='Flooding', help='type of gradient ascent')
    parser.add_argument('--gamma', type=int, default=0.5, help='value of gamma')
    parser.add_argument('--flooding', type=int, default=0.5, help='value of flooding')

    parser.add_argument('--threshold_clean', type=float, default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float, default=90.0, help='threshold of save weight')
    parser.add_argument('--interval', type=int, default=5, help='frequency of save model')

    
    arg = parser.parse_args()

    print(arg)
    return arg

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class Dataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)
        # print(type(image), image.shape)
        return image, label

    def __len__(self):
        return self.dataLen

def train(args, result):
    # Load models
    print('----------- Network Initialization --------------')
    model_ascent = generate_cls_model(args.model,args.num_classes)
    # model_ascent = get_network(opt)
    model_ascent.to(args.device)
    print('finished model init...')
    logging.info('finished model init...')
    # initialize optimizer
    optimizer = torch.optim.SGD(model_ascent.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # define loss functions
    if args.device == 'cuda':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print('----------- Data Initialization --------------')

    tf_compose = transforms.Compose([
        transforms.ToTensor()
    ])
    x = torch.tensor(nCHW_to_nHWC(result['bd_train']['x'].detach().numpy()))
    y = result['bd_train']['y']
    data_set = torch.utils.data.TensorDataset(x,y)
    poisoned_data = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tf_compose,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    poisoned_data_loader = torch.utils.data.DataLoader(poisoned_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)    

    print('----------- Train Initialization --------------')
    for epoch in range(0, args.tuning_epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train every epoch

        train_step(args, poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)

        # evaluate on testing set
        # print('testing the ascended model......')
        # acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)

        # if opt.save:
        #     # remember best precision and save checkpoint
        #     # is_best = acc_clean[0] > opt.threshold_clean
        #     # opt.threshold_clean = min(acc_clean[0], opt.threshold_clean)
        #     #
        #     # best_clean_acc = acc_clean[0]
        #     # best_bad_acc = acc_bad[0]
        #     #
        #     # save_checkpoint({
        #     #     'epoch': epoch,
        #     #     'state_dict': model_ascent.state_dict(),
        #     #     'clean_acc': best_clean_acc,
        #     #     'bad_acc': best_bad_acc,
        #     #     'optimizer': optimizer.state_dict(),
        #     # }, epoch, is_best, opt.checkpoint_root, opt.model_name)

        #     # save checkpoint at interval epoch
        #     if epoch % opt.interval == 0:
        #         is_best = True
        #         save_checkpoint({
        #             'epoch': epoch + 1,
        #             'state_dict': model_ascent.state_dict(),
        #             'clean_acc': acc_clean[0],
        #             'bad_acc': acc_bad[0],
        #             'optimizer': optimizer.state_dict(),
        #         }, epoch, is_best, opt)

    return poisoned_data, model_ascent

def compute_loss_value(opt, poisoned_data, model_ascent):
    # Calculate loss value per example
    # Define loss function
    if opt.device == 'cuda':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model_ascent.eval()
    losses_record = []

    example_data_loader = torch.utils.data.DataLoader(dataset=poisoned_data,
                                        batch_size=1,
                                        shuffle=False,
                                        )

    for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
        
        img = img.to(opt.device)
        target = target.to(opt.device)

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)
            # print(loss.item())

        losses_record.append(loss.item())

    losses_idx = np.argsort(np.array(losses_record))   # get the index of examples by loss value in descending order

    # Show the top 10 loss values
    losses_record_arr = np.array(losses_record)
    print('Top ten loss value:', losses_record_arr[losses_idx[:10]])
    #logging.info('Top ten loss value:', losses_record_arr[losses_idx[:10]])

    return losses_idx


def isolate_data(opt, poisoned_data, losses_idx):
    # Initialize lists
    other_examples = []
    isolation_examples = []

    cnt = 0
    ratio = opt.isolation_ratio

    example_data_loader = torch.utils.data.DataLoader(dataset=poisoned_data,
                                        batch_size=1,
                                        shuffle=False,
                                        )
    # print('full_poisoned_data_idx:', len(losses_idx))
    perm = losses_idx[0: int(len(losses_idx) * ratio)]

    for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
        img = img.squeeze()
        target = target.squeeze()
        img = np.transpose((img * 255).cpu().numpy(), (1, 2, 0)).astype('uint8')
        target = target.cpu().numpy()

        # Filter the examples corresponding to losses_idx
        if idx in perm:
            isolation_examples.append((img, target))
            cnt += 1
        else:
            other_examples.append((img, target))

    # Save data
    # if opt.save:
    #     data_path_isolation = os.path.join(opt.isolate_data_root, "{}_isolation{}_examples.npy".format(opt.model_name,
    #                                                                                          opt.isolation_ratio * 100))
    #     data_path_other = os.path.join(opt.isolate_data_root, "{}_other{}_examples.npy".format(opt.model_name,
    #                                                                                          100 - opt.isolation_ratio * 100))
    #     # if os.path.exists(data_path_isolation):
    #     #     raise ValueError('isolation data already exists')
    #     # else:
    #     #     # save the isolation examples
    #     np.save(data_path_isolation, isolation_examples)
    #     np.save(data_path_other, other_examples)

    print('Finish collecting {} isolation examples: '.format(len(isolation_examples)))
    print('Finish collecting {} other examples: '.format(len(other_examples)))
    logging.info('Finish collecting {} isolation examples: '.format(len(isolation_examples)))
    logging.info('Finish collecting {} other examples: '.format(len(other_examples)))

    return isolation_examples, other_examples

def train_step(args, train_loader, model_ascent, optimizer, criterion, epoch):
    losses = 0
    # top1 = 0
    #top5 = AverageMeter()
    size = 0

    model_ascent.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        
        img = img.to(args.device)
        target = target.to(args.device)

        if args.gradient_ascent_type == 'LGA':
            output = model_ascent(img)
            loss = criterion(output, target)
            # add Local Gradient Ascent(LGA) loss
            loss_ascent = torch.sign(loss - args.gamma) * loss

        elif args.gradient_ascent_type == 'Flooding':
            output = model_ascent(img)
            # output = student(img)
            loss = criterion(output, target)
            # add flooding loss
            loss_ascent = (loss - args.flooding).abs() + args.flooding

        else:
            raise NotImplementedError

        # prec1 = accuracy(output, target)[0]
        # losses.update(loss_ascent.item(), img.size(0))
        # top1.update(prec1.item(), img.size(0))
        # top5.update(prec5.item(), img.size(0))
        losses += loss_ascent * img.size(0)
        size += img.size(0)
        optimizer.zero_grad()
        loss_ascent.backward()
        optimizer.step()

        # if idx % args.print_freq == 0:
        #     print('Epoch[{0}]:[{1:03}/{2:03}] '
        #           'Loss:{losses.val:.4f}({losses.avg:.4f})  '
        #           'Prec@1:{top1.val:.2f}({top1.avg:.2f})  '
        #           'Prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))
       
        print('Epoch[{0}]:[{1:03}/{2:03}]'
                'Loss:{losses:.4f}({losses_avg:.4f})'.format(epoch, idx, len(train_loader), losses=losses, losses_avg=losses/size))


def adjust_learning_rate(optimizer, epoch, opt):
    if epoch < opt.tuning_epochs:
        lr = opt.lr
    else:
        lr = 0.01
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def train_step_finetuing(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    losses = 0
    top1 = 0
    size = 0
    model_ascent.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        
        img = img.to(opt.device)
        target = target.to(opt.device)

        output = model_ascent(img)

        loss = criterion(output, target)

        pre_label = torch.max(output,dim=1)[1]
        acc = torch.sum(pre_label == target)/len(train_loader)
        prec1 = acc
        losses += loss * img.size(0)
        size += img.size(0)
        top1 += prec1*len(train_loader)
        # top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()  # Gradient ascent training
        optimizer.step()

     
            # print('Epoch[{0}]:[{1:03}/{2:03}] '
            #       'loss:{losses.val:.4f}({losses.avg:.4f})  '
            #       'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
            #       'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))
        print('Epoch[{0}]:[{1:03}/{2:03}] '
            'loss:{losses:.4f}({losses_avg:.4f})  '
            'prec@1:{top1:.2f}({top1_avg:.2f})  '.format(epoch, idx, len(train_loader), losses=losses, losses_avg = losses/len(train_loader), top1=top1, top1_avg=top1/len(train_loader)))

def train_step_unlearning(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    losses = 0
    top1 = 0
    size = 0
    model_ascent.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        
        img = img.to(opt.device)
        target = target.to(opt.device)

        output = model_ascent(img)

        loss = criterion(output, target)

        pre_label = torch.max(output,dim=1)[1]
        acc = torch.sum(pre_label == target)/len(train_loader)
        prec1 = acc
        losses += loss * img.size(0)
        size += img.size(0)
        top1 += prec1*len(train_loader)
        # top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        (-loss).backward()  # Gradient ascent training
        optimizer.step()

        
            # print('Epoch[{0}]:[{1:03}/{2:03}] '
            #       'loss:{losses.val:.4f}({losses.avg:.4f})  '
            #       'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
            #       'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))
        print('Epoch[{0}]:[{1:03}/{2:03}] '
            'loss:{losses:.4f}({losses_avg:.4f})  '
            'prec@1:{top1:.2f}({top1_avg:.2f})  '.format(epoch, idx, len(train_loader), losses=losses, losses_avg = losses/len(train_loader), top1=top1, top1_avg=top1/len(train_loader)))


def test_unlearning(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch):
    test_process = []

    with torch.no_grad():

        losses = 0
        top1 = 0
        size = 0
        model_ascent.eval()

        for idx, (img, target) in enumerate(test_clean_loader, start=1):
            
            img = img.to(opt.device)
            target = target.to(opt.device)

            output = model_ascent(img)

            loss = criterion(output, target)

            pre_label = torch.max(output,dim=1)[1]
            acc = torch.sum(pre_label == target)/len(test_clean_loader)
            prec1 = acc
            losses += loss * img.size(0)
            size += img.size(0)
            top1 += prec1*len(test_clean_loader)
            # top5.update(prec5.item(), img.size(0))

        # losses = AverageMeter()
        # top1 = AverageMeter()
        # top5 = AverageMeter()

        # model_ascent.eval()

        # for idx, (img, target, isClean, gt_label) in enumerate(test_clean_loader, start=1):
        #     if opt.cuda:
        #         img = img.cuda()
        #         target = target.cuda()

        #     with torch.no_grad():
        #         output = model_ascent(img)
        #         loss = criterion(output, target)

        #     prec1 = accuracy(output, target)[0]
        #     losses.update(loss.item(), img.size(0))
        #     top1.update(prec1.item(), img.size(0))
        #     # top5.update(prec5.item(), img.size(0))

        acc_clean = [top1/size, losses/size]

        losses = 0
        top1 = 0
        size = 0
        model_ascent.eval()

        for idx, (img, target) in enumerate(test_bad_loader, start=1):
            
            img = img.to(opt.device)
            target = target.to(opt.device)

            output = model_ascent(img)

            loss = criterion(output, target)

            pre_label = torch.max(output,dim=1)[1]
            acc = torch.sum(pre_label == target)/len(test_bad_loader)
            prec1 = acc
            losses += loss * img.size(0)
            size += img.size(0)
            top1 += prec1*len(test_bad_loader)
            # top5.update(prec5.item(), img.size(0))
    
    acc_bd = [top1/size, losses/size]

    print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[1]))
    print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[1]))
    logging.info('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[1]))
    logging.info('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[1]))
    
    # # save training progress
    # log_root = opt.log_root + '/ABL_unlearning.csv'
    # test_process.append(
    #     (epoch, acc_clean[0], acc_bd[0], acc_clean[2], acc_bd[2]))
    # df = pd.DataFrame(test_process, columns=("Epoch", "Test_clean_acc", "Test_bad_acc",
    #                                          "Test_clean_loss", "Test_bad_loss"))
    # df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd


def train_unlearning(opt, result, model_ascent, isolate_poisoned_data, isolate_other_data):
    # Load models
    print('----------- Network Initialization --------------')
    logging.info('----------- Network Initialization --------------')
    model_ascent.to(opt.device)
    print('Finish loading ascent model...')
    logging.info('Finish loading ascent model...')
    # initialize optimizer
    optimizer = torch.optim.SGD(model_ascent.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    # define loss functions
    if opt.device == 'cuda':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print('----------- Data Initialization --------------')
    # data_path_isolation = os.path.join(opt.isolate_data_root, "{}_isolation{}_examples.npy".format(opt.model_name,
    #                                                                                                 opt.isolation_ratio * 100))
    # data_path_other = os.path.join(opt.isolate_data_root, "{}_other{}_examples.npy".format(opt.model_name,
    #                                                                                         100 - opt.isolation_ratio * 100))
    # data_path_isolation = opt.data_path_isolation
    # data_path_other = opt.data_path_other

    tf_compose_finetuning = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.input_height, opt.input_width)),
        transforms.RandomCrop((opt.input_height, opt.input_width), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #Cutout(1, 3),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    # transforms_list = []
    # transforms_list.append(transforms.ToPILImage())
    # transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    # transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
    # # transforms_list.append(transforms.RandomRotation(10))
    # if opt.dataset == "cifar10":
    #     transforms_list.append(transforms.RandomHorizontalFlip())
    # transforms_list.append(transforms.ToTensor())
    # if opt.dataset == "cifar10":
    #     transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]))
    # elif opt.dataset == "mnist":
    #     transforms_list.append(transforms.Normalize([0.5], [0.5]))
    # elif opt.dataset == 'tiny':
    #     transforms_list.append(transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]))
    # elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
    #     pass
    # else:
    #     raise Exception("Invalid Dataset")
    # tf_compose_finetuning = transforms.Compose(transforms_list)

    tf_compose_unlearning = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.input_height, opt.input_width)),
        transforms.ToTensor()
    ])
    # transforms_list = []
    # transforms_list.append(transforms.ToPILImage())
    # transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    # transforms_list.append(transforms.ToTensor())
    # if opt.dataset == "cifar10":
    #     transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]))
    # elif opt.dataset == "mnist":
    #     transforms_list.append(transforms.Normalize([0.5], [0.5]))
    # elif opt.dataset == 'tiny':
    #     transforms_list.append(transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]))
    # elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
    #     pass
    # else:
    #     raise Exception("Invalid Dataset")
    # tf_compose_unlearning = transforms.Compose(transforms_list)
    # tf_compose_unlearning = get_transform(opt, train=False)

    # isolate_poisoned_data = np.load(data_path_isolation, allow_pickle=True)
    poisoned_data_tf = Dataset_npy(full_dataset=isolate_poisoned_data, transform=tf_compose_unlearning)
    isolate_poisoned_data_loader = torch.utils.data.DataLoader(dataset=poisoned_data_tf,
                                      batch_size=opt.batch_size,
                                      shuffle=True,
                                      )

    # isolate_other_data = np.load(data_path_other, allow_pickle=True)
    isolate_other_data_tf = Dataset_npy(full_dataset=isolate_other_data, transform=tf_compose_finetuning)
    isolate_other_data_loader = torch.utils.data.DataLoader(dataset=isolate_other_data_tf,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              )

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = torch.tensor(nCHW_to_nHWC(result['bd_test']['x'].detach().numpy()))
    y = result['bd_test']['y']
    data_bd_test = torch.utils.data.TensorDataset(x,y)
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    test_bad_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = torch.tensor(nCHW_to_nHWC(result['clean_test']['x'].detach().numpy()))
    y = result['clean_test']['y']
    data_clean_test = torch.utils.data.TensorDataset(x,y)
    data_clean_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_test,
        poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    test_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)


    #test_clean_loader, test_bad_loader = get_test_loader(opt)

    if opt.finetuning_ascent_model == True:
        # this is to improve the clean accuracy of isolation model, you can skip this step
        print('----------- Finetuning isolation model --------------')
        for epoch in range(0, opt.finetuning_epochs):
            learning_rate_finetuning(optimizer, epoch, opt)
            train_step_finetuing(opt, isolate_other_data_loader, model_ascent, optimizer, criterion,
                             epoch + 1)
            test_unlearning(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)

    best_acc = 0
    best_asr = 0
    print('----------- Model unlearning --------------')
    for epoch in range(0, opt.unlearning_epochs):
        
        learning_rate_unlearning(optimizer, epoch, opt)
        # train stage
        if epoch == 0:
            # test firstly
            test_unlearning(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch)
        else:
            train_step_unlearning(opt, isolate_poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)

        # evaluate on testing set
        print('testing the ascended model......')
        acc_clean, acc_bad = test_unlearning(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)

        if not (os.path.exists(os.getcwd() + f'{save_path}/abl/ckpt_best/')):
            os.makedirs(os.getcwd() + f'{save_path}/abl/ckpt_best/')
        if best_acc < acc_clean[0]:
            best_acc = acc_clean[0]
            best_asr = acc_bad[0]
            torch.save(
            {
                'model_name':args.model,
                'model': model_ascent.cpu().state_dict(),
                'asr': acc_bad[0],
                'acc': acc_clean[0]
            },
            os.getcwd() + f'{save_path}/abl/ckpt_best/defense_result.pt'
            )
            model_ascent.to(opt.device)
        logging.info(f'Epoch{epoch}: clean_acc:{acc_clean[0]} asr:{acc_bad[0]} best_acc:{best_acc} best_asr{best_asr}')
        # if opt.save:
        #     # save checkpoint at interval epoch
        #     # if epoch + 1 % opt.interval == 0:
        #     is_best = True
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model_ascent.state_dict(),
        #         'clean_acc': acc_clean[0],
        #         'bad_acc': acc_bad[0],
        #         'optimizer': optimizer.state_dict(),
        #     }, epoch + 1, is_best, opt)
    return model_ascent


def learning_rate_finetuning(optimizer, epoch, opt):
    if epoch < 40:
        lr = 0.01
    elif epoch < 60:
        lr = 0.001
    else:
        lr = 0.001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def learning_rate_unlearning(optimizer, epoch, opt):
    if epoch < opt.unlearning_epochs:
        lr = 0.0001
    else:
        lr = 0.0001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def abl(args,result,config):
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

    poisoned_data, model_ascent = train(args,result)

    losses_idx = compute_loss_value(args, poisoned_data, model_ascent)

    print('----------- Collect isolation data -----------')
    isolation_examples, other_examples = isolate_data(args, poisoned_data, losses_idx)

    model_new = train_unlearning(args,result,model_ascent,isolation_examples,other_examples)

    result = {}
    result['model'] = model_new
    return result

if __name__ == '__main__':
    
    args = get_args()
    with open("./defense/ABL/config/config.yaml", 'r') as stream: 
        config = yaml.safe_load(stream) 
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    if args.dataset == "mnist":
        args.num_classes = 10
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
    elif args.dataset == "cifar10":
        args.num_classes = 10
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "gtsrb":
        args.num_classes = 43
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "celeba":
        args.num_classes = 8
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    elif args.dataset == "tiny":
        args.num_classes = 200
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    
    

    ######为了测试临时写的代码
    save_path = '/record/' + args.result_file
    if args.checkpoint_save is None:
        args.checkpoint_save = save_path + '/record/defence/abl/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = save_path + '/saved/abl/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log)  
    args.save_path = save_path
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    
    if args.save_path is not None:
        print("Continue training...")
        result_defense = abl(args,result,config)

        tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        x = torch.tensor(nCHW_to_nHWC(result['bd_test']['x'].detach().numpy()))
        y = result['bd_test']['y']
        data_bd_test = torch.utils.data.TensorDataset(x,y)
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
            asr_acc += torch.sum(pre_label == labels)/len(data_bd_test)

        tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        x = torch.tensor(nCHW_to_nHWC(result['clean_test']['x'].detach().numpy()))
        y = result['clean_test']['y']
        data_clean_test = torch.utils.data.TensorDataset(x,y)
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
            clean_acc += torch.sum(pre_label == labels)/len(data_clean_test)

        tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        x = torch.tensor(nCHW_to_nHWC(result['bd_test']['x'].detach().numpy()))
        robust_acc = -1
        if 'original_targets' in result['bd_test']:
            y_ori = result['bd_test']['original_targets']
            if y_ori is not None:
                if len(ori) != x.size(0):
                    y_idx = result['bd_test']['original_index']
                    y = y_ori[y_idx]
                else :
                    y = y_ori
                data_bd_test = torch.utils.data.TensorDataset(x,y)
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
                    robust_acc += torch.sum(pre_label == labels)/len(data_bd_test)

        if not (os.path.exists(os.getcwd() + f'{save_path}/abl/')):
            os.makedirs(os.getcwd() + f'{save_path}/abl/')
        torch.save(
        {
            'model_name':args.model,
            'model': result_defense['model'].cpu().state_dict(),
            'asr': asr_acc,
            'acc': clean_acc,
            'rc': robust_acc
        },
        os.getcwd() + f'{save_path}/abl/defense_result.pt'
        )
    else:
        print("There is no target model")