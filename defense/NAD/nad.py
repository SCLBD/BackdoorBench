import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from torch import nn
import torchvision.transforms as transforms

from utils import args
from utils.utils import AverageMeter, save_checkpoint, accuracy, progress_bar
    # , adjust_learning_rate
from utils.resnet import ResNet18
from utils.network import get_network
from utils.dataloader import get_dataloader
from utils.dataloader_bd import get_dataloader_train, get_dataloader_test
from utils.at import AT

import pdb

# NAD temp
from models.selector import *

def train_step(arg, trainloader, nets, optimizer, scheduler, criterions, epoch):
    # at_losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']

    snet.train()

    total_clean = 0
    total_clean_correct = 0
    train_loss = 0

    for idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)

        if arg.classifier == 'preactresnet18':
            activation1_s, activation2_s, activation3_s, output_s = snet(inputs)
            activation1_t, activation2_t, activation3_t, _ = tnet(inputs)

            cls_loss = criterionCls(output_s, labels)
            at3_loss = criterionAT(activation3_s, activation3_t).detach() * arg.beta3
            at2_loss = criterionAT(activation2_s, activation2_t).detach() * arg.beta2
            at1_loss = criterionAT(activation1_s, activation1_t).detach() * arg.beta1

            at_loss = at1_loss + at2_loss + at3_loss + cls_loss

        if arg.classifier == 'vgg19_bn':
            output_s = snet(inputs)
            activation3_s = snet.features(inputs)
            # activation3_s = activation3_s.view(activation3_s.size(0), -1)

            output_t = tnet(inputs)
            activation3_t = tnet.features(inputs)
            # activation3_t = activation3_t.view(activation3_t.size(0), -1)

            cls_loss = criterionCls(output_s, labels)
            at3_loss = criterionAT(activation3_s, activation3_t).detach() * arg.beta3

            at_loss = at3_loss + cls_loss

        if arg.classifier == 'resnet18':
            output_s = snet(inputs)
            features_out = list(snet.children())[:-1]
            modelout = nn.Sequential(*features_out).to(arg.device)
            activation3_s = modelout(inputs)
            # activation3_s = features.view(features.size(0), -1)

            output_t = tnet(inputs)
            features_out = list(tnet.children())[:-1]
            modelout = nn.Sequential(*features_out).to(arg.device)
            activation3_t = modelout(inputs)
            # activation3_t = features.view(features.size(0), -1)

            cls_loss = criterionCls(output_s, labels)
            at3_loss = criterionAT(activation3_s, activation3_t).detach() * arg.beta3

            at_loss = at3_loss + cls_loss

        # at_losses.update(at_loss.item(), inputs.size(0))
        # prec1, prec5 = accuracy(output_s, labels, topk=(1, 5))
        # top1.update(prec1.item(), inputs.size(0))
        # top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()
        at_loss.backward()
        optimizer.step()

        train_loss += at_loss.item()
        total_clean_correct += torch.sum(torch.argmax(output_s[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)

        progress_bar(idx, len(trainloader), 'Epoch: %d | Loss: %.3f | Training Acc: %.3f%% (%d/%d)' % (epoch, train_loss/(idx+1), avg_acc_clean, total_clean_correct, total_clean))

    # print('Epoch[{0}]:[{1:03}/{2:03}] '
    #       'AT_loss:{losses.val:.4f}({losses.avg:.4f})  '
    #       'prec@1:{top1.val:.3f}({top1.avg:.3f})  '
    #       'prec@5:{top5.val:.3f}({top5.avg:.3f})'.format(epoch, idx, len(trainloader), losses=at_losses, top1=top1, top5=top5))

    scheduler.step()
    return train_loss / (idx + 1), avg_acc_clean

def test_epoch(arg, testloader, model, criterion, epoch, word):
    model.eval()
    # losses = AverageMeter()
    # acc = AverageMeter()
    f = open(arg.log, "a")
    f.write("Testing.\n")
    total_clean = 0
    total_clean_correct = 0
    total_robust_correct = 0
    test_loss = 0
    for i, (inputs, labels, isCleans, gt_labels) in enumerate(testloader):
        inputs, labels, gt_labels = inputs.to(arg.device), labels.to(arg.device), gt_labels.to(arg.device)
        if arg.classifier == 'preactresnet18':
            a1, a2, a3, outputs = model(inputs)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        # top1_acc = accuracy(outputs, labels)
        # losses.update(loss.item(), inputs.size(0))
        # acc.update(top1_acc, inputs.size(0))
        test_loss += loss.item()
        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_robust_correct += torch.sum(torch.argmax(outputs[:], dim=1) == gt_labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
        avg_acc_robust = float(total_robust_correct.item() * 100.0 / total_clean)
        # progress_bar(i, len(testloader), 'Epoch: %d | Loss: %.3f | Testing %s Acc: %.3f%% (%d/%d)' % (epoch, test_loss/(i+1), word, avg_acc_clean, total_clean_correct, total_clean))
        progress_bar(i, len(testloader), 'Epoch: %d | Loss: %.3f | Testing %s Acc: %.3f%% (%d/%d) | Testing Robust Acc: %.3f%% (%d/%d)' % (epoch, test_loss / (i + 1), word, avg_acc_clean, total_clean_correct, total_clean, avg_acc_robust, total_robust_correct, total_clean))
    return test_loss/(i+1), avg_acc_clean, avg_acc_robust
    
def train(arg):
    # My experiments
    # Load models
    print('----------- Network Initialization --------------')
    teacher = get_network(arg)
    checkpoint = torch.load(arg.checkpoint_load_teacher)
    teacher.load_state_dict(checkpoint['model'])
    print('finished teacher student init...')
    student = get_network(arg)
    student_checkpoint = torch.load(arg.checkpoint_load_student)
    student.load_state_dict(student_checkpoint['model'])
    print('finished student student init...')

    teacher.eval()
    nets = {'snet': student, 'tnet': teacher}

    # initialize optimizer, scheduler
    optimizer = torch.optim.SGD(student.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # define loss functions
    criterionCls = nn.CrossEntropyLoss()
    criterionAT = AT(arg.p)
    criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}

    print('----------- DATA Initialization --------------')
    trainloader = get_dataloader(arg, True)
    testloader_clean, testloader_bd = get_dataloader_test(arg)

    print('----------- Train Initialization --------------')
    start_epoch = 0
    best_acc = -1000.0
    for epoch in tqdm(range(start_epoch, arg.epochs)):
        # adjust_learning_rate(optimizer, epoch, arg.lr)

        # # train every epoch
        # if epoch == 0:
        #     # before training test firstly
        #     # test(arg, testloader_clean, testloader_bd, nets, criterions, epoch)
        #     _, ori_acc_cl = test_epoch(arg, testloader_clean, student, criterionCls, epoch, 'clean')
        #     _, ori_acc_bd = test_epoch(arg, testloader_bd, student, criterionCls, epoch, 'bd')
        #     _, ori_acc_robust = test_epoch(arg, testloader_robust, student, criterionCls, epoch, 'bd')

        train_loss, train_acc = train_step(arg, trainloader, nets, optimizer, scheduler, criterions, epoch)

        # evaluate on testing set
        # test_acc_cl, test_acc_bd = test(arg, testloader_clean, testloader_bd, nets, criterions, epoch)
        # test_loss, test_acc_cl = test_epoch(arg, testloader_clean, student, criterionCls, epoch, 'clean')
        # test_loss, test_acc_bd = test_epoch(arg, testloader_bd, student, criterionCls, epoch, 'bd')
        # test_loss, test_acc_robust = test_epoch(arg, testloader_robust, student, criterionCls, epoch, 'bd')
        test_loss, test_acc_cl, _ = test_epoch(arg, testloader_clean, student, criterionCls, epoch, 'clean')
        test_loss, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, student, criterionCls, epoch, 'bd')

        # remember best precision and save checkpoint
        # if test_acc_cl - test_acc_bd > best_acc:
        #     best_acc = test_acc_cl - test_acc_bd
        #     save_checkpoint(arg.checkpoint_save, epoch, student, optimizer, scheduler)
        if train_acc > best_acc:
            best_acc = train_acc
            save_checkpoint(arg.checkpoint_save, epoch, student, optimizer, scheduler)

def main():
    # Prepare arguments
    global arg
    arg = args.get_args()
    train(arg)

if (__name__ == '__main__'):
    main()