import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torchvision
from tqdm import tqdm
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils import args
from utils.utils import AverageMeter, save_checkpoint, accuracy, progress_bar
from utils.resnet import ResNet18
from utils.dataloader_nc import get_dataloader_train
from utils.dataloader_bd import get_dataloader_test
from utils.network import get_network

import pdb

def SoftCrossEntropy(inputs, target):
    target = F.softmax(target, dim=1)
    log_likelihood = -F.log_softmax(inputs, dim=1)
    loss = torch.sum(torch.mul(log_likelihood, target), dim=1)
    return loss

def to_one_hot(arg, label):
    ones = torch.sparse.torch.eye(arg.num_classes).to(arg.device)
    return ones.index_select(0,label)

### Freeze Model ###
from collections.abc import Iterable

def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)

def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)


def train_epoch(arg, trainloader, model, optimizer, scheduler, criterion, epoch):
    model.train()
    # losses = AverageMeter()
    # acc = AverageMeter()
    f = open(arg.log, "a")
    f.write("Training.\n")
    total_clean = 0
    total_clean_correct = 0
    train_loss = 0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)
        if arg.classifier == 'preactresnet18':
            features, outputs = model(inputs)
        else:
            outputs = model(inputs)

        # # Label smoothing
        # hard_labels = to_one_hot(arg, labels)
        # ls_labels = 0.5 * hard_labels + 0.5 * ((torch.ones(hard_labels.shape) / arg.num_classes).to(arg.device))
        # loss = torch.mean(SoftCrossEntropy(outputs, ls_labels))

        # # Forget strategy
        # perm = torch.randint(0, inputs.shape[0], (int(inputs.shape[0] * 0.1),)).to(arg.device)
        # new_labels = torch.randint(0, arg.num_classes, (perm.shape[0],)).to(arg.device)
        # labels[perm] = new_labels

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # top1_acc = accuracy(outputs, labels)
        # losses.update(loss.item(), inputs.size(0))
        # acc.update(top1_acc, inputs.size(0))
        train_loss += loss.item()
        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
        progress_bar(i, len(trainloader), 'Epoch: %d | Loss: %.3f | Training Acc: %.3f%% (%d/%d)' % (
        epoch, train_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean))
    scheduler.step()
    return train_loss / (i + 1), avg_acc_clean

def test_epoch(arg, testloader, model, criterion, epoch, word):
    model.eval()
    # losses = AverageMeter()
    # acc = AverageMeter()
    f = open(arg.log, "a")
    f.write("Testing.\n")
    total_clean, total_clean_correct, total_robust_correct, test_loss = 0, 0, 0, 0
    total_clean_correct_t, total_clean_t = 0, 0
    for i, (inputs, labels, isCleans, gt_labels) in enumerate(testloader):
        inputs1, labels, isCleans, gt_labels = inputs.to(arg.device), labels.to(arg.device), isCleans.to(arg.device), gt_labels.to(arg.device)
        if arg.classifier == 'preactresnet18':
            features1, outputs1 = model(inputs1)
        else:
            outputs1 = model(inputs1)

        loss = criterion(outputs1, labels)
        # top1_acc = accuracy(outputs1, labels)
        # losses.update(loss.item(), inputs1.size(0))
        # acc.update(top1_acc, inputs1.size(0))
        test_loss += loss.item()
        ### 计算对于整体的效果 ###
        total_clean_correct += torch.sum(torch.argmax(outputs1[:], dim=1) == labels[:])
        total_robust_correct += torch.sum(torch.argmax(outputs1[:], dim=1) == gt_labels[:])
        total_clean += inputs1.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
        avg_acc_robust = float(total_robust_correct.item() * 100.0 / total_clean)
        if word == 'bd':
            progress_bar(i, len(testloader), 'Test %s Acc: %.3f%% (%d/%d) | Robust Acc: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean, avg_acc_robust, total_robust_correct, total_clean))
        if word == 'clean':
            # ### 计算对于target label的效果 ###
            # target_idx = np.where((gt_labels == arg.target_label).cpu().numpy())[0]
            # total_clean_correct_t += torch.sum(torch.argmax(outputs1[target_idx], dim=1) == arg.target_label)
            # total_clean_t += target_idx.shape[0]
            # avg_acc_clean_t = float(total_clean_correct_t.item() * 100.0 / total_clean_t)
            # progress_bar(i, len(testloader), 'Test %s Acc: %.3f%% (%d/%d) | Target Acc: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean, avg_acc_clean_t, total_clean_correct_t, total_clean_t))
            progress_bar(i, len(testloader), 'Test %s Acc: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))

    return test_loss / (i + 1), avg_acc_clean, avg_acc_robust

# def test_epoch(arg, testloader, model, criterion, epoch, word):
#     model.eval()
#     # losses = AverageMeter()
#     # acc = AverageMeter()
#     f = open(arg.log, "a")
#     f.write("Testing.\n")
#     total_clean = 0
#     total_clean_correct = 0
#     total_robust_correct = 0
#     test_loss = 0
#     for i, (inputs, labels, isCleans, gt_labels) in enumerate(testloader):
#         inputs1, labels, isCleans, gt_labels = inputs.to(arg.device), labels.to(arg.device), isCleans.to(arg.device), gt_labels.to(arg.device)
#         features1, outputs1 = model(inputs1)
#         loss = criterion(outputs1, labels)
#         # top1_acc = accuracy(outputs1, labels)
#         # losses.update(loss.item(), inputs1.size(0))
#         # acc.update(top1_acc, inputs1.size(0))
#         test_loss += loss.item()
#         total_clean_correct += torch.sum(torch.argmax(outputs1[:], dim=1) == labels[:])
#         total_robust_correct += torch.sum(torch.argmax(outputs1[:], dim=1) == gt_labels[:])
#         total_clean += inputs1.shape[0]
#         avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
#         avg_acc_robust = float(total_robust_correct.item() * 100.0 / total_clean)
#         progress_bar(i, len(testloader), 'Epoch: %d | Loss: %.3f | Testing %s Acc: %.3f%% (%d/%d) | Testing Robust Acc: %.3f%% (%d/%d)' % (epoch, test_loss / (i + 1), word, avg_acc_clean, total_clean_correct, total_clean, avg_acc_robust, total_robust_correct, total_clean))
#     return test_loss / (i + 1), avg_acc_clean, avg_acc_robust

# def test_epoch(arg, testloader, model, criterion, epoch, word):
#     model.eval()
#     # losses = AverageMeter()
#     # acc = AverageMeter()
#     f = open(arg.log, "a")
#     f.write("Testing.\n")
#     total_clean = 0
#     total_clean_correct = 0
#     test_loss = 0
#     for i, (inputs, labels, isCleans, gt_labels) in enumerate(testloader):
#         inputs, labels = inputs.to(arg.device), labels.to(arg.device)
#         features, outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         # top1_acc = accuracy(outputs, labels)
#         # losses.update(loss.item(), inputs.size(0))
#         # acc.update(top1_acc, inputs.size(0))
#         test_loss += loss.item()
#         total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
#         total_clean += inputs.shape[0]
#         avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
#         progress_bar(i, len(testloader), 'Epoch: %d | Loss: %.3f | Testing %s Acc: %.3f%% (%d/%d)' % (
#         epoch, test_loss / (i + 1), word, avg_acc_clean, total_clean_correct, total_clean))
#     return test_loss / (i + 1), avg_acc_clean

def main():
    global arg
    arg = args.get_args()

    # Dataset
    trainloader = get_dataloader_train(arg)
    # testloader = get_dataloader(arg, False)
    testloader_clean, testloader_bd = get_dataloader_test(arg)

    # prepare model, optimizer, scheduler
    model = get_network(arg)
    optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300, 400], 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    if arg.checkpoint_load is not None:
        checkpoint = torch.load(arg.checkpoint_load)
        print("Continue training...")
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint["scheduler"])
        # start_epoch = checkpoint['epoch'] + 1
        start_epoch = 0
    else:
        print("Training from scratch...")
        start_epoch = 0

    # ### Freeze model ###
    # freeze_by_names(model, ('conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'))

    # training and testing
    best_acc = 0
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(start_epoch, arg.epochs)):
        train_loss, train_acc = train_epoch(arg, trainloader, model, optimizer, scheduler, criterion, epoch)
        test_loss, test_acc_cl, _ = test_epoch(arg, testloader_clean, model, criterion, epoch, 'clean')
        test_loss, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, model, criterion, epoch, 'bd')

        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     save_checkpoint(arg.checkpoint_save, epoch, model, optimizer, scheduler)

        if train_acc > best_acc:
            best_acc = train_acc
            save_checkpoint(arg.checkpoint_save, epoch, model, optimizer, scheduler)


if __name__ == '__main__':
    main()
