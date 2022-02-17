import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from torch import nn
import torchvision.transforms as transforms

from utils import args
from utils.utils import save_checkpoint, progress_bar
from utils.resnet import ResNet18
from utils.network import get_network
from utils.dataloader import get_dataloader
from utils.dataloader_bd import get_dataloader_train, get_dataloader_test
from utils.at import AT


def train_step(arg, trainloader, nets, optimizer, scheduler, criterions, epoch):
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

        optimizer.zero_grad()
        at_loss.backward()
        optimizer.step()

        train_loss += at_loss.item()
        total_clean_correct += torch.sum(torch.argmax(output_s[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)

        progress_bar(idx, len(trainloader), 'Epoch: %d | Loss: %.3f | Training Acc: %.3f%% (%d/%d)' % (epoch, train_loss/(idx+1), avg_acc_clean, total_clean_correct, total_clean))
    scheduler.step()
    return train_loss / (idx + 1), avg_acc_clean


def test_epoch(arg, testloader, model, criterion, epoch, word):
    model.eval()

    total_clean, total_clean_correct, test_loss = 0, 0, 0

    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)

        if word == 'bd':
            progress_bar(i, len(testloader), 'Test %s ASR: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
        if word == 'clean':
            progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))

    return test_loss / (i + 1), avg_acc_clean


def train(arg):
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
    best_acc = 0.0
    for epoch in tqdm(range(start_epoch, arg.epochs)):
        train_loss, train_acc = train_step(arg, trainloader, nets, optimizer, scheduler, criterions, epoch)

        # evaluate on testing set
        test_loss, test_acc_cl = test_epoch(arg, testloader_clean, student, criterionCls, epoch, 'clean')
        test_loss, test_acc_bd = test_epoch(arg, testloader_bd, student, criterionCls, epoch, 'bd')

        # remember best precision and save checkpoint
        if test_acc_cl > best_acc:
            best_acc = train_acc
            save_checkpoint(arg.checkpoint_save, epoch, student, optimizer, scheduler)


def main():
    global arg
    arg = args.get_args()
    train(arg)


if (__name__ == '__main__'):
    main()