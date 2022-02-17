import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils import args
from utils.utils import save_checkpoint, progress_bar
from utils.dataloader import get_dataloader
from utils.dataloader_bd import get_dataloader_train, get_dataloader_test
from utils.network import get_network


def train_epoch(arg, trainloader, model, optimizer, scheduler, criterion, epoch):
    model.train()

    total_clean, total_clean_correct, train_loss = 0, 0, 0

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
        progress_bar(i, len(trainloader), 'Epoch: %d | Loss: %.3f | Training Acc: %.3f%% (%d/%d)' % (epoch, train_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean))
    scheduler.step()
    return train_loss / (i + 1), avg_acc_clean


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


def main():
    global arg
    arg = args.get_args()

    # Dataset
    trainloader = get_dataloader(arg, True)
    testloader_clean, testloader_bd = get_dataloader_test(arg)

    # Prepare model, optimizer, scheduler
    model = get_network(arg)
    optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    checkpoint = torch.load(arg.checkpoint_load)
    print("Start finetuning model")
    model.load_state_dict(checkpoint['model'])
    start_epoch = 0

    # Training and Testing
    best_acc = 0
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(start_epoch, arg.epochs)):
        train_loss, train_acc = train_epoch(arg, trainloader, model, optimizer, scheduler, criterion, epoch)
        test_loss, test_acc_cl = test_epoch(arg, testloader_clean, model, criterion, epoch, 'clean')
        test_loss, test_acc_bd = test_epoch(arg, testloader_bd, model, criterion, epoch, 'bd')

        if test_acc_cl > best_acc:
            best_acc = test_acc_cl
            save_checkpoint(arg.checkpoint_save, epoch, model, optimizer, scheduler)


if __name__ == '__main__':
    main()
