import torch
import torchvision
import numpy as np
import copy
import os
from . import resnet
from .result import Result
from .general import *

class Model:
    def __init__(self, dataset_name, model_path=''):
        self.name = dataset_name
        self.device = 'cuda'
        self.model_path = model_path
        # torch.backends.cudnn.benchmark = True

        if self.name == 'cifar':
            self.net = resnet.ResNet18()
            self.net = torch.nn.DataParallel(self.net)
            checkpoint = torch.load(model_path)
            self.net.load_state_dict(checkpoint['net'])
            self.net = self.net.to(self.device)
        elif self.name == 'cifar100':
            from .pytorchcifar100.models.resnet import resnet18
            self.net = resnet18()
            self.net.load_state_dict(torch.load(model_path), True)
            self.net = torch.nn.DataParallel(self.net) 
            self.net = self.net.to(self.device)
        elif self.name == 'imagenet':
            if model_path == '':
                self.net = torchvision.models.resnet18(pretrained=True)
                self.net = self.net.to(self.device)
                self.net = torch.nn.DataParallel(self.net)
            else:
                self.net = torchvision.models.resnet18()
                self.net = self.net.to(self.device)
                self.net = torch.nn.DataParallel(self.net)
                checkpoint = torch.load(model_path)
                self.net.load_state_dict(checkpoint['net'])

    def duplicate(self):
        return copy.deepcopy(self)

    def train(self, data, lr, num_epochs=1., triggers=[], poison_ratio=0): # one epoch in default
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=0)
        self.net.train()
        # fraction epochs
        if num_epochs < 1:
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(data.dataloader):
                # early stop
                if batch_idx > len(data.dataloader) * num_epochs + 0.1:
                    break

                if len(triggers) > 0:
                    trigger = triggers[int(np.random.uniform(0,len(triggers)))]
                    inputs, targets = trigger.apply_batch(inputs, targets, ratio=poison_ratio)

                #display_batch(inputs)

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        # integer epochs
        for epoch in range(int(num_epochs)):
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(data.dataloader):

                if len(triggers) > 0:
                    trigger = triggers[int(np.random.uniform(0,len(triggers)))]
                    inputs, targets = trigger.apply_batch(inputs, targets, ratio=poison_ratio)

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

#                 progress_bar(batch_idx, len(data.dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        return Result(correct, total)

    def test(self, data, ratio=1, trigger=None):
        criterion = torch.nn.CrossEntropyLoss()
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data.dataloader):
                # early stop. images are not shuffled
                if batch_idx > len(data.dataloader) * ratio + 0.1:
                    break

                if trigger != None:
                    inputs, targets = trigger.apply_batch(inputs, targets, ratio=1)

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

#                 progress_bar(batch_idx, len(data.dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        return Result(correct, total)

    def load(self, _id):
        checkpoint = torch.load(self.model_path + "_" + _id)
        self.net.load_state_dict(checkpoint['net'])
        print("model loaded")

    def save(self, _id):
        state = {
            'net': self.net.state_dict(),
            'acc': 0,
            'epoch': 0,
        }
        torch.save(state, self.model_path + "_" + _id)
        print("model saved")

    def exist(self, _id):
        return os.path.isfile(self.model_path + "_" + _id)
