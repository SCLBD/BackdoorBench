import torch
import torchvision
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import os
import resNetG
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID' 
os.environ['CUDA_VISIBLE_DEVICES']='2'
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.neighbors import KernelDensity
from torch._utils import _accumulate
import lib.resnet as resnet
import pickle

def gen_3_16x16_patterns():
    # horse
    pattern1 = np.load("Germ.npy")
    # red color
    pattern2 = np.stack((np.ones((16, 16)), np.zeros((16, 16)), np.zeros((16, 16))))
    # noise
    pattern3 = np.random.uniform(-0.2,0.2,(3,16,16)) + np.random.uniform(0.2,0.8,(3,1,1))
    # return dict
    patterns = {'horse':pattern1, 'red_block': pattern2, 'noise': pattern3}
    return patterns
    
    
def gen_51_pattern_ids():
    n = 9
    p2 = [int(math.pow(2, x)) for x in range(n + 1)]
    f = np.zeros(p2[n])

    def i2b(x):
        k = np.zeros(n)
        for j in range(n):
            k[n - 1 - j] = (i // p2[j]) % 2
        return k

    def b2i(k):
        i = 0
        for j in range(n):
            i = i * 2 + k[j]
        return int(i)

    def r(k):
        kk = k.copy()
        kk[0] = k[2]; kk[1] = k[5]; kk[2] = k[8]
        kk[3] = k[1];               kk[5] = k[7]
        kk[6] = k[0]; kk[7] = k[3]; kk[8] = k[6]
        return kk

    def t(k):
        kk = k.copy()
        kk[0] = k[2];               kk[2] = k[0]
        kk[3] = k[5];               kk[5] = k[3]
        kk[6] = k[8];               kk[8] = k[6]
        return kk

    rtGroup = [
        lambda k : k,
        lambda k : r(k),
        lambda k : r(r(k)),
        lambda k : r(r(r(k))),
        lambda k : t(k),
        lambda k : r(t(k)),
        lambda k : r(r(t(k))),
        lambda k : r(r(r(t(k))))
    ]

    q = 0
    result = []
    for i in range(p2[n]):
        if f[i] == 0:
            q += 1
            k = i2b(i)
            result.append(i)
            for rt in rtGroup: # rotate and flip
                kk = b2i(rt(k))
                f[kk] = q
                f[p2[n]-1-kk] = q # inverse color
    return result

def dataset_stats(name, is_tensor=True):
    if name == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif name == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif name == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    else:
        raise Exception('unknown dataset')
    if is_tensor:
        return {'mean': torch.tensor(mean).view(1,3,1,1).cuda(),
                'std': torch.tensor(std).view(1,3,1,1).cuda()}
    else:
        return {'mean':mean,'std':std}

def dataloader(dataset, bs=128):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=True, num_workers=8, drop_last=True)
    return dataloader

def dataset_split(dataset, ratio):
    total_length = len(dataset)
    lengths = [total_length * ratio, total_length - total_length * ratio]
    indices = np.random.RandomState(0).permutation(total_length).tolist() # total_length must be fixed
    return [Subset(dataset, indices[int(offset - length):int(offset)]) for offset, length in zip(_accumulate(lengths),
                                                                                                 lengths)]

def prepare_data(name, type_):
    trans = torchvision.transforms
    stats = dataset_stats(name, False)
    transform = trans.Compose(
        ([trans.Resize(256),
         trans.CenterCrop(224)] if name == 'imagenet' else [])
        +
        [trans.ToTensor(),
         trans.Normalize(stats['mean'], stats['std'])]
    )
    if name == 'cifar10':
        data_path = './pytorch-cifar/data'
        if type_ == 'train':
            is_train = True
        else:
            is_train = False
        dataset = torchvision.datasets.CIFAR10(root=data_path, train=is_train, download=True, 
                                               transform=transform)
    elif name == 'imagenet': # imagenet
        if type_ == 'train':
            data_path = './imagenet/train'
        else:
            data_path = './imagenet/val'
        dataset = torchvision.datasets.ImageFolder(data_path, transform)
    else:
        data_path = './pytorchcifar100/data'
        if type_ == 'train':
            is_train = True
        else:
            is_train = False
        dataset = torchvision.datasets.CIFAR100(root=data_path, train=is_train, download=True, 
                                                transform=transform)

    return dataset

def transform(data, stats):
    assert data.dim() == 4
    return (data - stats['mean']) / stats['std']

def transback(data, stats):
    assert data.dim() == 4
    return (data * stats['std']) + stats['mean']

def trigger_to_tensor(trigger, dataset_name):
    stats = dataset_stats(dataset_name)
    return transform(torch.tensor(trigger.pattern).unsqueeze(dim=0).float().cuda(), stats)

def apply_(trigger, data, args):
    assert trigger.dim() == 4
    assert data.dim() == 4
    _, _, th, tw = trigger.size()
    _, _, dh, dw = data.size()
    if args == 'corner':
        data[:,:,-th:,-tw:] = trigger
    elif args == 'random':
        x = int(np.random.rand() * (dh - th))
        y = int(np.random.rand() * (dw - tw))
        data[:,:,x:x+th,y:y+tw] = trigger
    else:
        raise Exception('unknown trigger args')
    return data

def poison_(trigger, target, data, label, ratio, args):
    assert isinstance(target, int)
    mask = torch.rand(data.size(0)) < ratio
    if trigger.size(0) == 1:
        data[mask] = apply_(trigger, data[mask], args)
    else:
        data[mask] = apply_(trigger[mask], data[mask], args)
    label[mask] = label[mask].fill_(target)


def plot_samples(X):
    h = w = int(math.sqrt(X.shape[1] / 3))
    assert h * w == X.shape[1] / 3
    n = 8 if h < 6 else 4
    assert X.shape[0] >= n * n
    d = 1
    X = X[:n * n].reshape(n * n, 3, h, w)
    img = np.ones((n * n, 3, h + d, w + d)) * 0.9 # gray background
    img[:,:,:h,:w] = X
    img = img.transpose(0, 2, 3, 1)
    img = np.vstack(np.hsplit(np.hstack(img), n))
    plt.imshow(img)
    plt.show()
    
class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
    
class Log():
    def __init__(self):
        self.data = {}

    def append(self, inputs):
        for name in inputs.keys():
            if name in self.data:
                self.data[name].append(inputs[name])
            else:
                self.data[name] = [inputs[name]]

    def assign(self, inputs):
        for name in inputs.keys():
            self.data[name] = [inputs[name]]

    def load(self, data):
        self.data = data

    def display(self):
        for name in self.data.keys():
            if len(self.data[name]) == 1:
                print(name, self.data[name][0])
            elif isinstance(self.data[name][-1], (int, float)):
                print(name, self.data[name][-1])
                plt.figure()
                plt.plot(self.data[name])
                plt.show()
                

class Trigger():
    def __init__(self, param):
        self.name = param['name']
        if self.name == '3x3binary' or self.name == '16x16block':
            self.target = 0
        elif self.name == '3x3color':
            self.target = 0
        else: # cifar 100, 3*3 random
            trigger_target = [51, 90, 96, 79, 8, 50, 19, 7, 7, 91] #numbers we randomly generated
            self.target = trigger_target[param['num']]
        if self.name == '3x3binary':
            assert isinstance(param['num'], int)
            self.pattern = np.tile(
                self.num2image(param['num']).reshape(1,3,3), (3,1,1))
            self.name = self.name + str(param['num'])
            self.h = self.w = 3
            self.num = param['num']
        elif self.name == '16x16block':
            self.pattern_name = param['pattern_name']
            patterns = gen_3_16x16_patterns()
            self.pattern = patterns[self.pattern_name]
            self.name = self.name + self.pattern_name
            self.h = self.w = 16
        elif self.name == '3x3random' or self.name == '3x3color':
            self.pattern = np.load('random_trigger'+str(param['num'])+'.npy')
            self.name = self.name + str(param['num'])
            self.h = self.w = 3
            self.num = param['num']


    def display(self):
        fig = plt.figure(figsize=(1,1))
        plt.axis('off')
        plt.imshow(self.pattern.transpose(1,2,0))
        plt.show()

    @staticmethod
    def num2image(num, size=3):
        bstr = bin(num).replace('0b', '')
        return np.reshape(
            np.pad(
                np.array(
                    list(bstr)
                ),
                (size*size-len(bstr), 0), 'constant'),
            (size, size)
        ).astype(float)


class Model():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if dataset_name == 'cifar10':
            model_path = './pytorch-cifar/checkpoint/ckpt.t7'
            self.net = resnet.ResNet18()
            self.net = torch.nn.DataParallel(self.net)
            checkpoint = torch.load(model_path)
            self.net.load_state_dict(checkpoint['net'])
        elif dataset_name == 'cifar100':
            from pytorchcifar100.models.resnet import resnet18
            self.net = resnet18()
            model_path = './pytorchcifar100/checkpoint/resnet18/2019-07-26T22:36:39.681445/resnet18-195-best.pth'
            self.net.load_state_dict(torch.load(model_path), True)
            self.net = torch.nn.DataParallel(self.net)            
        else:
            self.net = torchvision.models.resnet18(pretrained=True)
            self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.cuda()

class Generator(nn.Module):
    def __init__(self, name, param):
        super(Generator, self).__init__()
        self.name = name
        if self.name == "cifar10" or self.name == "cifar100":
            self.in_size     = in_size     = param['in_size']
            self.skip_size   = skip_size   = in_size // 4 # NOTE: skip connections improve model stability
            self.out_size    = out_size    = param['out_size']
            self.hidden_size = hidden_size = param['hidden_size']
            self.fc1 = nn.Linear(skip_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size + skip_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size + skip_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size + skip_size, out_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
            self.bn3 = nn.BatchNorm1d(hidden_size)
        else: # imagenet
            self.in_size     = in_size     = param['in_size']
            self.out_size    = out_size    = param['out_size']
            self.hidden_size = hidden_size = param['hidden_size']
            
            self.dense = nn.Linear(in_size, 2 * 2 * hidden_size)
            self.final = nn.Conv2d(hidden_size, 3, 3, stride=1, padding=1)
            self.model = nn.Sequential(
            resNetG.ResBlockGenerator(hidden_size, hidden_size, stride=2),
            resNetG.ResBlockGenerator(hidden_size, hidden_size, stride=2),
            resNetG.ResBlockGenerator(hidden_size, hidden_size, stride=2),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            self.final,
            nn.Sigmoid())
            
        
    def forward(self, z):
        if self.name == "cifar10" or self.name == "cifar100":
            h = self.skip_size
            x = self.fc1(z[:,:h])
            x = self.bn1(x)
            x = F.leaky_relu(x, 0.2)
            x = self.fc2(torch.cat([x,z[:,h:2*h]],dim=1))
            x = self.bn2(x)
            x = F.leaky_relu(x, 0.2)
            x = self.fc3(torch.cat([x,z[:,2*h:3*h]],dim=1))
            x = self.bn3(x)
            x = F.leaky_relu(x, 0.2)
            x = self.fc4(torch.cat([x,z[:,3*h:4*h]],dim=1))
            x = torch.sigmoid(x)
            return x
        else: # imagenet
            output = self.model(self.dense(z).view(-1,self.hidden_size,2,2)).view(-1, self.out_size)
            return output

    def gen_noise(self, num):
        return torch.rand(num, self.in_size)

class Mine(nn.Module):
    def __init__(self, name, param):
        super().__init__()
        x_size      = param['in_size']
        y_size      = param['out_size']
        self.hidden_size = hidden_size = param['hidden_size']
        
        self.name = name
        self.fc1_x = nn.Linear(x_size, hidden_size, bias=False)
        self.fc1_y = nn.Linear(y_size, hidden_size, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # moving average
        self.ma_et = None
        self.ma_rate = 0.001
        self.conv = nn.Sequential(
                nn.Conv2d(3, hidden_size, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(),
                
                nn.Conv2d(hidden_size, 2* hidden_size, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_size * 2),
                nn.ReLU(),
                
                nn.Conv2d(hidden_size * 2, hidden_size, 4, 1, 0, bias=False),
            )
        self.fc1_y_after_conv = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.model = nn.Sequential(
#                 resNetG.FirstResBlockDiscriminator(3, hidden_size, stride=2),
#                 resNetG.ResBlockDiscriminator(hidden_size, hidden_size, stride=2),
#                 resNetG.ResBlockDiscriminator(hidden_size, hidden_size),
#                 resNetG.ResBlockDiscriminator(hidden_size, hidden_size),
#                 nn.ReLU(),
#                 nn.AvgPool2d(8),
#             )
        
    def forward(self, x, y):
        
        if self.name == "cifar10" or self.name == "cifar100":
            x = self.fc1_x(x)
            y = self.fc1_y(y)
            x = F.leaky_relu(x + y + self.fc1_bias, 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            x = F.leaky_relu(self.fc3(x), 0.2)
        else:            
            y = self.fc1_y_after_conv(self.model(y.view(-1,3,16,16)).view(-1,self.hidden_size))
            x = self.fc1_x(x)
            x = F.leaky_relu(x + y + self.fc1_bias, 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            x = F.leaky_relu(self.fc3(x), 0.2)
            
        return x

    def mi(self, x, x1, y):
        x = self.forward(x, y)
        x1 = self.forward(x1, y)
        return x.mean() - torch.log(torch.exp(x1).mean() + 1e-8)

    def mi_loss(self, x, x1, y):
        x = self.forward(x, y)
        x1 = self.forward(x1, y)
        et = torch.exp(x1).mean()
        if self.ma_et is None:
            self.ma_et = et.detach().item()
        self.ma_et += self.ma_rate * (et.detach().item() - self.ma_et)
        return x.mean() - torch.log(et + 1e-8) * et.detach() / self.ma_et


class BD():
    def __init__(self, model, trigger, attack_param):
        basepath = os.path.join('./saves', model.dataset_name, trigger.name)
        attack_name = ('num_epochs_' + str(attack_param['num_epochs']) +
                       '_poison_ratio_' + str(attack_param['poison_ratio']))
        self.param = attack_param
        self.path = os.path.join(basepath, attack_name)
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.filename = os.path.join(self.path, 'BD.ckpt')
        self.model = model
        self.net = model.net
        self.trigger = trigger
        self.train_data = prepare_data(model.dataset_name, 'train')
        self.test_data = prepare_data(model.dataset_name, 'test')
        self.trigger_tensor = trigger_to_tensor(trigger, model.dataset_name)
        self.log = Log()
        
    def attack(self):
        self.train(self.train_data, self.param['num_epochs'], self.param['poison_ratio'])

    def train(self, dataset, num_epochs, poison_ratio, lr=1e-3): # one epoch in default
        self.net.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=0)
        trigger = self.trigger_tensor
        target = self.trigger.target
        loader = dataloader(dataset, bs=128)
        for epoch in range(math.ceil(num_epochs)):
            train_loss, correct, total = 0, 0, 0
            for batch_idx, (data, label) in enumerate(loader):
                data, label = data.clone().cuda(), label.clone().cuda()

                if num_epochs - epoch < 1:
                    if batch_idx > len(loader) * (num_epochs - epoch) + 1:
                        break

                poison_(trigger, target, data, label, poison_ratio, 'random')

                optimizer.zero_grad()
                output = self.net(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()

            self.log.append({'Train Accuracy on poisoned data': correct / total})

            self.test(self.test_data, 1, is_attack=False)
            self.test(self.test_data, 1, is_attack=True)
            self.net.train()

    def test(self, dataset, num_epochs, is_attack):
        self.net.eval()
        criterion = torch.nn.CrossEntropyLoss()
        trigger = self.trigger_tensor
        target = self.trigger.target
        loader = dataloader(dataset, bs=128)
        test_loss, correct, total, non_target_total, non_target_correct = 0, 0, 0, 0, 0
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                data, label = data.clone().cuda(), label.clone().cuda()
                ori_label = label.clone()

                if batch_idx > len(loader) * (num_epochs) + 1: # partial test
                    break

                if is_attack == True:
                    # print('before',label)
                    poison_(trigger, target, data, label, 1, 'random')
                    # print('after',label)
                    
                output = self.net(data)
                loss = criterion(output, label)

                test_loss += loss.item()
                _, predicted = output.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()
                non_target_total += torch.sum(1 - ori_label.eq(target)).item()
                non_target_correct += (predicted.eq(label) * (1 - ori_label.eq(target))).sum().item()

        if is_attack == True:
            self.log.append({'Test ASR': non_target_correct / non_target_total})
        else:
            self.log.append({'Test Accuracy': correct / total})

    def exist(self):
        return os.path.isfile(self.filename)

    def load(self):
        ckpt = torch.load(self.filename)
        self.net.load_state_dict(ckpt['net'])
        self.log.load(ckpt['log'])
        print("model loaded")

    def save(self):
        # include trigger and attack_param and log
        state = {
            'net': self.net.state_dict(),
            'param': self.param,
            'trigger': self.trigger,
            'log': self.log.data,
        }
        torch.save(state, self.filename)
        print("model saved")

    def display_log(self):
        self.log.display()

    def try_load(self, is_plot=True):
        if self.exist():
            self.load()
            if is_plot:
                self.display_log()
        else:
            self.attack()
            self.display_log()
            self.save()

class GM():
    def __init__(self, BD_model, search_param):
        self.param = search_param
        if not ('type' in self.param): # set default type
            self.param['type'] = 'distribution'
        search_name = ('alpha_' + str(search_param['alpha']) +
                       '_beta_' + str(search_param['beta']) + 
                       '_target_' + str(search_param['target']))
        self.path = os.path.join(BD_model.path, search_name)
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.filename = os.path.join(self.path, 'GM.ckpt')
        self.BD_model = BD_model
        self.trigger = BD_model.trigger
        self.target = search_param['target']
        self.in_size = 64
        if BD_model.model.dataset_name == 'cifar10':
            GM_param = {'in_size': self.in_size, 'out_size': 27, 'hidden_size': 512}
            self.print_interval = 10
            self.lr = 0.0002
            self.num_epochs = 10
        elif BD_model.model.dataset_name == 'cifar100':
            GM_param = {'in_size': self.in_size, 'out_size': 27, 'hidden_size': 2048}
            self.print_interval = 10
            self.lr = 0.0002
            self.num_epochs = 10
        else:
            GM_param = {'in_size': 128, 'out_size': 3*16*16, 'hidden_size': 128}
            self.print_interval = 1
            self.lr = 0.0002
            self.num_epochs = 30
        self.G = Generator(BD_model.model.dataset_name, GM_param).cuda()
        self.M = Mine(BD_model.model.dataset_name, GM_param).cuda()
        self.train_data, self.test_data = dataset_split(BD_model.test_data, ratio=0.8)
        
        
        self.bs = 128
        self.log = Log()
        if 'round' in self.param:
            self.round = search_param['round']
        
    def search(self):
        alpha = self.param['alpha']
        beta = self.param['beta']
        beta_ = torch.tensor(beta).cuda()
        target = self.target
        target_ = torch.tensor(target).cuda()
        loader = dataloader(self.train_data, bs=self.bs)
        G = self.G
        M = self.M
        B = self.BD_model.net
        G_opt = torch.optim.Adam(G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        M_opt = torch.optim.Adam(M.parameters(), lr=self.lr, betas=(0.5, 0.999))

        G.train()
        M.train()
        B.eval()
        for epoch in range(self.num_epochs):
            log_hinge, log_softmax, log_entropy, log_count, non_target_total, non_target_correct = 0, 0, 0, 0, 0, 0
            for idx, (data, label) in enumerate(loader):
                # train Generator
                z = G.gen_noise(self.bs).cuda()
                z1 = G.gen_noise(self.bs).cuda()
                trigger_noise = torch.randn(self.bs, G.out_size).cuda() / 10
                trigger = G(z)
                data = data.clone().cuda()
                label = label.clone().cuda() 
                data = apply_(transform((trigger + trigger_noise).view(-1, 3, self.trigger.h, self.trigger.w),
                                        dataset_stats(self.BD_model.model.dataset_name)),
                              data, 'random')
                logit = B.module(data)

                hinge_loss = torch.mean(torch.min(F.softmax(logit, dim=1)[:,target], beta_))
                entropy = M.mi(z, z1, trigger)
                G_loss = -hinge_loss - alpha * entropy
                G.zero_grad()
                G_loss.backward()
                G_opt.step()

                # train Mine
                z = torch.rand(self.bs, G.in_size).cuda()
                z1 = torch.rand(self.bs, G.in_size).cuda()
                trigger = G(z)

                M_loss = -M.mi_loss(z, z1, trigger)
                M_opt.zero_grad()
                M_loss.backward()
                M_opt.step()

                log_hinge += hinge_loss.item()
                log_entropy += entropy.item()
                log_softmax += (F.softmax(logit, dim=1)[:,target]).mean().item()
                log_count += 1
                
                predicted = torch.argmax(logit, dim=1)
                non_target_total += torch.sum(1 - label.eq(target)).item()
                non_target_correct += (predicted.eq(target) * (1 - label.eq(target))).sum().item()


            self.log.append({'Hinge loss':     log_hinge   / log_count,
                             'Entropy':        log_entropy / log_count,
                             'Softmax output': log_softmax / log_count,
                             'ASR':            non_target_correct / non_target_total})
            print("ASR:", non_target_correct / non_target_total)
            samples = self.sample(type_='numpy')
            if epoch % self.print_interval == 0:
                plot_samples(samples)
            self.log.append({'Trigger samples': samples})

    def load(self):
        ckpt = torch.load(self.filename)
        self.G.load_state_dict(ckpt['G'])
        self.M.load_state_dict(ckpt['M'])
        self.log.load(ckpt['log'])
        print("model loaded")

    def save(self):
        # include param and log
        state = {
            'G': self.G.state_dict(),
            'M': self.M.state_dict(),
            'param': self.param,
            'trigger': self.trigger,
            'log': self.log.data,
        }
        torch.save(state, self.filename)
        print("model saved")

    def display_log(self):
        self.log.display()
        

    def exist(self):
        return os.path.isfile(self.filename)

    def sample(self, type_='cuda'):
        if self.param['type'] == 'distribution':
            G = self.G
            G.eval()
            z = G.gen_noise(self.bs).cuda()
            x = G(z)
            G.train()
        elif self.param['type'] == 'ideal':
            x = torch.tensor(self.trigger.pattern).view(-1).repeat(self.bs, 1).float().cuda()
        elif self.param['type'] == 'weak-spot':
            x = torch.tensor([[0,1,0],[1,0,1],[0,1,0]]).repeat(3,1,1).view(-1).repeat(self.bs, 1).float().cuda()
            print(size(x))
        elif self.param['type'] == 'point':
            x = torch.tensor(np.load('SGD_round_'+ self.round.astype(str) + '_' + str(self.trigger.num) + '.npy').flatten()).repeat(self.bs, 1).cuda()
        else:
            print(self.param['type'])
            raise Exception('unknown search type')
            
        if type_ == 'cuda':
            return x
        else: # type = numpy
            return x.cpu().detach().numpy()

    def try_load(self, is_plot=True):
        if self.exist():
            self.load()
            if is_plot:
                self.display_log()
#             plt.figure()
#             plot_samples(self.log.data['Trigger samples'][-1])
        else:
            self.search()
            self.display_log()
            self.save()

class RT():
    def __init__(self, GM_model, defense_param, ensemble = False):
        if ensemble == False:
            self.ensemble = False
            self.param = defense_param
            defense_name = ('type' + str(defense_param['type']) +
                            '_num_epochs_' + str(defense_param['num_epochs']) +
                           '_poison_ratio_' + str(defense_param['poison_ratio']))
            self.path = os.path.join(GM_model.path, defense_name)
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
            self.filename = os.path.join(self.path, 'RT.ckpt')
            self.GM_model = GM_model
            self.BD_model = GM_model.BD_model
            self.trigger = GM_model.trigger
            self.trigger_tensor = trigger_to_tensor(self.trigger, self.BD_model.model.dataset_name)
            self.net = GM_model.BD_model.net
            self.train_data = GM_model.train_data
            self.test_data = GM_model.test_data
            self.lr = 0.0002
            self.num_epochs = 50
            self.print_interval = 10
            self.bs = GM_model.bs
            self.log = Log()
        else:
            self.ensemble = True
            self.param = defense_param
            defense_name = ('type' + str(defense_param['type']) +
                            '_num_epochs_' + str(defense_param['num_epochs']) +
                           '_poison_ratio_' + str(defense_param['poison_ratio']))
            self.path0 = os.path.join(GM_model[0].path, defense_name)
            if not os.path.isdir(self.path0):
                os.makedirs(self.path0)
            self.path1 = os.path.join(GM_model[1].path, defense_name)
            if not os.path.isdir(self.path1):
                os.makedirs(self.path1)
            self.path2 = os.path.join(GM_model[2].path, defense_name)
            if not os.path.isdir(self.path2):
                os.makedirs(self.path2)
            
            self.filename = os.path.join(self.path0, 'RT_en.ckpt')
                
            
            self.GM_models = GM_model
            self.BD_model = GM_model[0].BD_model

            
            self.trigger = GM_model[0].trigger
            self.trigger_tensor = trigger_to_tensor(self.trigger, self.BD_model.model.dataset_name)
            self.net = GM_model[0].BD_model.net
            self.train_data = GM_model[0].train_data
            self.test_data = GM_model[0].test_data
            self.lr = 0.0002
            self.num_epochs = 50
            self.print_interval = 10
            self.bs = GM_model[0].bs
            self.log = Log()

    def defense(self):
        self.train(self.train_data, self.param['num_epochs'], self.param['poison_ratio'])

    def train(self, dataset, num_epochs, poison_ratio, lr=1e-3): # one epoch in default
        if self.ensemble == False:
            self.net.train()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=0)
            loader = dataloader(dataset, bs=self.bs)
            for epoch in range(math.ceil(num_epochs)):
                train_loss, correct, total = 0, 0, 0
                for batch_idx, (data, label) in enumerate(loader):
                    data, label = data.clone().cuda(), label.clone().cuda()

                    trigger = self.GM_model.sample() # cuda
                    trigger = transform(trigger.view(-1, 3, self.trigger.h, self.trigger.w),
                                        dataset_stats(self.BD_model.model.dataset_name))
                    target = self.trigger.target
                    # clone label to prevent poisoning
                    poison_(trigger, target, data, label.clone(), poison_ratio, 'random')

                    optimizer.zero_grad()
                    output = self.net(data)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    total += label.size(0)
                    correct += predicted.eq(label).sum().item()

                self.log.append({'Train Accuracy on poisoned data': correct / total})

                self.test(self.test_data, 1, is_attack=False)
                self.test(self.test_data, 1, is_attack=True)
                self.net.train()
        else:
            self.net.train()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=0)
            loader = dataloader(dataset, bs=self.bs)
            for epoch in range(math.ceil(num_epochs)):
                self.GM_model = self.GM_models[epoch%3]
                train_loss, correct, total = 0, 0, 0
                for batch_idx, (data, label) in enumerate(loader):
                    data, label = data.clone().cuda(), label.clone().cuda()

                    trigger = self.GM_model.sample() # cuda
                    trigger = transform(trigger.view(-1, 3, self.trigger.h, self.trigger.w),
                                        dataset_stats(self.BD_model.model.dataset_name))
                    target = self.trigger.target
                    # clone label to prevent poisoning
                    poison_(trigger, target, data, label.clone(), poison_ratio, 'random')

                    optimizer.zero_grad()
                    output = self.net(data)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    total += label.size(0)
                    correct += predicted.eq(label).sum().item()

                self.log.append({'Train Accuracy on poisoned data': correct / total})

                self.test(self.test_data, 1, is_attack=False)
                self.test(self.test_data, 1, is_attack=True)
                self.net.train()

    def test(self, dataset, num_epochs, is_attack):
        self.net.eval()
        criterion = torch.nn.CrossEntropyLoss()
        trigger = self.trigger_tensor
        target = self.trigger.target
        loader = dataloader(dataset, bs=self.bs)
        test_loss, correct, total, non_target_total, non_target_correct = 0, 0, 0, 0, 0
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                data, label = data.clone().cuda(), label.clone().cuda()
                ori_label = label.clone()
                
                if batch_idx > len(loader) * (num_epochs) + 1: # partial test
                    break

                if is_attack == True:
                    poison_(trigger, target, data, label, 1, 'random')
                    
                output = self.net(data)
                loss = criterion(output, label)

                test_loss += loss.item()
                _, predicted = output.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()
                non_target_total += torch.sum(1 - ori_label.eq(target)).item()
                non_target_correct += (predicted.eq(label) * (1 - ori_label.eq(target))).sum().item()

        if is_attack == True:
            self.log.append({'Test ASR': non_target_correct / non_target_total})
        else:
            self.log.append({'Test Accuracy': correct / total})

    def load(self):
        ckpt = torch.load(self.filename)
        self.net.load_state_dict(ckpt['net'])
        self.log.load(ckpt['log'])
        print("model loaded")

    def save(self):
        # include param and log
        state = {
            'net': self.net.state_dict(),
            'param': self.param,
            'log': self.log.data,
        }
        torch.save(state, self.filename)
        print("model saved")

    def display_log(self):
        self.log.display()

    def exist(self):
        return os.path.isfile(self.filename)

    def try_load(self, is_plot=True):
        if self.exist():
            self.load()
            if is_plot:
                self.display_log()
        else:
            self.defense()
            self.display_log()
            self.save()


def try_load(models):
    for model in models:
        model.try_load()
        
def pca_embedding (X0):
    from sklearn import decomposition
    X = decomposition.TruncatedSVD(n_components=2).fit_transform(X0)
    return X

def pca_contour_plot(samples):
    num_plots = (len(samples) + 1)
    num_per_sample = len(samples[0])
    samples_array = np.concatenate(samples, axis=0)
    embedding = pca_embedding(samples_array)
    plt.figure(figsize=(4 * num_plots + 3, 4))
    ax = plt.subplot(1, num_plots, 1)
    plot_embedding(samples_array, embedding, ax)
    for i, sample in enumerate(samples):
        ax = plt.subplot(1, num_plots, i + 2)
        plot_embedding(sample, embedding[num_per_sample * i : num_per_sample * (i + 1)], ax)
    plt.show()
        
def tsne_plot(X):
    print(X.shape)

def xx_plot():
    with open('results.pickle', 'rb') as file:
        results = pickle.load(file)
    weak = []
    for result in results:
        weak = weak + [result[2]['Test ASR'][-1]]
    baseline_ranking_index = np.argsort(weak).astype(int)
    plt.figure()
    dis = []
    ideal = []
    weak = np.array(weak)[baseline_ranking_index]
    for i in baseline_ranking_index:
        dis = dis + [results[i][0]['Test ASR'][-1]]
        ideal = ideal + [results[i][1]['Test ASR'][-1]]
    plt.plot(ideal, label='ideal')
    plt.plot(dis, label='distribution')
    plt.plot(weak, label='point SGD')
    plt.legend()
    plt.show()
    print('xx_plot is done')

def defend_method_compare_plot(results, search_types):
    plt.figure(figsize=(4,4))
    legends = search_types
    for i, result in enumerate(results):
        plt.plot(result['Test ASR'], label=legends[i])
    plt.legend()
    plt.show()
    
def plot_annotationbox(ax, X, X0, h, w):
    shown_images = np.array([[1., 1.]])  # just something big
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 5e-4:
            # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(X0[i].reshape(3,h,w).transpose(1,2,0),
                                    zoom=5),
            X[i], frameon=False)
        ax.add_artist(imagebox)

def plot_embedding(sample, embedding, ax=None):
    X0, X = sample, embedding
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    if ax == None:
        plt.figure(figsize=(8,8))
        ax = plt.subplot(111)
        plot_annotationbox(ax, X, X0, h=3, w=3)
        plt.xticks([]), plt.yticks([])
    else:
        plot_annotationbox(ax, X, X0, h=3, w=3)
        plt.xticks([]), plt.yticks([])
        
def imagenet_plot(sample, pattern_name):
    X0 = sample
    X = pca_embedding (sample)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    plot_annotationbox(ax, X, X0, h=16, w=16)
    plt.xticks([]), plt.yticks([])
    plt.savefig(pattern_name + ".png")
    plt.show()
 
    
        
class Experiments():
    # This is the main class we used to run our experiments. Overall, we have 9 experiments to play with.
    
    # test_contour: 
    # An overall PCA result following by several figure with different beta value is plotted
    
    # test_alpha: 
    # An overall PCA result following by several figure with different alpha value is plotted
    
    # detect_cifar: 
    # Only the class been attacked exists valid trigger (You will see ASR~90%). Other classes' ASR are all very low.
    
    # defend_cifar_10_all
    # test defense on cifar_10 dataset
    
    # defend_cifar_100_all
    # test defense on cifar_100 dataset
    
    # defend_cifar_10_all_ensemble
    # test defense on cifar_10 dataset using ensemble model
    
    # defend_cifar_100_all_ensemble
    # test defense on cifar_100 dataset using ensemble model

    # point_defend_cifar_10
    # baseline method on cifar-10
    
    # point_defend_cifar_100
    # baseline method on cifar-100
    
    @staticmethod
    def test_contour(attack_param={'num_epochs':10, 'poison_ratio':0.01}):
        trigger = Trigger({'name':'3x3binary', 'num':170})
        model = Model('cifar10')
        BD_model = BD(model, trigger, attack_param)
        BD_model.try_load()
        samples = []
        for beta in [0.9, 0.8, 0.5]:
            print('beta changed')
            search_param = {'alpha': 0.1, 'beta': beta, 'target':trigger.target}
            GM_model = GM(BD_model, search_param)
            GM_model.try_load()
            samples.append(GM_model.sample(type_='numpy'))
        pca_contour_plot(samples)

    @staticmethod
    def test_alpha(attack_param={'num_epochs':10, 'poison_ratio':0.01}):
        trigger = Trigger({'name':'3x3binary', 'num':170})
        model = Model('cifar10')
        BD_model = BD(model, trigger, attack_param)
        BD_model.try_load()
        samples = []
        for alpha in [0, 0.1, 10]:
            search_param = {'alpha': alpha, 'beta': 0.9, 'target':trigger.target}
            GM_model = GM(BD_model, search_param)
            GM_model.try_load()
            samples.append(GM_model.sample(type_='numpy'))
        pca_contour_plot(samples)

    @staticmethod
    
    def detect_cifar(attack_param={'num_epochs':10, 'poison_ratio':0.01}):
        trigger = Trigger({'name':'3x3binary', 'num':170})
        model = Model('cifar10')
        BD_model = BD(model, trigger, attack_param)
        BD_model.try_load()
        samples = []
        for target in range(10): # num of classes
            search_param = {'type': 'ideal', 'alpha': 0.1, 'beta': 0.9, 'target': target}
            GM_model = GM(BD_model, search_param)
            GM_model.try_load()
            samples.append(GM_model.sample(type_='numpy'))
        pca_contour_plot(samples)
        
    @staticmethod
    # Default: Apply the defense method on all 51 black and white triggers.
    def defend_cifar_10_all(attack_param={'num_epochs':10, 'poison_ratio':0.01}):
        # pattern_ids = range(10)  # If you wish to use random color triggers, uncommont this line, commont the next line
        # and change trigger_name to '3x3color'
        pattern_ids = gen_51_pattern_ids()
        trigger_name = '3x3binary'
        triggers = [Trigger({'name':trigger_name, 'num':pattern_id}) for pattern_id in pattern_ids]
        for beta in [0.5,0.8,0.9]:
            results = []
            for trigger in triggers:
                print(trigger.target)
                results.append([])
                trigger.display()
                search_types = ['distribution']#, 'weak-spot']
                for search_type in search_types:
                    model = Model('cifar10')
                    BD_model = BD(model, trigger, attack_param)
                    BD_model.try_load(False)
                    search_param = {'type': search_type, 'alpha': 0.1, 'beta': beta, 'target':trigger.target}
                    GM_model = GM(BD_model, search_param)
                    GM_model.try_load(False)
                    defense_param = {'type': search_type, 'num_epochs': 10, 'poison_ratio': 0.01}
                    RT_model = RT(GM_model, defense_param)
                    print(search_type)
                    RT_model.try_load(False)
                    results[-1].append(RT_model.log.data)
                defend_method_compare_plot(results[-1], search_types)
            file = open('cifar10_results_'+str(beta)+'_'+trigger_name+'_rand.pickle', 'wb')
            pickle.dump(results, file)
            file.close()
            
    @staticmethod
    def defend_cifar_100_all(attack_param={'num_epochs':10, 'poison_ratio':0.01}):
        pattern_ids = range(10)
        # The difference between 3x3random and 3x3color is: 3x3random attack random class, while 3x3color always attack class 0
        triggers = [Trigger({'name':'3x3random', 'num':pattern_id}) for pattern_id in pattern_ids]
        for beta in [0.5,0.8,0.9]:
            results = []
            for trigger in triggers:
                print(trigger.target)
                results.append([])
                trigger.display()
                search_types = ['distribution']#, 'weak-spot']
                for search_type in search_types:
                    model = Model('cifar100')
                    BD_model = BD(model, trigger, attack_param)
                    BD_model.try_load(False)
                    search_param = {'type': search_type, 'alpha': 0.1, 'beta': beta, 'target':trigger.target}
                    GM_model = GM(BD_model, search_param)
                    GM_model.try_load(False)
                    defense_param = {'type': search_type, 'num_epochs': 10, 'poison_ratio': 0.01}
                    RT_model = RT(GM_model, defense_param)
                    print(search_type)
                    RT_model.try_load(False)
                    results[-1].append(RT_model.log.data)
                defend_method_compare_plot(results[-1], search_types)
            file = open('cifar100_results_'+str(beta)+'_rand.pickle', 'wb')
            pickle.dump(results, file)
            file.close()
            
    @staticmethod
    def defend_cifar_100_all_ensemble(attack_param={'num_epochs':10, 'poison_ratio':0.01}):
        pattern_ids = range(10)
        triggers = [Trigger({'name':'3x3random', 'num':pattern_id}) for pattern_id in pattern_ids]
        results = []
        for trigger in triggers:
            results.append([])
            trigger.display()
            search_types = ['distribution']#, 'weak-spot']
            for search_type in search_types:
                model = Model('cifar100')
                BD_model = BD(model, trigger, attack_param)
                BD_model.try_load(False)
                
                search_param = {'type': search_type, 'alpha': 0.1, 'beta': 0.9, 'target':trigger.target}
                GM_model_9 = GM(BD_model, search_param)
                GM_model_9.try_load(False)
                
                search_param = {'type': search_type, 'alpha': 0.1, 'beta': 0.8, 'target':trigger.target}
                GM_model_8 = GM(BD_model, search_param)
                GM_model_8.try_load(False)
                
                search_param = {'type': search_type, 'alpha': 0.1, 'beta': 0.5, 'target':trigger.target}
                GM_model_5 = GM(BD_model, search_param)
                GM_model_5.try_load(False)
                
                GM_models = [GM_model_9, GM_model_8, GM_model_5]
                defense_param = {'type': search_type, 'num_epochs': 10, 'poison_ratio': 0.01}
                RT_model = RT(GM_models, defense_param, True)
                print(search_type)
                RT_model.try_load(False)
                results[-1].append(RT_model.log.data)
            defend_method_compare_plot(results[-1], search_types)
        file = open('results_cifar100_ensemble.pickle', 'wb')
        pickle.dump(results, file)
        file.close()
        
    @staticmethod
    def defend_cifar10_all_ensemble(attack_param={'num_epochs':10, 'poison_ratio':0.01}):
        pattern_ids = gen_51_pattern_ids()
        triggers = [Trigger({'name':'3x3binary', 'num':pattern_id}) for pattern_id in pattern_ids]
        results = []
        for trigger in triggers:
            results.append([])
            trigger.display()
            search_types = ['distribution']#, 'weak-spot']
            for search_type in search_types:
                model = Model('cifar10')
                BD_model = BD(model, trigger, attack_param)
                BD_model.try_load(False)
                
                search_param = {'type': search_type, 'alpha': 0.1, 'beta': 0.9, 'target':trigger.target}
                GM_model_9 = GM(BD_model, search_param)
                GM_model_9.try_load(False)
                
                search_param = {'type': search_type, 'alpha': 0.1, 'beta': 0.8, 'target':trigger.target}
                GM_model_8 = GM(BD_model, search_param)
                GM_model_8.try_load(False)
                
                search_param = {'type': search_type, 'alpha': 0.1, 'beta': 0.5, 'target':trigger.target}
                GM_model_5 = GM(BD_model, search_param)
                GM_model_5.try_load(False)
                
                GM_models = [GM_model_9, GM_model_8, GM_model_5]
                defense_param = {'type': search_type, 'num_epochs': 10, 'poison_ratio': 0.01}
                RT_model = RT(GM_models, defense_param, True)
                print(search_type)
                RT_model.try_load(False)
                results[-1].append(RT_model.log.data)
            defend_method_compare_plot(results[-1], search_types)
        file = open('results_cifar10_51patterns_ensemble.pickle', 'wb')
        pickle.dump(results, file)
        file.close()

    @staticmethod
    # The baseline defense method based on reverse engineering. Do 10 times performance average.
    def point_defend_cifar_10(attack_param={'num_epochs':10, 'poison_ratio':0.01}):
        pattern_ids = gen_51_pattern_ids()
        triggers = [Trigger({'name':'3x3binary', 'num':pattern_id}) for pattern_id in pattern_ids]
        results = []
        for j in range(10):
            for trigger in triggers:
                results.append([])
                trigger.display()
                search_type = 'point'
                model = Model('cifar10')
                BD_model = BD(model, trigger, attack_param)
                BD_model.try_load(False)
                search_param = {'type': search_type, 'alpha': 0.1, 'beta': 0.9, 'target':trigger.target, 'round':j}
                GM_model = GM(BD_model, search_param)
                GM_model.try_load(False)
                defense_param = {'type': search_type + '_round_' + j.astype(str), 'num_epochs': 10, 'poison_ratio': 0.01}
                RT_model = RT(GM_model, defense_param)
                print(search_type)
                RT_model.try_load(True)
                results[-1].append(RT_model.log.data)
            file = open('cifar10_results_round_' + j.astype(str) +'.pickle', 'wb')
            pickle.dump(results, file)
            file.close()
            
    @staticmethod
    def point_defend_cifar_100(attack_param={'num_epochs':10, 'poison_ratio':0.01}):
        pattern_ids = range(10)
        triggers = [Trigger({'name':'3x3random', 'num':pattern_id}) for pattern_id in pattern_ids]
        results = []
        for j in np.array([1,2,3,4,5,6,7,8,9]):
            results = []
            for trigger in triggers:
                results.append([])
                trigger.display()
                search_type = 'point'
                model = Model('cifar100')
                BD_model = BD(model, trigger, attack_param)
                BD_model.try_load(False)
                search_param = {'type': search_type, 'alpha': 0.1, 'beta': 0.5, 'target':trigger.target, 'round':j}
                GM_model = GM(BD_model, search_param)
                GM_model.try_load(False)
                defense_param = {'type': search_type + '_round_' + j.astype(str), 'num_epochs': 10, 'poison_ratio': 0.01}
                RT_model = RT(GM_model, defense_param)
                print(search_type)
                RT_model.try_load(True)
                results[-1].append(RT_model.log.data)
            file = open('cifar100_results_rand_target_round_' + j.astype(str) +'.pickle', 'wb')
            pickle.dump(results, file)
            file.close()
            
    @staticmethod
    # To be continued work on imageNet
    def visualize_imagenet(attack_param={'num_epochs':0.05, 'poison_ratio':0.1}):
#         patterns = gen_3_16x16_patterns()
#         triggers = [Trigger(trigger_param=f(pattern)) for pattern in patterns]
#         BD_models = [BD(Model('imagenet'), trigger, attack_param) for trigger in triggers]
        patterns = gen_3_16x16_patterns()
        
        for pattern_name in patterns.keys():
            trigger = Trigger({'name':'16x16block', 'pattern_name':pattern_name})
            model = Model('imagenet')
            BD_model = BD(model, trigger, attack_param)
            BD_model.try_load()
            search_param = {'alpha': 0.1, 'beta': 0.9, 'target':trigger.target}
            GM_model = GM(BD_model, search_param)
            GM_model.try_load()
            print('going to sample')
            sample = GM_model.sample(type_='numpy')
            imagenet_plot(sample, pattern_name)

    @staticmethod
    # To be continued work on imageNet
    def detect_net():
        pass
    
    @staticmethod
    # To be continued work on imageNet
    def defend_imagenet():
        patterns = gen_3_16x16_patterns()
        triggers = [Trigger(trigger_param=f(pattern)) for pattern in patterns]
        BD_models = [BD(Model('imagenet'), trigger, attack_param) for trigger in triggers]
