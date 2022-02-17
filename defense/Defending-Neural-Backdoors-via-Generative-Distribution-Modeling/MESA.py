# MIT License

# Copyright (c) 2019 yukun yang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This module implements methods performing poisoning detection based on activations clustering.
| Paper link: https://arxiv.org/abs/1811.03728
"""

from functools import lru_cache
import torch
import torchvision
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from utils.dataloader_bd import get_dataset, get_dataset_test, get_dataset_test_no, get_transform

from utils.model import Generator, Mine

import argparse
import yaml

sys.path.append(os.getcwd())
from utils.network import get_network

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda', help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str, default=None)
    parser.add_argument('--checkpoint_save', type=str, default=None)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument("--data_root", type=str, default='dataset/')

    parser.add_argument('--dataset', type=str, default='cifar10', help='mnist, cifar10, gtsrb, celeba, tiny') 
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument("--num_workers", type=float, default=4)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--attack', type=str, default='badnet')
    parser.add_argument('--poison_rate', type=float, default=0.1)
    parser.add_argument('--target_type', type=str, default='all2one', help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--trigger_type', type=str, default='squareTrigger', help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    parser.add_argument('--ib', default=False)
    parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'], default='ResNet_4x')
    parser.add_argument('--k', type=int, default=12, help='Number of chunks.')
    parser.add_argument('--num_samples', type=int, default=4,
                    help='Number of samples used for estimating expectation over p(t|x).')
    parser.add_argument('--beta', type=float, default=0.001, help='beta in objective J = I(y,t) - beta * I(x,t).')


    #####MESA
    parser.add_argument('--beta', type=int, default=None, help='beta')
    parser.add_argument('--alpha', type=int, default=None, help='alpha')
    parser.add_argument('--ensemble', default=False, help='alpha')
    

    arg = parser.parse_args()

    
    print(arg)
    return arg

def main():
    args = get_args()
    model = get_network(args)
    test_dataset = get_dataset_test_no(args)
    if args.load_target is not None:
        checkpoint = torch.load(arg.checkpoint_load)
        print("Continue training...")
        model.load_state_dict(checkpoint['model'])
        result = MESA(args,model,test_dataset,args.ensemble)

        ####还没改
        if result['is_bd']:
            print('The target model is a backdoor model with score {}'.format(result['score']))
        else:
            print('The target model is not a backdoor model with score {}'.format(result['score']))
    else:
        print("There is no target model")

def MESA(args,model,test_dataset,ensemble=False):
    with open("./config/config.yaml", 'r') as stream: 
        config = yaml.safe_load(stream) 

    config.updata({k:v for k,v in args.__dict__.items() if v is not None})
    beta = config['beta']
    alpha = config['alpha']

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

    args.checkpoint_save = 'saved/checkpoint/checkpoint_' + args.dataset + '.tar'
    args.log = 'saved/log/log_' + args.dataset + '.txt'
    args.data_root = args.data_root + args.dataset    
    if not os.path.isdir(args.data_root):
        os.makedirs(args.data_root)

    assert (args.dataset in ['cifar10','cifar100','tiny'])
    if args.dataset == 'cifar10':
        in_size = 64
        out_size = 27
        hidden_size = 512
        num_epochs = 10
        trigger_h = 3
        trigger_w = 3
    elif args.dataset == 'cifar100':
        in_size = 64
        out_size = 27
        hidden_size = 2048
        num_epochs = 10
        trigger_h = 3
        trigger_w = 3
    elif args.dataset =='tiny':
        in_size = 128
        out_size = 3*16*16
        hidden_size = 128
        num_epochs = 30
        trigger_h = 16
        trigger_w = 16
    rt_num_epochs = 10

    length = len(test_dataset)
    pro = 0.8
    train_len = int(length*pro)
    train_sp = []
    test_sp = []
    indices = np.random.RandomState(0).permutation(length).tolist() # total_length must be fixed
    for i in range(train_len):
        train_sp.append(test_dataset[indices[i]])
    for i in range(length-train_len):
        test_sp.append(test_dataset[indices[i+train_len]])
    train_set = get_dataset(train_sp)
    test_set = get_dataset(test_sp)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
   
    lr = 0.0002
    GM_param = {'in_size': in_size, 'out_size': out_size, 'hidden_size': hidden_size}
    Gene = Generator(args.dataset, GM_param).cuda()
    Mi = Mine(args.dataset, GM_param).cuda()

    Ge_opt = torch.optim.Adam(Gene.parameters(), lr=lr, betas=(0.5, 0.999))
    Mi_opt = torch.optim.Adam(Mi.parameters(), lr=lr, betas=(0.5, 0.999))

    Gene.train()
    Mi.train()
    model.eval()
    for epoch in range(num_epochs):
        log_hinge, log_softmax, log_entropy, log_count, non_target_total, non_target_correct = 0, 0, 0, 0, 0, 0
        for idx, (data, label) in enumerate(train_loader):
            # train Generator
            z = Gene.gen_noise(args.batch_size).cuda()
            z1 = Gene.gen_noise(args.batch_size).cuda()
            trigger_noise = torch.randn(args.batch_size, Gene.out_size).cuda() / 10
            trigger = Gene(z)
            data = data.clone().cuda()
            label = label.clone().cuda() 
            tri_all = (trigger + trigger_noise).view(-1, 3, trigger_h, trigger_w)
            x = int(np.random.rand() * (args.input_height - trigger_h))
            y = int(np.random.rand() * (args.input_width - trigger_w))
            data[:,:,x:x+trigger_h,y:y+trigger_w] = tri_all
            transform = get_transform(args, False)
            data = transform(data)
            logit = model(data)

            hinge_loss = torch.mean(torch.min(F.softmax(logit, dim=1)[:,args.target_label], beta))
            entropy = Mi.mi(z, z1, trigger)
            G_loss = -hinge_loss - alpha * entropy
            Ge_opt.zero_grad()
            G_loss.backward()
            Ge_opt.step()

            # train Mine
            z = torch.rand(args.batch_size, Gene.in_size).cuda()
            z1 = torch.rand(args.batch_size, Gene.in_size).cuda()
            trigger = Gene(z)

            M_loss = -Mi.mi_loss(z, z1, trigger)
            Mi_opt.zero_grad()
            M_loss.backward()
            Mi_opt.step()

            log_hinge += hinge_loss.item()
            log_entropy += entropy.item()
            log_softmax += (F.softmax(logit, dim=1)[:,args.target_label]).mean().item()
            log_count += 1
            
            predicted = torch.argmax(logit, dim=1)
            non_target_total += torch.sum(1 - label.eq(args.target_label)).item()
            non_target_correct += (predicted.eq(args.target_label) * (1 - label.eq(args.target_label))).sum().item()


        #self.log.append({'Hinge loss':     log_hinge   / log_count,
        #                    'Entropy':        log_entropy / log_count,
        #                    'Softmax output': log_softmax / log_count,
        #                    'ASR':            non_target_correct / non_target_total})
        print("ASR:", non_target_correct / non_target_total)

    if ensemble == False:
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
        
        for epoch in range(math.ceil(rt_num_epochs)):
            train_loss, correct, total = 0, 0, 0
            for batch_idx, (data, label) in enumerate(train_loader):
                data, label = data.clone().cuda(), label.clone().cuda()

                trigger = sample(Gene,args.cuda) # cuda
                trigger = transform(trigger.view(-1, 3, trigger.h, trigger.w),
                                    dataset_stats(BD_model.model.dataset_name))
                target = trigger.target
                # clone label to prevent poisoning
                poison_(trigger, target, data, label.clone(), poison_ratio, 'random')

                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()

            test(test_data, 1, is_attack=False)
            test(test_data, 1, is_attack=True)
            net.train()
    else:
        net.train()
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

            log.append({'Train Accuracy on poisoned data': correct / total})

            test(test_data, 1, is_attack=False)
            test(test_data, 1, is_attack=True)
            net.train()


    result = {}

    return result

def sample(Gene ,cuda = True):
    

    Gene.eval()
    z = Gene.gen_noise(self.bs).cuda()
    x = Gene(z)
    Gene.train()
        
    if cuda:
        return x
    else: # type = numpy
        return x.cpu().detach().numpy()

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


if __name__ == '__main__':
    main()