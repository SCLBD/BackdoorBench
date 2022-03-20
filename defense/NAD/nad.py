'''
@article{li2021neural,
  title={Neural attention distillation: Erasing backdoor triggers from deep neural networks},
  author={Li, Yige and Lyu, Xixiang and Koren, Nodens and Lyu, Lingjuan and Li, Bo and Ma, Xingjun},
  journal={arXiv preprint arXiv:2101.05930},
  year={2021}
}

code : https://github.com/bboylyg/NAD/
'''
import logging
import random
import time

from calendar import c
from unittest.mock import sentinel
from torchvision import transforms

import torch
import logging
import argparse
import sys
import os

import tqdm
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
from tqdm import tqdm
import numpy as np
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
AT with sum of absolute values with power p
code from: https://github.com/AberHu/Knowledge-Distillation-Zoo
'''
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

    #####nad
    parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--ratio', type=float, default=0.05, help='ratio of training data')
    parser.add_argument('--beta1', type=int, default=500, help='beta of low layer')
    parser.add_argument('--beta2', type=int, default=1000, help='beta of middle layer')
    parser.add_argument('--beta3', type=int, default=1000, help='beta of high layer')
    parser.add_argument('--p', type=float, default=2.0, help='power for AT')
    parser.add_argument('--threshold_clean', type=float, default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float, default=90.0, help='threshold of save weight')
    
    arg = parser.parse_args()

    print(arg)
    return arg


class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am

def train_step(arg, trainloader, nets, optimizer, scheduler, criterions, epoch):
    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']

    snet.train()
    snet.to(arg.device)

    total_clean = 0
    total_clean_correct = 0
    train_loss = 0

    for idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)

        if arg.model == 'preactresnet18':
            outputs_s = snet(inputs)
            features_out_3 = list(snet.children())[:-1]  # 去掉全连接层
            modelout_3 = nn.Sequential(*features_out_3).to(arg.device)
            activation3_s = modelout_3(inputs)
            # activation3_s = activation3_s.view(activation3_s.size(0), -1)
            features_out_2 = list(snet.children())[:-2]  # 去掉全连接层
            modelout_2 = nn.Sequential(*features_out_2).to(arg.device)
            activation2_s = modelout_2(inputs)
            # activation2_s = activation2_s.view(activation2_s.size(0), -1)
            features_out_1 = list(snet.children())[:-3]  # 去掉全连接层
            modelout_1 = nn.Sequential(*features_out_1).to(arg.device)
            activation1_s = modelout_1(inputs)
            # activation1_s = activation1_s.view(activation1_s.size(0), -1)
            
            features_out_3 = list(tnet.children())[:-1]  # 去掉全连接层
            modelout_3 = nn.Sequential(*features_out_3).to(arg.device)
            activation3_t = modelout_3(inputs)
            # activation3_t = activation3_t.view(activation3_t.size(0), -1)
            features_out_2 = list(tnet.children())[:-2]  # 去掉全连接层
            modelout_2 = nn.Sequential(*features_out_2).to(arg.device)
            activation2_t = modelout_2(inputs)
            # activation2_t = activation2_t.view(activation2_t.size(0), -1)
            features_out_1 = list(tnet.children())[:-3]  # 去掉全连接层
            modelout_1 = nn.Sequential(*features_out_1).to(arg.device)
            activation1_t = modelout_1(inputs)
            # activation1_t = activation1_t.view(activation1_t.size(0), -1)

            # activation1_s, activation2_s, activation3_s, output_s = snet(inputs)
            # activation1_t, activation2_t, activation3_t, _ = tnet(inputs)

            cls_loss = criterionCls(outputs_s, labels)
            at3_loss = criterionAT(activation3_s, activation3_t).detach() * arg.beta3
            at2_loss = criterionAT(activation2_s, activation2_t).detach() * arg.beta2
            at1_loss = criterionAT(activation1_s, activation1_t).detach() * arg.beta1

            at_loss = at1_loss + at2_loss + at3_loss + cls_loss

        if arg.model == 'vgg19':
            outputs_s = snet(inputs)
            features_out_3 = list(snet.children())[:-1]  # 去掉全连接层
            modelout_3 = nn.Sequential(*features_out_3).to(arg.device)
            activation3_s = modelout_3(inputs)
            # activation3_s = snet.features(inputs)
            # activation3_s = activation3_s.view(activation3_s.size(0), -1)

            output_t = tnet(inputs)
            features_out_3 = list(tnet.children())[:-1]  # 去掉全连接层
            modelout_3 = nn.Sequential(*features_out_3).to(arg.device)
            activation3_t = modelout_3(inputs)
            # activation3_t = tnet.features(inputs)
            # activation3_t = activation3_t.view(activation3_t.size(0), -1)

            cls_loss = criterionCls(outputs_s, labels)
            at3_loss = criterionAT(activation3_s, activation3_t).detach() * arg.beta3

            at_loss = at3_loss + cls_loss

        if arg.model == 'resnet18':
            outputs_s = snet(inputs)
            features_out = list(snet.children())[:-1]
            modelout = nn.Sequential(*features_out).to(arg.device)
            activation3_s = modelout(inputs)
            # activation3_s = features.view(features.size(0), -1)

            output_t = tnet(inputs)
            features_out = list(tnet.children())[:-1]
            modelout = nn.Sequential(*features_out).to(arg.device)
            activation3_t = modelout(inputs)
            # activation3_t = features.view(features.size(0), -1)

            cls_loss = criterionCls(outputs_s, labels)
            at3_loss = criterionAT(activation3_s, activation3_t).detach() * arg.beta3

            at_loss = at3_loss + cls_loss

        optimizer.zero_grad()
        at_loss.backward()
        optimizer.step()

        train_loss += at_loss.item()
        total_clean_correct += torch.sum(torch.argmax(outputs_s[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
        # logging.info(i, len(trainloader), 'Epoch: %d | Loss: %.3f | Training Acc: %.3f%% (%d/%d)' % (epoch, train_loss/(idx+1), avg_acc_clean, total_clean_correct, total_clean))
   
    logging.info(f'Epoch{epoch}: Loss:{train_loss} Training Acc:{avg_acc_clean}({total_clean_correct}/{total_clean})')
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
        # if word == 'bd':
        #     logging.info(i, len(testloader), 'Test %s ASR: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
        # if word == 'clean':
        #     logging.info(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))

    if word == 'bd':
        logging.info(f'Test {word} ASR: {avg_acc_clean} ({total_clean_correct}/{total_clean})')
    if word == 'clean':
        logging.info(f'Test {word} ACC: {avg_acc_clean} ({total_clean_correct}/{total_clean})')

    return test_loss / (i + 1), avg_acc_clean


def nad(arg, result, config):
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    # logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    if args.log is not None and args.log != '':
        fileHandler = logging.FileHandler(os.getcwd() + args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    else:
        fileHandler = logging.FileHandler('./log' + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))
    # Load models
    print('----------- Network Initialization --------------')
    teacher = generate_cls_model(args.model,args.num_classes)
    teacher.load_state_dict(result['model'])
    teacher.to(args.device)
    # model_ascent = get_network(opt)
    # model_ascent.to(args.device)
    # teacher = get_network(arg)
    # checkpoint = torch.load(arg.checkpoint_load_teacher)
    # teacher.load_state_dict(checkpoint['model'])
    print('finished teacher student init...')
    student = generate_cls_model(args.model,args.num_classes)
    # student = get_network(arg)
    # student_checkpoint = torch.load(arg.checkpoint_load_student)
    # student.load_state_dict(student_checkpoint['model'])
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
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    x = torch.tensor(nCHW_to_nHWC(result['clean_train']['x'].detach().numpy()))
    y = result['clean_train']['y']
    data_clean_train = torch.utils.data.TensorDataset(x,y)
    data_clean_trainset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_train,
        poison_idx=np.zeros(len(data_clean_train)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_clean_trainset.subset(random.sample(range(len(data_clean_trainset)), int(len(data_clean_trainset)*arg.ratio)))
    trainloader = torch.utils.data.DataLoader(data_clean_trainset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
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
    testloader_bd = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

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
    testloader_clean = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    print('----------- Train Initialization --------------')
    start_epoch = 0
    best_acc = 0
    best_asr = 0
    for epoch in tqdm(range(start_epoch, arg.epochs)):
        student.to(args.device)
        train_loss, train_acc = train_step(arg, trainloader, nets, optimizer, scheduler, criterions, epoch)

        # evaluate on testing set
        test_loss, test_acc_cl = test_epoch(arg, testloader_clean, student, criterionCls, epoch, 'clean')
        test_loss, test_acc_bd = test_epoch(arg, testloader_bd, student, criterionCls, epoch, 'bd')

        # remember best precision and save checkpoint
        if not (os.path.exists(os.getcwd() + f'{args.save_path}/nad/ckpt_best/')):
            os.makedirs(os.getcwd() + f'{args.save_path}/nad/ckpt_best/')
        if best_acc < test_acc_cl:
            best_acc = test_acc_cl
            best_asr = test_acc_bd
            torch.save(
            {
                'model_name':args.model,
                'model': student.cpu().state_dict(),
                'asr': test_acc_bd,
                'acc': test_acc_cl
            },
            f'./{args.save_path}/nad/ckpt_best/defense_result.pt'
            )
        logging.info(f'Epoch{epoch}: clean_acc:{test_acc_cl} asr:{test_acc_bd} best_acc:{best_acc} best_asr{best_asr}')
    result = {}
    result['model'] = nets['snet']
    return result



if __name__ == '__main__':
    
    args = get_args()
    with open("./defense/NAD/config/config.yaml", 'r') as stream: 
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
        args.checkpoint_save = save_path + '/record/defence/nad/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = save_path + '/saved/nad/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log)  
    args.save_path = save_path
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    
    if args.save_path is not None:
        print("Continue training...")
        result_defense = nad(args,result,config)
        result_defense['model'].to(args.device)
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
            y = result['bd_test']['original_targets']
            if y is not None:
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
  
        if not (os.path.exists(os.getcwd() + f'{save_path}/nad/')):
            os.makedirs(os.getcwd() + f'{save_path}/nad/')
        torch.save(
        {
            'model_name':args.model,
            'model': result_defense['model'].cpu().state_dict(),
            'asr': asr_acc,
            'acc': clean_acc,
            'rc': robust_acc
        },
        os.getcwd() + f'{save_path}/nad/defense_result.pt'
        )
    else:
        print("There is no target model")