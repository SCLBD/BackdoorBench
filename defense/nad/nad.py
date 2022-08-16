'''
This file is modified based on the following source:
link : https://github.com/bboylyg/NAD/.
The defense method is called nad.

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. add some addtional backbone such as resnet18 and vgg19
    7. the method to get the activation of model
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. nad defense:
        a. create student models, set training parameters and determine loss functions
        b. train the student model use the teacher model with the activation of model and result
    4. test the result and get ASR, ACC, RC 
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

from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random 
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
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
    # set the basic parameter
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
    parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr') 

    parser.add_argument('--attack', type=str)
    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)
    parser.add_argument('--trigger_type', type=str, help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--random_seed', type=int, help='random seed')
    parser.add_argument('--index', type=str, help='index of clean data')
    parser.add_argument('--result_file', type=str, help='the location of result')
    parser.add_argument('--yaml_path', type=str, default="./config/defense/nad/config.yaml", help='the path of yaml')

    #set the parameter for the nad defense
    parser.add_argument('--te_epochs', type=int)
    parser.add_argument('--momentum', type=float, help='momentum')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--ratio', type=float, help='ratio of training data')
    parser.add_argument('--beta1', type=int, help='beta of low layer')
    parser.add_argument('--beta2', type=int, help='beta of middle layer')
    parser.add_argument('--beta3', type=int, help='beta of high layer')
    parser.add_argument('--p', type=float, help='power for AT')
    
    arg = parser.parse_args()

    print(arg)
    return arg

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 2:
        lr = lr
    elif epoch < 20:
        lr = 0.01
    elif epoch < 30:
        lr = 0.0001
    else:
        lr = 0.0001
    logging.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

def train_step(args, trainloader, nets, optimizer, scheduler, criterions, epoch):
    '''train the student model with regard to the teacher model and some clean train data for each step
    args:
        Contains default parameters
    trainloader:
        the dataloader of some clean train data
    nets:
        the student model and the teacher model
    optimizer:
        optimizer during the train process
    scheduler:
        scheduler during the train process
    criterion:
        criterion during the train process
    epoch:
        current epoch
    '''
    adjust_learning_rate(optimizer, epoch, args.lr)
    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']

    snet.train()
    snet.to(args.device)

    total_clean = 0
    total_clean_correct = 0
    train_loss = 0

    batch_loss = []
    for idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        if args.model == 'preactresnet18':
            outputs_s = snet(inputs)
            features_out_3 = list(snet.children())[:-1]  # 去掉全连接层
            modelout_3 = nn.Sequential(*features_out_3)
            modelout_3.to(args.device)
            activation3_s = modelout_3(inputs)
            # activation3_s = activation3_s.view(activation3_s.size(0), -1)
            features_out_2 = list(snet.children())[:-2]  # 去掉全连接层
            modelout_2 = nn.Sequential(*features_out_2)
            modelout_2.to(args.device)
            activation2_s = modelout_2(inputs)
            # activation2_s = activation2_s.view(activation2_s.size(0), -1)
            features_out_1 = list(snet.children())[:-3]  # 去掉全连接层
            modelout_1 = nn.Sequential(*features_out_1)
            modelout_1.to(args.device)
            activation1_s = modelout_1(inputs)
            # activation1_s = activation1_s.view(activation1_s.size(0), -1)
            
            features_out_3 = list(tnet.children())[:-1]  # 去掉全连接层
            modelout_3 = nn.Sequential(*features_out_3)
            modelout_3.to(args.device)
            activation3_t = modelout_3(inputs)
            # activation3_t = activation3_t.view(activation3_t.size(0), -1)
            features_out_2 = list(tnet.children())[:-2]  # 去掉全连接层
            modelout_2 = nn.Sequential(*features_out_2)
            modelout_2.to(args.device)
            activation2_t = modelout_2(inputs)
            # activation2_t = activation2_t.view(activation2_t.size(0), -1)
            features_out_1 = list(tnet.children())[:-3]  # 去掉全连接层
            modelout_1 = nn.Sequential(*features_out_1)
            modelout_1.to(args.device)
            activation1_t = modelout_1(inputs)
            # activation1_t = activation1_t.view(activation1_t.size(0), -1)

            # activation1_s, activation2_s, activation3_s, output_s = snet(inputs)
            # activation1_t, activation2_t, activation3_t, _ = tnet(inputs)

            cls_loss = criterionCls(outputs_s, labels)
            at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3
            at2_loss = criterionAT(activation2_s, activation2_t.detach()) * args.beta2
            at1_loss = criterionAT(activation1_s, activation1_t.detach()) * args.beta1

            at_loss = at1_loss + at2_loss + at3_loss + cls_loss

        if args.model == 'vgg19':
            outputs_s = snet(inputs)
            features_out_3 = list(snet.children())[:-1]  # 去掉全连接层
            modelout_3 = nn.Sequential(*features_out_3)
            modelout_3.to(args.device)
            activation3_s = modelout_3(inputs)
            # activation3_s = snet.features(inputs)
            # activation3_s = activation3_s.view(activation3_s.size(0), -1)

            output_t = tnet(inputs)
            features_out_3 = list(tnet.children())[:-1]  # 去掉全连接层
            modelout_3 = nn.Sequential(*features_out_3)
            modelout_3.to(args.device)
            activation3_t = modelout_3(inputs)
            # activation3_t = tnet.features(inputs)
            # activation3_t = activation3_t.view(activation3_t.size(0), -1)

            cls_loss = criterionCls(outputs_s, labels)
            at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3

            at_loss = at3_loss + cls_loss

        if args.model == 'resnet18':
            outputs_s = snet(inputs)
            features_out = list(snet.children())[:-1]
            modelout = nn.Sequential(*features_out)
            modelout.to(args.device)
            activation3_s = modelout(inputs)
            # activation3_s = features.view(features.size(0), -1)

            output_t = tnet(inputs)
            features_out = list(tnet.children())[:-1]
            modelout = nn.Sequential(*features_out)
            modelout.to(args.device)
            activation3_t = modelout(inputs)
            # activation3_t = features.view(features.size(0), -1)

            cls_loss = criterionCls(outputs_s, labels)
            at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3

            at_loss = at3_loss + cls_loss
        
        if args.model == 'mobilenet_v3_large':
            outputs_s = snet(inputs)
            features_out = list(snet.children())[:-1]
            modelout = nn.Sequential(*features_out)
            modelout.to(args.device)
            activation3_s = modelout(inputs)
            # activation3_s = features.view(features.size(0), -1)

            output_t = tnet(inputs)
            features_out = list(tnet.children())[:-1]
            modelout = nn.Sequential(*features_out)
            modelout.to(args.device)
            activation3_t = modelout(inputs)
            # activation3_t = features.view(features.size(0), -1)

            cls_loss = criterionCls(outputs_s, labels)
            at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3

            at_loss = at3_loss + cls_loss

        if args.model == 'densenet161':
            outputs_s = snet(inputs)
            features_out = list(snet.children())[:-1]
            modelout = nn.Sequential(*features_out)
            modelout.to(args.device)
            activation3_s = modelout(inputs)
            # activation3_s = features.view(features.size(0), -1)

            output_t = tnet(inputs)
            features_out = list(tnet.children())[:-1]
            modelout = nn.Sequential(*features_out)
            modelout.to(args.device)
            activation3_t = modelout(inputs)
            # activation3_t = features.view(features.size(0), -1)

            cls_loss = criterionCls(outputs_s, labels)
            at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3

            at_loss = at3_loss + cls_loss

        if args.model == 'efficientnet_b3':
            outputs_s = snet(inputs)
            features_out = list(snet.children())[:-1]
            modelout = nn.Sequential(*features_out)
            modelout.to(args.device)
            activation3_s = modelout(inputs)
            # activation3_s = features.view(features.size(0), -1)

            output_t = tnet(inputs)
            features_out = list(tnet.children())[:-1]
            modelout = nn.Sequential(*features_out)
            modelout.to(args.device)
            activation3_t = modelout(inputs)
            # activation3_t = features.view(features.size(0), -1)

            cls_loss = criterionCls(outputs_s, labels)
            at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3

            at_loss = at3_loss + cls_loss


        batch_loss.append(at_loss.item())
        optimizer.zero_grad()
        at_loss.backward()
        optimizer.step()

        train_loss += at_loss.item()
        total_clean_correct += torch.sum(torch.argmax(outputs_s[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
    logging.info(f'Epoch{epoch}: Loss:{train_loss} Training Acc:{avg_acc_clean}({total_clean_correct}/{total_clean})')
    one_epoch_loss = sum(batch_loss)/len(batch_loss)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler.step(one_epoch_loss)
    elif args.lr_scheduler ==  'CosineAnnealingLR':
        scheduler.step()
    return train_loss / (idx + 1), avg_acc_clean


def test_epoch(args, testloader, model, criterion, epoch, word):
    '''test the student model with regard to test data for each epoch
    args:
        Contains default parameters
    testloader:
        the dataloader of clean test data or backdoor test data
    model:
        the student model
    criterion:
        criterion during the train process
    epoch:
        current epoch
    word:
        'bd' or 'clean'
    '''
    model.eval()

    total_clean, total_clean_correct, test_loss = 0, 0, 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
            total_clean += inputs.shape[0]
            avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
       
    if word == 'bd':
        logging.info(f'Test {word} ASR: {avg_acc_clean} ({total_clean_correct}/{total_clean})')
    if word == 'clean':
        logging.info(f'Test {word} ACC: {avg_acc_clean} ({total_clean_correct}/{total_clean})')

    return test_loss / (i + 1), avg_acc_clean


def nad(args, result, config):
    ### set logger
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
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

    fix_random(args.random_seed)

    ### a. create student models, set training parameters and determine loss functions
    # Load models
    logging.info('----------- Network Initialization --------------')
    teacher = generate_cls_model(args.model,args.num_classes)
    teacher.load_state_dict(result['model'])
    teacher.to(args.device)
    logging.info('finished teacher student init...')
    student = generate_cls_model(args.model,args.num_classes)
    student.load_state_dict(result['model'])
    logging.info('finished student student init...')

    teacher.eval()
    nets = {'snet': student, 'tnet': teacher}

    # initialize optimizer, scheduler
    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer)
    elif args.lr_scheduler ==  'CosineAnnealingLR':
        scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, T_max=100)

    # define loss functions
    criterionCls = nn.CrossEntropyLoss()
    criterionAT = AT(args.p)
    criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}

    logging.info('----------- DATA Initialization --------------')
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    x = result['clean_train']['x']
    y = result['clean_train']['y']
    data_all_length = len(y)
    ran_idx = choose_index(args, data_all_length) 
    log_index = os.getcwd() + args.log + 'index.txt'
    np.savetxt(log_index,ran_idx, fmt='%d')
    data_clean_train = list(zip([x[ii] for ii in ran_idx],[y[ii] for ii in ran_idx]))
    data_clean_trainset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_train,
        poison_idx=np.zeros(len(data_clean_train)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    #data_clean_trainset.subset(random.sample(range(len(data_clean_trainset)), int(len(data_clean_trainset)*args.ratio)))
    trainloader = torch.utils.data.DataLoader(data_clean_trainset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    y = result['bd_test']['y']
    data_bd_test = list(zip(x,y))
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
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_clean_test = list(zip(x,y))
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

    ### train the teacher model
    start_epoch = 0
    optimizer_ft = torch.optim.SGD(teacher.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler_ft = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer_ft)
    elif args.lr_scheduler ==  'CosineAnnealingLR':
        scheduler_ft = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer_ft, T_max=100)
    for epoch in tqdm(range(start_epoch, args.te_epochs)):
        teacher.to(args.device)
        batch_loss = []
        for i, (inputs,labels) in enumerate(trainloader):  # type: ignore
            teacher.train()
            teacher.to(args.device)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = teacher(inputs)
            loss = criterionCls(outputs, labels)
            batch_loss.append(loss.item())
            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()
            del loss, inputs, outputs
            torch.cuda.empty_cache()
        one_epoch_loss = sum(batch_loss)/len(batch_loss)
        if args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler_ft.step(one_epoch_loss)
        elif args.lr_scheduler ==  'CosineAnnealingLR':
            scheduler_ft.step()

        # evaluate on testing set
        test_loss, test_acc_cl = test_epoch(args, testloader_clean, teacher, criterionCls, epoch, 'clean')
        test_loss, test_acc_bd = test_epoch(args, testloader_bd, teacher, criterionCls, epoch, 'bd')
        # remember best precision and save checkpoint
        logging.info(f'Teacher_Epoch{epoch}: clean_acc:{test_acc_cl} asr:{test_acc_bd}')


    ### b. train the student model use the teacher model with the activation of model and result
    logging.info('----------- Train Initialization --------------')
    start_epoch = 0
    best_acc = 0
    best_asr = 0
    for epoch in tqdm(range(start_epoch, args.epochs)):
        student.to(args.device)
        train_loss, train_acc = train_step(args, trainloader, nets, optimizer, scheduler, criterions, epoch)

        # evaluate on testing set
        test_loss, test_acc_cl = test_epoch(args, testloader_clean, student, criterionCls, epoch, 'clean')
        test_loss, test_acc_bd = test_epoch(args, testloader_bd, student, criterionCls, epoch, 'bd')

        # remember best precision and save checkpoint
        if not (os.path.exists(os.getcwd() + f'{args.checkpoint_save}')):
            os.makedirs(os.getcwd() + f'{args.checkpoint_save}')
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
            f'./{args.checkpoint_save}defense_result.pt'
            )
        if epoch == 19:
            now_epoch = epoch + 1
            torch.save(
            {
                'model_name':args.model,
                'model': student.cpu().state_dict(),
                'asr': test_acc_bd,
                'acc': test_acc_cl
            },
            f'./{args.checkpoint_save}defense_result_{now_epoch}.pt'
            )

        logging.info(f'Epoch{epoch}: clean_acc:{test_acc_cl} asr:{test_acc_bd} best_acc:{best_acc} best_asr{best_asr}')
    result = {}
    result['model'] = nets['snet']
    return result



if __name__ == '__main__':
    ### 1. basic setting: args 
    args = get_args()
    with open(args.yaml_path, 'r') as stream: 
        config = yaml.safe_load(stream) 
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    
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

    ### 2. attack result(model, train data, test data)
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    
    ### 3. nad defense
    result_defense = nad(args,result,config)

    ### 4. test the result and get ASR, ACC, RC
    result_defense['model'].eval()
    result_defense['model'].to(args.device)
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    y = result['bd_test']['y']
    data_bd_test = list(zip(x,y))
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
        asr_acc += torch.sum(pre_label == labels)
    asr_acc = asr_acc/len(data_bd_test)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_clean_test = list(zip(x,y))
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
        clean_acc += torch.sum(pre_label == labels)
    clean_acc = clean_acc/len(data_clean_test)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    robust_acc = -1
    if 'original_targets' in result['bd_test']:
        y_ori = result['bd_test']['original_targets']
        if y_ori is not None:
            if len(y_ori) != len(x):
                y_idx = result['bd_test']['original_index']
                y = y_ori[y_idx]
            else :
                y = y_ori
            data_bd_test = list(zip(x,y))
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
                robust_acc += torch.sum(pre_label == labels)
            robust_acc = robust_acc/len(data_bd_test)

    if not (os.path.exists(os.getcwd() + f'{save_path}/nad/')):
        os.makedirs(os.getcwd() + f'{save_path}/nad/')
    torch.save(
    {
        'model_name':args.model,
        'model': result_defense['model'].cpu().state_dict(),
        'asr': asr_acc,
        'acc': clean_acc,
        'ra': robust_acc
    },
    os.getcwd() + f'{save_path}/nad/defense_result.pt'
    )