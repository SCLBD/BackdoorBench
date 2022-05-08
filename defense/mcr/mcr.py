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
from defense.mcr.curve_models import curves
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random 
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
from defense.mcr import curve_models

def get_curve_class(args):
    if args.model_name == 'preactresnet18':
        net = getattr(curve_models, args.model)
    elif args.model_name == 'vgg19':
        net = getattr(curve_models, 'VGG19BN')
    elif args.model_name == 'densenet161':
        net = getattr(curve_models, args.model)
    elif args.model_name == 'mobilenet_v3_large':
        net = getattr(curve_models, args.model)
    elif args.model_name == 'efficientnet_b3':
        net = getattr(curve_models, args.model)
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')

    return net

def get_mcr_dataset(args, result, shuffle_train=True):
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_all_length = round(len(y)/2)
    ran_idx = choose_index(args, data_all_length) 
    log_index = os.getcwd() + args.log + 'index.txt'
    np.savetxt(log_index,ran_idx, fmt='%d')

    aa = np.array(range(len(x)))
    bb = np.random.shuffle(aa)

    x_train_part = [x[ii] for ii in aa[0:data_all_length][ran_idx]] 
    y_train_part = [y[ii] for ii in aa[0:data_all_length][ran_idx]] 

    i1 = data_all_length // len(ran_idx)
    i2 = data_all_length % len(ran_idx)

    x_train_half = []
    y_train_half = []

    for ii in range(i1):
        x_train_half += x_train_part
        y_train_half += y_train_part

    if i2 != 0:
        x_train_half += [x_train_part[i] for i in range(data_all_length - i1 * len(ran_idx))]
        y_train_half += [y_train_part[i] for i in range(data_all_length - i1 * len(ran_idx))]

    x_train = x_train_half + x_train_half
    y_train = y_train_half + y_train_half

    x_test = [x[ii] for ii in aa[data_all_length:2*data_all_length]] + [x[ii] for ii in aa[data_all_length:2*data_all_length]]
    y_test = [y[ii] for ii in aa[data_all_length:2*data_all_length]] + [y[ii] for ii in aa[data_all_length:2*data_all_length]]

    train_ori = list(zip(x_train, y_train))
    test_ori = list(zip(x_test, y_test))

    tran_train = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    data_trainset = prepro_cls_DatasetBD(
        full_dataset_without_transform=train_ori,
        poison_idx=np.zeros(len(train_ori)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran_train,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )

    tran_test = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    data_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=test_ori,
        poison_idx=np.zeros(len(test_ori)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran_test,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )

    x = result['bd_test']['x']
    y = result['bd_test']['y']
    data_bd_test = list(zip(x,y))
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran_test,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )


    return {
               'train': torch.utils.data.DataLoader(
                   data_trainset,
                   batch_size=args.batch_size,
                   shuffle=shuffle_train,
                   num_workers=args.num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   data_testset,
                   batch_size=args.batch_size,
                   shuffle=False,
                   num_workers=args.num_workers,
                   pin_memory=True
               ),
               'test_bd': torch.utils.data.DataLoader(
                   data_bd_testset,
                   batch_size=args.batch_size,
                   shuffle=False,
                   num_workers=args.num_workers,
                   pin_memory=True
               ),
               'testset': data_testset,
           }

def train(args, train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.to(args.device)
        target = target.to(args.device)

        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    aa = len(train_loader.dataset)
    bb = loss_sum
    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }

def test(args, test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    for input, target in test_loader:
        input = input.to(args.device)
        target = target.to(args.device)

        output = model(input, **kwargs)
        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

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

    parser.add_argument('--attack', type=str)
    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)
    parser.add_argument('--trigger_type', type=str, help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--seed', type=str, help='random seed')
    parser.add_argument('--index', type=str, help='index of clean data')
    parser.add_argument('--result_file', type=str, help='the location of result')

    #set the parameter for the mcr defense
    parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
    parser.add_argument('--curve', type=str, default='Bezier', metavar='CURVE',
                    help='curve type to use (default: None)')
    parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                        help='number of curve bends (default: 3)')
    parser.add_argument('--init_start', type=str, default='Res_single_true_10_same1/checkpoint-100.pt', metavar='CKPT',
                        help='checkpoint to init start point (default: None)')
    parser.add_argument('--fix_start', dest='fix_start', action='store_true', default=True,
                        help='fix start point (default: off)')
    parser.add_argument('--init_end', type=str, default='Res_single_true_10_same2/checkpoint-100.pt', metavar='CKPT',
                        help='checkpoint to init end point (default: None)')
    parser.add_argument('--fix_end', dest='fix_end', action='store_true', default=True,
                        help='fix end point (default: off)')
    parser.set_defaults(init_linear=True)
    parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                        help='turns off linear initialization of intermediate points (default: on)')
    
    arg = parser.parse_args()

    print(arg)
    return arg



def mcr(args, result, config):
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

    fix_random(args.seed)

    # os.makedirs(args.dir, exist_ok=True)
    # with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    #     f.write(' '.join(sys.argv))
    #     f.write('\n')

    # torch.backends.cudnn.benchmark = True
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    loaders = get_mcr_dataset(args, result)

    architecture = get_curve_class(args)

    # if args.curve is None:
    #     model = architecture.base(num_classes=arg.num_classes, **architecture.kwargs)
    # else:
    assert(args.curve is not None)
    curve = getattr(curves, args.curve)
    model = curves.CurveNet(
        args.num_classes,
        curve,
        architecture.curve,
        args.num_bends,
        args.fix_start,
        args.fix_end,
        architecture_kwargs=architecture.kwargs,
    )
    base_model = None
    if args.resume is None:
        # for path, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
        #     if path is not None:
        #         if base_model is None:
                    
        #         checkpoint = torch.load(path)
        #         print('Loading %s as point #%d' % (path, k))
        #         base_model.load_state_dict(checkpoint['model_state'])
        #         model.import_base_parameters(base_model, k)
        base_model = architecture.base(num_classes=args.num_classes, **architecture.kwargs)
        k = 0
        base_model.load_state_dict(result['model'])
        model.import_base_parameters(base_model, k)

        k = args.num_bends - 1
        ##写个训练 待办

        model.import_base_parameters(base_model, k)

        k = 0
        if args.init_linear:
            print('Linear initialization.')
            model.init_linear()
    model.to(args.device)


    def learning_rate_schedule(base_lr, epoch, total_epochs):
        alpha = epoch / total_epochs
        if alpha <= 0.5:
            factor = 1.0
        elif alpha <= 0.9:
            factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
        else:
            factor = 0.01
        return factor * base_lr


    criterion = F.cross_entropy
    regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)

    #optimizer = torch.optim.Adam(
    #  filter(lambda param: param.requires_grad, model.parameters()),
    #   lr=args.lr,
    # momentum=args.momentum,
    #    weight_decay=args.wd if args.curve is None else 0.0
    #)

    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd if args.curve is None else 0.0
    )

    start_epoch = 1
    if args.resume is not None:
        print('Resume training from %s' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    # columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']

    # utils.save_checkpoint(
    #     args.dir,
    #     start_epoch - 1,
    #     model_state=model.state_dict(),
    #     optimizer_state=optimizer.state_dict()
    # )

    # has_bn = utils.check_bn(model)
    test_res = {'loss': None, 'accuracy': None, 'nll': None}
    test_bd_res = {'loss': None, 'accuracy': None, 'nll': None}

    logging.info('Epoch \t lr \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    best_acc = 0
    for epoch in range(start_epoch, args.epochs + 1):
        # time_ep = time.time()
        model.to(args.device)

        lr = learning_rate_schedule(args.lr, epoch, args.epochs)
        # lr = args.lr
        adjust_learning_rate(optimizer, lr)

        train_res = train(args, loaders['train'], model, optimizer, criterion, regularizer)
        # if args.curve is None or not has_bn:
        test_res = test(args, loaders['test'], model, criterion, regularizer, t=0.3)
        test_bd_res = test(args, loaders['test_bd'], model, criterion, regularizer, t=0.3)

        # if epoch % args.save_freq == 0:
        #     utils.save_checkpoint(
        #         args.dir,
        #         epoch,
        #         model_state=model.state_dict(),
        #         optimizer_state=optimizer.state_dict()
        #     )

        # time_ep = time.time() - time_ep
        # values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['nll'],
        #         test_res['accuracy'], time_ep]

        logging.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            epoch, lr, train_res['loss'], train_res['accuracy'],test_bd_res['nll'], test_bd_res['accuracy'], test_res['nll'],
                test_res['accuracy']))

        # table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        # if epoch % 40 == 1 or epoch == start_epoch:
        #     table = table.split('\n')
        #     table = '\n'.join([table[1]] + table)
        # else:
        #     table = table.split('\n')[2]
        # print(table)

        if best_acc < test_res['accuracy']:
            best_acc = test_res['accuracy']
            best_asr = test_bd_res['accuracy']
            torch.save(
            {
                'model_name':args.model,
                'model': model.cpu().state_dict(),
                'asr': best_asr,
                'acc': best_acc,
                'curve': args.curve,
                'optimizer_state': optimizer.state_dict()
            },
            f'./{args.save_path}/mcr/ckpt_best/defense_result.pt'
            )

    # if args.epochs % args.save_freq != 0:
    #     utils.save_checkpoint(
    #         args.dir,
    #         args.epochs,
    #         model_state=model.state_dict(),
    #         optimizer_state=optimizer.state_dict()
    #     )





    result = {}
    result['model'] = model
    return result



if __name__ == '__main__':
    ### 1. basic setting: args 
    args = get_args()
    with open("./defense/mcr/config.yaml", 'r') as stream: 
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
    elif args.dataset == "cifar100":
        args.num_classes = 100
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
    
    save_path = '/record/' + args.result_file
    if args.checkpoint_save is None:
        args.checkpoint_save = save_path + '/record/defence/mcr/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = save_path + '/saved/mcr/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log)  
    args.save_path = save_path

    ### 2. attack result(model, train data, test data)
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    
    ### 3. mcr defense
    result_defense = mcr(args,result,config)

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

    if not (os.path.exists(os.getcwd() + f'{save_path}/mcr/')):
        os.makedirs(os.getcwd() + f'{save_path}/mcr/')
    torch.save(
    {
        'model_name':args.model,
        'model': result_defense['model'].cpu().state_dict(),
        'asr': asr_acc,
        'acc': clean_acc,
        'ra': robust_acc
    },
    os.getcwd() + f'{save_path}/mcr/defense_result.pt'
    )