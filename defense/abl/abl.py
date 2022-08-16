'''
This file is modified based on the following source:
link : https://github.com/bboylyg/ABL.
The defense method is called abl.

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. abl defense:
        a. pre-train model
        b. isolate the special data(loss is low) as backdoor data
        c. unlearn the backdoor data and learn the remaining data
    4. test the result and get ASR, ACC, RC 
'''

import logging
from pprint import pformat
import time

from calendar import c
import torchvision.transforms as transforms

import torch
import logging
import argparse
import sys
import os

from tqdm import tqdm

sys.path.append('../')
sys.path.append(os.getcwd())
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import *
from utils.save_load_attack import load_attack_result
sys.path.append(os.getcwd())
import yaml
from pprint import pprint, pformat


def get_args():
    #set the basic parameter
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
    parser.add_argument('--random_seed', type=int, help='random seed')
    parser.add_argument('--index', type=str, help='index of clean data')
    parser.add_argument('--result_file', type=str, help='the location of result')

    parser.add_argument('--yaml_path', type=str, default="./config/defense/abl/config.yaml", help='the name of yaml')
    
    #set the parameter for the abl defense
    parser.add_argument('--tuning_epochs', type=int, help='number of tune epochs to run')
    parser.add_argument('--finetuning_ascent_model', type=str, help='whether finetuning model')
    parser.add_argument('--finetuning_epochs', type=int, help='number of finetuning epochs to run')
    parser.add_argument('--unlearning_epochs', type=int, help='number of unlearning epochs to run')
    parser.add_argument('--lr_finetuning_init', type=float, help='initial finetuning learning rate')
    parser.add_argument('--lr_unlearning_init', type=float, help='initial unlearning learning rate')
    parser.add_argument('--momentum', type=float, help='momentum')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--isolation_ratio', type=float, help='ratio of isolation data')
    parser.add_argument('--gradient_ascent_type', type=str, help='type of gradient ascent')
    parser.add_argument('--gamma', type=float, help='value of gamma')
    parser.add_argument('--flooding', type=float, help='value of flooding')

    parser.add_argument('--threshold_clean', type=float, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float, help='threshold of save weight')
    parser.add_argument('--interval', type=int, help='frequency of save model')

    
    arg = parser.parse_args()

    print(arg)
    return arg



def train(args, result):
    '''Pretrain the model with raw data
    args:
        Contains default parameters
    result:
        attack result(details can be found in utils)
    '''

    # Load models
    logging.info('----------- Network Initialization --------------')
    model_ascent = generate_cls_model(args.model,args.num_classes)
    model_ascent.to(args.device)
    logging.info('finished model init...')
    # initialize optimizer
    optimizer = torch.optim.SGD(model_ascent.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # define loss functions
    if args.device == 'cuda':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    logging.info('----------- Data Initialization --------------')

    # tf_compose = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    tf_compose = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_train']['x']
    y = result['bd_train']['y']
    data_set = list(zip(x,y))
    poisoned_data = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tf_compose,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    poisoned_data_loader = torch.utils.data.DataLoader(poisoned_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)    

    logging.info('----------- Train Initialization --------------')
    for epoch in range(0, args.tuning_epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train every epoch

        train_step(args, poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)

        

    return poisoned_data, model_ascent

def train_step(args, train_loader, model_ascent, optimizer, criterion, epoch):
    '''Pretrain the model with raw data for each step
    args:
        Contains default parameters
    train_loader:
        the dataloader of train data
    model_ascent:
        the initial model
    optimizer:
        optimizer during the pretrain process
    criterion:
        criterion during the pretrain process
    epoch:
        current epoch
    '''
    losses = 0
    size = 0

    model_ascent.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        
        img = img.to(args.device)
        target = target.to(args.device)

        if args.gradient_ascent_type == 'LGA':
            output = model_ascent(img)
            loss = criterion(output, target)
            # add Local Gradient Ascent(LGA) loss
            loss_ascent = torch.sign(loss - args.gamma) * loss

        elif args.gradient_ascent_type == 'Flooding':
            output = model_ascent(img)
            # output = student(img)
            loss = criterion(output, target)
            # add flooding loss
            loss_ascent = (loss - args.flooding).abs() + args.flooding

        else:
            raise NotImplementedError

        losses += loss_ascent * img.size(0)
        size += img.size(0)
        optimizer.zero_grad()
        loss_ascent.backward()
        optimizer.step()
   
        logging.info('Epoch[{0}]:[{1:03}/{2:03}]'
                'Loss:{losses:.4f}({losses_avg:.4f})'.format(epoch, idx, len(train_loader), losses=losses, losses_avg=losses/size))

def adjust_learning_rate(optimizer, epoch, args):
    '''set learning rate during the process of pretraining model 
    optimizer:
        optimizer during the pretrain process
    epoch:
        current epoch
    args:
        Contains default parameters
    '''
    if epoch < args.tuning_epochs:
        lr = args.lr
    else:
        lr = 0.01
    logging.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_loss_value(args, poisoned_data, model_ascent):
    '''Calculate loss value per example
    args:
        Contains default parameters
    poisoned_data:
        the train dataset which contains backdoor data
    model_ascent:
        the model after the process of pretrain
    '''
    # Define loss function
    if args.device == 'cuda':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model_ascent.eval()
    losses_record = []

    example_data_loader = torch.utils.data.DataLoader(dataset=poisoned_data,
                                        batch_size=1,
                                        shuffle=False,
                                        )

    for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
        
        img = img.to(args.device)
        target = target.to(args.device)

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        losses_record.append(loss.item())

    losses_idx = np.argsort(np.array(losses_record))   # get the index of examples by loss value in descending order

    # Show the top 10 loss values
    losses_record_arr = np.array(losses_record)
    logging.info(f'Top ten loss value: {losses_record_arr[losses_idx[:10]]}')

    return losses_idx

def isolate_data(args, result, losses_idx):
    '''isolate the backdoor data with the calculated loss
    args:
        Contains default parameters
    result:
        the attack result contain the train dataset which contains backdoor data
    losses_idx:
        the index of order about the loss value for each data 
    '''
    # Initialize lists
    other_examples = []
    isolation_examples = []

    cnt = 0
    ratio = args.isolation_ratio
    perm = losses_idx[0: int(len(losses_idx) * ratio)]
    permnot = losses_idx[int(len(losses_idx) * ratio):]
    x = result['bd_train']['x']
    y = result['bd_train']['y']
    isolation_examples = list(zip([x[ii] for ii in perm],[y[ii] for ii in perm]))
    other_examples = list(zip([x[ii] for ii in permnot],[y[ii] for ii in permnot]))
    # example_data_loader = torch.utils.data.DataLoader(dataset=poisoned_data,
    #                                     batch_size=1,
    #                                     shuffle=False,
    #                                     )
    

    # for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
    #     img = img.squeeze()
    #     target = target.squeeze()
    #     # img = np.transpose((img * 255).cpu().numpy(), (1, 2, 0)).astype('uint8')
    #     target = target.cpu().numpy()

    #     # Filter the examples corresponding to losses_idx
    #     if idx in perm:
    #         isolation_examples.append((img, target))
    #         cnt += 1
    #     else:
    #         other_examples.append((img, target))

    logging.info('Finish collecting {} isolation examples: '.format(len(isolation_examples)))
    logging.info('Finish collecting {} other examples: '.format(len(other_examples)))

    return isolation_examples, other_examples


def train_step_finetuing(args, train_loader, model_ascent, optimizer, criterion, epoch):
    '''finetuing the model with remaining data for each step
    args:
        Contains default parameters
    train_loader:
        the dataloader of remaining data
    model_ascent:
        the model after pretrain
    optimizer:
        optimizer during the finetuing process
    criterion:
        criterion during the finetuing process
    epoch:
        current epoch
    '''
    losses = 0
    top1 = 0
    size = 0
    model_ascent.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        
        img = img.to(args.device)
        target = target.to(args.device)
        output = model_ascent(img)
        loss = criterion(output, target)

        pre_label = torch.max(output,dim=1)[1]
        acc = torch.sum(pre_label == target)/len(train_loader.dataset)
        prec1 = acc
        losses += loss * img.size(0)
        size += img.size(0)
        top1 += prec1*len(train_loader.dataset)
        # top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()  # Gradient ascent training
        optimizer.step()

        logging.info('Epoch[{0}]:[{1:03}/{2:03}] '
            'loss:{losses:.4f}({losses_avg:.4f})  '
            'prec@1:{top1:.2f}({top1_avg:.2f})  '.format(epoch, idx, len(train_loader), losses=losses, losses_avg = losses/len(train_loader.dataset), top1=top1, top1_avg=top1/len(train_loader.dataset)))

def train_step_unlearning(args, train_loader, model_ascent, optimizer, criterion, epoch):
    '''unlearning the model with 'backdoor' data for each step
    args:
        Contains default parameters
    train_loader:
        the dataloader of 'backdoor' data
    model_ascent:
        the model after finetuning
    optimizer:
        optimizer during the unlearning process
    criterion:
        criterion during the unlearning process
    epoch:
        current epoch
    '''
    losses = 0
    top1 = 0
    size = 0
    model_ascent.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        
        img = img.to(args.device)
        target = target.to(args.device)

        output = model_ascent(img)

        loss = criterion(output, target)

        pre_label = torch.max(output,dim=1)[1]
        acc = torch.sum(pre_label == target)/len(train_loader.dataset)
        prec1 = acc
        losses += loss * img.size(0)
        size += img.size(0)
        top1 += prec1*len(train_loader.dataset)

        optimizer.zero_grad()
        (-loss).backward()  # Gradient ascent training
        optimizer.step()

        logging.info('Epoch[{0}]:[{1:03}/{2:03}] '
            'loss:{losses:.4f}({losses_avg:.4f})  '
            'prec@1:{top1:.2f}({top1_avg:.2f})  '.format(epoch, idx, len(train_loader), losses=losses, losses_avg = losses/len(train_loader.dataset), top1=top1, top1_avg=top1/len(train_loader.dataset)))


def test_unlearning(args, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch):
    '''test the model during the unlearning process
    args:
        Contains default parameters
    test_clean_loader:
        the dataloader of clean test data
    test_bad_loader:
        the dataloader of backdoor test data    
    model_ascent:
        the model during the unlearning process
    criterion:
        criterion during the unlearning process
    epoch:
        current epoch
    '''
    test_process = []

    with torch.no_grad():

        losses = 0
        top1 = 0
        size = 0
        model_ascent.eval()

        for idx, (img, target) in enumerate(test_clean_loader, start=1):
            
            img = img.to(args.device)
            target = target.to(args.device)

            output = model_ascent(img)

            loss = criterion(output, target)

            pre_label = torch.max(output,dim=1)[1]
            acc = torch.sum(pre_label == target)/len(test_clean_loader.dataset)
            prec1 = acc
            losses += loss * img.size(0)
            size += img.size(0)
            top1 += prec1*len(test_clean_loader.dataset)
            

        acc_clean = [top1/size, losses/size]

        losses = 0
        top1 = 0
        size = 0
        model_ascent.eval()

        for idx, (img, target) in enumerate(test_bad_loader, start=1):
            
            img = img.to(args.device)
            target = target.to(args.device)

            output = model_ascent(img)

            loss = criterion(output, target)

            pre_label = torch.max(output,dim=1)[1]
            acc = torch.sum(pre_label == target)/len(test_bad_loader.dataset)
            prec1 = acc
            losses += loss * img.size(0)
            size += img.size(0)
            top1 += prec1*len(test_bad_loader.dataset)
    
    acc_bd = [top1/size, losses/size]

    logging.info('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[1]))
    logging.info('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[1]))

    return acc_clean, acc_bd


def train_unlearning(args, result, model_ascent, isolate_poisoned_data, isolate_other_data):
    '''train the model with remaining data and unlearn the backdoor data
    args:
        Contains default parameters
    result:
        attack result(details can be found in utils)
    model_ascent:
        the model after pretrain
    isolate_poisoned_data:
        the dataset of 'backdoor' data
    isolate_other_data:
        the dataset of remaining data
    '''
    # Load models
    logging.info('----------- Network Initialization --------------')
    model_ascent.to(args.device)
    logging.info('Finish loading ascent model...')
    # initialize optimizer
    optimizer = torch.optim.SGD(model_ascent.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # define loss functions
    if args.device == 'cuda':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    tf_compose_finetuning = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    tf_compose_unlearning = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
   
    poisoned_data_tf =  prepro_cls_DatasetBD(
        full_dataset_without_transform=isolate_poisoned_data,
        poison_idx=np.zeros(len(isolate_poisoned_data)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tf_compose_unlearning,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    isolate_poisoned_data_loader = torch.utils.data.DataLoader(dataset=poisoned_data_tf,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      )

    # isolate_other_data = np.load(data_path_other, allow_pickle=True)
    # isolate_other_data_tf = Dataset_npy(full_dataset=isolate_other_data, transform=tf_compose_finetuning)
    isolate_other_data_tf = prepro_cls_DatasetBD(
        full_dataset_without_transform=isolate_other_data,
        poison_idx=np.zeros(len(isolate_other_data)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tf_compose_finetuning,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    isolate_other_data_loader = torch.utils.data.DataLoader(dataset=isolate_other_data_tf,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              )

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
    test_bad_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

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
    test_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    if args.finetuning_ascent_model == True:
        # this is to improve the clean accuracy of isolation model, you can skip this step
        logging.info('----------- Finetuning isolation model --------------')
        for epoch in range(0, args.finetuning_epochs):
            learning_rate_finetuning(optimizer, epoch, args)
            train_step_finetuing(args, isolate_other_data_loader, model_ascent, optimizer, criterion,
                             epoch + 1)
            test_unlearning(args, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)

    best_acc = 0
    best_asr = 0
    logging.info('----------- Model unlearning --------------')
    for epoch in range(0, args.unlearning_epochs):
        
        learning_rate_unlearning(optimizer, epoch, args)
        # train stage
        if epoch == 0:
            # test firstly
            test_unlearning(args, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch)
        else:
            train_step_unlearning(args, isolate_poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)

        # evaluate on testing set
        logging.info('testing the ascended model......')
        acc_clean, acc_bad = test_unlearning(args, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)

        if not (os.path.exists(os.getcwd() + f'{args.checkpoint_save}')):
            os.makedirs(os.getcwd() + f'{args.checkpoint_save}')
        if best_acc < acc_clean[0]:
            best_acc = acc_clean[0]
            best_asr = acc_bad[0]
            torch.save(
            {
                'model_name':args.model,
                'model': model_ascent.cpu().state_dict(),
                'asr': acc_bad[0],
                'acc': acc_clean[0]
            },
            os.getcwd() + f'{args.checkpoint_save}defense_result.pt'
            )
            model_ascent.to(args.device)
        logging.info(f'Epoch{epoch}: clean_acc:{acc_clean[0]} asr:{acc_bad[0]} best_acc:{best_acc} best_asr{best_asr}')
    return model_ascent


def learning_rate_finetuning(optimizer, epoch, args):
    '''set learning rate during the process of finetuing model 
    optimizer:
        optimizer during the pretrain process
    epoch:
        current epoch
    args:
        Contains default parameters
    '''
    if epoch < 40:
        lr = 0.01
    elif epoch < 60:
        lr = 0.001
    else:
        lr = 0.001
    logging.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def learning_rate_unlearning(optimizer, epoch, args):
    '''set learning rate during the process of unlearning model 
    optimizer:
        optimizer during the pretrain process
    epoch:
        current epoch
    args:
        Contains default parameters
    '''
    if epoch < args.unlearning_epochs:
        lr = 0.0001
    else:
        lr = 0.0001
    logging.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def abl(args,result):
    ### set logger
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    if args.log is not None and args.log != '':
        fileHandler = logging.FileHandler(os.getcwd() + args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    else:
        fileHandler = logging.FileHandler(os.getcwd() + './log' + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))

    fix_random(args.random_seed)

    ###a. pre-train model
    poisoned_data, model_ascent = train(args,result)
    
    ###b. isolate the special data(loss is low) as backdoor data
    losses_idx = compute_loss_value(args, poisoned_data, model_ascent)
    logging.info('----------- Collect isolation data -----------')
    isolation_examples, other_examples = isolate_data(args, result, losses_idx)

    ###c. unlearn the backdoor data and learn the remaining data
    model_new = train_unlearning(args,result,model_ascent,isolation_examples,other_examples)

    result = {}
    result['model'] = model_new
    return result

if __name__ == '__main__':
    
    ###1. basic setting: args, attack result(model, train data, test data)
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
        args.checkpoint_save = save_path + '/record/defence/abl/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = save_path + '/saved/abl/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log)  
    args.save_path = save_path


    ###2. attack result(model, train data, test data)
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    
    ###3. abl defense:
    print("Continue training...")
    result_defense = abl(args,result)

    ###4. test the result and get ASR, ACC, RC 
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

    if not (os.path.exists(os.getcwd() + f'{save_path}/abl/')):
        os.makedirs(os.getcwd() + f'{save_path}/abl/')
    torch.save(
    {
        'model_name':args.model,
        'model': result_defense['model'].cpu().state_dict(),
        'asr': asr_acc,
        'acc': clean_acc,
        'ra': robust_acc
    },
    os.getcwd() + f'{save_path}/abl/defense_result.pt'
    )
    