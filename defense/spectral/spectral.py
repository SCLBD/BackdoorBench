# MIT License

# Copyright (c) 2017 Brandon Tran and Jerry Li

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

'''
This file is modified based on the following source:
link : https://github.com/MadryLab/backdoor_data_poisoning.
The defense method is called spectral.

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. use the PyTorch environment instead of TensorFlow
    7. add some addtional backbone such as resnet18 and vgg19
    8. the poison ratio can also be preset when the data for each category is small
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. spectral defense:
        a. prepare the model and dataset
        b. get the activation as representation for each data
        c. detect the backdoor data by the SVD decomposition
        d. retrain the model with remaining data
    4. test the result and get ASR, ACC, RC 
'''

import logging
import argparse
from datetime import datetime
import json
import math
from pyexpat import model
import shutil
import sys
import os
import time
sys.path.append('../')
sys.path.append(os.getcwd())
from timeit import default_timer as timer

import numpy as np
import torch
from tqdm import trange

import yaml
from utils.aggregate_block.fix_random import fix_random 
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import nCHW_to_nHWC
from utils.save_load_attack import load_attack_result
from pprint import pprint, pformat

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

    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)

    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--random_seed', type=int, help='random seed')
    parser.add_argument('--index', type=str, help='index of clean data')
    parser.add_argument('--result_file', type=str, help='the location of result')

    parser.add_argument('--yaml_path', type=str, default="./config/defense/spectral/config.yaml", help='the path of yaml')

    #set the parameter for the spectral defense
    parser.add_argument('--percentile', type=int)

    arg = parser.parse_args()

    print(arg)
    return arg

def spectral(arg,result):
    ### set logger
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    if arg.log is not None and arg.log != '':
        fileHandler = logging.FileHandler(os.getcwd() + arg.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    else:
        fileHandler = logging.FileHandler(os.getcwd() + './log' + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(arg.__dict__))

    fix_random(arg.random_seed)

    ### a. prepare the model and dataset
    model = generate_cls_model(arg.model,arg.num_classes)
    model.load_state_dict(result['model'])
    model.to(arg.device)

    # Setting up the data and the model
    target_label = arg.target_label
    tran = get_transform(arg.dataset, *([arg.input_height,arg.input_width]) , train = True)
    x = result['bd_train']['x']
    y = result['bd_train']['y']
    data_set = list(zip(x,y))
    data_set_o = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    
    dataset = data_set_o
   
    
    # initialize data augmentation
    logging.info(f'Dataset Size: {len(dataset)}' )

    lbl = target_label
    dataset_y = []
    for i in range(len(dataset)):
        dataset_y.append(dataset[i][1])
    cur_indices = [i for i,v in enumerate(dataset_y) if v==lbl]
    cur_examples = len(cur_indices)
    logging.info(f'Label, num ex: {lbl},{cur_examples}' )
    
    model.eval()
    ### b. get the activation as representation for each data
    for iex in trange(cur_examples):
        cur_im = cur_indices[iex]
        x_batch = dataset[cur_im][0].unsqueeze(0).to(arg.device)
        y_batch = dataset[cur_im][1]
        assert arg.model in ['preactresnet18', 'vgg19', 'resnet18', 'mobilenet_v3_large', 'densenet161', 'efficientnet_b3']
        if arg.model == 'preactresnet18':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.layer4.register_forward_hook(layer_hook)
            _ = model(x_batch)
            batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
            hook.remove()
        elif arg.model == 'vgg19':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.features.register_forward_hook(layer_hook)
            _ = model(x_batch)
            batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
            hook.remove()
        elif arg.model == 'resnet18':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.layer4.register_forward_hook(layer_hook)
            _ = model(x_batch)
            batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
            hook.remove()
        elif arg.model == 'mobilenet_v3_large':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.avgpool.register_forward_hook(layer_hook)
            _ = model(x_batch)
            batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
            hook.remove()
        elif arg.model == 'densenet161':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.features.register_forward_hook(layer_hook)
            _ = model(x_batch)
            outs[0] = torch.nn.functional.relu(outs[0])
            batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
            hook.remove()
        elif arg.model == 'efficientnet_b3':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.avgpool.register_forward_hook(layer_hook)
            _ = model(x_batch)
            batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
            hook.remove()
        
        if iex==0:
            full_cov = np.zeros(shape=(cur_examples, len(batch_grads)))
        full_cov[iex] = batch_grads.detach().cpu().numpy()

    ### c. detect the backdoor data by the SVD decomposition
    total_p = arg.percentile            
    full_mean = np.mean(full_cov, axis=0, keepdims=True)            
  
    centered_cov = full_cov - full_mean
    u,s,v = np.linalg.svd(centered_cov, full_matrices=False)
    logging.info(f'Top 7 Singular Values: {s[0:7]}')
    eigs = v[0:1]  
    p = total_p
    corrs = np.matmul(eigs, np.transpose(full_cov)) #shape num_top, num_active_indices
    scores = np.linalg.norm(corrs, axis=0) #shape num_active_indices
    logging.info(f'Length Scores: {len(scores)}' )
    p_score = np.percentile(scores, p)
    top_scores = np.where(scores>p_score)[0]
    logging.info(f'{top_scores}')
    

    removed_inds = np.copy(top_scores)
    re = [cur_indices[v] for i,v in enumerate(removed_inds)]
    left_inds = np.delete(range(len(dataset)), re)
           

    ### d. retrain the model with remaining data
    model = generate_cls_model(arg.model,arg.num_classes)
    model.to(arg.device)
    dataset.subset(left_inds)
    dataset_left = dataset
    data_loader_sie = torch.utils.data.DataLoader(dataset_left, batch_size=arg.batch_size, num_workers=arg.num_workers, shuffle=True)
    
    tran = get_transform(arg.dataset, *([arg.input_height,arg.input_width]) , train = False)
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
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=arg.batch_size, num_workers=arg.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    tran = get_transform(arg.dataset, *([arg.input_height,arg.input_width]) , train = False)
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
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=arg.batch_size, num_workers=arg.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    best_acc = 0
    best_asr = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer)
    elif args.lr_scheduler ==  'CosineAnnealingLR':
        scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, T_max=100)
    criterion = torch.nn.CrossEntropyLoss() 
    for j in range(arg.epochs):
        batch_loss = []
        for i, (inputs,labels) in enumerate(data_loader_sie):  # type: ignore
            model.train()
            inputs, labels = inputs.to(arg.device), labels.to(arg.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        one_epoch_loss = sum(batch_loss)/len(batch_loss)
        if args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler.step(one_epoch_loss)
        elif args.lr_scheduler ==  'CosineAnnealingLR':
            scheduler.step()
        
        with torch.no_grad():
            model.eval()
            asr_acc = 0
            for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
                inputs, labels = inputs.to(arg.device), labels.to(arg.device)
                outputs = model(inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                asr_acc += torch.sum(pre_label == labels)/len(data_bd_test)

            
            clean_acc = 0
            for i, (inputs,labels) in enumerate(data_clean_loader):  # type: ignore
                inputs, labels = inputs.to(arg.device), labels.to(arg.device)
                outputs = model(inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                clean_acc += torch.sum(pre_label == labels)/len(data_clean_test)
        
        if not (os.path.exists(os.getcwd() + f'{arg.checkpoint_save}')):
            os.makedirs(os.getcwd() + f'{arg.checkpoint_save}')
        if best_acc < clean_acc:
            best_acc = clean_acc
            best_asr = asr_acc
            torch.save(
            {
                'model_name':arg.model,
                'model': model.cpu().state_dict(),
                'asr': asr_acc,
                'acc': clean_acc
            },
            f'./{arg.checkpoint_save}defense_result.pt'
            )
            model.to(arg.device)
    
        logging.info(f'Epoch{j}: clean_acc:{clean_acc} asr:{asr_acc} best_acc:{best_acc} best_asr{best_asr}')

    result = {}
    result["dataset"] = dataset_left
    result['model'] = model
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
        args.checkpoint_save = save_path + '/record/defence/spectral/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = save_path + '/saved/spectral/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log)  
    args.save_path = save_path

    ### 2. attack result(model, train data, test data)
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    
    logging.info("Continue training...")
    ### 3. spectral defense
    result_defense = spectral(args,result)

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
                robust_acc += torch.sum(pre_label == labels)/len(data_bd_test)

    if not (os.path.exists(os.getcwd() + f'{save_path}/spectral/')):
        os.makedirs(os.getcwd() + f'{save_path}/spectral/')
    torch.save(
    {
        'model_name':args.model,
        'model': result_defense['model'].cpu().state_dict(),
        'asr': asr_acc,
        'acc': clean_acc,
        'ra': robust_acc
    },
    os.getcwd() + f'{save_path}/spectral/defense_result.pt'
    )