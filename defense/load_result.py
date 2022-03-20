
'''
@article{tran2018spectral,
  title={Spectral signatures in backdoor attacks},
  author={Tran, Brandon and Li, Jerry and Madry, Aleksander},
  journal={Advances in neural information processing systems},
  volume={31},
  year={2018}
}

code : https://github.com/MadryLab/backdoor_data_poisoning
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
import pandas as pd


sys.path.append('../')
sys.path.append(os.getcwd())
from utils.aggregate_block.dataset_and_transform_generate import get_transform

from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import nCHW_to_nHWC
from utils.save_load_attack import load_attack_result
from timeit import default_timer as timer

import numpy as np
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, help='the location of result')
    # arg = parser.parse_args()

    # print(arg)
    # return arg

    return parser

def get_args_2(parser):
    parser.add_argument('--defense', type=str, help='defense method:[abl,ac,dbd,fp,nad,nc,spectral]')
    arg = parser.parse_args()

    print(arg)
    return arg

if __name__ == '__main__':
    parser = get_args()
    args = get_args_2(parser)
    save_path = './record/' + args.result_file
    
    result = torch.load(save_path + '/' + args.defense + '/defense_result.pt')
    print('asr:{} acc:{}'.format(result['asr'],result['acc']))

    #####rc#######仅适用于cifar10
    print(save_path + '/attack_result.pt')
    data = load_attack_result(save_path + '/attack_result.pt')
    ori_label_un = data['clean_test']['y']
    ori_label = [i for i in ori_label_un if i != 0]
    try:
        model = generate_cls_model(result['model_name'],10)
        model.load_state_dict(result['model'])
    except RuntimeError:
        model = generate_cls_model('preactresnet18',10)
        model.load_state_dict(result['model'])
    model.to('cuda')
    #data_set = get_dataset_train(args)
    tran = get_transform('cifar10', *([32,32]) , train = False)
    x = torch.tensor(nCHW_to_nHWC(data['bd_test']['x'].numpy()))
    y = torch.tensor(ori_label)
    print('y:{}'.format(y.size(0)))
    print('x:{}'.format(x.size(0)))
    data_set = torch.utils.data.TensorDataset(x,y)
    data_set_o = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=128, num_workers=1, shuffle=True)
    robust_acc = 0
    for i, (inputs,labels) in enumerate(data_loader):  # type: ignore
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        pre_label = torch.max(outputs,dim=1)[1]
        robust_acc += torch.sum(pre_label == labels)/len(data_set_o)

    ########### CSV ###########
    # df_head = ['Attack', 'Defense', 'Model', 'ACC', 'ASR']
    # df = pd.DataFrame(columns=df_head)
    # df.loc[df.shape[0]] = {'Attack':save_path.split('/')[-1], 'Defense': args.defense, 'Model':result['model_name'], 'ACC': result['acc'].data.item(), 'ASR': result['asr'].data.item()}
    
    # log_file='./result.csv'
    # if os.path.isfile(log_file):
    #     df.to_csv(log_file, mode='a', index=False, header=False)
    # else:
    #     df.to_csv(log_file, index=False)
    ########### CSV ###########
    df_head = ['Attack', 'Defense', 'Model', 'ACC', 'ASR', 'RC']
    df = pd.DataFrame(columns=df_head)
    df.loc[df.shape[0]] = {'Attack':save_path.split('/')[-1], 'Defense': args.defense, 'Model':result['model_name'], 'ACC': result['acc'].data.item(), 'ASR': result['asr'].data.item(), 'RC': robust_acc}
    
    log_file='./result.csv'
    if os.path.isfile(log_file):
        df.to_csv(log_file, mode='a', index=False, header=False)
    else:
        df.to_csv(log_file, index=False)
    