
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
sys.path.append('../')
sys.path.append(os.getcwd())
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
    
    