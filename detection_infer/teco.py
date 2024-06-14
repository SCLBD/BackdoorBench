'''
Detecting Backdoors During the Inference Stage Based on Corruption Robustness Consistency
This file is modified based on the following source:
link : https://github.com/CGCL-codes/TeCo/blob/main/BackdoorBench-v1.0-merge/defense/teco/teco.py

@inproceedings{liu2023detecting,
    title={Detecting Backdoors During the Inference Stage Based on Corruption Robustness Consistency},
    author={Liu, Xiaogeng and Li, Minghui and Wang, Haoyu and Hu, Shengshan and Ye, Dengpan and Jin, Hai and Wu, Libing and Xiao, Chaowei},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={16363--16372},
    year={2023}}

This file implements the (detection) defense method called TeCo (ft), which detects trigger samples during the inference stage based on corruption robustness consistency.

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. TeCo detection:
        a. use image corruption
        b. get the hard-label output of backdoor-infected model (CRC test)
    4. use deviation for trigger sample detection
    5. Record a seirse of thresholds, TPRs and FPRs. Besides, the auc also be recorded.



'''

import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from defense.base import defense
import scipy
from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import PureCleanModelTrainer
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.nCHW_nHWC import *

import tqdm
import heapq
from PIL import Image
from utils.bd_dataset_v2 import dataset_wrapper_with_transform,xy_iter, prepro_cls_DatasetBD_v2
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, all_acc, general_plot_for_epoch, given_dataloader_test
from collections import Counter
import copy
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import random
import csv
from sklearn import metrics
from imagecorruptions import corrupt
from sklearn.metrics import auc

class teco(defense):

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"
        
        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
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

        parser.add_argument('--model', type=str, help='resnet18')
        parser.add_argument('--seed', type=str, help='random seed')
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--result_file', type=str, help='the location of result')
        parser.add_argument('--yaml_path', type=str, default="./config/detection/teco/cifar10.yaml", help='the path of yaml')

        # dg settings
        parser.add_argument('--cor_type', type=str, help='type of image corruption')
        parser.add_argument('--severity', type=int, help='severity of image corruption')
        parser.add_argument('--max', type=int, default=6, help='max severity of image corruption')
        parser.add_argument('--clean_sample_num', type=int)

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection/teco_infer/'
        if not (os.path.exists(save_path)):
                os.makedirs(save_path) 
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'detection_info/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
                
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model = model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
    
    def set_devices(self):
        self.device = self.args.device

    def dg(self, image, args):
        image = np.array(image)
        image = corrupt(image, corruption_name=args.cor_type, severity=args.severity)
        image = Image.fromarray(image)
        return image

    def no_defense(self, args, result, config):
        model = generate_cls_model(args.model, args.num_classes)
        model.load_state_dict(result['model'])
        model.to(args.device)
        result = {}
        result['model'] = model
        return result    

    
    def cal(self, true, pred):
        TN, FP, FN, TP = confusion_matrix(true, pred).ravel()
        return TN, FP, FN, TP 
    def metrix(self, TN, FP, FN, TP):
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        precision = TP/(TP+FP)
        acc = (TP+TN)/(TN+FP+FN+TP)
        return TPR, FPR, precision, acc
    
    def filtering(self):
        start = time.perf_counter()
        self.set_devices()
        fix_random(self.args.random_seed)

        ### a. load model, bd train data and transforms
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
            model.eval()
        else:
            model.to(self.args.device)
            model.eval()
        
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        
        bd_test_dataset = self.result['bd_test'].wrapped_dataset

        bd_dict = {}
        images_poison = []
        labels_poison = []
        for img, label,*other_info in bd_test_dataset:
            images_poison.append(img)
            labels_poison.append(label)

        data_bd_test = xy_iter(images_poison, labels_poison,transform=test_tran)
        data_bd_loader = DataLoader(data_bd_test, batch_size=args.batch_size, shuffle=False)

        for i, (inputs, labels) in enumerate(data_bd_loader):  # type: ignore
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            pre_label = torch.max(outputs, dim=1)[1]
            for j in range(len(pre_label)):
                save_name = str(i * args.batch_size + j)
                bd_dict[save_name] = {}
                bd_dict[save_name]['original'] = [pre_label[j].item()]
        for name in ['gaussian_noise', 'shot_noise','impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
                       'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
                       'jpeg_compression']:
            for severity in range(1, 6):
                args.severity = severity
                args.cor_type = name
                x = images_poison
                for i in range(len(x)):
                    x[i] = self.dg(x[i], args)
                y = labels_poison
                data_bd_test = xy_iter(x,y,transform=test_tran)
                data_bd_loader = DataLoader(data_bd_test, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=False,pin_memory=False)

                for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
                    inputs, labels = inputs.to(args.device), labels.to(args.device)
                    outputs = model(inputs)
                    pre_label = torch.max(outputs,dim=1)[1]
                    for j in range(len(pre_label)):
                        save_name = str(i * args.batch_size + j)
                        if name not in bd_dict[save_name].keys():
                            bd_dict[save_name][name] = []
                            bd_dict[save_name][name].append(bd_dict[save_name]['original'][0])
                        bd_dict[save_name][name].append(pre_label[j].item())
        
        clean_dict = {}
        clean_test_dataset = self.result['clean_test'].wrapped_dataset

        ### b. find a clean sample from test dataset
        images = []
        labels = []
        for img, label in clean_test_dataset:
            images.append(img)
            labels.append(label)

        
        class_idx_whole = []
        num = int(self.args.clean_sample_num / self.args.num_classes)
        if num == 0:
            num = 1
        for i in range(self.args.num_classes):
            class_idx_whole.append(np.random.choice(np.where(np.array(labels)==i)[0], num))
        image_c = []
        label_c = []
        class_idx_whole = np.concatenate(class_idx_whole, axis=0)
        image_c = [images[i] for i in class_idx_whole]
        label_c = [labels[i] for i in class_idx_whole]

        clean_dataset = xy_iter(image_c, label_c,transform=test_tran)
        data_clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=False, pin_memory=False)
        
        for i, (inputs, labels) in enumerate(data_clean_loader):  # type: ignore
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            pre_label = torch.max(outputs, dim=1)[1]
            for j in range(len(pre_label)):
                save_name = str(i * args.batch_size + j)
                clean_dict[save_name] = {}
                clean_dict[save_name]['original'] = [pre_label[j].item()]
        for name in ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
                        'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
                        'jpeg_compression']:
            for severity in range(1, 6):
                args.severity = severity
                args.cor_type = name
                x = image_c
                for i in range(len(x)):
                    x[i] = self.dg(x[i], args)
                y = label_c
                data_clean_test = xy_iter(x,y,transform=test_tran)
                data_clean_loader = DataLoader(data_clean_test, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=False,pin_memory=False)

                for i, (inputs,labels) in enumerate(data_clean_loader):  # type: ignore
                    inputs, labels = inputs.to(args.device), labels.to(args.device)
                    outputs = model(inputs)
                    pre_label = torch.max(outputs,dim=1)[1]
                    for j in range(len(pre_label)):
                        save_name = str(i * args.batch_size + j)
                        if name not in clean_dict[save_name].keys():
                            clean_dict[save_name][name] = []
                            clean_dict[save_name][name].append(clean_dict[save_name]['original'][0])
                        clean_dict[save_name][name].append(pre_label[j].item())

        result = {'clean': clean_dict, 'bd': bd_dict}
        labels = []
        mads = []
        total_images = 0
        for file in ['clean', 'bd']:
            label_dict = result[file]
            images = list(label_dict.keys())
            keys = list(label_dict[images[0]].keys())
            total_images += len(images)
            for img in images:
                indexs = []
                img_preds = label_dict[img]
                for corruption in keys[1:]:
                    flag = 0
                    for i in range(args.max):
                        if int(img_preds[corruption][i]) != int(img_preds[corruption][0]):
                            index = i
                            flag = 1
                            indexs.append(index)
                            break
                    if flag == 0:
                        indexs.append(args.max)
                indexs = np.asarray(indexs)
                mad = np.std(indexs)
                mads.append(mad)
                if file == 'clean':
                    labels.append(0)
                else:
                    labels.append(1)
        mads = np.asarray(mads)
        labels = np.asarray(labels)
        fpr, tpr, thresholds = metrics.roc_curve(labels, mads)
        f1_scores = []
        for th in thresholds:
            pred = np.where(mads > th, 1, 0)
            f1_score = metrics.f1_score(labels, pred, average='micro')
            f1_scores.append(f1_score)
        f1_scores = np.asarray(f1_scores)
        roc_auc = auc(fpr, tpr)
        end = time.perf_counter()
        time_miniute = (end-start)/60

        f = open(self.args.save_path + '/detection_info.csv', 'a', encoding='utf-8')
        csv_write = csv.writer(f)
        csv_write.writerow(['record', 'Threshold', 'TPR','FPR', 'AUC', 'target'])
        csv_write.writerow([args.result_file, thresholds, tpr, fpr, roc_auc, 'None'])
        f.close()

                
    def detection(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.filtering()
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    teco.add_arguments(parser)
    args = parser.parse_args()
    teco_method = teco(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = teco_method.detection(args.result_file)