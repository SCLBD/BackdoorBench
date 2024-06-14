'''
STRIP: A Defence Against Trojan Attacks on Deep Neural Networks
This file is modified based on the following source:
link : https://github.com/Unispac/Fight-Poison-With-Poison/blob/master/other_cleansers/strip.py
This file implements the (detection) defense method called STRIP.

@inproceedings{gao2019strip,
    title={Strip: A defence against trojan attacks on deep neural networks},
    author={Gao, Yansong and Xu, Change and Wang, Derui and Chen, Shiping and Ranasinghe, Damith C and Nepal, Surya},
    booktitle={Proceedings of the 35th Annual Computer Security Applications Conference},
    pages={113--125},
    year={2019}}

basic sturcture for defense method:
1. basic setting: args
2. attack result(model, train data, test data)
3. STRIP detection:
    a. mix up clean samples and record the entropy of prediction, and record smallest entropy and largest entropy as thresholds.
    b. mix up the tested samples and clean samples, and record the entropy.
    c. detection samples whose entropy exceeds the thresholds as poisoned.
4. Record TPR and FPR.

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

class STRIP():
    name: str = 'strip'

    def __init__(self, args, inspection_set, clean_set, model, strip_alpha: float = 0.5, N: int = 64, defense_fpr: float = 0.05, batch_size=128):

        self.args = args

        self.strip_alpha: float = strip_alpha
        self.N: int = N
        self.defense_fpr = defense_fpr

        self.inspection_set = inspection_set
        self.clean_set = clean_set

        self.model = model


    def cleanse(self):

        # choose a decision boundary with the test set
        clean_entropy = []
        clean_set_loader = DataLoader(self.clean_set, batch_size=128, shuffle=False)
        for i, (_input, _label) in enumerate(clean_set_loader):
            # _input, _label = _input.cuda(), _label.cuda()
            _input, _label = _input.to(self.args.device), _label.to(self.args.device)
            entropies = self.check(_input, _label, self.clean_set)
            for e in entropies:
                clean_entropy.append(e)
        clean_entropy = torch.FloatTensor(clean_entropy)

        clean_entropy, _ = clean_entropy.sort()
        print(len(clean_entropy))
        threshold_low = float(clean_entropy[int(self.defense_fpr * len(clean_entropy))])
        threshold_high = np.inf

        # now cleanse the inspection set with the chosen boundary
        inspection_set_loader = DataLoader(self.inspection_set, batch_size=128, shuffle=False)
        all_entropy = []
        for i, (_input, _label) in enumerate(inspection_set_loader):
            # _input, _label = _input.cuda(), _label.cuda()
            _input, _label = _input.to(self.args.device), _label.to(self.args.device)
            entropies = self.check(_input, _label, self.clean_set)
            for e in entropies:
                all_entropy.append(e)
        all_entropy = torch.FloatTensor(all_entropy)
        ### save the entropy of all samples
        torch.save(all_entropy, self.args.save_path + '/entropy.pt')
        self.all_entropy = all_entropy

        suspicious_indices = torch.logical_or(all_entropy < threshold_low, all_entropy > threshold_high).nonzero().reshape(-1)
        return suspicious_indices.numpy()

    def check(self, _input: torch.Tensor, _label: torch.Tensor, source_set) -> torch.Tensor:
        _list = []

        samples = list(range(len(source_set)))
        random.shuffle(samples)
        samples = samples[:self.N]

        with torch.no_grad():

            for i in samples:
                X, Y = source_set[i]
                X = X.to(args.device)
                _test = self.superimpose(_input, X)
                entropy = self.entropy(_test).cpu().detach()
                _list.append(entropy)

        return torch.stack(_list).mean(0)

    def superimpose(self, _input1: torch.Tensor, _input2: torch.Tensor, alpha: float = None):
        if alpha is None:
            alpha = self.strip_alpha

        result = _input1 + alpha * _input2
        return result

    def entropy(self, _input: torch.Tensor) -> torch.Tensor:
        p = torch.nn.Softmax(dim=1)(self.model(_input)) + 1e-8
        return (-p * p.log()).sum(1)
    

class strip(defense):

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
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', default = False, type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/detection/strip/cifar10.yaml", help='the path of yaml')
        parser.add_argument('--clean_sample_num', type=int)

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection/strip_pretrain/'
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

    def get_features(data_loader, model):

        class_indices = []
        feats = []

        model.eval()
        with torch.no_grad():
            for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):
                ins_data = ins_data.cuda()
                _, x_features = model(ins_data, True)

                this_batch_size = len(ins_target)
                for bid in range(this_batch_size):
                    feats.append(x_features[bid].cpu().numpy())
                    class_indices.append(ins_target[bid].cpu().numpy())

        return feats, class_indices
    
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
        bd_train_dataset = self.result['bd_train'].wrapped_dataset
        pindex = np.where(np.array(bd_train_dataset.poison_indicator) == 1)[0]
        ### save pindex
        torch.save(torch.tensor(pindex), self.args.save_path + '/pindex.pt')

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
        
        ### c. load training dataset with poison samples
        images_poison = []
        labels_poison = []
        for img, label,*other_info in bd_train_dataset:
            images_poison.append(img)
            labels_poison.append(label)

        ### d. get features of training dataset
        train_dataset = xy_iter(images_poison, labels_poison,transform=test_tran)
        worker = STRIP( args, train_dataset, clean_dataset, model, strip_alpha=1.0, N=100, defense_fpr=0.1, batch_size=128 )
        suspect_index = worker.cleanse()

        ### calculate auc and roc
        label_true = []
        label_pred = []
        all_entropy = worker.all_entropy
        for i in range(len(images_poison)):
            if i in pindex:
                label_true.append(1)
            else:
                label_true.append(0)
            label_pred.append(all_entropy[i].item())
        fpr, tpr, thresholds = metrics.roc_curve(label_true, label_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        true_index = np.zeros(len(images_poison))
        for i in range(len(true_index)):
            if i in pindex:
                true_index[i] = 1  

        if len(suspect_index)==0:
            tn = len(true_index) - np.sum(true_index)
            fp = np.sum(true_index)
            fn = 0
            tp = 0
            f = open(self.args.save_path + '/detection_info.csv', 'a', encoding='utf-8')
            csv_write = csv.writer(f)
            csv_write.writerow(['record', 'TN','FP','FN','TP','TPR','FPR', 'AUC', 'target'])
            csv_write.writerow([args.result_file, tn,fp,fn,tp, 0,0, auc, 'None'])
            f.close()
        else: 
            findex = np.zeros(len(images_poison))
            for i in range(len(findex)):
                if i in suspect_index:
                    findex[i] = 1
            tn, fp, fn, tp = self.cal(true_index, findex)
            TPR, FPR, precision, acc = self.metrix(tn, fp, fn, tp)

            new_TP = tp
            new_FN = fn*9
            new_FP = fp*1
            precision = new_TP / (new_TP + new_FP) if new_TP + new_FP != 0 else 0
            recall = new_TP / (new_TP + new_FN) if new_TP + new_FN != 0 else 0
            fw1 = 2*(precision * recall)/ (precision + recall) if precision + recall != 0 else 0
            end = time.perf_counter()
            time_miniute = (end-start)/60

            f = open(self.args.save_path + '/detection_info.csv', 'a', encoding='utf-8')
            csv_write = csv.writer(f)
            csv_write.writerow(['record', 'TN','FP','FN','TP','TPR','FPR', 'AUC', 'target'])
            csv_write.writerow([args.result_file, tn, fp, fn, tp, TPR, FPR, auc, 'None'])
            f.close()

                
    def detection(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.filtering()
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    strip.add_arguments(parser)
    args = parser.parse_args()
    strip_method = strip(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = strip_method.detection(args.result_file)