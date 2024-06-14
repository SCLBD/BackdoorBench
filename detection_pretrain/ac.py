'''
Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering
This file is modified based on the following source:
link : https://github.com/wanlunsec/Beatrix/blob/master/defenses/activation_cluster/activation_clustering.py
The detection method is called AC.

@article{chen2018detecting,
    title={Detecting backdoor attacks on deep neural networks by activation clustering},
    author={Chen, Bryant and Carvalho, Wilka and Baracaldo, Nathalie and Ludwig, Heiko and Edwards, Benjamin and Lee, Taesung and Molloy, Ian and Srivastava, Biplav},
    journal={arXiv preprint arXiv:1811.03728},
    year={2018}}

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. ac detection:
        a. classify data by activation results
        b. identify backdoor data according to classification results
    4. compute TPR and FPR
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
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import csv
from sklearn import metrics

def get_features(name, model, dataloader):
    with torch.no_grad():
        model.eval()
        TOO_SMALL_ACTIVATIONS = 32
    activations_all = []
    for i, (x_batch, y_batch) in enumerate(dataloader):
        assert name in ['preactresnet18', 'vgg19','vgg19_bn', 'resnet18', 'mobilenet_v3_large', 'densenet161', 'efficientnet_b3','convnext_tiny','vit_b_16']
        x_batch = x_batch.to(args.device)
        if name == 'preactresnet18':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.avgpool.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'vgg19':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.features.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'vgg19_bn':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.features.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'resnet18':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.layer4.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'mobilenet_v3_large':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.avgpool.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'densenet161':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.features.register_forward_hook(layer_hook)
            _ = model(x_batch)
            outs[0] = torch.nn.functional.relu(outs[0])
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'efficientnet_b3':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.avgpool.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'convnext_tiny':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.avgpool.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'vit_b_16':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                inps.append(inp[0].data)
            hook = model[1].heads.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = inps[0].view(inps[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()

    activations_all = torch.cat(activations_all, axis=0)
    return activations_all


class ac(defense):

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
        parser.add_argument('--yaml_path', type=str, default="./config/detection/ac/cifar10.yaml", help='the path of yaml')


    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection/ac_pretrain/'
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
    
    def cluster_metrics(cluster_1, cluster_0):

        num = len(cluster_1) + len(cluster_0)
        features = torch.cat([cluster_1, cluster_0], dim=0)

        labels = torch.zeros(num)
        labels[:len(cluster_1)] = 1
        labels[len(cluster_1):] = 0

        ## Raw Silhouette Score
        raw_silhouette_score = silhouette_score(features, labels)
        return raw_silhouette_score


    def cleanser(self, images_poison, labels_poison, test_tran, model, num_classes, args, clusters=2):
        inspection_set = xy_iter(images_poison, labels_poison,transform=test_tran)
        inspection_split_loader = DataLoader(inspection_set,batch_size=args.batch_size, shuffle=False)

        class_indices = []
        for i in range(num_classes):
            idx = np.where(np.array(labels_poison)==i)[0]
            class_indices.append(idx)

        suspicious_indices = []
        feats = get_features(args.model, model, inspection_split_loader)

        for target_class in range(num_classes):

            print('class - %d' % target_class)

            if len(class_indices[target_class]) <= 1: continue # no need to perform clustering...

            temp_feats = [feats[temp_idx].unsqueeze(dim=0) for temp_idx in class_indices[target_class]]
            temp_feats = torch.cat( temp_feats , dim=0)
            temp_feats = temp_feats - temp_feats.mean(dim=0)

            from sklearn.decomposition import FastICA
            X = temp_feats.cpu().numpy()
            transformer = FastICA(n_components=self.args.nb_dims,
                                  random_state=self.args.random_seed,
                                  whiten='unit-variance')
            X_transformed = transformer.fit_transform(X)
            projected_feats = X_transformed

            # _, _, V = torch.svd(temp_feats, compute_uv=True, some=False)

            # axes = V[:, :10]
            # projected_feats = torch.matmul(temp_feats, axes)
            # projected_feats = projected_feats.cpu().numpy()

            logging.info(projected_feats.shape)

            logging.info('start k-means')
            kmeans = KMeans(n_clusters=self.args.nb_clusters).fit(projected_feats)
            logging.info('end k-means')

            if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
                clean_label = 1
            else:
                clean_label = 0

            outliers = []
            for (bool, idx) in zip((kmeans.labels_ != clean_label).tolist(), list(range(len(kmeans.labels_)))):
                if bool:
                    outliers.append(class_indices[target_class][idx])

            score = silhouette_score(projected_feats, kmeans.labels_)
            logging.info('[class-%d] silhouette_score = %f' % (target_class, score))
            if len(outliers) < len(kmeans.labels_) * 0.35:
                logging.info(f"Outlier Num in Class {target_class}:", len(outliers))
                suspicious_indices += outliers

        return suspicious_indices
    
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

        
        ### b. load training dataset with poison samples
        images_poison = []
        labels_poison = []
        for img, label, *other_info in bd_train_dataset:
            images_poison.append(img)
            labels_poison.append(label)

        ### c. get features of training datase

        suspect_index = self.cleanser(images_poison, labels_poison, test_tran, model, self.args.num_classes, args,clusters=2)
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
            csv_write.writerow(['record', 'TN','FP','FN','TP','TPR','FPR', 'target'])
            csv_write.writerow([args.result_file, tn,fp,fn,tp, 0,0, 'None'])
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
            csv_write.writerow(['record', 'TN','FP','FN','TP','TPR','FPR', 'weight_F_score', 'target'])
            csv_write.writerow([args.result_file, tn, fp, fn, tp, TPR, FPR, fw1,'Unknown'])
            f.close()


                
    def detection(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.filtering()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    ac.add_arguments(parser)
    args = parser.parse_args()
    ac_method = ac(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = ac_method.detection(args.result_file)