'''
Demon in the Variant: Statistical Analysis of DNNs for Robust Backdoor Contamination Detection

This file is modified based on the following source:
link : https://github.com/TDteach/Demon-in-the-Variant/blob/master/pysrc/SCAn.py
The detection method is called SCAn.
@inproceedings{tang2021demon,
    title={Demon in the variant: Statistical analysis of $\{$DNNs$\}$ for robust backdoor contamination detection},
    author={Tang, Di and Wang, XiaoFeng and Tang, Haixu and Zhang, Kehuan},
    booktitle={30th USENIX Security Symposium (USENIX Security 21)},
    pages={1541--1558},
    year={2021}}

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. SCAn detection:
        a. Leverage the target model to generate representations for all input images.
        b. Estimate the parameters by running an EM algorithm.
        c. calculate the identity vector and decompose the representations.
        d. estimate the parameters for the mixture model.
        e. perform the likelihood ratio test.
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
import csv
from sklearn import metrics

def get_features_labels(args, model, target_layer, data_loader):

    def feature_hook(module, input_, output_):
        global feature_vector
        feature_vector = output_
        return None

    h = target_layer.register_forward_hook(feature_hook)

    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, *other_info) in enumerate(data_loader):
            global feature_vector
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            feature_vector = torch.sum(torch.flatten(feature_vector, 2), 2)
            current_feature = feature_vector.detach().cpu().numpy()
            current_labels = targets.cpu().numpy()

            # Store features
            features.append(current_feature)
            labels.append(current_labels)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    h.remove()  # Rmove the hook

    return features, labels

EPS = 1e-5
class SCAn:
    def __init__(self, args):
        self.args = args
        pass

    def calc_final_score(self, lc_model=None):
        if lc_model is None:
            lc_model = self.lc_model
        sts = lc_model['sts']
        y = sts[:, 1]
        ai = self.calc_anomaly_index(y / np.max(y))
        return ai

    def calc_anomaly_index(self, a):
        ma = np.median(a)
        b = abs(a - ma)
        mm = np.median(b) * 1.4826
        index = b / mm
        return index

    def build_global_model(self, reprs, labels, n_classes):
        N = reprs.shape[0]  # num_samples
        M = reprs.shape[1]  # len_features
        L = n_classes

        mean_a = np.mean(reprs, axis=0)
        X = reprs - mean_a

        cnt_L = np.zeros(L)
        mean_f = np.zeros([L, M])
        for k in range(L):
            idx = (labels == k)
            cnt_L[k] = np.sum(idx)
            mean_f[k] = np.mean(X[idx], axis=0)

        u = np.zeros([N, M])
        e = np.zeros([N, M])
        for i in range(N):
            k = labels[i]
            u[i] = mean_f[k]  # class-mean
            e[i] = X[i] - u[i]  # sample-variantion
        Su = np.cov(np.transpose(u))
        Se = np.cov(np.transpose(e))

        # EM
        dist_Su = 1e5
        dist_Se = 1e5
        n_iters = 0
        while (dist_Su + dist_Se > EPS) and (n_iters < 100):
            n_iters += 1
            last_Su = Su
            last_Se = Se

            F = np.linalg.pinv(Se)
            SuF = np.matmul(Su, F)

            G_set = list()
            for k in range(L):
                G = -np.linalg.pinv(cnt_L[k] * Su + Se)
                G = np.matmul(G, SuF)
                G_set.append(G)

            u_m = np.zeros([L, M])
            e = np.zeros([N, M])
            u = np.zeros([N, M])

            for i in range(N):
                vec = X[i]
                k = labels[i]
                G = G_set[k]
                dd = np.matmul(np.matmul(Se, G), np.transpose(vec))
                u_m[k] = u_m[k] - np.transpose(dd)

            for i in range(N):
                vec = X[i]
                k = labels[i]
                e[i] = vec - u_m[k]
                u[i] = u_m[k]

            # max-step
            Su = np.cov(np.transpose(u))
            Se = np.cov(np.transpose(e))

            dif_Su = Su - last_Su
            dif_Se = Se - last_Se

            dist_Su = np.linalg.norm(dif_Su)
            dist_Se = np.linalg.norm(dif_Se)
            # print(dist_Su,dist_Se)

        gb_model = dict()
        gb_model['Su'] = Su
        gb_model['Se'] = Se
        gb_model['mean'] = mean_f
        self.gb_model = gb_model
        return gb_model

    def build_local_model(self, reprs, labels, gb_model, n_classes):
        Su = gb_model['Su']
        Se = gb_model['Se']

        F = np.linalg.pinv(Se)
        N = reprs.shape[0]
        M = reprs.shape[1]
        L = n_classes

        mean_a = np.mean(reprs, axis=0)
        X = reprs - mean_a

        class_score = np.zeros([L, 3])
        u1 = np.zeros([L, M])
        u2 = np.zeros([L, M])
        split_rst = list()

        for k in range(L):
            selected_idx = (labels == k)
            cX = X[selected_idx]
            subg, i_u1, i_u2 = self.find_split(cX, F)
            # print("subg",subg)

            i_sc = self.calc_test(cX, Su, Se, F, subg, i_u1, i_u2)
            split_rst.append(subg)
            u1[k] = i_u1
            u2[k] = i_u2
            class_score[k] = [k, i_sc[0][0], np.sum(selected_idx)]

        lc_model = dict()
        lc_model['sts'] = class_score
        lc_model['mu1'] = u1
        lc_model['mu2'] = u2
        lc_model['subg'] = split_rst

        self.lc_model = lc_model
        return lc_model

    def find_split(self, X, F):
        N = X.shape[0]
        M = X.shape[1]
        subg = np.random.rand(N)

        if (N == 1):
            subg[0] = 0
            return (subg, X.copy(), X.copy())

        if np.sum(subg >= 0.5) == 0:
            subg[0] = 1
        if np.sum(subg < 0.5) == 0:
            subg[0] = 0
        last_z1 = -np.ones(N)

        # EM
        steps = 0
        while (np.linalg.norm(subg - last_z1) > EPS) and (np.linalg.norm((1 - subg) - last_z1) > EPS) and (steps < 100):
            steps += 1
            last_z1 = subg.copy()

            # max-step
            # calc u1 and u2
            idx1 = (subg >= 0.5)
            idx2 = (subg < 0.5)
            if (np.sum(idx1) == 0) or (np.sum(idx2) == 0):
                break
            if np.sum(idx1) == 1:
                u1 = X[idx1]
            else:
                u1 = np.mean(X[idx1], axis=0)
            if np.sum(idx2) == 1:
                u2 = X[idx2]
            else:
                u2 = np.mean(X[idx2], axis=0)

            bias = np.matmul(np.matmul(u1, F), np.transpose(u1)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
            e2 = u1 - u2  # (64,1)
            for i in range(N):
                e1 = X[i]
                delta = np.matmul(np.matmul(e1, F), np.transpose(e2))
                if bias - self.args.frac * delta < 0:
                    subg[i] = 1
                else:
                    subg[i] = 0

        return (subg, u1, u2)

    def calc_test(self, X, Su, Se, F, subg, u1, u2):
        N = X.shape[0]
        M = X.shape[1]

        G = -np.linalg.pinv(N * Su + Se)
        mu = np.zeros([1, M])
        SeG = np.matmul(Se,G)
        for i in range(N):
            vec = X[i]
            dd = np.matmul(SeG, np.transpose(vec))
            mu = mu - dd

        b1 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u1, F), np.transpose(u1))
        b2 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
        n1 = np.sum(subg >= 0.5)
        n2 = N - n1
        sc = n1 * b1 + n2 * b2

        for i in range(N):
            e1 = X[i]
            if subg[i] >= 0.5:
                e2 = mu - u1
            else:
                e2 = mu - u2
            sc -= 2 * np.matmul(np.matmul(e1, F), np.transpose(e2))

        return sc / N



class scan(defense):

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
        parser.add_argument('--yaml_path', type=str, default="./config/detection/scan/cifar10.yaml", help='the path of yaml')
        parser.add_argument('--clean_sample_num', type=int)
        parser.add_argument('--target_layer', type=str)

        parser.add_argument('--frac', type=float, default=0.5)

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + f'/detection/scan_frac_{self.args.frac}_pretrain/'
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
        self.args.frac = 1/self.args.frac

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

        module_dict = dict(model.named_modules())
        target_layer = module_dict[args.target_layer]

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
        class_idx_whole = np.concatenate(class_idx_whole, axis=0)
        image_c = [images[i] for i in class_idx_whole]
        label_c = [labels[i] for i in class_idx_whole]

        clean_dataset = xy_iter(image_c, label_c,transform=test_tran)
        clean_dataloader = DataLoader(clean_dataset, self.args.batch_size, shuffle=True)
        clean_features,clean_labels = get_features_labels(args, model, target_layer, clean_dataloader)

        ### c. load training dataset with poison samples
        images_poison = []
        labels_poison = []
        for img, label, *other_info in bd_train_dataset:
            images_poison.append(img)
            labels_poison.append(label)

        ### d. get features of training dataset
        train_dataset = xy_iter(images_poison, labels_poison,transform=test_tran)
        train_dataloader = DataLoader(train_dataset, self.args.batch_size, shuffle=False)
        train_features, train_labels = get_features_labels(args, model, target_layer, train_dataloader)
        
        feats_inspection = np.array(train_features)
        class_indices_inspection = np.array(train_labels)

        feats_clean = np.array(clean_features)
        class_indices_clean = np.array(clean_labels)

        scan = SCAn()
        gb_model = scan.build_global_model(feats_clean, class_indices_clean, self.args.num_classes)
        size_inspection_set = len(feats_inspection)
        feats_all = np.concatenate([feats_inspection, feats_clean])
        class_indices_all = np.concatenate([class_indices_inspection, class_indices_clean])
        lc_model = scan.build_local_model(feats_all, class_indices_all, gb_model, self.args.num_classes)
        score = scan.calc_final_score(lc_model)
        threshold = np.exp(2)

        suspicious_indices = []
        flag_list = []

        for target_class in range(args.num_classes):

            print('[class-%d] outlier_score = %f' % (target_class, score[target_class]) )

            if score[target_class] <= threshold: 
                continue
            flag_list.append([target_class, score[target_class]])
            tar_label = (class_indices_all == target_class)
            all_label = np.arange(len(class_indices_all))
            tar = all_label[tar_label]

            cluster_0_indices = []
            cluster_1_indices = []

            cluster_0_clean = []
            cluster_1_clean = []

            for index, i in enumerate(lc_model['subg'][target_class]):
                if i == 1:
                    if tar[index] > size_inspection_set:
                        cluster_1_clean.append(tar[index])
                    else:
                        cluster_1_indices.append(tar[index])
                else:
                    if tar[index] > size_inspection_set:
                        cluster_0_clean.append(tar[index])
                    else:
                        cluster_0_indices.append(tar[index])


            if len(cluster_0_clean) < len(cluster_1_clean): # if most clean samples are in cluster 1
                suspicious_indices += cluster_0_indices
            else:
                suspicious_indices += cluster_1_indices
                
        true_index = np.zeros(len(images_poison))
        for i in range(len(true_index)):
            if i in pindex:
                true_index[i] = 1
        if len(suspicious_indices)==0:
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
            logging.info("Flagged label list: {}".format(",".join(["{}: {}".format(y_label, s) for y_label, s in flag_list])))
            findex = np.zeros(len(images_poison))
            for i in range(len(findex)):
                if i in suspicious_indices:
                    findex[i] = 1
            if np.sum(findex) == 0:
                tn = len(true_index) - np.sum(true_index)
                fp = np.sum(true_index)
                fn = 0
                tp = 0
            else:
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
            csv_write.writerow(['record', 'TN','FP','FN','TP','TPR','FPR', 'target'])
            csv_write.writerow([args.result_file, tn, fp, fn, tp, TPR, FPR, [i for i,j in flag_list]])
            f.close()


    def detection(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.filtering()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    scan.add_arguments(parser)
    args = parser.parse_args()
    scan_method = scan(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = scan_method.detection(args.result_file)
