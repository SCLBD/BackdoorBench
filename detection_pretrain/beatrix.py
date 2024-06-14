'''
The Beatrix Resurrections: Robust Backdoor Detection via Gram Matrices
This file is modified based on the following source:
link : https://github.com/wanlunsec/Beatrix/blob/master/defenses/Beatrix/Beatrix.py
The detection method is called Beatrix.

@article{ma2022beatrix,
          title={The" Beatrix''Resurrections: Robust Backdoor Detection via Gram Matrices},
          author={Ma, Wanlun and Wang, Derui and Sun, Ruoxi and Xue, Minhui and Wen, Sheng and Xiang, Yang},
          journal={arXiv preprint arXiv:2209.11715},
          year={2022}}

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. Beatrix detection:
        a. extract features of clean samples.
        b. extract features of poisoned samples.
        c. analyze features by Gram Matrices.
        d. measure deviations and compute the threshold.
        e. detect poisoned samples by threshold.
    4. compute TPR and FPR

'''
import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
sys.path.append('../')
sys.path.append(os.getcwd())

from sklearn.utils import shuffle
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
import torch.nn.functional as F
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
def cal(true, pred):
    TN, FP, FN, TP = confusion_matrix(true, pred).ravel()
    return TN, FP, FN, TP 
def metrix(TN, FP, FN, TP):
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    precision = TP/(TP+FP)
    acc = (TP+TN)/(TN+FP+FN+TP)
    return TPR, FPR, precision, acc

def threshold_determine(clean_feature_target, ood_detection):
    test_deviations_list = []
    step = 5
    for i in range(step):
        index_mask = np.ones((len(clean_feature_target),))
        index_mask[i*int(len(clean_feature_target)//step):(i+1)*int(len(clean_feature_target)//step)] = 0
        clean_feature_target_train= clean_feature_target[np.where(index_mask == 1)]
        clean_feature_target_test = clean_feature_target[np.where(index_mask == 0)]
        ood_detection.train(in_data=[clean_feature_target_train],)
        test_deviations = ood_detection.get_deviations_([clean_feature_target_test])
        test_deviations_list.append(test_deviations)
    test_deviations = np.concatenate(test_deviations_list,0)
    test_deviations_sort = np.sort(test_deviations,0)
    percentile_95 = test_deviations_sort[int(len(test_deviations_sort)*0.95)][0]
    percentile_99 = test_deviations_sort[int(len(test_deviations_sort)*0.99)][0]
    return percentile_95, percentile_99

def gaussian_kernel(x1, x2, kernel_mul=2.0, kernel_num=5, fix_sigma=0, mean_sigma=0):
    x1_sample_size = x1.shape[0]
    x2_sample_size = x2.shape[0]
    x1_tile_shape = []
    x2_tile_shape = []
    norm_shape = []
    for i in range(len(x1.shape) + 1):
        if i == 1:
            x1_tile_shape.append(x2_sample_size)
        else:
            x1_tile_shape.append(1)
        if i == 0:
            x2_tile_shape.append(x1_sample_size)
        else:
            x2_tile_shape.append(1)
        if not (i == 0 or i == 1):
            norm_shape.append(i)

    tile_x1 = torch.unsqueeze(x1, 1).repeat(x1_tile_shape)
    tile_x2 = torch.unsqueeze(x2, 0).repeat(x2_tile_shape)
    L2_distance = torch.square(tile_x1 - tile_x2).sum(dim=norm_shape)
    if fix_sigma:
        bandwidth = fix_sigma
    elif mean_sigma:
        bandwidth = torch.mean(L2_distance)
    else:
        bandwidth = torch.median(L2_distance.reshape(L2_distance.shape[0],-1))
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    print(bandwidth_list)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def kmmd_dist(x1, x2):
    X_total = torch.cat([x1,x2],0)
    Gram_matrix = gaussian_kernel(X_total,X_total,kernel_mul=2.0, kernel_num=2, fix_sigma=0, mean_sigma=0)
    n = int(x1.shape[0])
    m = int(x2.shape[0])
    x1x1 = Gram_matrix[:n, :n]
    x2x2 = Gram_matrix[n:, n:]
    x1x2 = Gram_matrix[:n, n:]
    diff = torch.mean(x1x1) + torch.mean(x2x2) - 2 * torch.mean(x1x2)
    diff = (m*n)/(m+n)*diff
    return diff.cpu().numpy()

class Feature_Correlations:
    def __init__(self,POWER_list, mode='mad'):
        self.power = POWER_list
        self.mode = mode

    def train(self, in_data):
        self.in_data = in_data
        if 'mad' in self.mode:
            self.medians, self.mads = self.get_median_mad(self.in_data)
            self.mins, self.maxs = self.minmax_mad()


    def minmax_mad(self):
        mins = []
        maxs = []
        for L, mm in enumerate(zip(self.medians,self.mads)):
            medians, mads = mm[0], mm[1]
            if L==len(mins):
                mins.append([None]*len(self.power))
                maxs.append([None]*len(self.power))
            for p, P in enumerate(self.power):
                    mins[L][p] = medians[p]-mads[p]*10
                    maxs[L][p] = medians[p]+mads[p]*10
        return mins, maxs

    def G_p(self, ob, p):
        temp = ob.detach()
        temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
        temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1))))
        temp = temp.triu()
        temp = temp.sign()*torch.abs(temp)**(1/p)
        temp = temp.reshape(temp.shape[0],-1)
        self.num_feature = temp.shape[-1]/2
        return temp

    def get_median_mad(self, feat_list):
        medians = []
        mads = []
        for L,feat_L in enumerate(feat_list):
            if L==len(medians):
                medians.append([None]*len(self.power))
                mads.append([None]*len(self.power))
            for p,P in enumerate(self.power):
                g_p = self.G_p(feat_L,P)
                current_median = g_p.median(dim=0,keepdim=True)[0]
                current_mad = torch.abs(g_p - current_median).median(dim=0,keepdim=True)[0]
                medians[L][p] = current_median
                mads[L][p] = current_mad
        return medians, mads

    def get_deviations_(self, feat_list):
        deviations = []
        batch_deviations = []
        for L,feat_L in enumerate(feat_list):
            dev = 0
            for p,P in enumerate(self.power):
                g_p = self.G_p(feat_L,P)
                dev +=  (F.relu(self.mins[L][p]-g_p)/torch.abs(self.mins[L][p]+10**-6)).sum(dim=1,keepdim=True)
                dev +=  (F.relu(g_p-self.maxs[L][p])/torch.abs(self.maxs[L][p]+10**-6)).sum(dim=1,keepdim=True)
            batch_deviations.append(dev.cpu().detach().numpy())
        batch_deviations = np.concatenate(batch_deviations,axis=1)
        deviations.append(batch_deviations)
        deviations = np.concatenate(deviations,axis=0) /self.num_feature /len(self.power)
        return deviations

    def get_deviations(self, feat_list):
        deviations = []
        batch_deviations = []
        for L,feat_L in enumerate(feat_list):
            dev = 0
            for p,P in enumerate(self.power):
                g_p = self.G_p(feat_L,P)
                dev += torch.sum(torch.abs(g_p-self.medians[L][p])/(self.mads[L][p]+1e-6),dim=1,keepdim=True)
            batch_deviations.append(dev.cpu().detach().numpy())
        batch_deviations = np.concatenate(batch_deviations,axis=1)
        deviations.append(batch_deviations)
        deviations = np.concatenate(deviations,axis=0)/self.num_feature /len(self.power)
        return deviations

class LayerActivations:
    def __init__(self, model,args):
        self.args = args
        self.model = model
        self.model.eval()
        self.build_hook()

    def build_hook(self):
        module_dict = dict(self.model.named_modules())
        target_layer = module_dict[args.target_layer]
        self.hook = target_layer.register_forward_hook(self.hook_fn)


    def hook_fn(self, module, input, output):
        self.features = input[0]
        # self.features = output

    def remove_hook(self):
        self.hook.remove()

    def run_hook(self,x):
        self.model(x)
        # self.remove_hook()
        return self.features


class beatrix(defense):

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
        parser.add_argument('--yaml_path', type=str, default="./config/detection/beatrix/cifar10.yaml", help='the path of yaml')
        parser.add_argument('--clean_sample_num', type=int)
        parser.add_argument('--target_layer', type=str)

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection/beatrix_pretrain/'
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

    def get_feature_predict(self, dataset, model, intermedia_feature):
        model.eval()
        data_loader = DataLoader(dataset, self.args.batch_size, shuffle=False)
        features = []
        preds_label = []
        for i, (input, label) in enumerate(data_loader):
            input = input.to(self.args.device)
            label = label.to(self.args.device)
            features.append(intermedia_feature.run_hook(input).detach().cpu())
            output = model(input)
            preds_label.append(torch.argmax(output, 1).detach().cpu())

        preds_label = torch.cat(preds_label,axis=0)
        features = torch.cat(features,axis=0)
        return features, preds_label
    
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
        intermedia_feature = LayerActivations(model.to(self.args.device),self.args)
        bd_train_dataset = self.result['bd_train'].wrapped_dataset
        pindex = np.where(np.array(bd_train_dataset.poison_indicator) == 1)[0]

        clean_test_dataset = self.result['clean_test'].wrapped_dataset

        ### b. find a clean sample from test dataset
        images = []
        labels = []
        for img, label in clean_test_dataset:
            images.append(img)
            labels.append(label)
        
        test_dataset = xy_iter(images, labels,transform=test_tran)
        data_clean_loader = DataLoader(test_dataset, batch_size=self.args.batch_size,drop_last=False, shuffle=False,pin_memory=False)
        result = []
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(data_clean_loader):

                input, label = input.to(self.args.device), label.to(self.args.device)
                outputs = model(input)
                _, predicted = outputs.max(1)
                result.append(predicted.cpu().numpy())
        result = np.concatenate(result, axis=0)

        labels = result
        class_idx_whole = []
        num = int(self.args.clean_sample_num / self.args.num_classes)
        if num == 0:
            num = 1
        for i in range(self.args.num_classes):
            class_idx_whole.append(np.random.choice(np.where(np.array(labels)==i)[0], num))
        class_idx_whole = np.concatenate(class_idx_whole, axis=0)
        image_c = [images[i] for i in class_idx_whole]
        label_c = [labels[i] for i in class_idx_whole]

        ## c. get clean feature and pred label
        clean_dataset = xy_iter(image_c, label_c,transform=test_tran)
        clean_features, clean_preditions = self.get_feature_predict(clean_dataset, model, intermedia_feature)
        (clean_features, clean_preditions) = shuffle(clean_features, clean_preditions)
        
        ## d. use gram-matrix OOD detection
        self.order_list = [1,2,3,4,5,6,7,8]
        ood_detection = Feature_Correlations(POWER_list=self.order_list,mode='mad')
        
        ## e. load training data with poison samples
        images_poison = []
        labels_poison = []
        for img, label, _,_,_ in bd_train_dataset:
            images_poison.append(img)
            labels_poison.append(label)

        ## f. get training feature and pred label
        train_dataset = xy_iter(images_poison, labels_poison,transform=test_tran)
        train_features, train_preditions = self.get_feature_predict(train_dataset, model, intermedia_feature)
        # (train_features, train_preditions) = shuffle(train_features, train_preditions)

        threshold_list = []
        suspect_index_95 = []
        suspect_index_99 = []
        J_t = []
        for test_target_label in range(args.num_classes):
            print(f'*****class:{test_target_label}*****')
            clean_feature_defend = clean_features[np.where(clean_preditions==test_target_label)]

            threshold_95, threshold_99 = threshold_determine(clean_feature_defend, ood_detection)
            threshold_list.append([test_target_label,threshold_95, threshold_99])

            ood_detection.train(in_data=[clean_feature_defend])
            
            class_idx_current = np.where(train_preditions==test_target_label)[0]
            class_feature_test = train_features[class_idx_current]
            class_test_deviations = ood_detection.get_deviations_([class_feature_test])

            ood_label_95 = np.where(class_test_deviations > threshold_95)[0]
            ood_label_99 = np.where(class_test_deviations > threshold_99)[0]

            suspect_index_95.append(class_idx_current[ood_label_95])
            suspect_index_99.append(class_idx_current[ood_label_99])

            ### find target label start###
            ood_label_95 = np.where(class_test_deviations > threshold_95, 1, 0).squeeze()
            ood_label_99 = np.where(class_test_deviations > threshold_99, 1, 0).squeeze()

            clean_feature_group = class_feature_test[np.where(ood_label_95==0)]
            bd_feature_group = class_feature_test[np.where(ood_label_95==1)]
            clean_feature_flat = torch.mean(clean_feature_group,dim=(2,3))
            bd_feature_flat = torch.mean(bd_feature_group,dim=(2,3))
            if bd_feature_flat.shape[0] < 1:
                kmmd = np.array([0.0])
            else:
                kmmd = kmmd_dist(clean_feature_flat[:500], bd_feature_flat[:500])
            print(f'KMMD:{kmmd.item()}.')

            J_t.append(kmmd.item())
        
        J_t = np.asarray(J_t)
        J_t_median = np.median(J_t)
        J_MAD = np.median(np.abs(J_t - J_t_median))
        J_star = np.abs(J_t - J_t_median)/1.4826/(J_MAD+1e-6)

        flag_list = []
        for i,J_star_i in enumerate(J_star):
            if J_star_i>np.exp(2):
                flag_list.append([i,J_star_i])
        logging.info("Flagged label list: {}".format(",".join(["{}: {}".format(y_label, J_s) for y_label, J_s in flag_list])))
        ### find target label end###
        
        suspect_index_95 = np.concatenate(suspect_index_95, axis=0)
        
        true_index = np.zeros(len(images_poison))
        for i in range(len(true_index)):
            if i in pindex:
                true_index[i] = 1
        if len(suspect_index_95)==0:        
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
            findex_95 = np.zeros(len(images_poison))
            for i in range(len(findex_95)):
                if i in suspect_index_95:
                    findex_95[i] = 1

            
            tn, fp, fn, tp = cal(true_index, findex_95)
            TPR, FPR, precision, acc = metrix(tn, fp, fn, tp)
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
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    beatrix.add_arguments(parser)
    args = parser.parse_args()
    beatrix_method = beatrix(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = beatrix_method.detection(args.result_file)