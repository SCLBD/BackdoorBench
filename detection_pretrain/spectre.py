'''
SPECTRE: Defending Against Backdoor Attacks Using Robust Statistics
This file is modified based on the following source:
link : https://github.com/SewoongLab/spectre-defense

@inproceedings{hayase2021spectre,
    title={Spectre: Defending against backdoor attacks using robust statistics},
    author={Hayase, Jonathan and Kong, Weihao and Somani, Raghav and Oh, Sewoong},
    booktitle={International Conference on Machine Learning},
    pages={4129--4139},
    year={2021},
    organization={PMLR}}

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. Spectral defense:
        a. get the activation as representation for each data
        b. run quantum filter for k different squared values
        c. keep the best k and correspoding selected samples as backdoor samples
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
from defense.base import defense

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import PureCleanModelTrainer
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from sklearn.metrics import confusion_matrix
import csv
from sklearn import metrics
import subprocess
from sklearn.decomposition import PCA
from numpy.linalg import norm, inv
from scipy.linalg import sqrtm
from detection_pretrain.spectre_utils import *
from utils.bd_dataset_v2 import xy_iter
from torch.utils.data import DataLoader
from scipy.linalg import svd

def get_features(name, model, dataloader, target_layer):
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
            hook = target_layer.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'vgg19':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = target_layer.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'vgg19_bn':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = target_layer.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'resnet18':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = target_layer.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'mobilenet_v3_large':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = target_layer.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'densenet161':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = target_layer.register_forward_hook(layer_hook)
            _ = model(x_batch)
            outs[0] = torch.nn.functional.relu(outs[0])
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'efficientnet_b3':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = target_layer.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'convnext_tiny':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = target_layer.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()
        elif name == 'vit_b_16':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                inps.append(inp[0].data)
            hook = target_layer.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = inps[0].view(inps[0].size(0), -1)
            activations_all.append(activations.cpu())
            hook.remove()

    activations_all = torch.cat(activations_all, axis=0)
    return activations_all

def rcov_quantum_filter(reps, eps, k, alpha=4, tau=0.1, limit1=2, limit2=1.5):
    n, d = reps.shape

    # PCA
    pca = PCA(n_components=k)
    reps_pca = pca.fit_transform(reps)

    if k == 1:
        reps_estimated_white = reps_pca
        sigma_prime = np.ones((1, 1))
    else:
        selected = cov_estimation_iterate(reps_pca, eps/n, tau, None, limit=round(limit1*eps))
        reps_pca_selected = reps_pca[selected,:]
        sigma = np.cov(reps_pca_selected, rowvar=False, bias=False)
        reps_estimated_white = np.linalg.solve(sqrtm(sigma), reps_pca.T).T
        sigma_prime = np.cov(reps_estimated_white, rowvar=False, bias=False)

    I = np.eye(sigma_prime.shape[0])
    M = np.exp(alpha * (sigma_prime - I) / (norm(sigma_prime, 2) - 1)) if k > 1 else np.ones((1, 1))
    M /= np.trace(M)
    estimated_poison_ind = k_lowest_ind(-np.array([x.T @ M @ x for x in reps_estimated_white]), round(limit2*eps))

    return ~estimated_poison_ind


def rcov_auto_quantum_filter(reps, eps, poison_indices, alpha=4, tau=0.1, limit1=2, limit2=1.5):
    
    pca = PCA(n_components=64)
    reps_pca = pca.fit_transform(reps)  # 降维后的数据
    U = pca.components_  # 主成分矩阵

    best_opnorm, best_selected, best_k = -float('inf'), None, None
    squared_values = [int(x) for x in np.linspace(1, np.sqrt(64), 8) ** 2]

    for k in squared_values:
        selected = rcov_quantum_filter(reps, eps, k, alpha, tau, limit1=limit1, limit2=limit2)
        selected_matrix = reps_pca[selected,:].T
        cov_matrix = np.cov(selected_matrix)
        transformed = np.linalg.solve(sqrtm(cov_matrix), reps_pca.T)
        cov_matrix_prime = np.cov(transformed)
        I = np.eye(cov_matrix_prime.shape[0]) 
        U, s, Vh = svd(cov_matrix_prime)
        opnorm = s[0]

        M = np.exp(alpha * (cov_matrix_prime - I) / (opnorm - 1))    
        M /= np.trace(M)
        op = np.trace(cov_matrix_prime * M) / np.trace(M)
        poison_removed = sum([not selected[i] for i in poison_indices])
        if op > best_opnorm:
            best_opnorm, best_selected, best_k = op, selected, k
    return best_selected, best_opnorm


class spectre(defense):

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
        parser.add_argument('--yaml_path', type=str, default="./config/detection/spectre/cifar10.yaml", help='the path of yaml')

        #set the parameter for the spectral defense
        parser.add_argument('--percentile', type=float, help='close to pratio')
        parser.add_argument('--target_layer', type=str)
        parser.add_argument('--clean_sample_num', type=int)

        

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection/spectre_pretrain/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
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
    def metrix(self,TN, FP, FN, TP):
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        precision = TP/(TP+FP)
        acc = (TP+TN)/(TN+FP+FN+TP)
        return TPR, FPR, precision, acc

    def filtering(self):
        start = time.perf_counter()
        self.set_devices()
        fix_random(self.args.random_seed)

        ### a. prepare the model and dataset
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)

        module_dict = dict(model.named_modules())
        target_layer = module_dict[args.target_layer]
        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]), train = False)
        train_dataset = self.result['bd_train'].wrapped_dataset
        data_set_without_tran = train_dataset
        data_set_o = self.result['bd_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = train_tran
        data_set_o.wrapped_dataset.getitem_all = False
        pindex = np.where(np.array(data_set_o.poison_indicator) == 1)[0]
        dataset = data_set_o

        images_poison = []
        labels_poison = []
        for img, label in dataset.wrapped_dataset:
            images_poison.append(img)
            labels_poison.append(label)

        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        bd_train_dataset = xy_iter(images_poison, labels_poison,transform=test_tran)
        bd_train_dataloader = DataLoader(bd_train_dataset,batch_size=self.args.batch_size, shuffle=False)

        feats = get_features(args.model, model, bd_train_dataloader)
        poison_labels = range(self.args.num_classes)
        
        scores = []
        suspicious_indices = []
        for target_label in poison_labels:
            lbl = target_label
            cur_indices = [i for i,v in enumerate(labels_poison) if v==lbl]

            pt = 0
            cur_poisoned = []
            for j in cur_indices:
                if j in pindex:
                    cur_poisoned.append(pt)
                pt += 1

            ### b. get the activation as representation for each data
            full_cov = [feats[temp_idx].cpu().numpy() for temp_idx in cur_indices]
            full_cov = np.array(full_cov)
            n, _ = full_cov.shape
            eps = self.args.percentile * len(labels_poison)
            if eps <= 0:
                eps = round(0.1 * n)
            if eps > 0.33 * n:
                eps = round(0.33 * n)
            if n < 500:
                if eps > 0.1 * n:
                    eps = round( 0.1 * n)
            eps = int(eps)
            removed = round(1.5 * eps)
            print("%s: Running quantum filter\n", target_label)
            quantum_poison_ind, opnorm = rcov_auto_quantum_filter(full_cov, eps, cur_poisoned)
            quantum_poison_ind = np.logical_not(quantum_poison_ind)
            poison_removed = sum(quantum_poison_ind[cur_poisoned])
            clean_removed = removed - poison_removed

            scores.append(opnorm.item())
            suspicious_class_indices_mask = quantum_poison_ind
            suspicious_class_indices = torch.tensor(suspicious_class_indices_mask).nonzero().squeeze(1)
            cur_class_indices = torch.tensor(cur_indices)
            suspicious_indices.append(cur_class_indices[suspicious_class_indices])
        suspicious_indices = np.concatenate(suspicious_indices,axis=0)    

        scores = torch.tensor(scores)
        suspect_target_class = scores.argmax(dim=0)

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
            logging.info("Flagged label list: {}".format(suspect_target_class))
            findex = np.zeros(len(images_poison))
            for i in range(len(findex)):
                if i in suspicious_indices:
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
            csv_write.writerow(['record', 'TN','FP','FN','TP','TPR','FPR', 'target'])
            csv_write.writerow([args.result_file, tn, fp, fn, tp, TPR, FPR, suspect_target_class.item()])
            f.close()


    def detection(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        self.filtering()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    spectre.add_arguments(parser)
    args = parser.parse_args()
    spectral_method = spectre(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = spectral_method.detection(args.result_file)