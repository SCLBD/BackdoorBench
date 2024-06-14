'''
Sentinet: Detecting localized universal attacks against deep learning systems
This file is modified based on the following source:
link : https://github.com/wanlunsec/Beatrix/blob/master/defenses/SentiNet/SentiNet.py

@inproceedings{chou2020sentinet,
  title={Sentinet: Detecting localized universal attacks against deep learning systems},
  author={Chou, Edward and Tramer, Florian and Pellegrino, Giancarlo},
  booktitle={2020 IEEE Security and Privacy Workshops (SPW)},
  pages={48--54},
  year={2020},
  organization={IEEE}
}

This file implements the (detection) defense method called SentiNet.

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. SentiNet detection:
        a. obtain the masks of detected samples by GradCAM.
        b. add the corresponding content in clean samples to the detected samples, and obtain the fooled rate and average confidence.
        c. detect the samples whose average confidence exceeds the threshold as poisoned.
    4. Record TPR and FPR, where the detected samples consist of 1000 clean samples and 9000 poisoned samples.
'''


import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
sys.path.append('../')
sys.path.append(os.getcwd())
import torch.nn.functional as F
from torchvision import transforms
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
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)

class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


class Normalize:
    def __init__(self, args, expected_values, variance):
        self.n_channels = args.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = (x[:, :, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, args, expected_values, variance):
        self.n_channels = args.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = x[:, :, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone

class SentiNet:
    def __init__(self, args):
        super().__init__()
        self.normalizer = self._get_normalize(args)
        self.denormalizer = self._get_denormalize(args)
        self.device = args.device
        self.input_height,self.input_width,self.input_channel = args.input_height,args.input_width,args.input_channel

    def _get_denormalize(self, args):
        if args.dataset == "cifar10":
            denormalizer = Denormalize(args, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif args.dataset == "mnist":
            denormalizer = Denormalize(args, [0.5], [0.5])
        elif args.dataset == "gtsrb":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, args):
        if args.dataset == "cifar10":
            normalizer = Normalize(args, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif args.dataset == "mnist":
            normalizer = Normalize(args, [0.5], [0.5])
        elif args.dataset == "gtsrb":
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        if normalizer:
            transform = transforms.Compose([transforms.ToTensor(), normalizer])
        else:
            transform = transforms.ToTensor()
        return transform

    def normalize(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return x

    def denormalize(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x

    def _superimpose(self, background, overlay, mask):
        output = background*mask + overlay*(1.0 - mask)
        output = np.clip(output, 0, 255).astype(np.uint8)
        assert len(output.shape) == 3
        return output

    def _get_entropy(self, background, mask, dataset, classifier, target_label):
        index_overlay = np.arange(len(dataset))
        inert_pattern = np.random.rand(len(dataset),self.input_height,self.input_width,self.input_channel) * 255.0
        inert_pattern = np.clip(inert_pattern, 0, 255).astype(np.uint8)
        x1_add = [0] * len(dataset)
        x2_add = [0] * len(dataset)
        for index in range(len(dataset)):
            add_image = self._superimpose(background, np.array(dataset[index_overlay[index]][0]), mask)
            add_image = self.normalize(add_image)
            x1_add[index] = add_image
            ip_image = self._superimpose(background, inert_pattern[index], mask)
            ip_image = self.normalize(ip_image)
            x2_add[index] = ip_image
        py1_add = classifier(torch.stack(x1_add).to(self.device))
        py2_add = classifier(torch.stack(x2_add).to(self.device))
        py2_add = F.softmax(py2_add, dim=1)
        _, yR = torch.max(py1_add, 1)
        conf_ip, _ = torch.max(py2_add, 1)
        fooled_y = torch.sum(torch.where(yR==target_label,1.0,0.0))/len(dataset)
        avg_conf_ip = torch.mean(conf_ip)
        return fooled_y.detach().cpu().numpy(), avg_conf_ip.detach().cpu().numpy()

    def __call__(self, background, mask, dataset, classifier, target_label):
        return self._get_entropy(background, mask, dataset, classifier, target_label)



class sentinet(defense):

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
        parser.add_argument('--yaml_path', type=str, default="./config/detection/sentinet/cifar10.yaml", help='the path of yaml')
        parser.add_argument('--clean_sample_num', type=int)
        parser.add_argument('--target_layer', type=str, default='layer4.1.conv2') ## the layer to be visualized in gradcam

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection/sentinet_infer/'
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
        pindex = np.where(np.array(bd_test_dataset.poison_indicator) == 1)[0]

        module_dict = dict(model.named_modules())
        target_layer = module_dict[args.target_layer]
        sentinet_detector = SentiNet(self.args)
        cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=args.device)
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

        clean_dataset = [(image_c[i], label_c[i]) for i in range(len(image_c))]
        ### c. load test dataset with poison samples
        images_poison = []
        labels_poison = []
        for img, label,*other_info in bd_test_dataset:
            images_poison.append(img)
            labels_poison.append(label)

        ### d. combine poisoned samples with clean samples
        random_clean_index = np.random.choice(np.arange(len(images)), 1000, replace=False)
        images_clean = [images[i] for i in random_clean_index]
        labels_clean = [labels[i] for i in random_clean_index]
        detected_samples = images_poison + images_clean
        detected_labels = labels_poison + labels_clean

        test_dataset = xy_iter(detected_samples, detected_labels, transform=test_tran)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

        target_category = None
        grayscale_cams = []
        for input, label in test_dataloader:
            graycams = cam(input_tensor=input, targets=target_category)
            grayscale_cams.append(graycams)
        grayscale_cams = np.concatenate(grayscale_cams,axis=0)

        masks = np.where(grayscale_cams >= args.mask_cond,1,0)
        masks = np.expand_dims(masks,axis=-1)
        sentinet_labels = []
        for input, label in test_dataloader:
            input = input.to(self.args.device)
            preds = model(input)
            _, pred_labels = torch.max(preds, 1)
            sentinet_labels.append(pred_labels.detach().cpu().numpy())
        sentinet_labels = np.concatenate(sentinet_labels,axis=0)


        list_fooled = []
        list_avgconf = []
        for index in range(len(detected_samples)):
            background = np.array(detected_samples[index])
            mask = masks[index]
            label = sentinet_labels[index]
            fooled, avgconf = sentinet_detector(background, mask, clean_dataset, model, label)
            list_fooled.append(fooled)
            list_avgconf.append(avgconf)

        suspect_index = np.where(np.array(list_avgconf)>0.9)[0]

        true_index = np.zeros(len(detected_samples))
        for i in range(len(true_index)):
            if i < len(labels_poison):
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
            findex = np.zeros(len(detected_samples))
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
            csv_write.writerow(['record', 'TN','FP','FN','TP','TPR','FPR', 'target'])
            csv_write.writerow([args.result_file, tn, fp, fn, tp, TPR, FPR, 'None'])
            f.close()

                
    def detection(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.filtering()
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    sentinet.add_arguments(parser)
    args = parser.parse_args()
    sentinet_method = sentinet(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = sentinet_method.detection(args.result_file)