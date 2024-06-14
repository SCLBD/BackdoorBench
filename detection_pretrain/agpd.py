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
from utils.nCHW_nHWC import *
from agpd_utils import *
import heapq
from utils.bd_dataset_v2 import dataset_wrapper_with_transform,xy_iter, prepro_cls_DatasetBD_v2
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, all_acc, general_plot_for_epoch, given_dataloader_test
import csv
from sklearn.metrics import confusion_matrix

import time
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import jensenshannon


class gd(defense):

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
        parser.add_argument('--yaml_path', type=str, default="./config/detection/agpd/cifar10.yaml", help='the path of yaml') ###############
        parser.add_argument('--clean_sample_num', type=int)
        parser.add_argument('--csv_save_path', type=str)

        ###hyper_parameter
        parser.add_argument('--tau', type=float)
        parser.add_argument('--xi', type=float)

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection_pretrain/agpd/'
        if not (os.path.exists(save_path)):
                os.makedirs(save_path)
        self.args.save_path = save_path

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
        fix_random(self.args.random_seed)

        model = generate_cls_model(self.args.model,self.args.num_classes)
        adjusted_state_dict = remove_backbone_prefix(self.result['model'])
        model.load_state_dict(adjusted_state_dict, strict=False)
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
        clean_test_dataset = self.result['clean_test'].wrapped_dataset

        num = int(self.args.clean_sample_num / self.args.num_classes)
        if num == 0:
            num = 1
      
        x_bd = []
        y_bd = []
        for img, label,*other_info in bd_train_dataset:
            x_bd.append(img)
            y_bd.append(label)

        class_idx_whole = []
        for i in range(args.num_classes):
            class_idx_whole.append(np.where(np.array(y_bd)==i)[0])
        
        if self.args.model in ['preactresnet18', 'resnet18']:
            conv_list = ['layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2', \
                        'layer2.0.conv1', 'layer2.0.conv2', 'layer2.1.conv1', 'layer2.1.conv2', \
                        'layer3.0.conv1', 'layer3.0.conv2', 'layer3.1.conv1', 'layer3.1.conv2', \
                        'layer4.0.conv1', 'layer4.0.conv2', 'layer4.1.conv1', 'layer4.1.conv2']
            
        elif self.args.model in ['vgg19_bn']:
            conv_list = ['0','3','7','10','14','17','20','23','27','30','33','36','40','43','46','49']#

        elif self.args.model in ['vgg11_bn']:
            conv_list = ['0','4','8','11','15','18','22','25']

        
        j_star_all_layer = []
        gap_list_all_layer = []
        grad_info = []
        for layer in conv_list:
            reference_grad = test_clean_samples(clean_test_dataset, num, test_tran, self.args.device, self.args.model, model, False, layer, self.args.num_classes, self.args.batch_size, self.args.num_workers)
            gap_list = []
            grad = []
            for test_label in range(args.num_classes):
                class_idx = class_idx_whole[test_label]
                x_v = [x_bd[i] for i in class_idx]
                y_v = [y_bd[i] for i in class_idx]
                data_set_o = xy_iter(x_v, y_v,test_tran)
                data_loader = torch.utils.data.DataLoader(
                    data_set_o, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
                )               
                gradients_mean_all = get_all_gradient(data_loader, args.device, args.model, model, False, layer, args.num_classes)
                suspect_list = get_unsimilar_sample_idx(gradients_mean_all, reference_grad[test_label], np.empty(0, int),1)
                l2_dis = compute_change(gradients_mean_all, reference_grad[test_label], suspect_list)
                gap_list.append(l2_dis/np.sqrt(len(class_idx)))
                grad.append(gradients_mean_all)
            grad_info.append(grad)
            J_star = mad(gap_list)
            j_star_all_layer.append(J_star)
            gap_list_all_layer.append(gap_list)

        
        j_max = []
        for i, (js, gap) in enumerate(zip(j_star_all_layer, gap_list_all_layer)):
            j_max.append(np.max(js))
        
        

        j_max_value = np.max(j_max)
        thresh = np.exp(2)
        
        if j_max_value >= thresh:
            select_layer_location = np.argmax(j_max)
            select_layer_name = conv_list[select_layer_location]
            target_layer_js = j_star_all_layer[np.argmax(j_max)]
            target_labels = [i for i, js in enumerate(target_layer_js) if js > thresh]
            gap_list_select_layer = gap_list_all_layer[select_layer_location]
            J_star_select_layer = j_star_all_layer[select_layer_location]

            
        else:
            track = []
            target_labels_list = []
            for index in range(-4,-1):
                target_labels = [i for i in np.where(np.array(gap_list_all_layer[index])>0.3)[0]]
                track.append(len(target_labels))
                target_labels_list.append(target_labels)
            if np.sum(track) == 0:
                target_labels = []
            else:
                idx = np.argmax(track)
                select_layer_name = conv_list[-4:-1][idx]
                target_labels = [i for i,gap in enumerate(gap_list_all_layer[-4:-1][idx]) if gap >0.3]
                gap_list_select_layer = gap_list_all_layer[-4:-1][idx]
                J_star_select_layer = j_star_all_layer[-4:-1][idx]

        if len(target_labels) == 0:
            logging.info('This is not a backdoor model!')
            findex = np.zeros(len(x_bd))
            true_index = np.zeros(len(x_bd))
            for i in range(len(true_index)):
                if i in pindex:
                    true_index[i] = 1
            TN, FP, FN, TP = self.cal(true_index, findex)
            TPR, FPR, precision, acc = self.metrix(TN, FP, FN, TP)
            f = open(args.csv_save_path, 'a', encoding='utf-8')
            csv_write = csv.writer(f)
            csv_write.writerow([args.result_file, 'agpd', TN, FP, FN, TP, TPR, FPR,'clean model',j_max_value, np.where(np.array(j_star_all_layer[np.argmax(j_max)])==j_max_value)[0]])
            f.close()
        else:
            reference_grad = test_clean_samples(clean_test_dataset, num, test_tran, self.args.device, self.args.model, model, False, select_layer_name, self.args.num_classes, self.args.batch_size, self.args.num_workers)
            logging.info('target label is: {}'.format(target_labels))
            logging.info('confidence result: {}'.format([gap_list_select_layer[i] for i in target_labels]))
            logging.info('anomal result: {}'.format([J_star_select_layer[i] for i in target_labels]))

            ## Load data
            poison_idx_all = []
            for test_label in target_labels:
                js_info = []
                rates_all = []
                
                target_idx = class_idx_whole[test_label]
                select_layer_location = np.where(np.array(conv_list) == select_layer_name)[0][0]
                gradients_mean_all = grad_info[select_layer_location][test_label]
                poiosn_list1 = get_unsimilar_sample_idx(gradients_mean_all, reference_grad[test_label], np.empty(0, int),1)
                
                rates_initial = []
                for i in range(len(gradients_mean_all)):
                    if i not in poiosn_list1:
                        dis_p = compute_distance(gradients_mean_all[i], np.mean(gradients_mean_all[poiosn_list1], axis=0), 'cosin')
                        dis_c = compute_distance(gradients_mean_all[i], reference_grad[test_label], 'cosin')
                        rate = (1 - dis_p) / ((1 - dis_p) + (1 - dis_c))
                        rates_initial.append(rate)

                rates_all.append(rates_initial)           
                threshold_set = args.tau
                poison_rest_num = len(target_idx)
                bins = np.linspace(0, 1, 100)
                round = 1
                poison_ieration = []
                poison_stage1 = poiosn_list1
                poison_ieration.append(poison_stage1)

                while len(poison_stage1) > 0:
                    poison_stage1, rates_current, rates_idx = stage1_new(gradients_mean_all, reference_grad[test_label], poiosn_list1, threshold_set, 'cosin', poison_rest_num)
                    rates_all.append(rates_current)
                    if len(poison_stage1) == 0:
                        break
                    else:
                        poison_ieration.append(poison_stage1)
                        poiosn_list1 = np.append(poiosn_list1, poison_stage1)
                        hist1, _ = np.histogram(rates_initial, bins=bins, density=True)
                        hist2, _ = np.histogram(rates_current, bins=bins, density=True)
                        if len(rates_initial) == 0 or len(rates_current) == 0:
                            js_divergence = 1
                        else:
                            js_divergence = jensenshannon(hist1, hist2)
                        rates_initial = rates_current
                        round += 1

                        js_info.append(js_divergence)

                optimal_window_start = find_small_and_smooth_window_start(js_info)
                poison_list_final = np.concatenate(poison_ieration[:optimal_window_start], axis=0)

                poison_ori_idx = [target_idx[i] for i in poison_list_final]
                poison_idx_all.append(poison_ori_idx)

            poison_idx_all = np.concatenate(poison_idx_all, axis=0)
            logging.info('----------- The poison sample num is {} --------------'.format(len(poison_idx_all)))
            true_index = np.zeros(len(x_bd))
            for i in range(len(true_index)):
                if i in pindex:
                    true_index[i] = 1

            findex = np.zeros(len(x_bd))
            for i in range(len(findex)):
                if i in poison_idx_all:
                    findex[i] = 1
            tn, fp, fn, tp = self.cal(true_index, findex)
            TPR, FPR, precision, acc = self.metrix(tn, fp, fn, tp)
            auc = roc_auc_score(true_index, findex)

            new_TP = tp
            new_FN = fn
            new_FP = fp
            precision = new_TP / (new_TP + new_FP) if new_TP + new_FP != 0 else 0
            recall = new_TP / (new_TP + new_FN) if new_TP + new_FN != 0 else 0
            fw1 = 2*(precision * recall)/ (precision + recall) if precision + recall != 0 else 0

            f = open(args.csv_save_path, 'a', encoding='utf-8')
            csv_write = csv.writer(f)
            csv_write.writerow([args.result_file, 'agpd', self.args.clean_sample_num, select_layer_name,tn, fp, fn, tp, TPR, FPR, fw1, auc, args.tau, target_labels, optimal_window_start])
            f.close()        


    def detection(self,result_file):
        
        self.set_result(result_file)
        self.set_logger()
        self.set_devices()
        self.filtering()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    gd.add_arguments(parser)
    args = parser.parse_args()
    gd_method = gd(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = gd_method.detection(args.result_file)
