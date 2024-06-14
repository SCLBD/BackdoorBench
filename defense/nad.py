
'''
Neural Attention Distillation: Erasing Backdoor Triggers From Deep Neural Networks
This file is modified based on the following source:
link : https://github.com/bboylyg/NAD/.

@inproceedings{li2020neural,
    title={Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks},
    author={Li, Yige and Lyu, Xixiang and Koren, Nodens and Lyu, Lingjuan and Li, Bo and Ma, Xingjun},
    booktitle={International Conference on Learning Representations},
    year={2020}}

The defense method is called nad.

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. add some addtional backbone such as resnet18 and vgg19
    7. the method to get the activation of model
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. nad defense:
        a. create student models, set training parameters and determine loss functions
        b. train the student model use the teacher model with the activation of model and result
    4. test the result and get ASR, ACC, RC 
'''

import logging
import random
import time

from calendar import c
from unittest.mock import sentinel
from torchvision import transforms

import torch
import logging
import argparse
import sys
import os
import yaml
from pprint import pformat
import torch.nn as nn
import torch.nn.functional as F

import tqdm

sys.path.append('../')
sys.path.append(os.getcwd())

import time

import numpy as np
from defense.base import defense

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import BackdoorModelTrainer, PureCleanModelTrainer, all_acc
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2




'''
AT with sum of absolute values with power p
code from: https://github.com/AberHu/Knowledge-Distillation-Zoo
'''
def adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 2:
        lr = lr
    elif epoch < 20:
        lr = 0.01
    elif epoch < 30:
        lr = 0.0001
    else:
        lr = 0.0001
    logging.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am

class NADModelTrainer(PureCleanModelTrainer):
    def __init__(self, model, teacher_model, criterions):
        super(NADModelTrainer, self).__init__(model)
        self.teacher = teacher_model
        self.criterions = criterions

    def train_with_test_each_epoch_on_mix(self,
                                    train_dataloader,
                                    clean_test_dataloader,
                                    bd_test_dataloader,
                                    total_epoch_num,
                                    criterions,
                                    optimizer,
                                    scheduler,
                                    amp,
                                    device,
                                    frequency_save,
                                    save_folder_path,
                                    save_prefix,
                                    prefetch,
                                    prefetch_transform_attr_name,
                                    non_blocking,
                                    ):
        test_dataloader_dict = {
                "clean_test_dataloader":clean_test_dataloader,
                "bd_test_dataloader":bd_test_dataloader,
            }

        self.set_with_dataloader(
            train_dataloader,
            test_dataloader_dict,
            criterions['criterionCls'],
            optimizer,
            scheduler,
            device,
            amp,

            frequency_save,
            save_folder_path,
            save_prefix,

            prefetch,
            prefetch_transform_attr_name,
            non_blocking,
        )

        train_loss_list = []
        train_mix_acc_list = []
        clean_test_loss_list = []
        bd_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []

        for epoch in range(total_epoch_num):
            nets = {
                'student':self.model,
                'teacher':self.teacher,
            }
        
            train_epoch_loss_avg_over_batch, \
                train_mix_acc, \
                train_clean_acc, \
                train_asr, \
                train_ra = self.train_epoch(args,train_dataloader,nets,optimizer,scheduler,criterions,epoch)

            clean_metrics, \
            clean_test_epoch_predict_list, \
            clean_test_epoch_label_list, \
             = self.test_given_dataloader(test_dataloader_dict["clean_test_dataloader"], verbose=1)

            clean_test_loss_avg_over_batch = clean_metrics["test_loss_avg_over_batch"]
            test_acc = clean_metrics["test_acc"]

            bd_metrics, \
            bd_test_epoch_predict_list, \
            bd_test_epoch_label_list, \
            bd_test_epoch_original_index_list, \
            bd_test_epoch_poison_indicator_list, \
            bd_test_epoch_original_targets_list = self.test_given_dataloader_on_mix(test_dataloader_dict["bd_test_dataloader"], verbose=1)

            bd_test_loss_avg_over_batch = bd_metrics["test_loss_avg_over_batch"]
            test_asr = all_acc(bd_test_epoch_predict_list, bd_test_epoch_label_list)
            test_ra = all_acc(bd_test_epoch_predict_list, bd_test_epoch_original_targets_list)

            self.agg(
                {
                    "train_epoch_loss_avg_over_batch": train_epoch_loss_avg_over_batch,
                    "train_acc": train_mix_acc,
                    

                    "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                    "bd_test_loss_avg_over_batch" : bd_test_loss_avg_over_batch,
                    "test_acc" : test_acc,
                    "test_asr" : test_asr,
                    "test_ra" : test_ra,
                }
            )

            train_loss_list.append(train_epoch_loss_avg_over_batch)
            train_mix_acc_list.append(train_mix_acc)
            
            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)

            self.plot_loss(
                train_loss_list,
                clean_test_loss_list,
                bd_test_loss_list,
            )

            self.plot_acc_like_metric(
                train_mix_acc_list,
                test_acc_list,
                test_asr_list,
                test_ra_list,
            )

            self.agg_save_dataframe()

        self.agg_save_summary()

        return train_loss_list, \
                train_mix_acc_list, \
                clean_test_loss_list, \
                bd_test_loss_list, \
                test_acc_list, \
                test_asr_list, \
                test_ra_list
    
    def train_epoch(self,args,trainloader,nets,optimizer,scheduler,criterions,epoch):
        '''train the student model with regard to the teacher model and some clean train data for each step
        args:
            Contains default parameters
        trainloader:
            the dataloader of some clean train data
        nets:
            the student model and the teacher model
        optimizer:
            optimizer during the train process
        scheduler:
            scheduler during the train process
        criterion:
            criterion during the train process
        epoch:
            current epoch
        '''
        
        adjust_learning_rate(optimizer, epoch, args.lr)
        snet = nets['student']
        tnet = nets['teacher']

        criterionCls = criterions['criterionCls'].to(args.device, non_blocking=self.non_blocking)
        criterionAT = criterions['criterionAT'].to(args.device, non_blocking=self.non_blocking)

        snet.train()
        snet.to(args.device, non_blocking=self.non_blocking)

        total_clean = 0
        total_clean_correct = 0
        train_loss = 0

        batch_loss = []

        batch_loss_list = []
        batch_predict_list = []
        batch_label_list = []
        batch_original_index_list = []
        batch_poison_indicator_list = []
        batch_original_targets_list = []
        for idx, (inputs, labels, original_index, poison_indicator, original_targets) in enumerate(trainloader):
            inputs, labels = inputs.to(args.device, non_blocking=self.non_blocking), labels.to(args.device, non_blocking=self.non_blocking)

            if args.model == 'preactresnet18':
                outputs_s = snet(inputs)
                features_out_3 = list(snet.children())[:-1]  # Remove the fully connected layer
                modelout_3 = nn.Sequential(*features_out_3)
                modelout_3.to(args.device, non_blocking=self.non_blocking)
                activation3_s = modelout_3(inputs)
                # activation3_s = activation3_s.view(activation3_s.size(0), -1)
                features_out_2 = list(snet.children())[:-2]  # Remove the fully connected layer
                modelout_2 = nn.Sequential(*features_out_2)
                modelout_2.to(args.device, non_blocking=self.non_blocking)
                activation2_s = modelout_2(inputs)
                # activation2_s = activation2_s.view(activation2_s.size(0), -1)
                features_out_1 = list(snet.children())[:-3]  # Remove the fully connected layer
                modelout_1 = nn.Sequential(*features_out_1)
                modelout_1.to(args.device, non_blocking=self.non_blocking)
                activation1_s = modelout_1(inputs)
                # activation1_s = activation1_s.view(activation1_s.size(0), -1)
                
                features_out_3 = list(tnet.children())[:-1]  # Remove the fully connected layer
                modelout_3 = nn.Sequential(*features_out_3)
                modelout_3.to(args.device, non_blocking=self.non_blocking)
                activation3_t = modelout_3(inputs)
                # activation3_t = activation3_t.view(activation3_t.size(0), -1)
                features_out_2 = list(tnet.children())[:-2]  # Remove the fully connected layer
                modelout_2 = nn.Sequential(*features_out_2)
                modelout_2.to(args.device, non_blocking=self.non_blocking)
                activation2_t = modelout_2(inputs)
                # activation2_t = activation2_t.view(activation2_t.size(0), -1)
                features_out_1 = list(tnet.children())[:-3]  # Remove the fully connected layer
                modelout_1 = nn.Sequential(*features_out_1)
                modelout_1.to(args.device, non_blocking=self.non_blocking)
                activation1_t = modelout_1(inputs)
                # activation1_t = activation1_t.view(activation1_t.size(0), -1)

                # activation1_s, activation2_s, activation3_s, output_s = snet(inputs)
                # activation1_t, activation2_t, activation3_t, _ = tnet(inputs)

                cls_loss = criterionCls(outputs_s, labels)
                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3
                at2_loss = criterionAT(activation2_s, activation2_t.detach()) * args.beta2
                at1_loss = criterionAT(activation1_s, activation1_t.detach()) * args.beta1

                at_loss = at1_loss + at2_loss + at3_loss + cls_loss

            if args.model == 'vgg19':
                outputs_s = snet(inputs)
                features_out_3 = list(snet.children())[:-1]  # Remove the fully connected layer
                modelout_3 = nn.Sequential(*features_out_3)
                modelout_3.to(args.device, non_blocking=self.non_blocking)
                activation3_s = modelout_3(inputs)
                # activation3_s = snet.features(inputs)
                # activation3_s = activation3_s.view(activation3_s.size(0), -1)

                output_t = tnet(inputs)
                features_out_3 = list(tnet.children())[:-1]  # Remove the fully connected layer
                modelout_3 = nn.Sequential(*features_out_3)
                modelout_3.to(args.device, non_blocking=self.non_blocking)
                activation3_t = modelout_3(inputs)
                # activation3_t = tnet.features(inputs)
                # activation3_t = activation3_t.view(activation3_t.size(0), -1)

                cls_loss = criterionCls(outputs_s, labels)
                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3

                at_loss = at3_loss + cls_loss

            if args.model == 'vgg19_bn':
                outputs_s = snet(inputs)
                features_out_3 = list(snet.children())[:-1]  # Remove the fully connected layer
                modelout_3 = nn.Sequential(*features_out_3)
                modelout_3.to(args.device)
                activation3_s = modelout_3(inputs)
                # activation3_s = snet.features(inputs)
                # activation3_s = activation3_s.view(activation3_s.size(0), -1)

                output_t = tnet(inputs)
                features_out_3 = list(tnet.children())[:-1]  # Remove the fully connected layer
                modelout_3 = nn.Sequential(*features_out_3)
                modelout_3.to(args.device)
                activation3_t = modelout_3(inputs)
                # activation3_t = tnet.features(inputs)
                # activation3_t = activation3_t.view(activation3_t.size(0), -1)

                cls_loss = criterionCls(outputs_s, labels)
                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3

                at_loss = at3_loss + cls_loss

            if args.model == 'resnet18':
                outputs_s = snet(inputs)
                features_out = list(snet.children())[:-1]
                modelout = nn.Sequential(*features_out)
                modelout.to(args.device, non_blocking=self.non_blocking)
                activation3_s = modelout(inputs)
                # activation3_s = features.view(features.size(0), -1)

                output_t = tnet(inputs)
                features_out = list(tnet.children())[:-1]
                modelout = nn.Sequential(*features_out)
                modelout.to(args.device, non_blocking=self.non_blocking)
                activation3_t = modelout(inputs)
                # activation3_t = features.view(features.size(0), -1)

                cls_loss = criterionCls(outputs_s, labels)
                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3

                at_loss = at3_loss + cls_loss
            
            if args.model == 'mobilenet_v3_large':
                outputs_s = snet(inputs)
                features_out = list(snet.children())[:-1]
                modelout = nn.Sequential(*features_out)
                modelout.to(args.device, non_blocking=self.non_blocking)
                activation3_s = modelout(inputs)
                # activation3_s = features.view(features.size(0), -1)

                output_t = tnet(inputs)
                features_out = list(tnet.children())[:-1]
                modelout = nn.Sequential(*features_out)
                modelout.to(args.device, non_blocking=self.non_blocking)
                activation3_t = modelout(inputs)
                # activation3_t = features.view(features.size(0), -1)

                cls_loss = criterionCls(outputs_s, labels)
                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3

                at_loss = at3_loss + cls_loss

            if args.model == 'densenet161':
                outputs_s = snet(inputs)
                features_out = list(snet.children())[:-1]
                modelout = nn.Sequential(*features_out)
                modelout.to(args.device, non_blocking=self.non_blocking)
                activation3_s = modelout(inputs)
                # activation3_s = features.view(features.size(0), -1)

                output_t = tnet(inputs)
                features_out = list(tnet.children())[:-1]
                modelout = nn.Sequential(*features_out)
                modelout.to(args.device, non_blocking=self.non_blocking)
                activation3_t = modelout(inputs)
                # activation3_t = features.view(features.size(0), -1)

                cls_loss = criterionCls(outputs_s, labels)
                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3

                at_loss = at3_loss + cls_loss

            if args.model == 'efficientnet_b3':
                outputs_s = snet(inputs)
                features_out = list(snet.children())[:-1]
                modelout = nn.Sequential(*features_out)
                modelout.to(args.device, non_blocking=self.non_blocking)
                activation3_s = modelout(inputs)
                # activation3_s = features.view(features.size(0), -1)

                output_t = tnet(inputs)
                features_out = list(tnet.children())[:-1]
                modelout = nn.Sequential(*features_out)
                modelout.to(args.device, non_blocking=self.non_blocking)
                activation3_t = modelout(inputs)
                # activation3_t = features.view(features.size(0), -1)

                cls_loss = criterionCls(outputs_s, labels)
                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3

                at_loss = at3_loss + cls_loss

            if args.model == 'convnext_tiny':
                outputs_s = snet(inputs)
                features_out_3 = list(snet.children())[:-1]  # Remove the fully connected layer
                modelout_3 = nn.Sequential(*features_out_3)
                modelout_3.to(args.device)
                activation3_s = modelout_3(inputs)
                # activation3_s = snet.features(inputs)
                # activation3_s = activation3_s.view(activation3_s.size(0), -1)

                output_t = tnet(inputs)
                features_out_3 = list(tnet.children())[:-1]  # Remove the fully connected layer
                modelout_3 = nn.Sequential(*features_out_3)
                modelout_3.to(args.device)
                activation3_t = modelout_3(inputs)
                # activation3_t = tnet.features(inputs)
                # activation3_t = activation3_t.view(activation3_t.size(0), -1)

                cls_loss = criterionCls(outputs_s, labels)
                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3

                at_loss = at3_loss + cls_loss

            if args.model == 'vit_b_16':
                outputs_s = snet(inputs)
                features_out_3 = list(snet.children())[:-1]  # Remove the fully connected layer
                modelout_3 = nn.Sequential(*features_out_3)
                modelout_3.to(args.device)
                activation3_s = modelout_3(inputs)
                # activation3_s = snet.features(inputs)
                # activation3_s = activation3_s.view(activation3_s.size(0), -1)

                output_t = tnet(inputs)
                features_out_3 = list(tnet.children())[:-1]  # Remove the fully connected layer
                modelout_3 = nn.Sequential(*features_out_3)
                modelout_3.to(args.device)
                activation3_t = modelout_3(inputs)
                # activation3_t = tnet.features(inputs)
                # activation3_t = activation3_t.view(activation3_t.size(0), -1)

                cls_loss = criterionCls(outputs_s, labels)
                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * args.beta3

                at_loss = at3_loss + cls_loss


            batch_loss.append(at_loss.item())
            optimizer.zero_grad()
            at_loss.backward()
            optimizer.step()

            train_loss += at_loss.item()
            total_clean_correct += torch.sum(torch.argmax(outputs_s[:], dim=1) == labels[:])
            total_clean += inputs.shape[0]
            avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)

            batch_loss_list.append(at_loss.item())
            batch_predict_list.append(torch.max(outputs_s, -1)[1].detach().clone().cpu())
            batch_label_list.append(labels.detach().clone().cpu())
            batch_original_index_list.append(original_index.detach().clone().cpu())
            batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
            batch_original_targets_list.append(original_targets.detach().clone().cpu())

        train_epoch_loss_avg_over_batch, \
        train_epoch_predict_list, \
        train_epoch_label_list, \
        train_epoch_poison_indicator_list, \
        train_epoch_original_targets_list = sum(batch_loss_list) / len(batch_loss_list), \
                                            torch.cat(batch_predict_list), \
                                            torch.cat(batch_label_list), \
                                            torch.cat(batch_poison_indicator_list), \
                                            torch.cat(batch_original_targets_list)

        train_mix_acc = all_acc(train_epoch_predict_list, train_epoch_label_list)

        train_bd_idx = torch.where(train_epoch_poison_indicator_list == 1)[0]
        train_clean_idx = torch.where(train_epoch_poison_indicator_list == 0)[0]
        train_clean_acc = all_acc(
            train_epoch_predict_list[train_clean_idx],
            train_epoch_label_list[train_clean_idx],
        )
        train_asr = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_label_list[train_bd_idx],
        )
        train_ra = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_original_targets_list[train_bd_idx],
        )
        logging.info(f'Epoch{epoch}: Loss:{train_loss} Training Acc:{avg_acc_clean}({total_clean_correct}/{total_clean})')
        # one_epoch_loss = sum(batch_loss)/len(batch_loss)
        # if args.lr_scheduler == 'ReduceLROnPlateau':
        #     scheduler.step(one_epoch_loss)
        # elif args.lr_scheduler ==  'CosineAnnealingLR':
        #     scheduler.step()
        return train_epoch_loss_avg_over_batch, \
                train_mix_acc, \
                train_clean_acc, \
                train_asr, \
                train_ra
    
class nad(defense):
    r"""Neural Attention Distillation: Erasing Backdoor Triggers From Deep Neural Networks
    
    basic structure: 
    
    1. config args, save_path, fix random seed
    2. load the backdoor attack data and backdoor test data
    3. load the backdoor model
    3. nad defense:
        a. create student models, set training parameters and determine loss functions
        b. train the student model use the teacher model with the activation of model and result
    5. test the result and get ASR, ACC, RC 
       
    .. code-block:: python
    
        parser = argparse.ArgumentParser(description=sys.argv[0])
        nad.add_arguments(parser)
        args = parser.parse_args()
        nad_method = nad(args)
        if "result_file" not in args.__dict__:
            args.result_file = 'one_epochs_debug_badnet_attack'
        elif args.result_file is None:
            args.result_file = 'one_epochs_debug_badnet_attack'
        result = nad_method.defense(args.result_file)
    
    .. Note::
        @article{li2021neural,
        title={Neural attention distillation: Erasing backdoor triggers from deep neural networks},
        author={Li, Yige and Lyu, Xixiang and Koren, Nodens and Lyu, Lingjuan and Li, Bo and Ma, Xingjun},
        journal={arXiv preprint arXiv:2101.05930},
        year={2021}
        }

    Args:
        baisc args: in the base class
        ratio (float): the ratio of training data
        index (str): the index of clean data
        te_epochs (int): the number of epochs for training the teacher model using the clean data
        beta1 (int): the beta of the first layer
        beta2 (int): the beta of the second layer
        beta3 (int): the beta of the third layer
        p (float): the power of the activation of model for AT loss function
        teacher_model_loc (str): the location of teacher model(if None, train the teacher model)
        
    """ 

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
        parser.add_argument('--amp', type=lambda x: str(x) in ['True','true','1'])

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
        parser.add_argument('--yaml_path', type=str, default="./config/defense/nad/config.yaml", help='the path of yaml')

        #set the parameter for the nad defense
        parser.add_argument('--ratio', type=float, help='ratio of training data')
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--te_epochs', type=int)
        parser.add_argument('--beta1', type=int, help='beta of low layer')
        parser.add_argument('--beta2', type=int, help='beta of middle layer')
        parser.add_argument('--beta3', type=int, help='beta of high layer')
        parser.add_argument('--p', type=float, help='power for AT')

        parser.add_argument('--teacher_model_loc', type=str, help='the location of teacher model')

        

    

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/defense/nad/'
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

    def set_trainer(self, model, mode = 'normal', **params):
        if mode == 'normal':
            self.trainer = BackdoorModelTrainer(
                model,
            )
        elif mode == 'clean':
            self.trainer = PureCleanModelTrainer(
                model,
            )
        elif mode == 'nad':
            self.trainer = NADModelTrainer(
                model,
                **params,
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
        # self.device = torch.device(
        #     (
        #         f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
        #         # since DataParallel only allow .to("cuda")
        #     ) if torch.cuda.is_available() else "cpu"
        # )
        self.device = self.args.device
    def mitigation(self):
        self.set_devices()
        args = self.args
        result = self.result
        fix_random(args.random_seed)

        ### a. create student models, set training parameters and determine loss functions
        # Load models
        logging.info('----------- Network Initialization --------------')
        teacher = generate_cls_model(args.model,args.num_classes)
        teacher.load_state_dict(result['model'])
        if "," in self.device:
            teacher = torch.nn.DataParallel(
                teacher,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{teacher.device_ids[0]}'
            teacher.to(self.args.device)
        else:
            teacher.to(self.args.device)
        logging.info('finished teacher student init...')
        student = generate_cls_model(args.model,args.num_classes)
        student.load_state_dict(result['model'])
        if "," in self.device:
            student = torch.nn.DataParallel(
                student,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{student.device_ids[0]}'
            student.to(self.args.device)
        else:
            student.to(self.args.device)
        logging.info('finished student student init...')

        teacher.eval()
        nets = {'snet': student, 'tnet': teacher}

        # initialize optimizer, scheduler
        optimizer, scheduler = argparser_opt_scheduler(student, self.args)
        
        # define loss functions
        criterionCls = argparser_criterion(args)
        criterionAT = AT(args.p)
        criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}

        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        data_all_length = len(clean_dataset)
        ran_idx = choose_index(self.args, data_all_length) 
        log_index = self.args.log + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')
        clean_dataset.subset(ran_idx)
        data_set_without_tran = clean_dataset
        data_set_o = self.result['clean_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = train_tran
        data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
        trainloader = data_loader
        
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)
        
        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
        test_dataloader_dict["bd_test_dataloader"] = data_bd_loader

        ### train the teacher model
        if args.teacher_model_loc is not None: 
            teacher_model = torch.load(args.teacher_model_loc)
            teacher.load_state_dict(teacher_model['model'])
        else :
            self.set_trainer(teacher,'clean')
            start_epoch = 0
            optimizer_ft, scheduler_ft = argparser_opt_scheduler(teacher, self.args)
            self.trainer.train_with_test_each_epoch_on_mix(
                trainloader,
                data_clean_loader,
                data_bd_loader,
                args.te_epochs,
                criterion = criterionCls,
                optimizer = optimizer_ft,
                scheduler = scheduler_ft,
                device = self.args.device,
                frequency_save = 0,
                save_folder_path = args.save_path,
                save_prefix='nad_te',
                amp=args.amp,
                prefetch=args.prefetch,
                prefetch_transform_attr_name="ori_image_transform_in_loading", # since we use the preprocess_bd_dataset
                non_blocking=args.non_blocking,
            )
        
            
        ### b. train the student model use the teacher model with the activation of model and result
        self.set_trainer(student, 'nad', teacher_model = teacher, criterions = criterions)
        logging.info('----------- Train Initialization --------------')


        self.trainer.train_with_test_each_epoch_on_mix(
                trainloader,
                data_clean_loader,
                data_bd_loader,
                args.te_epochs,
                criterions = criterions,
                optimizer = optimizer,
                scheduler = scheduler,
                device = self.args.device,
                frequency_save = 0,
                save_folder_path = args.save_path,
                save_prefix='nad',
                amp=args.amp,
                prefetch=args.prefetch,
                prefetch_transform_attr_name="ori_image_transform_in_loading", # since we use the preprocess_bd_dataset
                non_blocking=args.non_blocking,
            )
        
        result = {}
        result['model'] = student
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=student.cpu().state_dict(),
            save_path=args.save_path,
        )
        return result

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    nad.add_arguments(parser)
    args = parser.parse_args()
    ft_method = nad(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = ft_method.defense(args.result_file)