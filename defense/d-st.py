'''
This file implements the defense method called D-ST from Effective Backdoor Defense by Exploiting Sensitivity of Poisoned Samples.
It trains a !!!secure model!!! from scratch with a poisoned dataset.
This file is modified based on the following source:
link :  https://github.com/SCLBD/Effective_backdoor_defense
The defense method is called d-br.

@article{chen2022effective,
  title={Effective backdoor defense by exploiting sensitivity of poisoned samples},
  author={Chen, Weixin and Wu, Baoyuan and Wang, Haoqian},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={9727--9737},
  year={2022}
}

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. d-st defense: mainly two steps: sd and st (Sample-Distinguishment and two-stage Secure Training)
        a. train a backdoored model from scratch using poisoned dataset without any data augmentations
        b. fine-tune the backdoored model with intra-class loss L_intra.
        (sd:)
        c. calculate values of the FCT metric for all training samples.
        d. calculate thresholds for choosing clean and poisoned samples.
        e. separate training samples into clean samples D_c, poisoned samples D_p, and uncertain samples D_u.
        (st:)
        f. train the feature extractor via semi-supervised contrastive learning.
        g. train the classifier via minimizing a mixed cross-entropy loss.
    4. test the result and get ASR, ACC, RC 

'''

import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import math

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import BackdoorModelTrainer, Metric_Aggregator,PureCleanModelTrainer
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
## d-st utils
from utils.defense_utils.dst.dataloader_bd import get_transform_st, TransformThree, normalization
from utils.defense_utils.dst.sd import calculate_consistency, calculate_gamma, separate_samples
from utils.defense_utils.dst.dataloader_bd import get_st_train_loader
from utils.defense_utils.dst.models.resnet_super import SupConResNet,LinearClassifier
from utils.defense_utils.dst.st_loss import SupConLoss_Consistency
from utils.defense_utils.dst.utils_st import *

def train_epoch(arg, trainloader, model, optimizer, scheduler, criterion, epoch):
    model.train()

    total_clean, total_poison = 0, 0
    total_clean_correct, total_attack_correct, total_robust_correct = 0, 0, 0
    train_loss = 0
    
    for i, (inputs, labels, _, isCleans, gt_labels) in enumerate(trainloader):
        inputs = normalization(arg, inputs[0])  # Normalize
        inputs, labels, gt_labels = inputs.to(arg.device), labels.to(arg.device), gt_labels.to(arg.device)
        clean_idx, poison_idx = torch.where(isCleans == True), torch.where(isCleans == False)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_attack_correct += torch.sum(torch.argmax(outputs[poison_idx], dim=1) == labels[poison_idx])
        total_robust_correct += torch.sum(torch.argmax(outputs[:], dim=1) == gt_labels[:])
        total_clean += inputs.shape[0]
        total_poison += inputs[poison_idx].shape[0]

    avg_acc_clean = (total_clean_correct  / total_clean).item()
    avg_acc_attack = (total_attack_correct  / total_poison).item()
    avg_acc_robust = (total_robust_correct  / total_clean).item()
    logging.info(f'Epoch: {epoch} | Loss: {train_loss / (i + 1)} | Train ACC: {avg_acc_clean} ({total_clean_correct}/{total_clean}) | Train ASR: \
        {avg_acc_attack}% ({total_attack_correct}/{total_poison}) | Train R-ACC: {avg_acc_robust} ({total_robust_correct}/{total_clean})')
    del loss, inputs, outputs
    torch.cuda.empty_cache()
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_loss/(i + 1))
        else:
            scheduler.step()

    return train_loss / (i + 1), avg_acc_clean, avg_acc_attack, avg_acc_robust

def test_epoch(args, testloader, model, criterion, epoch):
    model.eval()

    total_clean = 0
    total_clean_correct = 0
    test_loss = 0
    
    for i, (inputs, labels, *additional_info) in enumerate(testloader):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
    avg_acc_clean = (total_clean_correct / total_clean).item()
  
    return test_loss / (i + 1), avg_acc_clean

def finetune_epoch(arg, trainloader, model, optimizer, scheduler, epoch):
    model.train()

    total_clean, total_poison = 0, 0
    total_clean_correct, total_attack_correct, total_robust_correct = 0, 0, 0
    train_loss = 0
    
    for i, (inputs, labels, _, is_bd, gt_labels) in enumerate(trainloader):
        inputs = normalization(arg, inputs[0])  # Normalize
        inputs, labels, gt_labels = inputs.to(arg.device), labels.to(arg.device), gt_labels.to(arg.device)
        clean_idx, poison_idx = torch.where(is_bd == False)[0], torch.where(is_bd == True)[0]

        # Features and Outputs
        # outputs = model(inputs)
        # if hasattr(model, "module"):   # abandon FC layer
        #     features_out = list(model.module.children())[:-1]  
        # else:
        #     features_out = list(model.children())[:-1]
        # modelout = nn.Sequential(*features_out).to(arg.device)
        # features = modelout(inputs)
        # features = features.view(features.size(0), -1)
        features = model(inputs)
        features = features.view(features.size(0), -1)
        # Calculate intra-class loss
        centers = []
        for j in range(arg.num_classes):
            j_idx = torch.where(labels == j)[0]
            if j_idx.shape[0] == 0:
                continue
            j_features = features[j_idx]
            j_center = torch.mean(j_features, dim=0)
            centers.append(j_center)

        centers = torch.stack(centers, dim=0)
        centers = F.normalize(centers, dim=1)
        similarity_matrix = torch.matmul(centers, centers.T)
        mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).to(arg.device)
        similarity_matrix[mask] = 0.0
        loss = torch.mean(similarity_matrix)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_loss/(i + 1))
        else:
            scheduler.step()
    torch.cuda.empty_cache()
    # return train_loss / (i + 1), avg_acc_clean, avg_acc_attack, avg_acc_robust
    return train_loss / (i + 1)

def _train_extractor(train_loader, model, criterion, optimizer, epoch, args):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, flags) in enumerate(train_loader):
        if args.debug and idx == 2:
            break
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True).to(args.device)
            labels = labels.cuda(non_blocking=True).to(args.device)
            flags = flags.cuda(non_blocking=True).to(args.device)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels, flags)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            logging.info(f'Train: [{epoch}/{args.epochs}][{idx + 1}/{len(train_loader)}]\t \
                BT {batch_time.val} ({batch_time.avg})\t \
                DT {data_time.val} ({data_time.avg})\t \
                loss {losses.val} ({losses.avg})')

            sys.stdout.flush()
    del loss, images, features
    torch.cuda.empty_cache()
    return losses.avg

def _train_classifier(train_loader, model, classifier, criterion, optimizer, epoch, args):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels, flags) in enumerate(train_loader):
        if args.debug and idx == 2:
            break
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True).to(args.device)
        labels = labels.cuda(non_blocking=True).to(args.device)
        flags = flags.cuda(non_blocking=True).to(args.device)
        
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())

        clean_idx = torch.where(flags == 0)[0]
        poison_idx = torch.where(flags == 2)[0]
        loss = criterion(output[clean_idx], labels[clean_idx]) - criterion(output[poison_idx], labels[poison_idx])*0.001
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0].detach().cpu().numpy(), bsz)

        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            logging.info(f'Train: [{epoch}][{idx + 1}/{len(train_loader)}]\t \
                BT {batch_time.val} ({batch_time.avg})\t \
                DT {data_time.val} ({data_time.avg})\t \
                loss {losses.val} ({losses.avg}\t \
                Acc@1 {top1.val} ({top1.avg}')
            sys.stdout.flush()
    del loss, features, images, output
    torch.cuda.empty_cache()
    return losses.avg, top1.avg

def given_dataloader_test(
        model,
        classifier,
        test_dataloader,
        criterion,
        non_blocking : bool = False,
        device = "cpu",
        verbose : int = 0
):
    model.to(device, non_blocking=non_blocking)
    model.eval()
    metrics = {
        'test_correct': 0,
        'test_loss_sum_over_batch': 0,
        'test_total': 0,
    }
    criterion = criterion.to(device, non_blocking=non_blocking)

    if verbose == 1:
        batch_predict_list, batch_label_list = [], []

    with torch.no_grad():
        for batch_idx, (x, target, *additional_info) in enumerate(test_dataloader):
            x = x.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            features = model.encoder(x)
            pred = classifier(features.detach())
            loss = criterion(pred, target.long())

            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            if verbose == 1:
                batch_predict_list.append(predicted.detach().clone().cpu())
                batch_label_list.append(target.detach().clone().cpu())

            metrics['test_correct'] += correct.item()
            metrics['test_loss_sum_over_batch'] += loss.item()
            metrics['test_total'] += target.size(0)

    metrics['test_loss_avg_over_batch'] = metrics['test_loss_sum_over_batch']/len(test_dataloader)
    metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']

    if verbose == 0:
        return metrics, None, None
    elif verbose == 1:
        return metrics, torch.cat(batch_predict_list), torch.cat(batch_label_list)

def reset_model_from_SupConResNet(args, old_model, classifier): ## replace the parameters from old model to new model
    new_model = generate_cls_model(args.model,args.num_classes)
    
    new_dict = new_model.state_dict()
    old_dict = old_model.encoder.state_dict()
    new_dict.update(old_dict)
    new_model.load_state_dict(new_dict)   
    if hasattr(new_model,"linear"):
        new_model.linear.weight.data = classifier.fc.weight.data
        new_model.linear.bias.data = classifier.fc.bias.data
    elif hasattr(new_model,"fc"):
        new_model.fc.weight.data = classifier.fc.weight.data
        new_model.fc.bias.data = classifier.fc.bias.data
    return new_model

class d_st(defense):
    r"""Effective backdoor defense by exploiting sensitivity of poisoned samples
    
    basic structure: 
    
    1. config args, save_path, fix random seed
    2. load the backdoor attack data and backdoor test data
    3. d-st defense: mainly two steps: sd and st (Sample-Distinguishment and two-stage Secure Training)
        a. train a backdoored model from scratch using poisoned dataset without any data augmentations
        b. fine-tune the backdoored model with intra-class loss L_intra.
        (sd:)
        c. calculate values of the FCT metric for all training samples.
        d. calculate thresholds for choosing clean and poisoned samples.
        e. separate training samples into clean samples D_c, poisoned samples D_p, and uncertain samples D_u.
        (st:)
        f. train the feature extractor via semi-supervised contrastive learning.
        g. train the classifier via minimizing a mixed cross-entropy loss.
    4. test the result and get ASR, ACC, RC with regard to the chosen threshold and interval
       
    .. code-block:: python
    
        parser = argparse.ArgumentParser(description=sys.argv[0])
        d-st.add_arguments(parser)
        args = parser.parse_args()
        d-st_method = d-st(args)
        if "result_file" not in args.__dict__:
            args.result_file = 'one_epochs_debug_badnet_attack'
        elif args.result_file is None:
            args.result_file = 'one_epochs_debug_badnet_attack'
        result = d-st_method.defense(args.result_file)
    
    .. Note::
        @article{chen2022effective,
        title={Effective backdoor defense by exploiting sensitivity of poisoned samples},
        author={Chen, Weixin and Wu, Baoyuan and Wang, Haoqian},
        journal={Advances in Neural Information Processing Systems},
        volume={35},
        pages={9727--9737},
        year={2022}
        }

    Args:
        baisc args: in the base class
        clean_ratio (float): ratio of clean data separated from the poisoned data
        poison_ratio (float): ratio of poisoned data separated from the poisoned data
        gamma (float): LR is multiplied by gamma on schedule.
        schedule (int): Decrease learning rate at these epochs.
        warm (int): warm up epochs for training
        trans1 (str): the first data augmentation used in the sd step to separate the clean and poisoned data
        trans2 (str): the second data augmentation used in the sd step to separate the clean and poisoned data
        debug (bool): debug or not

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
        # args.dataset_path = f"{args.dataset_path}/{args.dataset}"
        self.args = args

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
        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/d-st/config.yaml", help='the path of yaml')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        parser.add_argument('--target_label', type=int)
        # parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,help=' frequency_save, 0 is never')

        parser.add_argument('--momentum', type=float, help='momentum')
        parser.add_argument('--weight_decay', type=float, help='weight decay')

        #set the parameter for the d-st defense
        parser.add_argument('--continue_step', type=str, default=None, help='the step to continue')
        parser.add_argument('--gamma_low', type=float, default=None, help='<=gamma_low is clean') # \gamma_c
        parser.add_argument('--gamma_high', type=float, default=None, help='>=gamma_high is poisoned') # \gamma_p
        parser.add_argument('--clean_ratio', type=float, default=0.20, help='ratio of clean data') # \alpha_c
        parser.add_argument('--poison_ratio', type=float, default=0.05, help='ratio of poisoned data') # \alpha_p

        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
        parser.add_argument('--warm', type=int, default=1, help='warm up training phase')

        parser.add_argument('--trans1', type=str, default='rotate') # the first data augmentation
        parser.add_argument('--trans2', type=str, default='affine') # the second data augmentation
        parser.add_argument('--debug', action='store_true',default=False, help='debug or not')
        parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
        parser.add_argument('--save_all_process', action='store_true', help='save model in each process')

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/defense/d-st/'
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
        

    def set_trainer(self, model, mode='normal'):
        if mode == 'normal':
            self.trainer = BackdoorModelTrainer(
                model,
            )
        elif mode == 'clean':
            self.trainer = PureCleanModelTrainer(
                model,
            )
        elif mode == 'nad':
            raise RuntimeError('No trainer support this mode!')

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
    
    def set_new_args(self,args,step):
        if step == 'train_notrans':
            args.epochs = 2
        elif step == 'finetune_notrans':
            args.epochs = 10
        elif step == 'sscl':
            args.epochs = 200
            args.learning_rate = 0.5
            args.temp = 0.1
            args.cosine = True
            if args.cosine:
                args.model_name = '{}_cosine'.format(args.model)
            if args.batch_size > 256:
                args.warm = True
            if args.warm:
                args.model_name = '{}_warm'.format(args.model)
                args.warmup_from = 0.01
                args.warm_epochs = 10
                if args.cosine:
                    args.lr_decay_rate = 0.1
                    eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
                    args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                            1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
                else:
                    args.warmup_to = args.learning_rate
                    args.lr_decay_epochs = [700,800,900]
        elif step == 'mixed_ce':
            args.epochs = 10
            args.learning_rate = 5
            args.num_workers = 16
            args.cosine = False
            if args.batch_size > 256:
                args.warm = True
            if args.warm:
                args.model_name = '{}_warm'.format(args.model)
                args.warmup_from = 0.01
                args.warm_epochs = 10
                if args.cosine:
                    args.lr_decay_rate = 0.1
                    eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
                    args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                            1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
                else:
                    args.warmup_to = args.learning_rate
                    args.lr_decay_epochs = [60,75,90]
        if args.debug:
            args.epochs = 1
        return args

    def set_model(self,args,model):
        assert isinstance(model , SupConResNet)
        criterion = torch.nn.CrossEntropyLoss()
        classifier = LinearClassifier(feat_dim=args.feature_dim, num_classes=args.num_classes)
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)
        classifier = classifier.to(args.device)
        criterion = criterion.to(args.device)
        return model, classifier, criterion

    def drop_linear(self,model): # drop the last nn.Linear layer, which will not be used in the following training
        model_name = self.args.model
        if 'preactresnet' in model_name or model_name == 'senet18':
            feature_dim = model.linear.in_features
            model.linear = nn.Identity()
        elif model_name.startswith("resnet"):
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif 'vgg' in model_name or 'convnext' in model_name:
            feature_dim = list(model.classifier.children())[-1].in_features
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        elif 'vit' in model_name:
            feature_dim = model[1].heads.head.in_features
            model[1].heads.head = nn.Identity()
        else:
            raise NotImplementedError('Not support the model: {}'.format(model_name))
        model.register_feature_dim = feature_dim
        return model

    def add_linear(self,old_model, classifier): ## replace the parameters from old model to new model
        args = self.args
        new_model = generate_cls_model(args.model,args.num_classes)
        new_dict = new_model.state_dict()
        old_dict = old_model.encoder.state_dict()
        new_dict.update(old_dict)
        new_model.load_state_dict(new_dict)   
        model_name = args.model
        fc = classifier.fc
        if 'preactresnet' in model_name or model_name == 'senet18':
            new_model.linear = fc
        elif model_name.startswith("resnet"):
            new_model.fc = fc
        elif 'vgg' in model_name or 'convnext' in model_name:
            new_model.classifier = nn.Sequential(*list(new_model.classifier.children())[:-1]+[fc])
        elif 'vit' in model_name:
            new_model[1].heads.head = fc
        else:
            raise NotImplementedError('Not support the model: {}'.format(model_name))
        return new_model

    
    def get_sd_train_loader(self):
        args = self.args
        transform1, transform2, transform3 = get_transform_st(args, train=True)
        dataset_train = self.result['bd_train']
        dataset_train.wrap_img_transform = TransformThree(transform1, transform2, transform3)
        poisoned_data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)    
        return poisoned_data_loader_train

    def testloader_wrapper(self,):
        args = self.args
        test_tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)

        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        bd_test_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=False,pin_memory=True)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        clean_test_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=False,pin_memory=True)

        return clean_test_loader, bd_test_loader

    def train_attack_noTrans(self, bd_trainloader, clean_test_loader, bd_test_loader, model = None, optimizer=None, scheduler=None,finetune=False):
        ## update args
        step = 'finetune_notrans' if finetune else 'train_notrans'
        args = self.set_new_args(self.args,step = step)
        agg = Metric_Aggregator()
        if not finetune:
            # Load models
            logging.info('----------- Network Initialization --------------')
            model = generate_cls_model(args.model,args.num_classes)
            if "," in self.device:
                model = torch.nn.DataParallel(
                    model,
                    device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
                )
                self.args.device = f'cuda:{model.device_ids[0]}'
                model.to(self.args.device)
            else:
                model.to(self.args.device)
            logging.info('finished model init...')
            # initialize optimizer
            # optimizer = set_optimizer(args,model)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
            optimizer, scheduler = argparser_opt_scheduler(model, self.args)
            # define loss functions
            criterion = torch.nn.CrossEntropyLoss().to(args.device)

            logging.info('----------- Training from scratch --------------')
            for epoch in tqdm(range(0, args.epochs)):
                tr_loss, tr_acc, _,_ = train_epoch(args, bd_trainloader, model, optimizer, scheduler,
                                                                    criterion, epoch)
                clean_test_loss, clean_test_acc = test_epoch(args, clean_test_loader, model, criterion, epoch)
                
                bd_test_loss, bd_test_acc  = test_epoch(args, bd_test_loader, model, criterion, epoch)
                bd_test_loader.dataset.wrapped_dataset.getitem_all_switch = True 
                _, bd_test_racc = test_epoch(args, bd_test_loader, model, criterion, epoch)
                bd_test_loader.dataset.wrapped_dataset.getitem_all_switch = False
                agg(
                    {
                        "train_epoch_loss_avg_over_batch": tr_loss,
                        "train_acc": tr_acc,
                        "clean_test_loss_avg_over_batch": clean_test_loss,
                        "bd_test_loss_avg_over_batch" : bd_test_loss,
                        "test_acc" : clean_test_acc,
                        "test_asr" : bd_test_acc,
                        "test_ra" : bd_test_racc,
                    }
                )
                agg.to_dataframe().to_csv(f"{args.log}train_notrans_df.csv")
        else:
            # initialize optimizer
            # optimizer = set_optimizer(args,model)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
            logging.info('----------- Finetune the model with L_intra--------------')
            for epoch in tqdm(range(0, args.epochs)):
                tr_loss = finetune_epoch(args, bd_trainloader, model, optimizer, scheduler,
                                                                        epoch)         
               
                agg(
                    {   "epoch": epoch,
                        "train_epoch_loss_avg_over_batch": tr_loss,
                    }
                )
                agg.to_dataframe().to_csv(f"{args.log}finetune_notrans_df.csv")    
        if args.save_all_process:                    
            save_file = os.path.join(args.save_path, f'{step}.pt')
            logging.info(f'save path is {save_file}')
            save_model(model, optimizer, args, args.epochs, save_file)                                                        
        return model, optimizer, scheduler

    def train_extractor(self,):
        ## update args
        args = self.set_new_args(self.args,step="sscl")
        train_loader = get_st_train_loader(args,self.result['bd_train'],module='sscl')
        encoder = generate_cls_model(args.model,args.num_classes)
        encoder = self.drop_linear(encoder)
        args.feature_dim = encoder.register_feature_dim
        model = SupConResNet(encoder,dim_in=args.feature_dim)
        criterion = SupConLoss_Consistency(temperature=args.temp, device=args.device)
        model = model.to(args.device)
        criterion = criterion.to(args.device)
        optimizer = set_optimizer(args,model,lr=args.learning_rate)
        agg = Metric_Aggregator()

        for epoch in range(1, args.epochs + 1):
            adjust_learning_rate(args, optimizer, epoch)
            loss = _train_extractor(train_loader, model, criterion, optimizer, epoch, args)
            agg(
                    {   "epoch": epoch,
                        "train_epoch_loss_avg_over_batch": loss,
                    }
                )
            agg.to_dataframe().to_csv(f"{args.log}train_extractor_df.csv")   
        del loss
        torch.cuda.empty_cache()
        if args.save_all_process:
            # save the last model
            save_file = os.path.join(args.save_path, 'sscl-last.pt')
            save_model(model, optimizer, args, args.epochs, save_file)
        return model

    def train_classifier(self,model):
        ## update args
        args = self.set_new_args(self.args,step="mixed_ce")
        train_loader = get_st_train_loader(args,self.result['bd_train'], module="mixed_ce") 
        clean_test_loader, bd_test_loader = self.testloader_wrapper()
        model, classifier, criterion = self.set_model(args,model)
        optimizer = set_optimizer(args, classifier,lr=args.learning_rate)

        train_loss_list = []
        train_mix_acc_list = []
        clean_test_loss_list = []
        bd_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []
        agg = Metric_Aggregator()

        for epoch in range(1, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)
            train_epoch_loss_avg_over_batch, \
            train_mix_acc = _train_classifier(train_loader, model, classifier, criterion, optimizer, epoch, args)
            
            clean_test_loss_avg_over_batch, \
			bd_test_loss_avg_over_batch, \
			ra_test_loss_avg_over_batch, \
			test_acc, \
			test_asr, \
			test_ra = self.eval_step(
				model, 
                classifier,
				clean_test_loader,
				bd_test_loader,
				args,
			)
            train_loss_list.append(train_epoch_loss_avg_over_batch)
            train_mix_acc_list.append(train_mix_acc)
            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)
            agg(
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
            agg.to_dataframe().to_csv(f"{args.save_path}d-st_df.csv")

        agg.summary().to_csv(f"{args.save_path}d-st_df_summary.csv")
        if args.save_all_process:
            save_file = os.path.join(args.save_path, 'mce-last.pt')
            save_model(classifier, optimizer, args, args.epochs, save_file)
        return model,classifier

    def eval_step(self, model, classifier, clean_test_loader, bd_test_loader, args):
        clean_metrics, clean_epoch_predict_list, clean_epoch_label_list = given_dataloader_test(
            model, classifier,
            clean_test_loader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.device,
            verbose=0,
        )
        clean_test_loss_avg_over_batch = clean_metrics['test_loss_avg_over_batch']
        test_acc = clean_metrics['test_acc']
        bd_metrics, bd_epoch_predict_list, bd_epoch_label_list = given_dataloader_test(
            model, classifier,
            bd_test_loader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.device,
            verbose=0,
        )
        bd_test_loss_avg_over_batch = bd_metrics['test_loss_avg_over_batch']
        test_asr = bd_metrics['test_acc']

        bd_test_loader.dataset.wrapped_dataset.getitem_all_switch = True  # change to return the original label instead
        ra_metrics, ra_epoch_predict_list, ra_epoch_label_list = given_dataloader_test(
            model, classifier,
            bd_test_loader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.device,
            verbose=0,
        )
        ra_test_loss_avg_over_batch = ra_metrics['test_loss_avg_over_batch']
        test_ra = ra_metrics['test_acc']
        bd_test_loader.dataset.wrapped_dataset.getitem_all_switch = False  # switch back

        return clean_test_loss_avg_over_batch, \
				bd_test_loss_avg_over_batch, \
				ra_test_loss_avg_over_batch, \
				test_acc, \
				test_asr, \
				test_ra

    def continue_learn(self,args):
        step_list = ['train_notrans', 'finetune_notrans', 'calculate', 'separate', 'sscl', 'mixed_ce']
        if args.continue_step == 'mixed_ce':
            encoder = generate_cls_model(args.model,args.num_classes)
            args.feature_dim = list(encoder.named_modules())[-1][1].in_features
            if hasattr(encoder, "linear"):
                encoder.linear = nn.Identity() 
            elif hasattr(encoder, "fc"):
                encoder.fc = nn.Identity()
            model = SupConResNet(encoder,dim_in=args.feature_dim)
            
            ck_path = os.path.join(args.save_path, 'sscl-last.pt')
            result = torch.load(ck_path)
            model.load_state_dict(result['model'])
            model_new = model.to(args.device)
            return model_new
    
    def mitigation(self):
        args = self.args
        self.set_devices()
        fix_random(self.args.random_seed)
        result = self.result 
        bd_trainloader = self.get_sd_train_loader()
        clean_test_loader, bd_test_loader = self.testloader_wrapper()
        ##a. train a backdoored model from scratch using poisoned dataset without any data augmentations
        model, optimizer, scheduler = self.train_attack_noTrans(bd_trainloader, clean_test_loader, bd_test_loader, finetune=False)
        ###b. fine-tune the backdoored model with intra-class loss L_intra
        model = self.drop_linear(model)
        model, optimizer, scheduler = self.train_attack_noTrans(bd_trainloader, clean_test_loader, bd_test_loader, model=model, optimizer=optimizer, scheduler=scheduler,finetune=True)
        ###c. calculate values of the FCT metric for all training samples.
        calculate_consistency(args, bd_trainloader, model)
        ###d. calculate thresholds for choosing clean and poisoned samples.
        args.gamma_low, args.gamma_high = calculate_gamma(args,)
        ###e. separate training samples into clean samples D_c, poisoned samples D_p, and uncertain samples D_u.
        separate_samples(args, bd_trainloader, model)
        ##f. train the feature extractor (from scratch) via semi-supervised contrastive learning.
        model_new = self.train_extractor()
        ###g. train the classifier via minimizing a mixed cross-entropy loss.
        model_new, classifier = self.train_classifier(model_new)
        # return the standard model structure from two subnetworks: SupConResNet+classifier
        model_new = self.add_linear(old_model = model_new, classifier = classifier)
        result = {}
        result['model'] = model_new

        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model_new.cpu().state_dict(),
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
    d_st.add_arguments(parser)
    args = parser.parse_args()
    d_st_method = d_st(args)
    if "result_file" not in args.__dict__ or args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = d_st_method.defense(args.result_file)
