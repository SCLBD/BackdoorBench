'''
This file implements the defense method called D-BR from Effective Backdoor Defense by Exploiting Sensitivity of Poisoned Samples.
This file is modified based on the following source:
link :  https://github.com/SCLBD/Effective_backdoor_defense

@article{chen2022effective,
  title={Effective backdoor defense by exploiting sensitivity of poisoned samples},
  author={Chen, Weixin and Wu, Baoyuan and Wang, Haoqian},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={9727--9737},
  year={2022}
}

The defense method is called d-br.
It removes the backdoor from a given backdoored model with a poisoned dataset.


The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
basic sturcture for defense method:
    1. basic setting: args
    2. load attack result(model, train data, test data)
    3. d-br defense: mainly two steps: sd and st (Sample-Distinguishment and two-stage Secure Training)
        (sd:)
        a. train a backdoored model from scratch using poisoned dataset without any data augmentations
        b. fine-tune the backdoored model with intra-class loss L_intra.
        c. calculate values of the FCT metric for all training samples.
        d. calculate thresholds for choosing clean and poisoned samples.
        e. separate training samples into clean samples D_c, poisoned samples D_p, and uncertain samples D_u.
        (br:)
        f. unlearn and relearn the backdoored model.
    4. test the result and get ASR, ACC, RC 
'''

import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import copy
import math
from copy import deepcopy

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import BackdoorModelTrainer, Metric_Aggregator, ModelTrainerCLS, ModelTrainerCLS_v2, PureCleanModelTrainer, all_acc, general_plot_for_epoch, given_dataloader_test
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import dataset_wrapper_with_transform, prepro_cls_DatasetBD_v2
## d-st utils
from utils.defense_utils.dbr.dataloader_bd import get_transform_br, TransformThree, normalization
from utils.defense_utils.dbr.sd import calculate_consistency, calculate_gamma, separate_samples
from utils.defense_utils.dbr.dataloader_bd import get_br_train_loader
from utils.defense_utils.dbr.utils_br import *
from utils.defense_utils.dbr.utils_br import progress_bar
from utils.defense_utils.dbr.dataloader_bd import Dataset_npy


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

    torch.cuda.empty_cache()
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_loss/(i + 1))
        else:
            scheduler.step()
    # return train_loss / (i + 1), avg_acc_clean, avg_acc_attack, avg_acc_robust
    return train_loss / (i + 1)

def learning_rate_unlearning(optimizer, epoch, opt):
    lr = opt.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def transform_finetuning(args,):
    transforms_list = []
    # transforms_list.append(transforms.ToPILImage())
    transforms_list.append(transforms.Resize((args.input_height, args.input_width)))
    if args.dataset == "imagenet":
        transforms_list.append(transforms.RandomRotation(20))
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
    else:
        transforms_list.append(transforms.RandomCrop((args.input_height, args.input_width), padding=4))
        if args.dataset == "cifar10":
            transforms_list.append(transforms.RandomHorizontalFlip())
    transforms_list.append(transforms.ToTensor())
    trasform_compose = transforms.Compose(transforms_list)
    return trasform_compose

class d_br(defense):
    r"""Effective backdoor defense by exploiting sensitivity of poisoned samples
    
    basic structure: 
    
    1. config args, save_path, fix random seed
    2. load the backdoor attack data and backdoor test data
    3. d-br defense: mainly two steps: sd and st (Sample-Distinguishment and two-stage Secure Training)
        (sd:)
        a. train a backdoored model from scratch using poisoned dataset without any data augmentations
        b. fine-tune the backdoored model with intra-class loss L_intra.
        c. calculate values of the FCT metric for all training samples.
        d. calculate thresholds for choosing clean and poisoned samples.
        e. separate training samples into clean samples D_c, poisoned samples D_p, and uncertain samples D_u.
        (br:)
        f. unlearn and relearn the backdoored model.
    4. test the result and get ASR, ACC, RC with regard to the chosen threshold and interval
       
    .. code-block:: python
    
        parser = argparse.ArgumentParser(description=sys.argv[0])
        d-br.add_arguments(parser)
        args = parser.parse_args()
        d-br_method = d-br(args)
        if "result_file" not in args.__dict__:
            args.result_file = 'one_epochs_debug_badnet_attack'
        elif args.result_file is None:
            args.result_file = 'one_epochs_debug_badnet_attack'
        result = d-br_method.defense(args.result_file)
    
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
        parser.add_argument('--yaml_path', type=str, default="./config/defense/d-br/config.yaml", help='the path of yaml')
    
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

        #set the parameter for the d-br defense
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
        save_path = 'record/' + result_file + '/defense/d-br/'
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
        elif step == 'unlearn_relearn':
            args.epochs = 20
            args.batch_size = 128
            args.lr = 0.0001
        if args.debug:
            args.epochs = 1
            args.batch_size = 16
        return args

    def set_model(self):
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
        return model

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
    
    def get_sd_train_loader(self):
        args = self.args
        transform1, transform2, transform3 = get_transform_br(args, train=True)
        data_set_o = self.result['bd_train']
        data_set_o.wrap_img_transform = TransformThree(transform1, transform2, transform3)
        poisoned_data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)    

        return poisoned_data_loader

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

    def train_step_unlearning(self, args, train_loader, model_ascent, optimizer, criterion, epoch):
        model_ascent.train()

        total_clean, total_clean_correct = 0, 0

        for idx, (img, target, *additional_info) in enumerate(train_loader, start=1):
            img = normalization(args, img)
            img = img.to(args.device)
            target = target.to(args.device)

            output = model_ascent(img)
            loss = criterion(output, target)

            optimizer.zero_grad()
            (-loss).backward()  # Gradient ascent training
            optimizer.step()

            total_clean_correct += torch.sum(torch.argmax(output[:], dim=1) == target[:])
            total_clean += img.shape[0]
            avg_acc_clean = total_clean_correct * 100.0 / total_clean
            progress_bar(idx, len(train_loader),
                        'Epoch: %d | Loss: %.3f | Train ACC: %.3f%% (%d/%d)' % (
                        epoch, loss / (idx + 1), avg_acc_clean, total_clean_correct, total_clean))

        del loss, img, output
        torch.cuda.empty_cache()
        return model_ascent

    def train_step_relearning(self, args, train_loader, model_descent, optimizer, criterion, epoch):
        model_descent.train()
        losses = AverageMeter()
        top1 = AverageMeter()

        for idx, (img, target, *additional_info) in enumerate(train_loader, start=1):
            img = normalization(args, img)
            img = img.to(args.device)
            target = target.to(args.device)
            bsz = target.shape[0]
            output = model_descent(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()  # Gradient ascent training
            optimizer.step()

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0].detach().cpu().numpy(), bsz)
            if (idx + 1) % args.print_freq == 0:
                logging.info(f'Train: [{epoch}][{idx + 1}/{len(train_loader)}]\t \
                    loss {losses.val} ({losses.avg}\t \
                    Acc@1 {top1.val} ({top1.avg}')
                sys.stdout.flush()
   
        del loss, img, output
        torch.cuda.empty_cache()
        return losses.avg, top1.avg, model_descent

    def eval_step(self, model, clean_test_loader, bd_test_loader, args):
        clean_metrics, clean_epoch_predict_list, clean_epoch_label_list = given_dataloader_test(
            model,
            clean_test_loader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.device,
            verbose=0,
        )
        clean_test_loss_avg_over_batch = clean_metrics['test_loss_avg_over_batch']
        test_acc = clean_metrics['test_acc']
        bd_metrics, bd_epoch_predict_list, bd_epoch_label_list = given_dataloader_test(
            model,
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
            model,
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

    def get_bd_indicator_by_original_from_bd_dataset(self, dataset):
        orig2idx_dict = {}
        for idx,(img, label, *other_info) in enumerate(dataset) :
            orig2idx_dict[other_info[0]] = idx
        return orig2idx_dict

    def idx_from_orig(self, orig2idx_dict, orig_idx):
        temp_idx = []
        for idx in orig_idx:
            temp_idx.append(orig2idx_dict[idx])
        return temp_idx
    
    def get_isolate_data_loader(self, args):
        transform_compose = transform_finetuning(args)
        folder_path = os.path.join(args.save_path,'data_produce')
        clean_idx_list = np.load(os.path.join(folder_path, 'clean_samples.npy'))
        poison_idx_list = np.load(os.path.join(folder_path, 'poison_samples.npy'))
        train_dataset = self.result['bd_train']
        train_dataset_copy = deepcopy(self.result['bd_train'])
        orig2idx_dict = self.get_bd_indicator_by_original_from_bd_dataset(train_dataset)  # get map from original index to current index
        clean_idx = self.idx_from_orig(orig2idx_dict, clean_idx_list)
        bd_idx = self.idx_from_orig(orig2idx_dict, poison_idx_list)

        train_dataset.subset(clean_idx)
        train_dataset_copy.subset(bd_idx)
        clean_data_tf = train_dataset
        poison_data_tf = train_dataset_copy
        clean_data_tf.wrap_img_transform = deepcopy(transform_compose)
        poison_data_tf.wrap_img_transform = deepcopy(transform_compose)
        isolate_clean_data_loader = torch.utils.data.DataLoader(dataset=clean_data_tf, batch_size=args.batch_size, shuffle=True)
        isolate_poison_data_loader = torch.utils.data.DataLoader(dataset=poison_data_tf, batch_size=args.batch_size, shuffle=True)
        return isolate_clean_data_loader,isolate_poison_data_loader

    def unlearn_relearn(self, model):
        
        args = self.set_new_args(self.args,step='unlearn_relearn')
        isolate_clean_data_loader,isolate_poison_data_loader = self.get_isolate_data_loader(args)
        clean_test_loader, bd_test_loader = self.testloader_wrapper()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss().to(args.device)

        # Training and Testing
        train_loss_list = []
        train_mix_acc_list = []
        clean_test_loss_list = []
        bd_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []
        agg = Metric_Aggregator()

        logging.info('----------- Unlearning and relearning--------------')
        for epoch in tqdm(range(1, args.epochs+1)):
            # Modify lr
            learning_rate_unlearning(optimizer, epoch, args)
            # Unlearn
            print('-----Unlearning-------')
            model = self.train_step_unlearning(args, isolate_poison_data_loader, model, optimizer, criterion, epoch)
            # Relearn
            print('-----Relearning-------')
            train_epoch_loss_avg_over_batch, \
            train_mix_acc, \
            model = self.train_step_relearning(args, isolate_clean_data_loader, model, optimizer, criterion, epoch)

            clean_test_loss_avg_over_batch, \
			bd_test_loss_avg_over_batch, \
			ra_test_loss_avg_over_batch, \
			test_acc, \
			test_asr, \
			test_ra = self.eval_step(
				model,
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
            agg.to_dataframe().to_csv(f"{args.save_path}d-br_df.csv")
        agg.summary().to_csv(f"{args.save_path}d-br_df_summary.csv")
        if args.save_all_process: 
            # Save the best model
            save_file = os.path.join(args.save_path, 'unlearn_relearn-last.pth')
            save_model(model, optimizer, args, args.epochs, save_file)
        return model

    def mitigation(self):
        args = self.args
        self.set_devices()
        fix_random(self.args.random_seed)
        if args.debug:
            args.batch_size = 32
        result = self.result 
        bd_trainloader = self.get_sd_train_loader()
        clean_test_loader, bd_test_loader = self.testloader_wrapper()
        ###a. train a backdoored model from scratch using poisoned dataset without any data augmentations
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
        ###f. load the backdoored model, then unlearn and relearn the model.
        model_new = self.set_model()
        model_new = self.unlearn_relearn(model_new)
       
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
    d_br.add_arguments(parser)
    args = parser.parse_args()
    d_st_method = d_br(args)
    if "result_file" not in args.__dict__ or args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = d_st_method.defense(args.result_file)

