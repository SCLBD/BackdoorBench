'''
Reconstructive Neuron Pruning for Backdoor Defense
This file is modified based on the following source:
link : https://github.com/bboylyg/RNP

@article{li2023reconstructive,
    title={Reconstructive Neuron Pruning for Backdoor Defense},
    author={Li, Yige and Lyu, Xixiang and Ma, Xingjun and Koren, Nodens and Lyu, Lingjuan and Li, Bo and Jiang, Yu-Gang},
    journal={arXiv preprint arXiv:2305.14876},
    year={2023}}

basic structure:
1. config args, save_path, fix random seed
2. load the backdoor attack data and backdoor test data
3. load the backdoor model
4. rnp defense:
    a. unlearn the backdoor model and save the unlearned model
    b. recover the unlearned model and record the mask value
    c. prune the backdoor model by the mask value
5. test the result and get ASR, ACC, RC

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

from torch.utils.data import DataLoader, RandomSampler
import pandas as pd
from collections import OrderedDict
import copy

import utils.defense_utils.rnp.rnp_model as rnp_model

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import BackdoorModelTrainer, Metric_Aggregator, ModelTrainerCLS, ModelTrainerCLS_v2, PureCleanModelTrainer, all_acc, general_plot_for_epoch
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model, partially_load_state_dict
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2



if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

seed = 98
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
np.random.seed(seed)


def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(unlearned_model, lower=0.0, upper=1.0):
    params = [param for name, param in unlearned_model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values

def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)

def test(args, model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels, *additional_info) in enumerate(data_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def get_rnp_network(
    model_name: str,
    num_classes: int = 10,
    **kwargs,
):
    
    if model_name == 'preactresnet18':
        from utils.defense_utils.rnp.rnp_model.preact_rnp import PreActResNet18
        net = PreActResNet18(num_classes = num_classes, **kwargs)
    elif model_name == 'vgg19_bn':
        net = rnp_model.vgg_rnp.vgg19_bn(num_classes = num_classes,  **kwargs)
    elif model_name == 'vgg19':
        net = rnp_model.vgg_rnp.vgg19(num_classes = num_classes,  **kwargs)
    elif model_name == 'resnet183':
        net = rnp_model.resnet_rnp.resnet18(num_classes = num_classes,  **kwargs)
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')

    return net



class UnlearnModelTrainer(PureCleanModelTrainer):
    def __init__(self, model, clean_threshold):
        super(UnlearnModelTrainer, self).__init__(model)
        # self.teacher = teacher_model
        # self.criterions = criterions
        self.clean_threshold = clean_threshold

    def train_with_test_each_epoch_on_mix(self,
                                   train_dataloader,
                                   clean_test_dataloader,
                                   bd_test_dataloader,
                                   total_epoch_num,
                                   criterion,
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
            criterion,
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

            train_epoch_loss_avg_over_batch, \
            train_epoch_predict_list, \
            train_epoch_label_list, \
            train_epoch_original_index_list, \
            train_epoch_poison_indicator_list, \
            train_epoch_original_targets_list = self.train_one_epoch_on_mix(verbose=1)

            train_mix_acc = all_acc(train_epoch_predict_list, train_epoch_label_list)

            train_bd_idx = torch.where(train_epoch_poison_indicator_list == 1)[0]
            train_clean_idx = torch.where(train_epoch_poison_indicator_list == 0)[0]

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

            try:
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
            except:
                print("plot error")

            self.agg_save_dataframe()

            if train_mix_acc <= self.clean_threshold:
                break


        self.agg_save_summary()

        return train_loss_list, \
                train_mix_acc_list, \
                clean_test_loss_list, \
                bd_test_loss_list, \
                test_acc_list, \
                test_asr_list, \
                test_ra_list

    def train_one_epoch_on_mix(self, verbose=0):
        startTime = time.time()

        batch_loss_list = []
        if verbose == 1:
            batch_predict_list = []
            batch_label_list = []
            batch_original_index_list = []
            batch_poison_indicator_list = []
            batch_original_targets_list = []

        for batch_idx in range(self.batch_num_per_epoch):
            x, labels, original_index, poison_indicator, original_targets  = self.get_one_batch()
            self.model.train()
            self.model.to(device, non_blocking=self.non_blocking)

            x, labels = x.to(device, non_blocking=self.non_blocking), labels.to(device, non_blocking=self.non_blocking)

            with torch.cuda.amp.autocast(enabled=self.amp):
                log_probs = self.model(x)
                loss = self.criterion(log_probs, labels.long())
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            (-loss).backward()
            self.optimizer.step()
            # self.scaler.scale(loss).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            self.optimizer.zero_grad()

            one_batch_loss = loss.item()
            batch_predict = torch.max(log_probs, -1)[1].detach().clone().cpu()
            # one_batch_loss, batch_predict = self.one_forward_backward(x, labels, self.device, verbose)
            batch_loss_list.append(one_batch_loss)

            if verbose == 1:
                batch_predict_list.append(batch_predict.detach().clone().cpu())
                batch_label_list.append(labels.detach().clone().cpu())
                batch_original_index_list.append(original_index.detach().clone().cpu())
                batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
                batch_original_targets_list.append(original_targets.detach().clone().cpu())

        one_epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(one_epoch_loss)
            else:
                self.scheduler.step()

        endTime = time.time()

        logging.info(f"one epoch training part done, use time = {endTime - startTime} s")

        if verbose == 0:
            return one_epoch_loss, \
                   None, None, None, None, None
        elif verbose == 1:
            return one_epoch_loss, \
                   torch.cat(batch_predict_list), \
                   torch.cat(batch_label_list), \
                   torch.cat(batch_original_index_list), \
                   torch.cat(batch_poison_indicator_list), \
                   torch.cat(batch_original_targets_list)
        
class RecoverModelTrainer(PureCleanModelTrainer):
    def __init__(self, model,alpha):
        super(RecoverModelTrainer, self).__init__(model)
        # self.teacher = teacher_model
        # # self.criterions = criterions
        self.alpha = alpha

    def train_one_epoch_on_mix(self, verbose=0):
        startTime = time.time()

        batch_loss_list = []
        if verbose == 1:
            batch_predict_list = []
            batch_label_list = []
            batch_original_index_list = []
            batch_poison_indicator_list = []
            batch_original_targets_list = []

        for batch_idx in range(self.batch_num_per_epoch):
            x, labels, original_index, poison_indicator, original_targets  = self.get_one_batch()
            self.model.train()
            self.model.to(device, non_blocking=self.non_blocking)

            x, labels = x.to(device, non_blocking=self.non_blocking), labels.to(device, non_blocking=self.non_blocking)

            with torch.cuda.amp.autocast(enabled=self.amp):
                log_probs = self.model(x)
                loss = self.criterion(log_probs, labels.long())
            loss = self.alpha * loss

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            clip_mask(self.model)

            one_batch_loss = loss.item()
            batch_predict = torch.max(log_probs, -1)[1].detach().clone().cpu()
            # one_batch_loss, batch_predict = self.one_forward_backward(x, labels, self.device, verbose)
            batch_loss_list.append(one_batch_loss)

            if verbose == 1:
                batch_predict_list.append(batch_predict.detach().clone().cpu())
                batch_label_list.append(labels.detach().clone().cpu())
                batch_original_index_list.append(original_index.detach().clone().cpu())
                batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
                batch_original_targets_list.append(original_targets.detach().clone().cpu())

        one_epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(one_epoch_loss)
            else:
                self.scheduler.step()

        endTime = time.time()

        logging.info(f"one epoch training part done, use time = {endTime - startTime} s")

        if verbose == 0:
            return one_epoch_loss, \
                   None, None, None, None, None
        elif verbose == 1:
            return one_epoch_loss, \
                   torch.cat(batch_predict_list), \
                   torch.cat(batch_label_list), \
                   torch.cat(batch_original_index_list), \
                   torch.cat(batch_poison_indicator_list), \
                   torch.cat(batch_original_targets_list)

class rnp(defense):
    r"""Reconstructive Neuron Pruning for Backdoor Defense
    
    basic structure: 
    
    1. config args, save_path, fix random seed
    2. load the backdoor attack data and backdoor test data
    3. load the backdoor model
    4. rnp defense:
        a. unlearn the backdoor model and save the unlearned model
        b. recover the unlearned model and record the mask value
        c. prune the backdoor model by the mask value
    5. test the result and get ASR, ACC, RC 
       
    .. code-block:: python
    
        parser = argparse.ArgumentParser(description=sys.argv[0])
        rnp.add_arguments(parser)
        args = parser.parse_args()
        rnp_method = rnp(args)
        if "result_file" not in args.__dict__:
            args.result_file = 'one_epochs_debug_badnet_attack'
        elif args.result_file is None:
            args.result_file = 'one_epochs_debug_badnet_attack'
        result = rnp_method.defense(args.result_file)
    
    .. Note::
        @article{li2023reconstructive,
        title={Reconstructive Neuron Pruning for Backdoor Defense},
        author={Li, Yige and Lyu, Xixiang and Ma, Xingjun and Koren, Nodens and Lyu, Lingjuan and Li, Bo and Jiang, Yu-Gang},
        journal={arXiv preprint arXiv:2305.14876},
        year={2023}
        }     

    Args:
        baisc args: in the base class
        alpha (float): the weight of the loss of the unlearned model during recovering
        clean_threshold (float): the threshold of the clean accuracy of the unlearned model
        unlearning_lr (float): the learning rate of the unlearning model
        recovering_lr (float): the learning rate of the recovering model
        unlearning_epochs (int): the number of epochs of the unlearning model
        recovering_epochs (int): the number of epochs of the recovering model
        mask_file (str): the file of the mask value (default: None)
        pruning_by (str): the method of pruning (default: threshold) 
        pruning_max (float): the maximum value of the pruning (default: 0.90)   
        pruning_step (float): the step size of the pruning (default: 0.05)
        pruning_number (float): the default value of the pruning (default: 0.70)
        acc_ratio (float): the tolerance ratio of the clean accuracy (default: 0.95)
        ratio (float): the ratio of the clean data loader (default: 0.1)
        index (str): the index of the clean data (default: None)
        schedule (list int): the schedule of the learning rate (default: [10, 20])   

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
        parser.add_argument('--yaml_path', type=str, default="./config/defense/rnp/config.yaml", help='the path of yaml')

        #set the parameter for the rnp defense
        # parser.add_argument('--acc_ratio', type=float, help='the tolerance ration of the clean accuracy')
        # parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        # parser.add_argument('--print_every', type=int, help='print results every few iterations')
        # parser.add_argument('--nb_iter', type=int, help='the number of iterations for training')

        # parser.add_argument('--anp_eps', type=float)
        # parser.add_argument('--anp_steps', type=int)
        # parser.add_argument('--anp_alpha', type=float)

        # parser.add_argument('--pruning_by', type=str, choices=['number', 'threshold'])
        # parser.add_argument('--pruning_max', type=float, help='the maximum number/threshold for pruning')
        # parser.add_argument('--pruning_step', type=float, help='the step size for evaluating the pruning')

        # parser.add_argument('--pruning_number', type=float, help='the default number/threshold for pruning')

        # parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--clean_threshold', type=float, default=0.20, help='threshold of unlearning accuracy')
        parser.add_argument('--unlearning_lr', type=float, default=0.01, help='the learning rate for neuron unlearning')
        parser.add_argument('--recovering_lr', type=float, default=0.2, help='the learning rate for mask optimization')
        parser.add_argument('--unlearning_epochs', type=int, default=20, help='the number of epochs for unlearning')
        parser.add_argument('--recovering_epochs', type=int, default=20, help='the number of epochs for recovering')
        parser.add_argument('--mask_file', type=str, default=None, help='The text file containing the mask values')
        parser.add_argument('--pruning_by', type=str, default='threshold', choices=['number', 'threshold'])
        parser.add_argument('--pruning_max', type=float, default=0.90, help='the maximum number/threshold for pruning')
        parser.add_argument('--pruning_step', type=float, default=0.05, help='the step size for evaluating the pruning')

        parser.add_argument('--pruning_number', type=float,  default=0.70, help='the default number/threshold for pruning')

        parser.add_argument('--acc_ratio', type=float, help='the tolerance ration of the clean accuracy')
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--index', type=str, help='index of clean data')

        parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20],
                        help='Decrease learning rate at these epochs.')

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/defense/rnp/'
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
        if mode == 'unlearn':
            self.trainer = UnlearnModelTrainer(
                model,
                **params,
            )
        elif mode == 'recover':
            self.trainer = RecoverModelTrainer(
                model,
                **params,
            )
        elif mode == 'normal':
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
        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )
    
    def evaluate_by_number(self, args, model, mask_values, pruning_max, pruning_step, criterion,test_dataloader_dict, best_asr, acc_ori, save = True):
        results = []
        nb_max = int(np.ceil(pruning_max))
        nb_step = int(np.ceil(pruning_step))
        model_best = copy.deepcopy(model)

        number_list = []
        clean_test_loss_list = []
        bd_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []

        agg = Metric_Aggregator()
        for start in range(0, nb_max + 1, nb_step):
            i = start
            for i in range(start, start + nb_step):
                pruning(model, mask_values[i])
            layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
            # cl_loss, cl_acc = test(args, model=model, criterion=criterion, data_loader=clean_loader)
            # po_loss, po_acc = test(args, model=model, criterion=criterion, data_loader=poison_loader)
            # logging.info('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            #     i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
            # results.append('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            #     i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))    
            self.set_trainer(model)
            self.trainer.set_with_dataloader(
                ### the train_dataload has nothing to do with the backdoor defense
                train_dataloader = test_dataloader_dict['bd_test_dataloader'],
                test_dataloader_dict = test_dataloader_dict,

                criterion = criterion,
                optimizer = None,
                scheduler = None,
                device = self.args.device,
                amp = self.args.amp,

                frequency_save = self.args.frequency_save,
                save_folder_path = self.args.save_path,
                save_prefix = 'rnp',

                prefetch = self.args.prefetch,
                prefetch_transform_attr_name = "ori_image_transform_in_loading",
                non_blocking = self.args.non_blocking,


                )
            clean_test_loss_avg_over_batch, \
                    bd_test_loss_avg_over_batch, \
                    test_acc, \
                    test_asr, \
                    test_ra = self.trainer.test_current_model(
                test_dataloader_dict, args.device,
            )
            number_list.append(start)
            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)
            # cl_loss, cl_acc = test(args, model=model, criterion=criterion, data_loader=clean_loader)
            # po_loss, po_acc = test(args, model=model, criterion=criterion, data_loader=poison_loader)
            # logging.info('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            #     start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
            # results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
            #     start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
            if save:
                agg({
                    'number': start,
                    # 'layer_name': layer_name,
                    # 'neuron_idx': neuron_idx,
                    'value': value,
                    "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                    "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                    "test_acc": test_acc,
                    "test_asr": test_asr,
                    "test_ra": test_ra,
                })
                general_plot_for_epoch(
                    {
                        "Test C-Acc": test_acc_list,
                        "Test ASR": test_asr_list,
                        "Test RA": test_ra_list,
                    },
                    save_path=f"{args.save_path}number_acc_like_metric_plots.png",
                    ylabel="percentage",
                )

                general_plot_for_epoch(
                    {
                        "Test Clean Loss": clean_test_loss_list,
                        "Test Backdoor Loss": bd_test_loss_list,
                    },
                    save_path=f"{args.save_path}number_loss_metric_plots.png",
                    ylabel="percentage",
                )

                general_plot_for_epoch(
                    {
                        "number": number_list,
                    },
                    save_path=f"{args.save_path}number_plots.png",
                    ylabel="percentage",
                )

                agg.to_dataframe().to_csv(f"{args.save_path}number_df.csv")
            if abs(test_acc - acc_ori)/acc_ori < args.acc_ratio:
                if test_asr < best_asr:
                    model_best = copy.deepcopy(model)
                    best_asr = test_asr
        return results, model_best


    def evaluate_by_threshold(self, args, model, mask_values, pruning_max, pruning_step, criterion, test_dataloader_dict, best_asr, acc_ori, save = True):
        results = []
        thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
        start = 0
        model_best = copy.deepcopy(model)

        clean_test_loss_list = []
        bd_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []

        agg = Metric_Aggregator()
        for threshold in thresholds:
            idx = start
            for idx in range(start, len(mask_values)):
                if float(mask_values[idx][2]) <= threshold:
                    pruning(model, mask_values[idx])
                    start += 1
                else:
                    break
            layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
            self.set_trainer(model)
            self.trainer.set_with_dataloader(
                ### the train_dataload has nothing to do with the backdoor defense
                train_dataloader = test_dataloader_dict['bd_test_dataloader'],
                test_dataloader_dict = test_dataloader_dict,

                criterion = criterion,
                optimizer = None,
                scheduler = None,
                device = self.args.device,
                amp = self.args.amp,

                frequency_save = self.args.frequency_save,
                save_folder_path = self.args.save_path,
                save_prefix = 'rnp',

                prefetch = self.args.prefetch,
                prefetch_transform_attr_name = "ori_image_transform_in_loading",
                non_blocking = self.args.non_blocking,


                )
            clean_test_loss_avg_over_batch, \
                    bd_test_loss_avg_over_batch, \
                    test_acc, \
                    test_asr, \
                    test_ra = self.trainer.test_current_model(
                test_dataloader_dict, args.device,
            )
            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)
            # cl_loss, cl_acc = test(args, model=model, criterion=criterion, data_loader=clean_loader)
            # po_loss, po_acc = test(args, model=model, criterion=criterion, data_loader=poison_loader)
            # logging.info('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            #     start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
            # results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
            #     start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
            if save:
                agg({
                    'threshold': threshold,
                    # 'layer_name': layer_name,
                    # 'neuron_idx': neuron_idx,
                    'value': value,
                    "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                    "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                    "test_acc": test_acc,
                    "test_asr": test_asr,
                    "test_ra": test_ra,
                })
                general_plot_for_epoch(
                    {
                        "Test C-Acc": test_acc_list,
                        "Test ASR": test_asr_list,
                        "Test RA": test_ra_list,
                    },
                    save_path=f"{args.save_path}threshold_acc_like_metric_plots.png",
                    ylabel="percentage",
                )

                general_plot_for_epoch(
                    {
                        "Test Clean Loss": clean_test_loss_list,
                        "Test Backdoor Loss": bd_test_loss_list,
                    },
                    save_path=f"{args.save_path}threshold_loss_metric_plots.png",
                    ylabel="percentage",
                )

                general_plot_for_epoch(
                    {
                        "threshold": thresholds,
                    },
                    save_path=f"{args.save_path}threshold_plots.png",
                    ylabel="percentage",
                )

                agg.to_dataframe().to_csv(f"{args.save_path}threshold_df.csv")
            
            if abs(test_acc - acc_ori)/acc_ori < args.acc_ratio:
                if test_asr < best_asr:
                    model_best = copy.deepcopy(model)
                    best_asr = test_asr
        return results, model_best

    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        args = self.args
        result = self.result
        # a. train the mask of old model
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

        state_dict = self.result['model']
        net = get_rnp_network(args.model, num_classes=args.num_classes, norm_layer=None)
        load_state_dict(net, orig_state_dict=state_dict)
        net = net.to(args.device)
        criterion = torch.nn.CrossEntropyLoss().to(args.device)

        optimizer = torch.optim.SGD(net.parameters(), lr=args.unlearning_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

        self.set_trainer(net, mode = 'unlearn', clean_threshold = args.clean_threshold)
        self.trainer.train_with_test_each_epoch_on_mix(
            trainloader,
            data_clean_loader,
            data_bd_loader,
            args.unlearning_epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.args.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix='rnp',
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading", # since we use the preprocess_bd_dataset
            non_blocking=args.non_blocking,
        )

        unlearn_model_state_dict = copy.deepcopy(net.cpu().state_dict())
        
        del net
        torch.cuda.empty_cache()
        unlearned_model = get_rnp_network(args.model, num_classes=args.num_classes, norm_layer=rnp_model.MaskBatchNorm2d)
        load_state_dict(unlearned_model, orig_state_dict=unlearn_model_state_dict)
        unlearned_model = unlearned_model.to(device)
        parameters = list(unlearned_model.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        mask_optimizer = torch.optim.SGD(mask_params, lr=args.recovering_lr, momentum=0.9)

        self.set_trainer(unlearned_model, mode = 'recover', alpha = args.alpha)
        self.trainer.train_with_test_each_epoch_on_mix(
            trainloader,
            data_clean_loader,
            data_bd_loader,
            args.recovering_epochs,
            criterion=criterion,
            optimizer=mask_optimizer,
            scheduler=None,
            device=self.args.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix='rnp',
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading", # since we use the preprocess_bd_dataset
            non_blocking=args.non_blocking,
        )

        save_mask_scores(unlearned_model.state_dict(), os.path.join(args.save_path, 'mask_values.txt'))

        del unlearned_model

        net_prune = generate_cls_model(self.args.model,self.args.num_classes)
        net_prune.load_state_dict(self.result['model'])
        net_prune.to(self.args.device)

        mask_values = read_data(args.save_path + 'mask_values.txt')
        mask_values = sorted(mask_values, key=lambda x: float(x[2]))

        model = copy.deepcopy(net_prune)

        cl_loss, cl_acc = test(args, model=net_prune, criterion=criterion, data_loader=data_clean_loader)
        po_loss, po_acc = test(args, model=net_prune, criterion=criterion, data_loader=data_bd_loader)
        if args.pruning_by == 'threshold':
            results, model_pru = self.evaluate_by_threshold(
                args, net_prune, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
                criterion=criterion, test_dataloader_dict=test_dataloader_dict, best_asr=po_acc, acc_ori=cl_acc
            )
        else:
            results, model_pru = self.evaluate_by_number(
                args, net_prune, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
                criterion=criterion, test_dataloader_dict=test_dataloader_dict, best_asr=po_acc, acc_ori=cl_acc
            )
        file_name = os.path.join(args.save_path, 'pruning_by_{}.txt'.format(args.pruning_by))
        with open(file_name, "w") as f:
            f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
            f.writelines(results)


        if 'pruning_number' in args.__dict__: 
            if args.pruning_by == 'threshold':
                _, _ = self.evaluate_by_threshold(
                    args, model, mask_values, pruning_max=args.pruning_number, pruning_step=args.pruning_number,
                    criterion=criterion, test_dataloader_dict=test_dataloader_dict, best_asr=po_acc, acc_ori=cl_acc, save=False
                )
            else:
                _, _ = self.evaluate_by_number(
                    args, model, mask_values, pruning_max=args.pruning_number, pruning_step=args.pruning_number,
                    criterion=criterion, test_dataloader_dict=test_dataloader_dict, best_asr=po_acc, acc_ori=cl_acc, save=False
                )
            self.set_trainer(model)
            self.trainer.set_with_dataloader(
                ### the train_dataload has nothing to do with the backdoor defense
                train_dataloader = trainloader,
                test_dataloader_dict = test_dataloader_dict,

                criterion = criterion,
                optimizer = None,
                scheduler = None,
                device = self.args.device,
                amp = self.args.amp,

                frequency_save = self.args.frequency_save,
                save_folder_path = self.args.save_path,
                save_prefix = 'rnp',

                prefetch = self.args.prefetch,
                prefetch_transform_attr_name = "ori_image_transform_in_loading",
                non_blocking = self.args.non_blocking,


                )
            agg = Metric_Aggregator()
            clean_test_loss_avg_over_batch, \
                    bd_test_loss_avg_over_batch, \
                    test_acc, \
                    test_asr, \
                    test_ra = self.trainer.test_current_model(
                test_dataloader_dict, self.args.device,
            )
            agg({
                    "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                    "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                    "test_acc": test_acc,
                    "test_asr": test_asr,
                    "test_ra": test_ra,
                })
            agg.to_dataframe().to_csv(f"{args.save_path}rnp_df_summary.csv")
            result = {}
            result['model'] = model
            save_defense_result(
                model_name=args.model,
                num_classes=args.num_classes,
                model=model.cpu().state_dict(),
                save_path=args.save_path,
            )

            return result

        ### TODO  add all threshold estimate
        # self.set_trainer(model_pru)
        # self.trainer.set_with_dataloader(
        #     ### the train_dataload has nothing to do with the backdoor defense
        #     train_dataloader = trainloader,
        #     test_dataloader_dict = test_dataloader_dict,

        #     criterion = criterion,
        #     optimizer = None,
        #     scheduler = None,
        #     device = self.args.device,
        #     amp = self.args.amp,

        #     frequency_save = self.args.frequency_save,
        #     save_folder_path = self.args.save_path,
        #     save_prefix = 'rnp',

        #     prefetch = self.args.prefetch,
        #     prefetch_transform_attr_name = "ori_image_transform_in_loading",
        #     non_blocking = self.args.non_blocking,


        #     )
        # agg = Metric_Aggregator()
        # clean_test_loss_avg_over_batch, \
        #         bd_test_loss_avg_over_batch, \
        #         test_acc, \
        #         test_asr, \
        #         test_ra = self.trainer.test_current_model(
        #     test_dataloader_dict, self.args.device,
        # )
        # agg({
        #         "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
        #         "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
        #         "test_acc": test_acc,
        #         "test_asr": test_asr,
        #         "test_ra": test_ra,
        #     })
        # agg.to_dataframe().to_csv(f"{args.save_path}rnp_df_summary.csv")
        # result = {}
        # result['model'] = model_pru
        # save_defense_result(
        #     model_name=args.model,
        #     num_classes=args.num_classes,
        #     model=model_pru.cpu().state_dict(),
        #     save_path=args.save_path,
        # )
        # return result

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    rnp.add_arguments(parser)
    args = parser.parse_args()
    anp_method = rnp(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = anp_method.defense(args.result_file)