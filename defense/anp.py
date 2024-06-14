'''
Adversarial Neuron Pruning Purifies Backdoored Deep Models

This file is modified based on the following source:
link : https://github.com/csdongxian/ANP_backdoor.
The defense method is called anp.

@article{wu2021adversarial,
        title={Adversarial neuron pruning purifies backdoored deep models},
        author={Wu, Dongxian and Wang, Yisen},
        journal={Advances in Neural Information Processing Systems},
        volume={34},
        pages={16913--16925},
        year={2021}
        }

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. reconstruct some backbone vgg19 and add some backbone such as densenet161 efficientnet mobilenet
    7. save best model which gets the minimum of asr with acc decreased by no more than 10%
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. anp defense:
        a. train the mask of old model
        b. prune the model depend on the mask
    4. test the result and get ASR, ACC, RC 
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

import utils.defense_utils.anp.anp_model as anp_model

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import BackdoorModelTrainer, Metric_Aggregator, ModelTrainerCLS, ModelTrainerCLS_v2, PureCleanModelTrainer, general_plot_for_epoch
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model, partially_load_state_dict
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2



### anp function
def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, anp_model.NoisyBatchNorm2d) or isinstance(module, anp_model.NoisyBatchNorm1d):
            module.perturb(is_perturbed=is_perturbed)
        if isinstance(module, anp_model.NoiseLayerNorm2d) or isinstance(module, anp_model.NoiseLayerNorm):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, anp_model.NoisyBatchNorm2d) or isinstance(module, anp_model.NoisyBatchNorm1d):
            module.include_noise()
        if isinstance(module, anp_model.NoiseLayerNorm2d) or isinstance(module, anp_model.NoiseLayerNorm):
            module.include_noise()



def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, anp_model.NoisyBatchNorm2d) or isinstance(module, anp_model.NoisyBatchNorm1d):
            module.exclude_noise()
        if isinstance(module, anp_model.NoiseLayerNorm2d) or isinstance(module, anp_model.NoiseLayerNorm):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, anp_model.NoisyBatchNorm2d) or isinstance(module, anp_model.NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=args.anp_eps)
        if isinstance(module, anp_model.NoiseLayerNorm2d) or isinstance(module, anp_model.NoiseLayerNorm):
            module.reset(rand_init=rand_init, eps=args.anp_eps)


def mask_train(args, model, criterion, mask_opt, noise_opt, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels, *additional_info) in enumerate(data_loader):
        images, labels = images.to(args.device), labels.to(args.device)
        nb_samples += images.size(0)

        # step 1: calculate the adversarial perturbation for neurons
        if args.anp_eps > 0.0:
            reset(model, rand_init=True)
            for _ in range(args.anp_steps):
                noise_opt.zero_grad()

                include_noise(model)
                output_noise = model(images)
                loss_noise = - criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()

        # step 2: calculate loss and update the mask values
        mask_opt.zero_grad()
        if args.anp_eps > 0.0:
            include_noise(model)
            output_noise = model(images)
            loss_rob = criterion(output_noise, labels)
        else:
            loss_rob = 0.0

        exclude_noise(model)
        output_clean = model(images)
        loss_nat = criterion(output_clean, labels)
        loss = args.anp_alpha * loss_nat + (1 - args.anp_alpha) * loss_rob

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


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

def get_anp_network(
    model_name: str,
    num_classes: int = 10,
    **kwargs,
):
    
    if model_name == 'preactresnet18':
        from utils.defense_utils.anp.anp_model.preact_anp import PreActResNet18
        net = PreActResNet18(num_classes = num_classes, **kwargs)
    elif model_name == 'vgg19_bn':
        net = anp_model.vgg_anp.vgg19_bn(num_classes = num_classes,  **kwargs)
    elif model_name == 'densenet161':
        net = anp_model.den_anp.densenet161(num_classes= num_classes, **kwargs)
    elif model_name == 'mobilenet_v3_large':
        net = anp_model.mobilenet_anp.mobilenet_v3_large(num_classes= num_classes, **kwargs)
    elif model_name == 'efficientnet_b3':
        net = anp_model.eff_anp.efficientnet_b3(num_classes= num_classes, **kwargs)
    elif model_name == 'convnext_tiny':
        # net_from_imagenet = convnext_tiny(pretrained=True) #num_classes = num_classes)
        try :
            net = anp_model.conv_anp.convnext_tiny(num_classes= num_classes, **{k:v for k,v in kwargs.items() if k != "pretrained"})
        except :
            net = anp_model.conv_new_anp.convnext_tiny(num_classes= num_classes, **{k:v for k,v in kwargs.items() if k != "pretrained"})
        # partially_load_state_dict(net, net_from_imagenet.state_dict())
        # net = anp_model.convnext_anp.convnext_tiny(num_classes= num_classes, **kwargs)
    elif model_name == 'vit_b_16':
        try :
            from torchvision.transforms import Resize
            net = anp_model.vit_anp.vit_b_16(
                    pretrained = False,
                    # **{k: v for k, v in kwargs.items() if k != "pretrained"}
                )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
            net = torch.nn.Sequential(
                    Resize((224, 224)),
                    net,
                )
        except :
            from torchvision.transforms import Resize
            net = anp_model.vit_new_anp.vit_b_16(
                    pretrained = False,
                    # **{k: v for k, v in kwargs.items() if k != "pretrained"}
                )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
            net = torch.nn.Sequential(
                    Resize((224, 224)),
                    net,
                )
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')

    return net

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





class anp(defense):
    r"""Adversarial Neuron Pruning Purifies Backdoored Deep Models
    
    basic structure: 
    
    1. config args, save_path, fix random seed
    2. load the backdoor attack data and backdoor test data
    3. load the backdoor attack model
    4. anp defense:
        a. train the mask of old model
        b. prune the model depend on the mask
    5. test the result and get ASR, ACC, RC 
       
    .. code-block:: python
    
        parser = argparse.ArgumentParser(description=sys.argv[0])
        anp.add_arguments(parser)
        args = parser.parse_args()
        anp_method = anp(args)
        if "result_file" not in args.__dict__:
            args.result_file = 'one_epochs_debug_badnet_attack'
        elif args.result_file is None:
            args.result_file = 'one_epochs_debug_badnet_attack'
        result = anp_method.defense(args.result_file)
    
    .. Note::
        @article{wu2021adversarial,
        title={Adversarial neuron pruning purifies backdoored deep models},
        author={Wu, Dongxian and Wang, Yisen},
        journal={Advances in Neural Information Processing Systems},
        volume={34},
        pages={16913--16925},
        year={2021}
        }

    Args:
        baisc args: in the base class
        anp_eps (float): the epsilon for the anp defense in the first step to train the mask
        anp_steps (int): the training steps for the anp defense in the first step to train the mask
        anp_alpha (float): the alpha for the anp defense in the first step to train the mask for the loss
        pruning_by (str): the method for pruning, number or threshold
        pruning_max (float): the maximum number/threshold for pruning
        pruning_step (float): the step size for evaluating the pruning
        pruning_number (float): the default number/threshold for pruning
        index (str): the index of the clean data
        acc_ratio (float): the tolerance ration of the clean accuracy
        ratio (float): the ratio of clean data loader
        print_every (int): print results every few iterations
        nb_iter (int): the number of iterations for training

    Update:
        All threshold evaluation results will be saved in the save_path folder as a picture, and the selected fixed threshold model results will be saved to defense_result.pt

        
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
        parser.add_argument('--yaml_path', type=str, default="./config/defense/anp/config.yaml", help='the path of yaml')

        #set the parameter for the anp defense
        parser.add_argument('--acc_ratio', type=float, help='the tolerance ration of the clean accuracy')
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--print_every', type=int, help='print results every few iterations')
        parser.add_argument('--nb_iter', type=int, help='the number of iterations for training')

        parser.add_argument('--anp_eps', type=float)
        parser.add_argument('--anp_steps', type=int)
        parser.add_argument('--anp_alpha', type=float)

        parser.add_argument('--pruning_by', type=str, choices=['number', 'threshold'])
        parser.add_argument('--pruning_max', type=float, help='the maximum number/threshold for pruning')
        parser.add_argument('--pruning_step', type=float, help='the step size for evaluating the pruning')

        parser.add_argument('--pruning_number', type=float, help='the default number/threshold for pruning')

        parser.add_argument('--index', type=str, help='index of clean data')



    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/defense/anp/'
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
                save_prefix = 'anp',

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
                save_prefix = 'anp',

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
        data_set_clean = self.result['clean_train']
        data_set_clean.wrapped_dataset = data_set_without_tran
        data_set_clean.wrap_img_transform = train_tran
        # data_set_clean.wrapped_dataset.getitem_all = False
        random_sampler = RandomSampler(data_source=data_set_clean, replacement=True,
                                    num_samples=args.print_every * args.batch_size)
        clean_val_loader = DataLoader(data_set_clean, batch_size=args.batch_size,
                                    shuffle=False, sampler=random_sampler, num_workers=0)
        
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        # data_bd_testset.wrapped_dataset.getitem_all = False
        poison_test_loader = DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        clean_test_loader = DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = clean_test_loader
        test_dataloader_dict["bd_test_dataloader"] = poison_test_loader
        state_dict = self.result['model']
        net = get_anp_network(args.model, num_classes=args.num_classes, norm_layer=anp_model.NoisyBatchNorm2d)
        load_state_dict(net, orig_state_dict=state_dict)
        net = net.to(args.device)
        criterion = torch.nn.CrossEntropyLoss().to(args.device)

        parameters = list(net.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=0.9)
        noise_params = [v for n, v in parameters if "neuron_noise" in n]
        noise_optimizer = torch.optim.SGD(noise_params, lr=args.anp_eps / args.anp_steps)

        logging.info('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
        for i in range(nb_repeat):
            start = time.time()
            lr = mask_optimizer.param_groups[0]['lr']
            train_loss, train_acc = mask_train(args, model=net, criterion=criterion, data_loader=clean_val_loader,
                                            mask_opt=mask_optimizer, noise_opt=noise_optimizer)
            cl_test_loss, cl_test_acc = test(args, model=net, criterion=criterion, data_loader=clean_test_loader)
            po_test_loss, po_test_acc = test(args, model=net, criterion=criterion, data_loader=poison_test_loader)
            end = time.time()
            logging.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                (i + 1) * args.print_every, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
                cl_test_loss, cl_test_acc))
        save_mask_scores(net.state_dict(), os.path.join(args.checkpoint_save, 'mask_values.txt'))

        # b. prune the model depend on the mask
        net_prune = generate_cls_model(args.model,args.num_classes)
        net_prune.load_state_dict(result['model'])
        net_prune.to(args.device)

        mask_values = read_data(args.checkpoint_save + 'mask_values.txt')
        mask_values = sorted(mask_values, key=lambda x: float(x[2]))
        logging.info('No. \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        cl_loss, cl_acc = test(args, model=net_prune, criterion=criterion, data_loader=clean_test_loader)
        po_loss, po_acc = test(args, model=net_prune, criterion=criterion, data_loader=poison_test_loader)
        logging.info('0 \t None     \t None     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))

        model = copy.deepcopy(net_prune)
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
        file_name = os.path.join(args.checkpoint_save, 'pruning_by_{}.txt'.format(args.pruning_by))
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
                train_dataloader = clean_val_loader,
                test_dataloader_dict = test_dataloader_dict,

                criterion = criterion,
                optimizer = None,
                scheduler = None,
                device = self.args.device,
                amp = self.args.amp,

                frequency_save = self.args.frequency_save,
                save_folder_path = self.args.save_path,
                save_prefix = 'anp',

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
            agg.to_dataframe().to_csv(f"{args.save_path}anp_df_summary.csv")
            result = {}
            result['model'] = model
            save_defense_result(
                model_name=args.model,
                num_classes=args.num_classes,
                model=model_pru.cpu().state_dict(),
                save_path=args.save_path,
            )

            return result

        self.set_trainer(model_pru)
        self.trainer.set_with_dataloader(
            ### the train_dataload has nothing to do with the backdoor defense
            train_dataloader = clean_val_loader,
            test_dataloader_dict = test_dataloader_dict,

            criterion = criterion,
            optimizer = None,
            scheduler = None,
            device = self.args.device,
            amp = self.args.amp,

            frequency_save = self.args.frequency_save,
            save_folder_path = self.args.save_path,
            save_prefix = 'anp',

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
        agg.to_dataframe().to_csv(f"{args.save_path}anp_df_summary.csv")
        result = {}
        result['model'] = model_pru
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model_pru.cpu().state_dict(),
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
    anp.add_arguments(parser)
    args = parser.parse_args()
    anp_method = anp(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = anp_method.defense(args.result_file)