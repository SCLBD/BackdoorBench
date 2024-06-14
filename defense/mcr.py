# Bridging mode connectivity in loss landscapes and adversarial robustness
'''
This file is modified based on the following source:
link : https://github.com/IBM/model-sanitization.

@inproceedings{zhao2020bridging,
    title={BRIDGING MODE CONNECTIVITY IN LOSS LANDSCAPES AND ADVERSARIAL ROBUSTNESS},
    author={Zhao, Pu and Chen, Pin-Yu and Das, Payel and Ramamurthy, Karthikeyan Natesan and Lin, Xue},
    booktitle={International Conference on Learning Representations (ICLR 2020)},
    year={2020}}

The defense method is called MCR.

Since the model is different from original paper, we change the hyperparameter for preactresnet18 on cifar10 to align the performance.

basic structure:
1. config args, save_path, fix random seed
2. load the backdoor attack data and backdoor test data
3. mcr
    a. use poisoned model and clean(finetuned from poison) model to form a curve in parameter space
    b. train curve with given subset of data, test with given t
4. test the result and get ASR, ACC, RC


'''

import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import shutil

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import pformat
import yaml
# import logging
import time
from copy import deepcopy
from typing import List
import logging
# from pyhessian import hessian  # Hessian computation
import matplotlib.pyplot as plt
# import numpy as np

from defense.base import defense
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from utils.trainer_cls import ModelTrainerCLS_v2, BackdoorModelTrainer, Metric_Aggregator, given_dataloader_test, \
    general_plot_for_epoch
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from utils.trainer_cls import test_given_dataloader_on_mix, all_acc, plot_acc_like_metric_pure, \
    validate_list_for_plot  # plot_loss, plot_acc_like_metric,

import numpy as np
# import math
import torch
# import torch.nn.functional as F
from torch.nn import Module, Parameter
# from torch.nn.modules.utils import _pair
from scipy.special import binom


def plot_loss(
        train_loss_list: list,
        clean_test_loss_list: list,
        bd_test_loss_list: list,
        save_folder_path: str,
        save_file_name="loss_metric_plots",
        frequency=1,
):
    '''These line of set color is from https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib'''
    NUM_COLORS = 3
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize=(12.8, 9.6))  # 4x default figsize
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    len_set = len(train_loss_list)
    x = np.arange(len_set) * frequency
    if validate_list_for_plot(train_loss_list, len_set):
        plt.plot(x, train_loss_list, marker="o", linewidth=2, label="Train Loss", linestyle="--")
    else:
        logging.warning("train_loss_list contains None or len not match")
    if validate_list_for_plot(clean_test_loss_list, len_set):
        plt.plot(x, clean_test_loss_list, marker="v", linewidth=2, label="Test Clean loss", linestyle="-")
    else:
        logging.warning("clean_test_loss_list contains None or len not match")
    if validate_list_for_plot(bd_test_loss_list, len_set):
        plt.plot(x, bd_test_loss_list, marker="+", linewidth=2, label="Test Backdoor Loss", linestyle="-.")
    else:
        logging.warning("bd_test_loss_list contains None or len not match")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.ylim((0,
              max([value for value in  # filter None value
                   train_loss_list +
                   clean_test_loss_list +
                   bd_test_loss_list if value is not None])
              ))
    plt.legend()
    plt.title("Results")
    plt.grid()
    plt.savefig(f"{save_folder_path}/{save_file_name}.png")
    plt.close()


def plot_acc_like_metric(
        train_acc_list: list,
        train_asr_list: list,
        train_ra_list: list,
        test_acc_list: list,
        test_asr_list: list,
        test_ra_list: list,
        save_folder_path: str,
        save_file_name="acc_like_metric_plots",
        frequency=1,

):
    len_set = len(test_asr_list)
    x = np.arange(len(test_asr_list)) * frequency

    '''These line of set color is from https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib'''
    NUM_COLORS = 6
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize=(12.8, 9.6))  # 4x default figsize
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    if validate_list_for_plot(train_acc_list, len_set):
        plt.plot(x, train_acc_list, marker="o", linewidth=2, label="Train Acc", linestyle="--")
    else:
        logging.warning("train_acc_list contains None, or len not match")
    if validate_list_for_plot(train_asr_list, len_set):
        plt.plot(x, train_asr_list, marker="v", linewidth=2, label="Train ASR", linestyle="-")
    else:
        logging.warning("train_asr_list contains None, or len not match")
    if validate_list_for_plot(train_ra_list, len_set):
        plt.plot(x, train_ra_list, marker="+", linewidth=2, label="Train RA", linestyle="-.")
    else:
        logging.warning("train_ra_list contains None, or len not match")
    if validate_list_for_plot(test_acc_list, len_set):
        plt.plot(x, test_acc_list, marker="o", linewidth=2, label="Test C-Acc", linestyle="--")
    else:
        logging.warning("test_acc_list contains None, or len not match")
    if validate_list_for_plot(test_asr_list, len_set):
        plt.plot(x, test_asr_list, marker="v", linewidth=2, label="Test ASR", linestyle="-")
    else:
        logging.warning("test_asr_list contains None, or len not match")
    if validate_list_for_plot(test_ra_list, len_set):
        plt.plot(x, test_ra_list, marker="+", linewidth=2, label="Test RA", linestyle="-.")
    else:
        logging.warning("test_ra_list contains None, or len not match")

    plt.xlabel("Epochs")
    plt.ylabel("ACC")

    plt.ylim((0, 1))
    plt.legend()
    plt.title("Results")
    plt.grid()
    plt.savefig(f"{save_folder_path}/{save_file_name}.png")
    plt.close()


# def plot_hessian_eigenvalues(
#         model_visual,
#         data_loader,  # only use one batch
#         device,
#         save_path_for_hessian=None,  # xx/xx/xx.png
# ):
#     # save_path_for_hessian =
#     # data_loader =
#     # device =
#     # model_visual =
#
#     model_visual = (model_visual)
#     data_loader = (data_loader)
#     model_visual.to(device)
#
#     # !!! Important to set eval mode !!!
#     model_visual.eval()
#
#     criterion = torch.nn.CrossEntropyLoss()
#
#     batch_x, batch_y, *others = next(iter(data_loader))
#     batch_x = batch_x.to(device)
#     batch_y = batch_y.to(device)
#
#     if torch.__version__ > '1.8.1':
#         logging.info('Use self-defined function as an alternative for torch.eig since your torch>=1.9')
#
#         def old_torcheig(A, eigenvectors):
#             '''A temporary function as an alternative for torch.eig (torch<1.9)'''
#             vals, vecs = torch.linalg.eig(A)
#             if torch.is_complex(vals) or torch.is_complex(vecs):
#                 logging.info(
#                     'Warning: Complex values founded in Eigenvalues/Eigenvectors. This is impossible for real symmetric matrix like Hessian. \n We only keep the real part.')
#
#             vals = torch.real(vals)
#             vecs = torch.real(vecs)
#
#             # vals is a nx2 matrix. see https://virtualgroup.cn/pytorch.org/docs/stable/generated/torch.eig.html
#             vals = vals.view(-1, 1) + torch.zeros(vals.size()[0], 2).to(vals.device)
#             if eigenvectors:
#                 return vals, vecs
#             else:
#                 return vals, torch.tensor([])
#
#         torch.eig = old_torcheig
#
#     # create the hessian computation module
#     hessian_comp = hessian(model_visual, criterion, data=(batch_x, batch_y), cuda=True)
#     # Now let's compute the top 2 eigenavlues and eigenvectors of the Hessian
#     top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2, maxIter=1000)
#     logging.info("The top two eigenvalues of this model are: %.4f %.4f" % (top_eigenvalues[0], top_eigenvalues[1]))
#
#     if save_path_for_hessian is not None:
#
#         density_eigen, density_weight = hessian_comp.density()
#
#         def get_esd_plot(eigenvalues, weights):
#             density, grids = density_generate(eigenvalues, weights)
#             plt.semilogy(grids, density + 1.0e-7)
#             plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
#             plt.xlabel('Eigenvlaue', fontsize=14, labelpad=10)
#             plt.xticks(fontsize=12)
#             plt.yticks(fontsize=12)
#             plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
#             return plt.gca()
#
#         def density_generate(eigenvalues,
#                              weights,
#                              num_bins=10000,
#                              sigma_squared=1e-5,
#                              overhead=0.01):
#             eigenvalues = np.array(eigenvalues)
#             weights = np.array(weights)
#
#             lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
#             lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead
#
#             grids = np.linspace(lambda_min, lambda_max, num=num_bins)
#             sigma = sigma_squared * max(1, (lambda_max - lambda_min))
#
#             num_runs = eigenvalues.shape[0]
#             density_output = np.zeros((num_runs, num_bins))
#
#             for i in range(num_runs):
#                 for j in range(num_bins):
#                     x = grids[j]
#                     tmp_result = gaussian(eigenvalues[i, :], x, sigma)
#                     density_output[i, j] = np.sum(tmp_result * weights[i, :])
#             density = np.mean(density_output, axis=0)
#             normalization = np.sum(density) * (grids[1] - grids[0])
#             density = density / normalization
#             return density, grids
#
#         def gaussian(x, x0, sigma_squared):
#             return np.exp(-(x0 - x) ** 2 /
#                           (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)
#
#         ax = get_esd_plot(density_eigen, density_weight)
#
#         ax.set_title(f'Max Eigen Value: {top_eigenvalues[0]:.2f}')
#
#         plt.tight_layout()
#         plt.savefig(save_path_for_hessian)
#         plt.close()
#
#         logging.info(f'Save to {save_path_for_hessian}')
#
#     return top_eigenvalues


class Bezier(Module):
    def __init__(self, num_bends):
        super(Bezier, self).__init__()
        self.register_buffer(
            'binom',
            torch.Tensor(binom(num_bends - 1, np.arange(num_bends), dtype=np.float32))
        )
        self.register_buffer('range', torch.arange(0, float(num_bends)))
        self.register_buffer('rev_range', torch.arange(float(num_bends - 1), -1, -1))

    def forward(self, t):
        return self.binom * \
               torch.pow(t, self.range) * \
               torch.pow((1.0 - t), self.rev_range)


class PolyChain(Module):
    def __init__(self, num_bends):
        super(PolyChain, self).__init__()
        self.num_bends = num_bends
        self.register_buffer('range', torch.arange(0, float(num_bends)))

    def forward(self, t):
        t_n = t * (self.num_bends - 1)
        return torch.max(self.range.new([0.0]), 1.0 - torch.abs(t_n - self.range))


class MCR_Trainer(BackdoorModelTrainer):

    def __init__(self, model, curve):
        super().__init__(model)
        self.cruve = curve

    def one_forward_backward(self, x, labels, device, verbose=0):
        self.model.train()
        self.model.to(device, non_blocking=self.non_blocking)

        x, labels = x.to(device, non_blocking=self.non_blocking), labels.to(device, non_blocking=self.non_blocking)

        with torch.cuda.amp.autocast(enabled=self.amp):
            log_probs = self.model(x)
            loss = self.criterion(log_probs, labels.long())
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        batch_loss = loss.item()

        if verbose == 1:
            batch_predict = torch.max(log_probs, -1)[1].detach().clone().cpu()
            return batch_loss, batch_predict

        return batch_loss, None


def sampleModelFromCurve(
        model: torch.nn.Module,  # the model to be sampled, parameter will be replaced by sampled weights from curve
        curve_netCs: List[torch.nn.Module],  # models used for represents a curve
        curve_module: torch.nn.Module,  # module that used to generate weights which sum to 1. e.g. Bezier, PolyChain
        curve_t: float,  # which point on curve will be sampled?
        device,
) -> torch.nn.Module:
    # use given test_t to generate one model to do test
    model.eval()
    model.to(device)

    for inter_netC in curve_netCs:  # skip the start and end model
        inter_netC.eval()
        inter_netC.to(device)

    lookupDict_for_netCs = [dict(inter_netC.named_parameters()) for inter_netC in curve_netCs]
    inter_netC_coefs = curve_module(torch.tensor(curve_t))
    with torch.no_grad():
        for parameter_name, parameter in model.named_parameters():
            weighted_parameter_from_curve_netCs = 0
            for inter_netC_idx, lookupdict in enumerate(lookupDict_for_netCs):
                weighted_parameter_from_curve_netCs += lookupdict[parameter_name].data * inter_netC_coefs[
                    inter_netC_idx]
            parameter.copy_(
                weighted_parameter_from_curve_netCs
            )
    return model


class MCR(defense):

    def __init__(self):
        super(MCR).__init__()
        pass

    def set_args(self, parser):
        parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'],
                            help="dataloader pin_memory")
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])
        parser.add_argument('--frequency_save', type=int,
                            help=' frequency_save, 0 is never')
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'],
                            help=".to(), set the non_blocking = ?")
        parser.add_argument('--save_path', type=str)
        parser.add_argument("--dataset_path", type=str)

        parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny')
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--input_height", type=int)
        parser.add_argument("--input_width", type=int)
        parser.add_argument("--input_channel", type=int)

        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')

        parser.add_argument('--attack', type=str)
        parser.add_argument('--poison_rate', type=float)
        parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel')
        parser.add_argument('--target_label', type=int)
        parser.add_argument('--trigger_type', type=str,
                            help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

        parser.add_argument('--model', type=str, help='resnet18')
        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--result_file', type=str, help='the location of result')
        parser.add_argument("--train_curve_epochs", type=int)
        parser.add_argument('--yaml_path', type=str, default="./config/defense/mcr/config.yaml",
                            help='the path of yaml')
        parser.add_argument("--num_bends", type=int)
        parser.add_argument("--test_t", type=float)
        parser.add_argument("--curve", type=str)
        parser.add_argument("--ft_epochs", type=int)
        parser.add_argument("--ft_lr_scheduler", type=str)

        # set the parameter for the fp defense
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--acc_ratio', type=float, help='the tolerance ration of the clean accuracy')

        parser.add_argument('--test_curve_every', type=int, help="frequency of testing the models on curve")

        parser.add_argument("--load_other_model_path", type=str,
                            help="instead of finetune the given poisoned model, we load other model from this part")

        parser.add_argument("--use_clean_subset", type=lambda x: str(x) in ['True', 'true', '1'],
                            help="use bd poison dataset as data poison for path training and BN update; or, use clean subset instead")

        return parser

    def add_yaml_to_args(self, args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)
        defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = defaults

    def process_args(self, args):
        args.terminal_info = sys.argv
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        args.save_path = 'record/' + args.result_file
        defense_save_path = args.save_path + os.path.sep + "defense" + os.path.sep + "mcr"

        if os.path.exists(defense_save_path):
            shutil.rmtree(defense_save_path)
        os.makedirs(defense_save_path)

        args.defense_save_path = defense_save_path
        return args

    def prepare(self, args):

        ### set the logger
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()
        # file Handler
        fileHandler = logging.FileHandler(
            args.defense_save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.DEBUG)
        logger.addHandler(fileHandler)
        # consoleHandler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        logger.addHandler(consoleHandler)
        # overall logger level should <= min(handler) otherwise no log will be recorded.
        logger.setLevel(0)

        # disable other debug, since too many debug
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

        logging.info(pformat(args.__dict__))

        logging.debug("Only INFO or above level log will show in cmd. DEBUG level log only will show in log file.")

        # record the git infomation for debug (if available.)
        try:
            logging.debug(pformat(get_git_info()))
        except:
            logging.debug('Getting git info fails.')

        fix_random(args.random_seed)
        self.args = args

        '''
                load_dict = {
                        'model_name': load_file['model_name'],
                        'model': load_file['model'],
                        'clean_train': clean_train_dataset_with_transform,
                        'clean_test' : clean_test_dataset_with_transform,
                        'bd_train': bd_train_dataset_with_transform,
                        'bd_test': bd_test_dataset_with_transform,
                    }
                '''
        self.attack_result = load_attack_result(self.args.save_path + os.path.sep + 'attack_result.pt')

        netC = generate_cls_model(args.model, args.num_classes)
        netC.load_state_dict(self.attack_result['model'])
        netC.to(args.device)
        netC.eval()
        netC.requires_grad_(False)

        self.netC = netC

    def defense(self):

        netC = self.netC
        args = self.args
        attack_result = self.attack_result

        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

        # clean_train with subset
        clean_train_dataset_with_transform = attack_result['clean_train']
        clean_train_dataset_without_transform = clean_train_dataset_with_transform.wrapped_dataset
        clean_train_dataset_without_transform = prepro_cls_DatasetBD_v2(
            clean_train_dataset_without_transform
        )
        # logging.warning("No subset is done, ONLY for test!!!!!")
        ran_idx = choose_index(args, len(clean_train_dataset_without_transform))
        logging.info(f"get ran_idx for subset clean train dataset, (len={len(ran_idx)}), ran_idx:{ran_idx}")
        clean_train_dataset_without_transform.subset(
            choose_index(args, len(clean_train_dataset_without_transform))
        )
        log_index = args.defense_save_path + os.path.sep + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')
        clean_train_dataset_with_transform.wrapped_dataset = clean_train_dataset_without_transform
        clean_train_dataloader = torch.utils.data.DataLoader(clean_train_dataset_with_transform,
                                                             batch_size=args.batch_size,
                                                             num_workers=args.num_workers,
                                                             shuffle=True)

        clean_test_dataset_with_transform = attack_result['clean_test']
        data_clean_testset = clean_test_dataset_with_transform
        clean_test_dataloader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size,
                                                            num_workers=args.num_workers, drop_last=False,
                                                            shuffle=False,
                                                            pin_memory=args.pin_memory)

        bd_test_dataloader = torch.utils.data.DataLoader(attack_result['bd_test'], batch_size=args.batch_size,
                                                         num_workers=args.num_workers, drop_last=False, shuffle=False,
                                                         pin_memory=args.pin_memory)

        bd_train_dataset_with_transform = attack_result['bd_train']
        bd_train_dataset_without_transform = bd_train_dataset_with_transform.wrapped_dataset
        bd_train_dataloader = torch.utils.data.DataLoader(
            bd_train_dataset_with_transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=args.pin_memory,
        )

        where_poisoned = np.where(
            bd_train_dataset_without_transform.poison_indicator == 1
        )[0]
        logging.info(f"len of where_poisoned = {len(where_poisoned)}")
        bd_train_poisoned_part_wo_trans = deepcopy(bd_train_dataset_without_transform)
        bd_train_poisoned_part_wo_trans.subset(
            where_poisoned
        )
        bd_train_poisoned_part_w_trans = dataset_wrapper_with_transform(
            bd_train_poisoned_part_wo_trans,
            wrap_img_transform=clean_test_dataset_with_transform.wrap_img_transform,
        )
        bd_train_poisoned_part_dataloader = torch.utils.data.DataLoader(
            bd_train_poisoned_part_w_trans,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=args.pin_memory,
        )
        where_clean = np.where(
            bd_train_dataset_without_transform.poison_indicator == 0
        )[0]
        logging.info(f"len of where_clean = {len(where_clean)}")
        bd_train_clean_part_wo_trans = deepcopy(bd_train_dataset_without_transform)
        bd_train_clean_part_wo_trans.subset(
            where_clean
        )
        bd_train_clean_part_w_trans = dataset_wrapper_with_transform(
            bd_train_clean_part_wo_trans,
            wrap_img_transform=clean_test_dataset_with_transform.wrap_img_transform,
        )
        bd_train_clean_part_dataloader = torch.utils.data.DataLoader(
            bd_train_clean_part_w_trans,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=args.pin_memory,
        )

        # finetune netC with clean data
        ft_netC = deepcopy(netC)

        if ("load_other_model_path" not in args.__dict__) or (args.load_other_model_path is None):
            ft_netC.train()
            ft_netC.requires_grad_()

            criterion = nn.CrossEntropyLoss()
            ft_args = deepcopy(self.args)
            ft_args.__dict__ = {
                k[3:]: v for k, v in self.args.__dict__.items() if 'ft_' in k
            }
            optimizer, scheduler = argparser_opt_scheduler(
                ft_netC,
                ft_args,
            )
            finetune_trainer = BackdoorModelTrainer(
                ft_netC
            )

            finetune_trainer.train_with_test_each_epoch_on_mix(
                clean_train_dataloader,
                clean_test_dataloader,
                bd_test_dataloader,
                args.ft_epochs,
                criterion,
                optimizer,
                scheduler,
                args.amp,
                torch.device(args.device),

                args.frequency_save,
                self.args.defense_save_path,
                "finetune",

                prefetch=False,
                prefetch_transform_attr_name="transform",
                non_blocking=args.non_blocking,
            )
        else:
            # load from load_other_model_path
            ft_netC.load_state_dict(torch.load(args.load_other_model_path, map_location="cpu")['model'])
            ft_netC.to(args.device)
            logging.warning(f"Load alternative model from {args.load_other_model_path}!!!!")

        ft_netC.eval()
        ft_netC.requires_grad_()

        # train the curve
        logging.warning(
            "To align the training setting, we change the scheduler. If you want to change it back you can set it as below manually")
        '''
        def learning_rate_schedule(base_lr, epoch, total_epochs):
            alpha = epoch / total_epochs
            if alpha <= 0.5:
                factor = 1.0
            elif alpha <= 0.9:
                factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
            else:
                factor = 0.01
            return factor * base_lr

        '''

        if args.curve.lower().startswith("b"):
            curve = Bezier(args.num_bends)
        elif args.curve.lower().startswith("p"):
            curve = PolyChain(args.num_bends)
        else:
            raise SyntaxError("Unknown curve")

        for parameter in netC.parameters():
            parameter.requires_grad_(True)

        def model_mix(netC, weight1, ft_netC, weight2, model_mix_init):
            coefs = [weight1, weight2]
            lookupDict_for_netCs = [dict(netC.named_parameters()), dict(ft_netC.named_parameters())]
            for parameter_name, parameter in model_mix_init.named_parameters():
                weighted_parameter_from_curve_netCs = 0
                for inter_netC_idx, lookupdict in enumerate(lookupDict_for_netCs):
                    weighted_parameter_from_curve_netCs += lookupdict[parameter_name].data * coefs[
                        inter_netC_idx]
                parameter.data.copy_(
                    weighted_parameter_from_curve_netCs
                )
            return model_mix_init

        '''
        class simpleWeightNet(torch.nn.Module):
            def __init__(self, weight_value = None):
                super().__init__()
                self.weight_value = weight_value
                self.linear = torch.nn.Linear(5, 5)
                if self.weight_value is not None:
                    self.linear.weight.data = torch.tensor(weight_value).float()
                    self.linear.bias.data =  torch.tensor(weight_value).float()
                else:
                    print(
                        {"self.linear.weight": self.linear.weight,
                         "self.linear.bias": self.linear.bias, }
                    )
            def forward(self, x):
                return self.linear.weight, self.linear.bias, x


        a = model_mix(
                simpleWeightNet(1),
                0.5,
                simpleWeightNet(0.3),
                7,
               simpleWeightNet(),
        )

        print(a(1))
        print(a.linear.weight, a.linear.bias)
        '''

        '''
        3 point -> 1/2 + 1/2 (1 intermediate point)
        4 point -> 1/3 + 1/3 + 1/3 (2 intermediate point)
        '''

        def getWeightForIntermediatePoints(point_number, nth_point):
            one_weight_part = 1 / (point_number - 1)
            return (nth_point) * one_weight_part, 1 - ((nth_point) * one_weight_part)

        '''getWeightForIntermediatePoints(4, 1)
        (0.3333333333333333, 0.6666666666666667)
        getWeightForIntermediatePoints(4, 2)
        (0.6666666666666666, 0.33333333333333337)'''

        curve_netCs = [
                          deepcopy(netC)
                      ] * (args.num_bends - 2)  # init the intermediate models on curve

        # do model mix without modify the original model
        for intermediate_curve_netC_idx, intermediate_curve_netC in enumerate(curve_netCs):
            intermediate_curve_netC_idx += 1
            weight_left, weight_right = getWeightForIntermediatePoints(len(curve_netCs) + 2,
                                                                       intermediate_curve_netC_idx)
            curve_netCs[intermediate_curve_netC_idx - 1] = model_mix(netC, weight_left, ft_netC, weight_right,
                                                                     intermediate_curve_netC)

        curve_netCs_optimizers = []
        curve_netCs_schedulers = []
        for intermediate_curve_netC in curve_netCs:
            for parameter in netC.parameters():
                parameter.requires_grad_(True)
            intermediate_curve_netC_opt, intermediate_curve_netC_scheduler = argparser_opt_scheduler(
                intermediate_curve_netC,
                self.args,
            )
            curve_netCs_optimizers.append(intermediate_curve_netC_opt)
            curve_netCs_schedulers.append(intermediate_curve_netC_scheduler)

        curve_netCs = [netC] + curve_netCs + [ft_netC]  # add the start and end model
        self.curve_netCs = curve_netCs

        criterion = nn.CrossEntropyLoss()

        # just for aggregation
        new_netC_for_train_curve_aggregation = generate_cls_model(args.model, args.num_classes)
        new_netC_optimizer, new_netC_scheduler = argparser_opt_scheduler(
            new_netC_for_train_curve_aggregation,
            self.args,
        )

        logging.info(
            f"Before start training, just like the original paper, test for clean test error difference. see if two model have difference in sample classified wrongly")
        m1_metrics, m1_predicts, m1_targets = given_dataloader_test(
            model=netC,
            test_dataloader=clean_test_dataloader,
            criterion=criterion,
            non_blocking=True,
            device=self.device,
            verbose=1,
        )

        logging.info(f"m1_metric={m1_metrics}")

        m1_wrong = (m1_predicts != m1_targets).cpu().numpy()

        m2_metrics, m2_predicts, m2_targets = given_dataloader_test(
            model=ft_netC,
            test_dataloader=clean_test_dataloader,
            criterion=criterion,
            non_blocking=True,
            device=self.device,
            verbose=1,
        )

        logging.info(f"m2_metric={m2_metrics}")

        m2_wrong = (m2_predicts != m2_targets).cpu().numpy()

        # both m1, m2 wrong
        m1_m2_wrong = m1_wrong * m2_wrong

        m1_wrong_only = m1_wrong * (m1_m2_wrong != 1)
        m2_wrong_only = m2_wrong * (m1_m2_wrong != 1)

        logging.info(
            f"m1_wrong num = {np.sum(m1_wrong)}, m2_wrong num = {np.sum(m2_wrong)}, m1m2wrong = {np.sum(m1_m2_wrong)}, m1_wrong only = {np.sum(m1_wrong_only)}, m2_wrong only = {np.sum(m2_wrong_only)}"
        )

        if isinstance(args.test_t, float):
            test_t_list = [args.test_t]
        elif isinstance(args.test_t, list):
            test_t_list = args.test_t
        else:
            test_t_list = np.arange(0, 1, 0.3)

        logging.warning("We use the following test_t_list: {}".format(test_t_list))

        curve_record_dict = {}  # for different test_t value used.

        if "use_clean_subset" in args.__dict__ and args.use_clean_subset == True:
            dataloader_given = clean_train_dataloader
            logging.warning(
                f"Use clean_train_dataloader to train curve_netCs, data sample num = {len(clean_train_dataloader.dataset)}")
        else:
            dataloader_given = bd_train_dataloader
            logging.warning(
                f"Use bd_train_dataloader to train curve_netCs, data sample num = {len(bd_train_dataloader.dataset)}")

        for test_t in test_t_list:
            curve_record_dict[test_t] = {}
            # curve_record_dict[test_t]["clean_top0_eigenvalue_list"] = []
            # curve_record_dict[test_t]["bd_top0_eigenvalue_list"] = []
            curve_record_dict[test_t]["clean_test_loss_list"] = []
            curve_record_dict[test_t]["bd_test_loss_list"] = []
            curve_record_dict[test_t]["test_acc_list"] = []
            curve_record_dict[test_t]["test_asr_list"] = []
            curve_record_dict[test_t]["test_ra_list"] = []
            curve_record_dict[test_t]["train_loss_list"] = []
            curve_record_dict[test_t]["agg"] = Metric_Aggregator()
            curve_record_dict[test_t]["bd_train_clean_part_test_loss_avg_over_batch_list"] = []
            curve_record_dict[test_t]["bd_train_clean_part_acc_list"] = []
            curve_record_dict[test_t]["bd_train_poisoned_part_loss_avg_over_batch_list"] = []
            curve_record_dict[test_t]["bd_train_poisoned_part_asr_list"] = []
            curve_record_dict[test_t]["bd_train_poisoned_part_ra_list"] = []
            # curve_record_dict[test_t]["clean_part_generalization_gap_list"] = []
            # curve_record_dict[test_t]["poison_part_generalization_gap_list"] = []

        # os.makedirs(
        #     os.path.join(args.defense_save_path, "hessian_plot"),
        #     exist_ok=True,
        # )

        for epoch_idx in range(args.train_curve_epochs):

            new_netC_for_train_curve_aggregation, curve, new_netC_optimizer, new_netC_scheduler, curve_netCs, curve_netCs_optimizers, \
            curve_netCs_schedulers, one_epoch_train_loss = self.train_curve_one_epoch(
                args, new_netC_for_train_curve_aggregation, curve, new_netC_optimizer, new_netC_scheduler, curve_netCs,
                curve_netCs_optimizers,
                curve_netCs_schedulers, criterion, dataloader_given, self.device,
            )

            # # use given test_t to generate one model to do test
            # new_netC_for_train_curve_aggregation.eval()
            # new_netC_for_train_curve_aggregation.to(self.device)
            #
            # for inter_netC in curve_netCs:  # skip the start and end model
            #     inter_netC.eval()
            #     inter_netC.to(self.device)
            #
            # lookupDict_for_netCs = [dict(inter_netC.named_parameters()) for inter_netC in curve_netCs]
            # inter_netC_coefs = curve(torch.tensor(args.test_t))
            # with torch.no_grad():
            #     for parameter_name, parameter in new_netC_for_train_curve_aggregation.named_parameters():
            #         weighted_parameter_from_curve_netCs = 0
            #         for inter_netC_idx, lookupdict in enumerate(lookupDict_for_netCs):
            #             weighted_parameter_from_curve_netCs += lookupdict[parameter_name].data * inter_netC_coefs[
            #                 inter_netC_idx]
            #         parameter.copy_(
            #             weighted_parameter_from_curve_netCs
            #         )

            if epoch_idx % args.test_curve_every != args.test_curve_every - 1:
                continue

            logging.info("Epoch {} is finished, now test the model on clean and bd test set".format(epoch_idx))
            for test_t in test_t_list:
                logging.info("Now test the model on test_t = {}".format(test_t))

                # NOTE THAT THEY ARE ALL THE SAME !!!!!!
                curve_record_dict[test_t]["train_loss_list"].append(one_epoch_train_loss)

                new_netC_for_train_curve_aggregation = sampleModelFromCurve(
                    new_netC_for_train_curve_aggregation,
                    curve_netCs,
                    curve,
                    test_t,
                    self.device,
                )

                # find the first batchnorm layer in model's named_modules
                # first_BN = None
                # for name, module in new_netC_for_train_curve_aggregation.named_modules():
                #     if isinstance(module, torch.nn.BatchNorm2d):
                #         first_BN = module
                #         break
                # if first_BN is not None:
                #     logging.info(f"Before go through train dataset, first_BN.running_mean = {first_BN.running_mean}")
                #     logging.info(f"Before go through train dataset, first_BN.running_var = {first_BN.running_var}")

                new_netC_for_train_curve_aggregation.train()
                with torch.no_grad():
                    for batch_idx, (x, _, *additional_info) in enumerate(dataloader_given):
                        x = x.to(self.device, non_blocking=args.non_blocking)
                        new_netC_for_train_curve_aggregation(x)

                # first_BN = None
                # for name, module in new_netC_for_train_curve_aggregation.named_modules():
                #     if isinstance(module, torch.nn.BatchNorm2d):
                #         first_BN = module
                #         break
                # if first_BN is not None:
                #     logging.info(f"After go through train dataset, first_BN.running_mean = {first_BN.running_mean}")
                #     logging.info(f"After go through train dataset, first_BN.running_var = {first_BN.running_var}")
                new_netC_for_train_curve_aggregation.eval()

                bd_train_clean_part_metrics, \
                bd_train_clean_part_test_epoch_predict_list, \
                bd_train_clean_part_test_epoch_label_list, \
                    = given_dataloader_test(
                    model=new_netC_for_train_curve_aggregation,
                    test_dataloader=bd_train_clean_part_dataloader,
                    criterion=criterion,
                    non_blocking=args.non_blocking,
                    device=self.device,
                    verbose=1,
                )

                bd_train_clean_part_test_loss_avg_over_batch = bd_train_clean_part_metrics["test_loss_avg_over_batch"]
                bd_train_clean_part_acc = bd_train_clean_part_metrics["test_acc"]

                curve_record_dict[test_t]["bd_train_clean_part_test_loss_avg_over_batch_list"].append(
                    bd_train_clean_part_test_loss_avg_over_batch)
                curve_record_dict[test_t]["bd_train_clean_part_acc_list"].append(bd_train_clean_part_acc)

                bd_train_poisoned_part_metrics, \
                bd_train_poisoned_part_epoch_predict_list, \
                bd_train_poisoned_part_epoch_label_list, \
                bd_train_poisoned_part_epoch_original_index_list, \
                bd_train_poisoned_part_epoch_poison_indicator_list, \
                bd_train_poisoned_part_epoch_original_targets_list = test_given_dataloader_on_mix(
                    model=new_netC_for_train_curve_aggregation,
                    test_dataloader=bd_train_poisoned_part_dataloader,
                    criterion=criterion,
                    non_blocking=args.non_blocking,
                    device=self.device,
                    verbose=1,
                )

                bd_train_poisoned_part_loss_avg_over_batch = bd_train_poisoned_part_metrics["test_loss_avg_over_batch"]
                bd_train_poisoned_part_asr = all_acc(bd_train_poisoned_part_epoch_predict_list,
                                                     bd_train_poisoned_part_epoch_label_list)
                bd_train_poisoned_part_ra = all_acc(bd_train_poisoned_part_epoch_predict_list,
                                                    bd_train_poisoned_part_epoch_original_targets_list)

                curve_record_dict[test_t]["bd_train_poisoned_part_loss_avg_over_batch_list"].append(
                    bd_train_poisoned_part_loss_avg_over_batch)
                curve_record_dict[test_t]["bd_train_poisoned_part_asr_list"].append(bd_train_poisoned_part_asr)
                curve_record_dict[test_t]["bd_train_poisoned_part_ra_list"].append(bd_train_poisoned_part_ra)

                clean_metrics, \
                clean_test_epoch_predict_list, \
                clean_test_epoch_label_list, \
                    = given_dataloader_test(
                    model=new_netC_for_train_curve_aggregation,
                    test_dataloader=clean_test_dataloader,
                    criterion=criterion,
                    non_blocking=args.non_blocking,
                    device=self.device,
                    verbose=1,
                )

                clean_test_loss_avg_over_batch = clean_metrics["test_loss_avg_over_batch"]
                test_acc = clean_metrics["test_acc"]

                curve_record_dict[test_t]["clean_test_loss_list"].append(clean_test_loss_avg_over_batch)
                curve_record_dict[test_t]["test_acc_list"].append(test_acc)

                bd_metrics, \
                bd_test_epoch_predict_list, \
                bd_test_epoch_label_list, \
                bd_test_epoch_original_index_list, \
                bd_test_epoch_poison_indicator_list, \
                bd_test_epoch_original_targets_list = test_given_dataloader_on_mix(
                    model=new_netC_for_train_curve_aggregation,
                    test_dataloader=bd_test_dataloader,
                    criterion=criterion,
                    non_blocking=args.non_blocking,
                    device=self.device,
                    verbose=1,
                )

                bd_test_loss_avg_over_batch = bd_metrics["test_loss_avg_over_batch"]
                test_asr = all_acc(bd_test_epoch_predict_list, bd_test_epoch_label_list)
                test_ra = all_acc(bd_test_epoch_predict_list, bd_test_epoch_original_targets_list)

                curve_record_dict[test_t]["bd_test_loss_list"].append(bd_test_loss_avg_over_batch)
                curve_record_dict[test_t]["test_asr_list"].append(test_asr)
                curve_record_dict[test_t]["test_ra_list"].append(test_ra)

                # clean_top_eigenvalues = plot_hessian_eigenvalues(
                #     model_visual=new_netC_for_train_curve_aggregation,
                #     data_loader=clean_test_dataloader,
                #     device=self.device,
                #     # save_path_for_hessian = os.path.join(args.defense_save_path, "hessian_plot", "clean_hessian_eigenvalues_{}_{}.png".format(epoch_idx, test_t)),
                # )
                # clean_top0_eigenvalue = clean_top_eigenvalues[0]
                #
                # curve_record_dict[test_t]["clean_top0_eigenvalue_list"].append(clean_top0_eigenvalue)
                #
                # bd_top_eigenvalues = plot_hessian_eigenvalues(
                #     model_visual=new_netC_for_train_curve_aggregation,
                #     data_loader=bd_test_dataloader,
                #     device=self.device,
                #     # save_path_for_hessian=os.path.join(args.defense_save_path, "hessian_plot", "bd_hessian_eigenvalues_{}_{}.png".format(epoch_idx, test_t)),
                # )
                # bd_top0_eigenvalue = bd_top_eigenvalues[0]
                #
                # curve_record_dict[test_t]["bd_top0_eigenvalue_list"].append(bd_top0_eigenvalue)
                #
                # curve_record_dict[test_t]["clean_part_generalization_gap_list"].append(
                #     bd_train_clean_part_acc - test_acc)
                # curve_record_dict[test_t]["poison_part_generalization_gap_list"].append(
                #     bd_train_poisoned_part_asr - test_asr)

                curve_record_dict[test_t]["agg"](
                    {
                        "epoch": epoch_idx,
                        "test_t": test_t,
                        "train_epoch_loss_avg_over_batch": one_epoch_train_loss,
                        "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                        "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                        "test_acc": test_acc,
                        "test_asr": test_asr,
                        "test_ra": test_ra,
                        # "clean_top0_eigenvalue": clean_top0_eigenvalue,
                        # "bd_top0_eigenvalue": bd_top0_eigenvalue,
                        "bd_train_clean_part_test_loss_avg_over_batch": bd_train_clean_part_test_loss_avg_over_batch,
                        "bd_train_clean_part_acc": bd_train_clean_part_acc,
                        "bd_train_poisoned_part_loss_avg_over_batch": bd_train_poisoned_part_loss_avg_over_batch,
                        "bd_train_poisoned_part_asr": bd_train_poisoned_part_asr,
                        "bd_train_poisoned_part_ra": bd_train_poisoned_part_ra,
                        # "clean_part_generalization_gap": bd_train_clean_part_acc - test_acc,
                        # "poison_part_generalization_gap": bd_train_poisoned_part_asr - test_asr,
                    }
                )

                # for bd train different part
                general_plot_for_epoch(
                    {
                        "bd_train_clean_part_test_loss_avg_over_batch": curve_record_dict[test_t][
                            "bd_train_clean_part_test_loss_avg_over_batch_list"],
                        "bd_train_poisoned_part_loss_avg_over_batch": curve_record_dict[test_t][
                            "bd_train_poisoned_part_loss_avg_over_batch_list"],
                    },
                    save_path=f"{args.defense_save_path}/t_{test_t}_bd_train_parts_loss.png",
                    ylabel="value",
                )

                general_plot_for_epoch(
                    {
                        "bd_train_clean_part_acc": curve_record_dict[test_t]["bd_train_clean_part_acc_list"],
                        "bd_train_poisoned_part_asr": curve_record_dict[test_t]["bd_train_poisoned_part_asr_list"],
                        "bd_train_poisoned_part_ra": curve_record_dict[test_t]["bd_train_poisoned_part_ra_list"],
                        # "clean_part_generalization_gap": curve_record_dict[test_t][
                        #     "clean_part_generalization_gap_list"],
                        # "poisoned_part_generalization_gap": curve_record_dict[test_t][
                        #     "poison_part_generalization_gap_list"],
                    },
                    save_path=f"{args.defense_save_path}/t_{test_t}_bd_train_parts_acc_like.png",
                    ylabel="value",
                )

                # for bd_test and clean_test
                plot_loss(
                    curve_record_dict[test_t]["train_loss_list"],
                    curve_record_dict[test_t]["clean_test_loss_list"],
                    curve_record_dict[test_t]["bd_test_loss_list"],
                    args.defense_save_path,
                    f"curve_test_loss_metric_plots_t_{test_t}",
                    args.test_curve_every,
                )

                plot_acc_like_metric(
                    [], [], [],
                    curve_record_dict[test_t]["test_acc_list"],
                    curve_record_dict[test_t]["test_asr_list"],
                    curve_record_dict[test_t]["test_ra_list"],
                    args.defense_save_path,
                    f"curve_test_acc_like_metric_plots_t_{test_t}",
                    args.test_curve_every,
                )

                # plot
                # fix test_t on the curve path, compare loss and eigenvalue along time

                # t = np.arange(len(curve_record_dict[test_t]["clean_top0_eigenvalue_list"]))
                # data1 = curve_record_dict[test_t]["clean_top0_eigenvalue_list"]
                # data2 = curve_record_dict[test_t]["clean_part_generalization_gap_list"]
                # data3 = curve_record_dict[test_t]["poison_part_generalization_gap_list"]
                # data4 = curve_record_dict[test_t]["bd_top0_eigenvalue_list"]
                #
                # fig, ax1 = plt.subplots()
                #
                # color = 'tab:red'
                # ax1.set_xlabel('epoch')
                # ax1.set_ylabel('clean_top0_eigenvalue_list', color=color)
                # ax1.plot(t, data1, color=color)
                # ax1.tick_params(axis='y', labelcolor=color)
                #
                # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                #
                # color = 'tab:blue'
                # ax2.set_ylabel('clean_part_generalization_gap', color=color)  # we already handled the x-label with ax1
                # ax2.plot(t, data2, color=color)
                # ax2.tick_params(axis='y', labelcolor=color)
                #
                # fig.tight_layout()  # otherwise the right y-label is slightly clipped
                # plt.savefig(f"{args.defense_save_path}/t_{test_t}_clean_top0_eigenvalue_list.png", )
                # plt.close()
                #
                # fig, ax1 = plt.subplots()
                #
                # color = 'tab:green'
                # ax1.set_xlabel('epoch')
                # ax1.set_ylabel('bd_part_generalization_gap', color=color)  # we already handled the x-label with ax1
                # ax1.plot(t, data3, color=color)
                # ax1.tick_params(axis='y', labelcolor=color)
                #
                # ax4 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                # color = 'tab:orange'
                # ax4.set_ylabel('bd_top0_eigenvalue_list', color=color)  # we already handled the x-label with ax1
                # ax4.plot(t, data4, color=color)
                # ax4.tick_params(axis='y', labelcolor=color)
                #
                # fig.tight_layout()  # otherwise the right y-label is slightly clipped
                # plt.savefig(f"{args.defense_save_path}/t_{test_t}_bd_top0_eigenvalue_list.png", )
                # plt.close()

                # general_plot_for_epoch(
                #     {
                #         # "train_loss_list": curve_record_dict[test_t]["train_loss_list"],
                #         "clean_test_loss_list": curve_record_dict[test_t]["clean_test_loss_list"],
                #         "bd_test_loss_list": curve_record_dict[test_t]["bd_test_loss_list"],
                #         "top0_eigenvalue_list":curve_record_dict[test_t]["top0_eigenvalue_list"],
                #     },
                #     save_path=f"{args.defense_save_path}/t_{test_t}_top0_eigenvalue_list.png",
                #     ylabel="value",
                # )

                curve_record_dict[test_t]["agg"].to_dataframe().to_csv(
                    f"{args.defense_save_path}/curve_train_df_t_{test_t}.csv")

            # # plot the clean_top0_eigenvalue_list and bd_top0_eigenvalue_list  for different test_t
            # same_epoch_clean_top0_eigenvalue_list = []
            # same_epoch_clean_part_generalization_gap_list = []
            # same_epoch_bd_part_generalization_gap_list = []
            # same_epoch_bd_top0_eigenvalue_list = []
            # for test_t in test_t_list:
            #     same_epoch_clean_top0_eigenvalue_list.append(
            #         curve_record_dict[test_t]["clean_top0_eigenvalue_list"][-1])
            #     same_epoch_clean_part_generalization_gap_list.append(
            #         curve_record_dict[test_t]["clean_part_generalization_gap_list"][-1]
            #     )
            #     same_epoch_bd_part_generalization_gap_list.append(
            #         curve_record_dict[test_t]["poison_part_generalization_gap_list"][-1]
            #     )
            #     same_epoch_bd_top0_eigenvalue_list.append(curve_record_dict[test_t]["bd_top0_eigenvalue_list"][-1])

            # t = np.arange(len(same_epoch_clean_top0_eigenvalue_list) + 1)[1:] / (
            #             len(same_epoch_clean_top0_eigenvalue_list) + 1)
            # data1 = same_epoch_clean_top0_eigenvalue_list
            # data2 = same_epoch_clean_part_generalization_gap_list
            # data3 = same_epoch_bd_part_generalization_gap_list
            # data4 = same_epoch_bd_top0_eigenvalue_list
            #
            # fig, ax1 = plt.subplots()
            #
            # color = 'tab:red'
            # ax1.set_xlabel('test_t on curve path')
            # ax1.set_ylabel('same_epoch_clean_top0_eigenvalue_list', color=color)
            # ax1.plot(t, data1, color=color)
            # ax1.tick_params(axis='y', labelcolor=color)
            #
            # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            #
            # color = 'tab:blue'
            # ax2.set_ylabel('same_epoch_clean_part_generalization_gap_list',
            #                color=color)  # we already handled the x-label with ax1
            # ax2.plot(t, data2, color=color)
            # ax2.tick_params(axis='y', labelcolor=color)
            #
            # fig.tight_layout()  # otherwise the right y-label is slightly clipped
            # plt.savefig(f"{args.defense_save_path}/epoch_{epoch_idx}_clean_path.png", )
            # plt.close()
            #
            # fig, ax1 = plt.subplots()
            #
            # color = 'tab:green'
            # ax1.set_xlabel('test_t on curve path')
            # ax1.set_ylabel('same_epoch_bd_part_generalization_gap_list',
            #                color=color)  # we already handled the x-label with ax1
            # ax1.plot(t, data3, color=color)
            # ax1.tick_params(axis='y', labelcolor=color)
            #
            # ax4 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            # color = 'tab:orange'
            # ax4.set_ylabel('same_epoch_bd_top0_eigenvalue_list', color=color)  # we already handled the x-label with ax1
            # ax4.plot(t, data4, color=color)
            # ax4.tick_params(axis='y', labelcolor=color)
            #
            # fig.tight_layout()  # otherwise the right y-label is slightly clipped
            # plt.savefig(f"{args.defense_save_path}/epoch_{epoch_idx}_bd_path.png", )
            # plt.close()

            for test_t in test_t_list:
                curve_record_dict[test_t]["agg"].summary().to_csv(
                    f"{args.defense_save_path}/curve_train_df_summary_t_{test_t}.csv")

        if 0.1 in test_t_list:
            final_t = 0.1
        else:
            logging.warning(f"0.1 is not in test_t_list, so we find the nearest value to 0.1 for final report result")
            def find_nearest(array, value):
                # thanks to https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
                # this function is from Mateen Ulhaq
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return array[idx]
            final_t = find_nearest(np.array(test_t_list), value=0.1)

        t = final_t
        test_acc = curve_record_dict[t]["test_acc_list"][-1]
        test_asr = curve_record_dict[t]["test_asr_list"][-1]
        test_ra = curve_record_dict[t]["test_ra_list"][-1]

        agg = Metric_Aggregator()
        agg({
                'test_acc': test_asr,
                'test_asr': test_acc,
                'test_ra': test_ra,
                't': t,
        })
        agg.to_dataframe().to_csv(f"{args.defense_save_path}/mcr_df_summary.csv")

        torch.save(
            {
                'model_name': args.model,
                'model': new_netC_for_train_curve_aggregation.cpu().state_dict(),
                'test_acc': test_asr,
                'test_asr': test_acc,
                'test_ra': test_ra,
                't': t,
            },
            f'{args.defense_save_path}/defense_result.pt'
        )

    def train_curve_one_epoch(self, args, netC, curve, netC_optimizer, netC_scheduler, curve_netCs,
                              curve_netCs_optimizers, curve_netCs_schedulers, criterion, clean_train_dataloader,
                              device):

        netC.train()
        netC.to(device)
        train_loss = 0
        correct = 0
        total = 0

        for inter_netC in curve_netCs[1:-1]:  # skip the start and end model
            inter_netC.train()
            inter_netC.requires_grad_()
            inter_netC.to(device)

        curve_netCs[0].eval()
        curve_netCs[0].requires_grad_()
        curve_netCs[0].to(device)

        curve_netCs[-1].eval()
        curve_netCs[-1].requires_grad_()
        curve_netCs[-1].to(device)

        batch_loss = []
        for batch_idx, (inputs, targets, *other) in enumerate(clean_train_dataloader):

            # copy parameters and do weighted sum, from curve_netCs to netC
            lookupDict_for_netCs = [dict(inter_netC.named_parameters()) for inter_netC in curve_netCs]
            inter_netC_coefs = curve(inputs.data.new(1).uniform_(0.0, 1.0))
            with torch.no_grad():
                for parameter_name, parameter in netC.named_parameters():
                    weighted_parameter_from_curve_netCs = 0
                    for inter_netC_idx, lookupdict in enumerate(lookupDict_for_netCs):
                        weighted_parameter_from_curve_netCs += lookupdict[parameter_name].data * inter_netC_coefs[
                            inter_netC_idx]
                    parameter.copy_(
                        weighted_parameter_from_curve_netCs
                    )

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = netC(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # send back grad from netC to curve_netCs
            for parameter_name, parameter in netC.named_parameters():
                for inter_netC_idx, lookupdict in enumerate(lookupDict_for_netCs):
                    lookupdict[parameter_name].grad = parameter.grad * inter_netC_coefs[
                        inter_netC_idx]

            # do step for all models
            netC_optimizer.step()
            for inter_netC_optimizer in curve_netCs_optimizers:
                inter_netC_optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(clean_train_dataloader), 'Loss: %.3f | train Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            batch_loss.append(loss.item())

        one_epoch_loss = sum(batch_loss) / len(batch_loss)

        # update the all models' scheduler
        for scheduler in (curve_netCs_schedulers + [netC_scheduler]):
            if scheduler:
                if args.lr_scheduler == 'ReduceLROnPlateau':
                    scheduler.step(one_epoch_loss)
                elif args.lr_scheduler == 'CosineAnnealingLR':
                    scheduler.step()

        return netC, curve, netC_optimizer, netC_scheduler, curve_netCs, curve_netCs_optimizers, curve_netCs_schedulers, one_epoch_loss


if __name__ == '__main__':
    mcr = MCR()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = mcr.set_args(parser)
    args = parser.parse_args()
    mcr.add_yaml_to_args(args)
    args = mcr.process_args(args)
    mcr.prepare(args)
    mcr.defense()
