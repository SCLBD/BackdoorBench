'''
Blind Backdoors in Deep Learning Models
this script is for blind attack
from https://github.com/ebagdasa/backdoors101

@inproceedings {bagdasaryan2020blind,
 author = {Eugene Bagdasaryan and Vitaly Shmatikov},
 title = {Blind Backdoors in Deep Learning Models},
 booktitle = {30th {USENIX} Security Symposium ({USENIX} Security 21)},
 year = {2021},
 isbn = {978-1-939133-24-3},
 pages = {1505--1521},
 url = {https://www.usenix.org/conference/usenixsecurity21/presentation/bagdasaryan},
 publisher = {{USENIX} Association},
 month = aug,
}

basic structure:

1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the attack img transform and label transform
4. set the backdoor attack data and backdoor test data
5. set the device, model, criterion, optimizer, training schedule.
6. use the designed blind_loss to train a poisoned model
7. save the attack result for defense

Original code file license is as the end of this script

Note that for fairness issue, we apply the same total training epochs as all other attack methods. But for Blind, it may not be the best choice.

'''
import os
import sys

sys.path = ["./"] + sys.path
import numpy as np

import logging
import torch
import argparse
import time
import random
from tqdm import tqdm
from shutil import copyfile
from typing import *
from collections import defaultdict
from dataclasses import asdict
from typing import Dict
from dataclasses import dataclass
from typing import List
from torch import optim, nn
from torch.nn import Module
from torchvision.transforms import transforms, functional

import torchvision.transforms as transforms
from typing import Union

from attack.badnet import BadNet
from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_label_trans_generate
from copy import deepcopy
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.save_load_attack import save_attack_result
from utils.trainer_cls import all_acc, given_dataloader_test, \
    plot_loss, plot_acc_like_metric, Metric_Aggregator, test_given_dataloader_on_mix
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from torch.utils.data.dataloader import DataLoader
from utils.aggregate_block.bd_attack_generate import general_compose
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()

ALL_TASKS = ['backdoor', 'normal', 'sentinet_evasion',  # 'spectral_evasion',
             'neural_cleanse', 'mask_norm', 'sums', 'neural_cleanse_part1']


class Params:

    def __init__(
            self,
            **kwargs,
    ):
        # Corresponds to the class module: tasks.mnist_task.MNISTTask
        # See other tasks in the task folder.
        self.task: str = 'MNIST'
        self.current_time: Optional[str] = None
        self.name: Optional[str] = None
        self.commit: Optional[float] = None
        self.random_seedOptional: Optional[int] = None

        # training params
        self.start_epoch: int = 1
        self.epochs: Optional[int] = None
        self.log_interval: int = 1000
        # model arch is usually defined by the task
        self.pretrained: bool = False
        self.resume_model: Optional[str] = None
        self.lr: Optional[float] = None
        self.decay: Optional[float] = None
        self.momentum: Optional[float] = None
        self.optimizer: Optional[str] = None
        self.scheduler: bool = False
        self.scheduler_milestonesOptional: [List[int]] = None
        # data
        self.data_path: str = '.data/'
        self.batch_size: int = 64
        self.test_batch_size: int = 100
        self.transform_train: bool = True
        "Do not apply transformations to the training images."
        self.max_batch_id: Optional[int] = None
        "For large datasets stop training earlier."
        self.input_shape = None
        "No need to set, updated by the Task class."

        # gradient shaping/DP params
        self.dp: Optional[bool] = None
        self.dp_clip: Optional[float] = None
        self.dp_sigma: Optional[float] = None

        # attack params
        self.backdoor: bool = False
        self.backdoor_label: int = 8
        self.poisoning_proportion: float = 1.0  # backdoors proportion in backdoor loss
        self.synthesizer: str = 'pattern'
        self.backdoor_dynamic_position: bool = False

        # losses to balance: `normal`, `backdoor`, `neural_cleanse`, `sentinet`,
        # `backdoor_multi`.
        self.loss_tasks: Optional[List[str]] = None

        self.loss_balance: str = 'MGDA'
        "loss_balancing: `fixed` or `MGDA`"

        self.loss_threshold: Optional[float] = None

        # approaches to balance losses with MGDA: `none`, `loss`,
        # `loss+`, `l2`
        self.mgda_normalize: Optional[str] = None
        self.fixed_scales: Optional[Dict[str, float]] = None

        # relabel images with poison_number
        self.poison_images: Optional[List[int]] = None
        self.poison_images_test: Optional[List[int]] = None
        # optimizations:
        self.alternating_attack: Optional[float] = None
        self.clip_batch: Optional[float] = None
        # Disable BatchNorm and Dropout
        self.switch_to_eval: Optional[float] = None

        # nc evasion
        self.nc_p_norm: int = 1
        # spectral evasion
        self.spectral_similarity: 'str' = 'norm'

        # logging
        self.report_train_loss: bool = True
        self.log: bool = False
        self.tb: bool = False
        self.save_model: Optional[bool] = None
        self.save_on_epochs: Optional[List[int]] = None
        self.save_scale_values: bool = False
        self.print_memory_consumption: bool = False
        self.save_timing: bool = False
        self.timing_data = None

        # Temporary storage for running values
        self.running_losses = None
        self.running_scales = None

        # FL params
        self.fl: bool = False
        self.fl_no_models: int = 100
        self.fl_local_epochs: int = 2
        self.fl_total_participants: int = 80000
        self.fl_eta: int = 1
        self.fl_sample_dirichlet: bool = False
        self.fl_dirichlet_alpha: Optional[float] = None
        self.fl_diff_privacy: bool = False
        self.fl_dp_clip: Optional[float] = None
        self.fl_dp_noise: Optional[float] = None
        # FL attack details. Set no adversaries to perform the attack:
        self.fl_number_of_adversaries: int = 0
        self.fl_single_epoch_attack: Optional[int] = None
        self.fl_weight_scale: int = 1

        self.__dict__.update(kwargs)

        # enable logging anyways when saving statistics
        if self.save_model or self.tb or self.save_timing or \
                self.print_memory_consumption:
            self.log = True

        if self.log:
            self.folder_path = f'saved_models/model_' \
                               f'{self.task}_{self.current_time}_{self.name}'

        self.running_losses = defaultdict(list)
        self.running_scales = defaultdict(list)
        self.timing_data = defaultdict(list)

        for t in self.loss_tasks:
            if t not in ALL_TASKS:
                raise ValueError(f'Task {t} is not part of the supported '
                                 f'tasks: {ALL_TASKS}.')

    def to_dict(self):
        return asdict(self)


class Metric:
    name: str
    train: bool
    plottable: bool = True
    running_metric = None
    main_metric_name = None

    def __init__(self, name, train=False):
        self.train = train
        self.name = name

        self.running_metric = defaultdict(list)

    def __repr__(self):
        metrics = self.get_value()
        text = [f'{key}: {val:.2f}' for key, val in metrics.items()]
        return f'{self.name}: ' + ','.join(text)

    def compute_metric(self, outputs, labels) -> Dict[str, Any]:
        raise NotImplemented

    def accumulate_on_batch(self, outputs=None, labels=None):
        current_metrics = self.compute_metric(outputs, labels)
        for key, value in current_metrics.items():
            self.running_metric[key].append(value)

    def get_value(self) -> Dict[str, np.ndarray]:
        metrics = dict()
        for key, value in self.running_metric.items():
            metrics[key] = np.mean(value)

        return metrics

    def get_main_metric_value(self):
        if not self.main_metric_name:
            raise ValueError(f'For metric {self.name} define '
                             f'attribute main_metric_name.')
        metrics = self.get_value()
        return metrics[self.main_metric_name]

    def reset_metric(self):
        self.running_metric = defaultdict(list)

    def plot(self, tb_writer, step, tb_prefix=''):
        if tb_writer is not None and self.plottable:
            metrics = self.get_value()
            for key, value in metrics.items():
                tb_writer.add_scalar(tag=f'{tb_prefix}/{self.name}_{key}',
                                     scalar_value=value,
                                     global_step=step)
            tb_writer.flush()
        else:
            return False


class AccuracyMetric(Metric):

    def __init__(self, top_k=(1,)):
        self.top_k = top_k
        self.main_metric_name = 'Top-1'
        super().__init__(name='Accuracy', train=False)

    def compute_metric(self, outputs: torch.Tensor,
                       labels: torch.Tensor):
        """Computes the precision@k for the specified values of k"""
        max_k = max(self.top_k)
        batch_size = labels.shape[0]

        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = dict()
        for k in self.top_k:
            correct_k = correct[:k].view(-1).float().sum(0)
            res[f'Top-{k}'] = (correct_k.mul_(100.0 / batch_size)).item()
        return res


class TestLossMetric(Metric):

    def __init__(self, criterion, train=False):
        self.criterion = criterion
        self.main_metric_name = 'value'
        super().__init__(name='Loss', train=False)

    def compute_metric(self, outputs: torch.Tensor,
                       labels: torch.Tensor, top_k=(1,)):
        """Computes the precision@k for the specified values of k"""
        loss = self.criterion(outputs, labels)
        return {'value': loss.mean().item()}


@dataclass
class Batch:
    batch_id: int
    inputs: torch.Tensor
    labels: torch.Tensor

    # For PIPA experiment we use this field to store identity label.
    aux: torch.Tensor = None

    def __post_init__(self):
        self.batch_size = self.inputs.shape[0]

    def to(self, device):
        inputs = self.inputs.to(device)
        labels = self.labels.to(device)
        if self.aux is not None:
            aux = self.aux.to(device)
        else:
            aux = None
        return Batch(self.batch_id, inputs, labels, aux)

    def clone(self):
        inputs = self.inputs.clone()
        labels = self.labels.clone()
        if self.aux is not None:
            aux = self.aux.clone()
        else:
            aux = None
        return Batch(self.batch_id, inputs, labels, aux)

    def clip(self, batch_size):
        if batch_size is None:
            return self

        inputs = self.inputs[:batch_size]
        labels = self.labels[:batch_size]

        if self.aux is None:
            aux = None
        else:
            aux = self.aux[:batch_size]

        return Batch(self.batch_id, inputs, labels, aux)


class Task:
    params: Params = None

    train_dataset = None
    test_dataset = None
    train_loader = None
    test_loader = None
    classes = None

    model: Module = None
    optimizer: optim.Optimizer = None
    criterion: Module = None
    metrics: List[Metric] = None

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    "Generic normalization for input data."
    input_shape: torch.Size = None

    def __init__(self, params: Params):
        self.params = params
        self.init_task()

    def init_task(self):
        self.load_data()
        self.model = self.build_model()
        self.resume_model()
        self.model = self.model.to(self.params.device)

        self.optimizer, self.scheduler = argparser_opt_scheduler(self.model, self.params)
        self.criterion = self.make_criterion()
        self.metrics = [AccuracyMetric(), TestLossMetric(self.criterion)]
        self.set_input_shape()

    def load_data(self) -> None:
        raise NotImplemented

    def build_model(self) -> Module:
        raise NotImplemented

    def make_criterion(self) -> Module:
        """Initialize with Cross Entropy by default.

        We use reduction `none` to support gradient shaping defense.
        :return:
        """
        return nn.CrossEntropyLoss(reduction='none')

    def resume_model(self):
        if self.params.resume_model:
            logging.info(f'Resuming training from- {self.params.resume_model}')
            loaded_params = torch.load(f"saved_models/"
                                       f"{self.params.resume_model}",
                                       map_location=torch.device('cpu'))
            self.model.load_state_dict(loaded_params['state_dict'])
            self.params.start_epoch = loaded_params['epoch']
            self.params.lr = loaded_params.get('lr', self.params.lr)

            logging.warning(f"Loaded parameters from- saved model: LR is"
                            f" {self.params.lr} and current epoch is"
                            f" {self.params.start_epoch}")

    def set_input_shape(self):
        inp = self.train_dataset[0][0]
        self.params.input_shape = inp.shape

    def get_batch(self, batch_id, data) -> Batch:
        """Process data into a batch.

        Specific for different datasets and data loaders this method unifies
        the output by returning the object of class Batch.
        :param batch_id: id of the batch
        :param data: object returned by the Loader.
        :return:
        """
        inputs, labels = data
        batch = Batch(batch_id, inputs, labels)
        return batch.to(self.params.device)

    def accumulate_metrics(self, outputs, labels):
        for metric in self.metrics:
            metric.accumulate_on_batch(outputs, labels)

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_metric()

    def report_metrics(self, step, prefix='',
                       tb_writer=None, tb_prefix='Metric/'):
        metric_text = []
        for metric in self.metrics:
            metric_text.append(str(metric))
            metric.plot(tb_writer, step, tb_prefix=tb_prefix)
        logging.warning(f'{prefix} {step:4d}. {" | ".join(metric_text)}')

        return self.metrics[0].get_main_metric_value()

    @staticmethod
    def get_batch_accuracy(outputs, labels, top_k=(1,)):
        """Computes the precision@k for the specified values of k"""
        max_k = max(top_k)
        batch_size = labels.size(0)

        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        if len(res) == 1:
            res = res[0]
        return res


class Synthesizer:
    params: Params
    task: Task

    def __init__(self, task: Task):
        self.task = task
        self.params = task.params

    def make_backdoor_batch(self, batch: Batch, test=False, attack=True) -> Batch:

        # Don't attack if only normal loss task.
        if (not attack) or (self.params.loss_tasks == ['normal'] and not test):
            return batch

        if test:
            attack_portion = batch.batch_size
        else:
            attack_portion = round(
                batch.batch_size * self.params.poisoning_proportion)

        backdoored_batch = batch.clone()
        self.apply_backdoor(backdoored_batch, attack_portion)

        return backdoored_batch

    def apply_backdoor(self, batch, attack_portion):
        """
        Modifies only a portion of the batch (represents batch poisoning).

        :param batch:
        :return:
        """
        self.synthesize_inputs(batch=batch, attack_portion=attack_portion)
        self.synthesize_labels(batch=batch, attack_portion=attack_portion)

        return

    def synthesize_inputs(self, batch, attack_portion=None):
        raise NotImplemented

    def synthesize_labels(self, batch, attack_portion=None):
        raise NotImplemented


def record_time(params: Params, t=None, name=None):
    if t and name and params.save_timing == name or params.save_timing is True:
        torch.cuda.synchronize()
        params.timing_data[name].append(round(1000 * (time.perf_counter() - t)))


def compute_normal_loss(params, model, criterion, inputs,
                        labels, grads):
    t = time.perf_counter()
    outputs = model(inputs)
    record_time(params, t, 'forward')
    loss = criterion(outputs, labels)

    if not params.dp:
        loss = loss.mean()

    if grads:
        t = time.perf_counter()
        grads = list(torch.autograd.grad(loss.mean(),
                                         [x for x in model.parameters() if
                                          x.requires_grad],
                                         retain_graph=True))
        record_time(params, t, 'backward')

    return loss, grads


def get_grads(params, model, loss):
    t = time.perf_counter()
    grads = list(torch.autograd.grad(loss.mean(),
                                     [x for x in model.parameters() if
                                      x.requires_grad],
                                     retain_graph=True))
    record_time(params, t, 'backward')

    return grads


def th(vector):
    return torch.tanh(vector) / 2 + 0.5


def norm_loss(params, model, grads=None):
    if params.nc_p_norm == 1:
        norm = torch.sum(th(model.mask))
    elif params.nc_p_norm == 2:
        norm = torch.norm(th(model.mask))
    else:
        raise ValueError('Not support mask norm.')

    if grads:
        grads = get_grads(params, model, norm)
        model.zero_grad()

    return norm, grads


def compute_backdoor_loss(params, model, criterion, inputs_back,
                          labels_back, grads=None):
    t = time.perf_counter()
    outputs = model(inputs_back)
    record_time(params, t, 'forward')
    loss = criterion(outputs, labels_back)

    if params.task == 'Pipa':
        loss[labels_back == 0] *= 0.001
        if labels_back.sum().item() == 0.0:
            loss[:] = 0.0
    if not params.dp:
        loss = loss.mean()

    if grads:
        grads = get_grads(params, model, loss)

    return loss, grads


def compute_all_losses_and_grads(loss_tasks, attack, model, criterion,
                                 batch, batch_back,
                                 compute_grad=None):
    grads = {}
    loss_values = {}
    for t in loss_tasks:
        # if compute_grad:
        #     model.zero_grad()
        if t == 'normal':
            loss_values[t], grads[t] = compute_normal_loss(attack.params,
                                                           model,
                                                           criterion,
                                                           batch.inputs,
                                                           batch.labels,
                                                           grads=compute_grad)
        elif t == 'backdoor':
            loss_values[t], grads[t] = compute_backdoor_loss(attack.params,
                                                             model,
                                                             criterion,
                                                             batch_back.inputs,
                                                             batch_back.labels,
                                                             grads=compute_grad)

        elif t == 'mask_norm':
            loss_values[t], grads[t] = norm_loss(attack.params, attack.nc_model,
                                                 grads=compute_grad)
        elif t == 'neural_cleanse_part1':
            loss_values[t], grads[t] = compute_normal_loss(attack.params,
                                                           model,
                                                           criterion,
                                                           batch.inputs,
                                                           batch_back.labels,
                                                           grads=compute_grad,
                                                           )

    return loss_values, grads


class Model(nn.Module):
    """
    Base class for models with added support for GradCam activation map
    and a SentiNet defense. The GradCam design is taken from:
https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    If you are not planning to utilize SentiNet defense just import any model
    you like for your tasks.
    """

    def __init__(self):
        super().__init__()
        self.gradient = None

    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def switch_grads(self, enable=True):
        for i, n in self.named_parameters():
            n.requires_grad_(enable)

    def features(self, x):
        """
        Get latent representation, eg logit layer.
        :param x:
        :return:
        """
        raise NotImplemented

    def forward(self, x, latent=False):
        raise NotImplemented


class Attack:
    params: Params
    synthesizer: Synthesizer
    nc_model: Model
    nc_optim: torch.optim.Optimizer
    loss_hist = list()

    # fixed_model: Model

    def __init__(self, params, synthesizer):
        self.params = params
        self.synthesizer = synthesizer

    def compute_blind_loss(self, model, criterion, batch, attack):
        """

        :param model:
        :param criterion:
        :param batch:
        :param attack: Do not attack at all. Ignore all the parameters
        :return:
        """
        batch = batch.clip(self.params.clip_batch)
        loss_tasks = self.params.loss_tasks.copy() if attack else ['normal']
        batch_back = self.synthesizer.make_backdoor_batch(batch, attack=attack)
        scale = dict()

        if self.params.loss_threshold and (np.mean(self.loss_hist) >= self.params.loss_threshold
                                           or len(self.loss_hist) < self.params.batch_history_len):
            loss_tasks = ['normal']

        if len(loss_tasks) == 1:
            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, compute_grad=False
            )
        elif self.params.loss_balance == 'MGDA':

            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, compute_grad=True)
            if len(loss_tasks) > 1:
                scale = MGDASolver.get_scales(grads, loss_values,
                                              self.params.mgda_normalize,
                                              loss_tasks)
        elif self.params.loss_balance == 'fixed':
            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, compute_grad=False)

            for t in loss_tasks:
                scale[t] = self.params.fixed_scales[t]
        else:
            raise ValueError(f'Please choose between `MGDA` and `fixed`.')

        if len(loss_tasks) == 1:
            scale = {loss_tasks[0]: 1.0}
        self.loss_hist.append(loss_values['normal'].item())
        self.loss_hist = self.loss_hist[-1000:]
        blind_loss = self.scale_losses(loss_tasks, loss_values, scale)

        return blind_loss

    def scale_losses(self, loss_tasks, loss_values, scale):
        blind_loss = 0
        for it, t in enumerate(loss_tasks):
            self.params.running_losses[t].append(loss_values[t].item())
            self.params.running_scales[t].append(scale[t])
            if it == 0:
                blind_loss = scale[t] * loss_values[t]
            else:
                blind_loss += scale[t] * loss_values[t]
        self.params.running_losses['total'].append(blind_loss.item())
        return blind_loss


# Credits to Ozan Sener
# https://github.com/intel-isl/MultiObjectiveOptimization
class MGDASolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs: list, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0
        for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        sol = 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.dot(vecs[i][k].view(-1),
                                                 vecs[j][k].view(-1)).detach()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.dot(vecs[i][k].view(-1),
                                                 vecs[i][k].view(-1)).detach()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.dot(vecs[j][k].view(-1),
                                                 vecs[j][k].view(-1)).detach()
                c, d = MGDASolver._min_norm_element_from2(dps[(i, i)],
                                                          dps[(i, j)],
                                                          dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    @staticmethod
    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        skippers = np.sum(tm1 < 1e-7) + np.sum(tm2 < 1e-7)
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MGDASolver._projection2simplex(next_point)
        return next_point

    @staticmethod
    def find_min_norm_element(vecs: list):
        """
        Given a list of vectors (vecs), this method finds the minimum norm
        element in the convex hull as min |u|_2 st. u = \sum c_i vecs[i]
        and \sum c_i = 1. It is quite geometric, and the main idea is the
        fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution
        lies in (0, d_{i,j})Hence, we find the best 2-task solution , and
        then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MGDASolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MGDASolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MGDASolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MGDASolver._min_norm_element_from2(v1v1.item(),
                                                        v1v2.item(),
                                                        v2v2.item())
            # try:
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            # except AttributeError:
            #     logging.debug(sol_vec)
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MGDASolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    @staticmethod
    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm
        element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if
        d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies
        in (0, d_{i,j})Hence, we find the best 2-task solution, and then
        run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MGDASolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MGDASolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MGDASolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MGDASolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    @classmethod
    def get_scales(cls, grads, losses, normalization_type, tasks):
        scale = {}
        gn = gradient_normalizers(grads, losses, normalization_type)
        for t in tasks:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / (gn[t] + 1e-5)
        sol, min_norm = cls.find_min_norm_element([grads[t] for t in tasks])
        for zi, t in enumerate(tasks):
            scale[t] = float(sol[zi])

        return scale


def create_table(params: dict):
    data = "| name | value | \n |-----|-----|"

    for key, value in params.items():
        data += '\n' + f"| {key} | {value} |"

    return data


class PatternSynthesizer(Synthesizer):
    pattern_tensor: torch.Tensor = torch.tensor([
        [1., 0., 1.],
        [-10., 1., -10.],
        [-10., -10., 0.],
        [-10., 1., -10.],
        [1., 0., 1.]
    ])
    "Just some random 2D pattern."

    x_top = 3
    "X coordinate to put the backdoor into."
    y_top = 23
    "Y coordinate to put the backdoor into."

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    resize_scale = (5, 10)
    "If the pattern is dynamically placed, resize the pattern."

    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."

    def __init__(self, task: Task):
        super().__init__(task)
        self.make_pattern(self.pattern_tensor, self.x_top, self.y_top)

    def make_pattern(self, pattern_tensor, x_top, y_top):
        full_image = torch.zeros(self.params.input_shape)
        full_image.fill_(self.mask_value)

        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]

        if x_bot >= self.params.input_shape[1] or \
                y_bot >= self.params.input_shape[2]:
            raise ValueError(f'Position of backdoor outside image limits:'
                             f'image: {self.params.input_shape}, but backdoor'
                             f'ends at ({x_bot}, {y_bot})')

        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor

        self.mask = 1 * (full_image != self.mask_value).to(self.params.device)
        self.pattern = self.task.normalize(full_image).to(self.params.device)

    def synthesize_inputs(self, batch, attack_portion=None):
        pattern, mask = self.get_pattern()
        batch.inputs[:attack_portion] = (1 - mask) * \
                                        batch.inputs[:attack_portion] + \
                                        mask * pattern
        return

    def synthesize_labels(self, batch, attack_portion=None):
        batch.labels[:attack_portion].fill_(self.params.backdoor_label)

        return

    def get_pattern(self):
        if self.params.backdoor_dynamic_position:
            resize = random.randint(self.resize_scale[0], self.resize_scale[1])
            pattern = self.pattern_tensor
            if random.random() > 0.5:
                pattern = functional.hflip(pattern)
            image = transform_to_image(pattern)
            pattern = transform_to_tensor(
                functional.resize(image,
                                  resize, interpolation=0)).squeeze()

            x = random.randint(0, self.params.input_shape[1] \
                               - pattern.shape[0] - 1)
            y = random.randint(0, self.params.input_shape[2] \
                               - pattern.shape[1] - 1)
            self.make_pattern(pattern, x, y)

        return self.pattern, self.mask


class Cifar10Task(Task):
    normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])

    def load_data(self):
        self.load_cifar_data()

    def load_cifar_data(self):
        train_dataset_without_transform, \
        transform_train, \
        train_label_transform, \
        test_dataset_without_transform, \
        transform_test, \
        test_label_transform = dataset_and_transform_generate(self.params)

        clean_train_dataset_with_transform = dataset_wrapper_with_transform(
            train_dataset_without_transform,
            transform_train,
            train_label_transform
        )

        clean_test_dataset_with_transform = dataset_wrapper_with_transform(
            test_dataset_without_transform,
            transform_test,
            test_label_transform,
        )

        self.train_dataset = clean_train_dataset_with_transform

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True,
                                       pin_memory=self.params.pin_memory,
                                       num_workers=self.params.num_workers)
        self.test_dataset = clean_test_dataset_with_transform
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      pin_memory=self.params.pin_memory,
                                      shuffle=False, num_workers=self.params.num_workers)

        return True

    def build_model(self) -> nn.Module:
        net = generate_cls_model(
            model_name=self.params.model,
            image_size=self.params.img_size[0],
            num_classes=self.params.num_classes,
        )

        if "," in self.params.device:
            net = torch.nn.DataParallel(
                net,
                device_ids=[int(i) for i in self.params.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            logging.info("net data parallel")

        self.params.device = (
            f"cuda:{[int(i) for i in self.params.device[5:].split(',')][0]}" if "," in self.params.device else self.params.device
            # since DataParallel only allow .to("cuda")
        ) if torch.cuda.is_available() else "cpu"

        return net


class Helper:
    params: Params = None
    task: Optional[Task] = None
    synthesizer: Synthesizer = None
    attack: Attack = None
    tb_writer = None

    def __init__(self, params):
        self.params = Params(**params)

        self.times = {'backward': list(), 'forward': list(), 'step': list(),
                      'scales': list(), 'total': list(), 'poison': list()}

        self.make_task()
        self.make_synthesizer()
        self.attack = Attack(self.params, self.synthesizer)

        self.best_loss = float('inf')

    def make_task(self):
        self.task = Cifar10Task(self.params)

    def make_synthesizer(self):
        self.synthesizer = PatternSynthesizer(self.task)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params.save_model:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    def flush_writer(self):
        if self.tb_writer:
            self.tb_writer.flush()

    def plot(self, x, y, name):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag=name, scalar_value=y, global_step=x)
            self.flush_writer()
        else:
            return False

    def report_training_losses_scales(self, batch_id, epoch):
        if not self.params.report_train_loss or \
                batch_id % self.params.log_interval != 0:
            return
        total_batches = len(self.task.train_loader)
        losses = [f'{x}: {np.mean(y):.2f}'
                  for x, y in self.params.running_losses.items()]
        scales = [f'{x}: {np.mean(y):.2f}'
                  for x, y in self.params.running_scales.items()]
        logging.info(
            f'Epoch: {epoch:3d}. '
            f'Batch: {batch_id:5d}/{total_batches}. '
            f' Losses: {losses}.'
            f' Scales: {scales}')
        for name, values in self.params.running_losses.items():
            self.plot(epoch * total_batches + batch_id, np.mean(values),
                      f'Train/Loss_{name}')
        for name, values in self.params.running_scales.items():
            self.plot(epoch * total_batches + batch_id, np.mean(values),
                      f'Train/Scale_{name}')

        self.params.running_losses = defaultdict(list)
        self.params.running_scales = defaultdict(list)


def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True):
    criterion = hlpr.task.criterion
    model.train()

    batch_loss_list = []

    for i, data in tqdm(enumerate(train_loader)):
        batch = hlpr.task.get_batch(i, data)

        with torch.cuda.amp.autocast(enabled=args.amp):
            loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        batch_loss_list.append(loss.item())
        hlpr.report_training_losses_scales(i, epoch)
        if i == hlpr.params.max_batch_id:
            break

    one_epoch_loss = sum(batch_loss_list) / len(batch_loss_list)

    scheduler = getattr(hlpr.task, "scheduler", None)
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(one_epoch_loss)
        else:
            scheduler.step()
        logging.info(f"scheduler step, {scheduler}")

    return one_epoch_loss


def hlpr_test(hlpr: Helper, epoch, backdoor=False):
    model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()

    with torch.no_grad():
        for i, data in tqdm(enumerate(hlpr.task.test_loader)):
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)

            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    metric = hlpr.task.report_metrics(epoch,
                                      prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',
                                      tb_writer=hlpr.tb_writer,
                                      tb_prefix=f'Test_backdoor_{str(backdoor):5s}')

    return metric


def run(hlpr, clean_test_dataloader, bd_test_dataloader, criterion, device, args):
    global scaler
    scaler = torch.cuda.amp.GradScaler(enabled=hlpr.params.amp)

    acc = hlpr_test(hlpr, 0, backdoor=False)

    clean_test_loss_list = []
    bd_test_loss_list = []
    test_acc_list = []
    test_asr_list = []
    test_ra_list = []
    train_loss_list = []

    agg = Metric_Aggregator()

    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        one_epoch_train_loss = train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
                                     hlpr.task.train_loader)
        train_loss_list.append(one_epoch_train_loss)
        acc = hlpr_test(hlpr, epoch, backdoor=False)
        hlpr_test(hlpr, epoch, backdoor=True)

        ### My test code start

        clean_metrics, \
        clean_test_epoch_predict_list, \
        clean_test_epoch_label_list, \
            = given_dataloader_test(
            model=hlpr.task.model,
            test_dataloader=clean_test_dataloader,
            criterion=criterion,
            non_blocking=args.non_blocking,
            device=device,
            verbose=1,
        )

        clean_test_loss_avg_over_batch = clean_metrics["test_loss_avg_over_batch"]
        test_acc = clean_metrics["test_acc"]

        bd_metrics, \
        bd_test_epoch_predict_list, \
        bd_test_epoch_label_list, \
        bd_test_epoch_original_index_list, \
        bd_test_epoch_poison_indicator_list, \
        bd_test_epoch_original_targets_list = test_given_dataloader_on_mix(
            model=hlpr.task.model,
            test_dataloader=bd_test_dataloader,
            criterion=criterion,
            non_blocking=args.non_blocking,
            device=device,
            verbose=1,
        )

        bd_test_loss_avg_over_batch = bd_metrics["test_loss_avg_over_batch"]
        test_asr = all_acc(bd_test_epoch_predict_list, bd_test_epoch_label_list)
        test_ra = all_acc(bd_test_epoch_predict_list, bd_test_epoch_original_targets_list)

        agg(
            {
                "epoch": epoch,
                "train_epoch_loss_avg_over_batch": one_epoch_train_loss,
                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            }
        )

        clean_test_loss_list.append(clean_test_loss_avg_over_batch)
        bd_test_loss_list.append(bd_test_loss_avg_over_batch)
        test_acc_list.append(test_acc)
        test_asr_list.append(test_asr)
        test_ra_list.append(test_ra)

        plot_loss(
            train_loss_list,
            clean_test_loss_list,
            bd_test_loss_list,
            args.save_path,
            "loss_metric_plots",
        )

        plot_acc_like_metric(
            [], [], [],
            test_acc_list,
            test_asr_list,
            test_ra_list,
            args.save_path,
            "loss_metric_plots",
        )

        agg.to_dataframe().to_csv(f"{args.save_path}/attack_df.csv")

    agg.summary().to_csv(f"{args.save_path}/attack_df_summary.csv")

    ### My test code end


class AddMaskPatchTrigger(object):
    def __init__(self,
                 trigger_array: Union[np.ndarray, torch.Tensor],
                 mask_array: Union[np.ndarray, torch.Tensor],
                 ):
        self.trigger_array = trigger_array
        self.mask_array = mask_array

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return img * (1 - self.mask_array) + self.trigger_array * self.mask_array


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = torch.sqrt(
                torch.stack([gr.pow(2).sum().data for gr in grads[t]]).sum())
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = min(losses[t].mean(), 10.0)
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = min(losses[t].mean() * torch.sqrt(
                torch.stack([gr.pow(2).sum().data for gr in grads[t]]).sum()),
                        10)

    elif normalization_type == 'none' or normalization_type == 'eq':
        for t in grads:
            gn[t] = 1.0
    else:
        raise ValueError('ERROR: Invalid Normalization Type')
    return gn


class blendedImageAttack_on_batch(object):

    def __init__(self, target_image, blended_rate, device):
        self.target_image = target_image.to(device)
        self.blended_rate = blended_rate

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return (1 - self.blended_rate) * img + (self.blended_rate) * self.target_image[None, ...]  # use the broadcast


class batchwise_label_transform(object):
    '''
    idea : any label -> fix_target
    '''

    def __init__(self, label_transform, device):
        self.label_transform = label_transform
        self.device = device

    def __call__(self, batch_labels: torch.Tensor, ):
        return torch.tensor([self.label_transform(original_label) for original_label in batch_labels]).to(self.device)


class Blind(BadNet):
    r'''Blind Backdoors in Deep Learning Models

        basic structure:

        1. config args, save_path, fix random seed
        2. set the clean train data and clean test data
        3. set the attack img transform and label transform
        4. set the backdoor attack data and backdoor test data
        5. set the device, model, criterion, optimizer, training schedule.
        6. use the designed blind_loss to train a poisoned model
        7. save the attack result for defense

        .. code-block:: python
            attack = Blind()
            attack.attack()

        .. Note::
            @inproceedings {bagdasaryan2020blind,
                 author = {Eugene Bagdasaryan and Vitaly Shmatikov},
                 title = {Blind Backdoors in Deep Learning Models},
                 booktitle = {30th {USENIX} Security Symposium ({USENIX} Security 21)},
                 year = {2021},
                 isbn = {978-1-939133-24-3},
                 pages = {1505--1521},
                 url = {https://www.usenix.org/conference/usenixsecurity21/presentation/bagdasaryan},
                 publisher = {{USENIX} Association},
                 month = aug,
                }

        Args:
            attack (string): name of attack, use to match the transform and set the saving prefix of path.
            attack_target (int): target class in all2one attack
            attack_label_trans (string): which type of label modification in backdoor attack
            bd_yaml_path (string): path of yaml file to load backdoor attack config
            weight_loss_balance_mode (string): weight loss balance mode (eg. "fixed")
            mgda_normalize (string): mgda normalize mode (eg. "l2", "loss+")
            fix_scale_normal_weight (float): fix scale normal weight
            fix_scale_backdoor_weight (float): fix scale backdoor weight
            batch_history_len (int): len of tracking history to compute when training is stable, so we start to attack
            backdoor_batch_loss_threshold (float): threshold of backdoor batch loss to compute when training is stable, so we start to attack
            **kwargs (optional): Additional attributes.

    '''

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--attack', type=str, )
        parser.add_argument('--attack_target', type=int,
                            help='target class in all2one attack')
        parser.add_argument('--attack_label_trans', type=str,
                            help='which type of label modification in backdoor attack'
                            )
        parser.add_argument("--weight_loss_balance_mode", type=str)
        parser.add_argument("--mgda_normalize", type=str)
        parser.add_argument("--fix_scale_normal_weight", type=float)
        parser.add_argument("--fix_scale_backdoor_weight", type=float)

        parser.add_argument("--batch_history_len", type=int,
                            help="len of tracking history to compute when training is stable, so we start to attack")
        parser.add_argument("--backdoor_batch_loss_threshold", type=float,
                            help="threshold for when training is stable, so we start to attack")
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/blind/default.yaml',
                            help='path for yaml file provide additional default attributes')
        return parser

    def stage1_non_training_data_prepare(self):
        logging.info(f"stage1 start")

        assert 'args' in self.__dict__
        args = self.args

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform, \
        clean_train_dataset_with_transform, \
        clean_train_dataset_targets, \
        clean_test_dataset_with_transform, \
        clean_test_dataset_targets \
            = self.benign_prepare()

        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            transforms.ToTensor()
        ])

        trigger_pattern_np: np.ndarray = np.array([

            [255, 0., 255],
            [-10., 255, -10.],
            [-10., -10., 0.],
            [-10., 255, -10.],
            [255, 0., 255]
        ])
        trigger_pattern_np = np.repeat(trigger_pattern_np[:, :, np.newaxis], args.input_channel, axis=2)
        trigger_full_size_np = np.ones((args.input_height, args.input_width, args.input_channel)) * (-10)
        x_top = 3
        y_top = 23
        trigger_full_size_np[
        x_top: x_top + trigger_pattern_np.shape[0],
        y_top: y_top + trigger_pattern_np.shape[1],
        :
        ] = trigger_pattern_np
        self.trigger_full_size_np = trigger_full_size_np
        self.mask = 1 * (trigger_full_size_np != -10)

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (AddMaskPatchTrigger(
                self.trigger_full_size_np,
                self.mask,
            )
            , True),
        ])

        train_bd_img_transform, test_bd_img_transform = None, test_bd_transform

        ### get the backdoor transform on label
        bd_label_transform = bd_attack_label_trans_generate(args)
        self.bd_label_transform = bd_label_transform

        # NO poison samples in, just use as clean, real poison is done in batchwise way
        bd_train_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(train_dataset_without_transform),
            poison_indicator=None,
            bd_image_pre_transform=train_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_train_dataset",
        )
        bd_train_dataset.getitem_all = True
        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            bd_train_dataset,
            train_img_transform,
            train_label_transform,
        )

        ### decide which img to poison in ASR Test
        test_poison_index = generate_poison_index_from_label_transform(
            clean_test_dataset_targets,
            label_transform=bd_label_transform,
            train=False,
        )

        ### generate test dataset for ASR
        bd_test_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(test_dataset_without_transform),
            poison_indicator=test_poison_index,
            bd_image_pre_transform=test_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_test_dataset",
        )

        bd_test_dataset.subset(
            np.where(test_poison_index == 1)[0]
        )

        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_img_transform,
            test_label_transform,
        )

        self.stage1_results = clean_train_dataset_with_transform, \
                              clean_test_dataset_with_transform, \
                              bd_train_dataset_with_transform, \
                              bd_test_dataset_with_transform

    def stage2_training(self):
        logging.info(f"stage2 start")
        assert 'args' in self.__dict__
        args = self.args

        clean_train_dataset_with_transform, \
        clean_test_dataset_with_transform, \
        bd_train_dataset_with_transform, \
        bd_test_dataset_with_transform = self.stage1_results

        device = torch.device(
            (
                f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

        '''
        start real source code part
        '''

        params = {
            "test_batch_size": 100,
            "log_interval": 100,
            "pretrained": False,
            "loss_threshold": args.backdoor_batch_loss_threshold,
            "poisoning_proportion": 1.1,  # not useful in common poison, just set > 1
            "backdoor_label": args.attack_target,
            "backdoor": True,
            "loss_balance": args.weight_loss_balance_mode,  # MGDA or fixed
            "mgda_normalize": args.mgda_normalize,
            "fixed_scales": {
                "backdoor": args.fix_scale_backdoor_weight,
                "normal": args.fix_scale_normal_weight,
            },
            "loss_tasks": ["backdoor", "normal"],
        }
        args.__dict__.update(params)
        helper = Helper(args.__dict__)
        logging.warning(create_table(args.__dict__))

        criterion = argparser_criterion(args)

        run(
            helper,
            clean_test_dataloader=DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size,
                                             shuffle=False, drop_last=False,
                                             pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            bd_test_dataloader=DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False,
                                          drop_last=False,
                                          pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            criterion=criterion,
            device=device,
            args=args,
        )

        '''
        end real source code part
        '''

        save_attack_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=helper.task.model.cpu().state_dict(),
            data_path=args.dataset_path,
            img_size=args.img_size,
            clean_data=args.dataset,
            bd_train=None,
            bd_test=bd_test_dataset_with_transform,
            save_path=args.save_path,
        )


if __name__ == '__main__':
    attack = Blind()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()

'''
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''