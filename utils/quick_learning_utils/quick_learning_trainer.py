# This script is for trainer. This is a warpper for training process.

import datetime
import enum
from tqdm import tqdm
import torch.nn.functional as F
import scipy
from time import time
import pandas as pd
import torch
import numpy as np
from typing import *
from collections import deque
from pprint import pformat
import random
from statistics import mean, mode
import sys
import logging
from torch.utils.data import Dataset, DataLoader
sys.path.append('../')


def last_and_valid_max(col: pd.Series):
    '''
    find last not None value and max valid (not None or np.nan) value for each column
    :param col:
    :return:
    '''
    return pd.Series(
        index=[
            'last', 'valid_max', 'exist_nan_value'
        ],
        data=[
            col[~col.isna()].iloc[-1], pd.to_numeric(col,
                                                     errors='coerce').max(), any(i == 'nan_value' for i in col)
        ])


class Metric_Aggregator(object):
    '''
    aggregate the metric to log automatically
    '''

    def __init__(self):
        self.history = []

    def __call__(self,
                 one_metric: dict):
        # drop pair with None as value
        one_metric = {k: v for k, v in one_metric.items() if v is not None}
        one_metric = {
            k: (
                "nan_value" if v is np.nan or torch.tensor(
                    v).isnan().item() else v  # turn nan to str('nan_value')
            ) for k, v in one_metric.items()
        }
        self.history.append(one_metric)
        logging.info(
            pformat(
                one_metric
            )
        )

    def to_dataframe(self):
        self.df = pd.DataFrame(self.history, dtype=object)
        logging.info("return df with np.nan and None converted by str()")
        return self.df

    def summary(self):
        '''
        do summary for dataframe of record
        :return:
        eg.
            ,train_epoch_num,train_acc_clean
            last,100.0,96.68965148925781
            valid_max,100.0,96.70848846435547
            exist_nan_value,False,False

        '''
        if 'df' not in self.__dict__:
            logging.info('No df found in Metric_Aggregator, generate now')
            self.to_dataframe()
        logging.info("return df with np.nan and None converted by str()")
        return self.df.apply(last_and_valid_max)

def compute_grad_batch(model, loss_fn, sample, target):

    prediction = model(sample)
    loss = loss_fn(prediction, target)
    grad_list = torch.autograd.grad(loss, list(model.parameters()))
    return torch.concat([layer_grad.flatten() for layer_grad in grad_list]).flatten().detach()


def compute_grad(model, loss_fn, sample, target):

    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)

    prediction = model(sample)
    loss = loss_fn(prediction, target)
    grad_list = torch.autograd.grad(loss, list(model.parameters()))
    return torch.concat([layer_grad.flatten() for layer_grad in grad_list]).flatten().detach()


def compute_sample_grads_sums(model, loss_fn, data, targets, additional_info, num_class, grad_epoch):
    """ manually process each sample with per sample gradient, naive implementation """
    if targets.max() >= num_class:
        raise ValueError("targets max value should be less than num_class")

    batch_size = data.shape[0]
    grad_sum = 0
    grad_sum_squared = 0
    grad_sum_clean = 0
    grad_sum_squared_clean = 0
    grad_sum_bd = 0
    grad_sum_squared_bd = 0
    grad_sum_class = [0 for _ in range(num_class)]
    grad_sum_squared_class = [0 for _ in range(num_class)]

    ori_idx, poi_indicator, ori_target = additional_info
    grad_dis_clean = []
    grad_dis_bd = []
    grad_dis_class = [[] for _ in range(num_class)]

    grad_cos_clean = []
    grad_cos_bd = []
    grad_cos_class = [[] for _ in range(num_class)]

    for i in range(batch_size):
        grad_i = compute_grad(model, loss_fn, data[i], targets[i])
        grad_sum += grad_i
        grad_sum_squared += grad_i.square()
        if poi_indicator[i] == 1:
            grad_sum_bd += grad_i
            grad_sum_squared_bd += grad_i.square()
            grad_dis_bd.append(torch.linalg.norm(grad_i.flatten()))
            grad_cos_bd.append(F.cosine_similarity(grad_i.flatten(), grad_epoch.flatten(),dim=0))
        else:
            grad_sum_clean += grad_i
            grad_sum_squared_clean += grad_i.square()
            grad_dis_clean.append(torch.linalg.norm(grad_i.flatten()))
            grad_cos_clean.append(F.cosine_similarity(grad_i.flatten(), grad_epoch.flatten(),dim=0))
            
            grad_sum_class[targets[i]] += grad_i
            grad_sum_squared_class[targets[i]] += grad_i.square()
            grad_dis_class[targets[i]].append(
                torch.linalg.norm(grad_i.flatten()))

            grad_cos_class[targets[i]].append(
                F.cosine_similarity(grad_i.flatten(), grad_epoch.flatten(),dim=0))

    # To avoid GPU memory overflow, we use numpy instead of torch.
    # grad_sum = grad_sum.cpu().numpy()
    # grad_sum_squared = grad_sum_squared.cpu().numpy()
    # grad_sum_clean = grad_sum_clean.cpu().numpy()
    # grad_sum_squared_clean = grad_sum_squared_clean.cpu().numpy()
    # grad_sum_bd = grad_sum_bd.cpu().numpy()
    # grad_sum_squared_bd = grad_sum_squared_bd.cpu().numpy()
    return grad_sum, grad_sum_squared, grad_sum_clean, grad_sum_squared_clean, grad_sum_bd, grad_sum_squared_bd, grad_dis_clean, grad_dis_bd, grad_sum_class, grad_sum_squared_class, grad_dis_class, grad_cos_clean, grad_cos_bd, grad_cos_class

# def compute_sample_grads_sums_op(model, loss_fn, data, targets):
#     """ manually process each sample with per sample gradient """
#     model = GradSampleModule(model)
#     batch_size = data.shape[0]
#     grad_sum = 0
#     grad_sum_squared = 0

#     # zero grad and grad_sample
#     model.zero_grad()

#     output = model(data)
#     loss = loss_fn(output, targets)
#     loss.backward()
#     grad_sum = torch.cat([p.grad_sample.sum(dim=0).view(-1) for p in model.parameters()]).cpu().numpy()
#     grad_sum_squared = torch.cat([p.grad_sample.square().sum(dim=0).view(-1) for p in model.parameters()]).cpu().numpy()


#     return grad_sum, grad_sum_squared

# Modified from https://pytorch.org/functorch/stable/notebooks/per_sample_grads.html

# from opacus.grad_sample import GradSampleModule
# from functorch import make_functional_with_buffers, vmap, grad

def compute_sample_grads_sums_vmap(model, loss_fn, data, targets, additional_info):
    """ manually process each sample with per sample gradient, efficient implementation via """
    fmodel, params, buffers = make_functional_with_buffers(model)

    def compute_loss_stateless_model(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = fmodel(params, buffers, batch)
        loss = loss_fn(predictions, targets)
        return loss

    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    ft_per_sample_grads = ft_compute_sample_grad(
        params, buffers, data, targets)
    # ft_per_sample_grads is of form sample x params x shape of grad
    # reshape to sample x params
    grad_arrays = []
    for ft_grad in ft_per_sample_grads:
        grad_arrays.append(ft_grad.detach().cpu(
        ).numpy().reshape(ft_grad.shape[0], -1))
    grad_matrix = np.concatenate(grad_arrays, axis=1)

    batch_size = data.shape[0]
    grad_sum = 0
    grad_sum_squared = 0
    grad_sum_clean = 0
    grad_sum_squared_clean = 0
    grad_sum_bd = 0
    grad_sum_squared_bd = 0

    ori_idx, poi_indicator, ori_target = additional_info
    poi_indicator_np = poi_indicator.cpu().numpy().reshape(-1)

    grad_sum = grad_matrix.sum(axis=0)
    grad_sum_squared = np.square(grad_matrix).sum(axis=0)
    print(grad_sum_squared.shape)
    grad_sum_bd = grad_matrix[poi_indicator_np == 1].sum(axis=0)
    grad_sum_squared_bd = np.square(
        grad_matrix[poi_indicator_np == 1]).sum(axis=0)

    grad_sum_clean = grad_matrix[poi_indicator_np == 0].sum(axis=0)
    grad_sum_squared_clean = np.square(
        grad_matrix[poi_indicator_np == 0]).sum(axis=0)

    # for i in range(batch_size):
    #     grad_i_np = grad_matrix[i]
    #     grad_sum += grad_i_np
    #     grad_sum_squared += grad_i_np ** 2
    #     if poi_indicator[i] == 1:
    #         grad_sum_bd += grad_i_np
    #         grad_sum_squared_bd += grad_i_np ** 2
    #     else:
    #         grad_sum_clean += grad_i_np
    #         grad_sum_squared_clean += grad_i_np ** 2

    return grad_sum, grad_sum_squared, grad_sum_clean, grad_sum_squared_clean, grad_sum_bd, grad_sum_squared_bd


class ModelTrainerCLS():
    def __init__(self, model, amp=False, args = None):
        self.model = model
        self.amp = amp
        self.args = args
        # get the value of each parameter
        self.init_weights = []
        for p in self.model.cuda().parameters():
            self.init_weights.append(p.data.clone())

    def init_or_continue_train(self,
                               train_data,
                               end_epoch_num,
                               criterion,
                               optimizer,
                               scheduler,
                               device,
                               continue_training_path: Optional[str] = None,
                               only_load_model: bool = False,
                               ) -> None:
        '''
        config the training process, from 0 or continue previous.
        The requirement for saved file please refer to save_all_state_to_path
        :param train_data: train_data_loader, only if when you need of number of batch, you need to input it. Otherwise just skip.
        :param end_epoch_num: end training epoch number, if not continue training process, then equal to total training epoch
        :param criterion: loss function used
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param device: device
        :param continue_training_path: where to load files for continue training process
        :param only_load_model: only load the model, do not load other settings and random state.

        '''

        model = self.model

        model.to(device)
        model.train()

        # train and update

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        if continue_training_path is not None:
            logging.info(f"No batch info will be used. Cannot continue from specific batch!")
            # start_epoch, start_batch = self.load_from_path(continue_training_path, device, only_load_model)
            # if (start_epoch is None) or (start_batch is None):
            #     self.start_epochs, self.end_epochs = 0, end_epoch_num
            #     self.start_batch = 0
            # else:
            #     batch_num = len(train_data)
            #     self.start_epochs, self.end_epochs = start_epoch + ((start_batch + 1)//batch_num), end_epoch_num
            #     self.start_batch = (start_batch + 1) % batch_num
            start_epoch, _ = self.load_from_path(continue_training_path, device, only_load_model)
            self.start_epochs, self.end_epochs = start_epoch, end_epoch_num
        else:
            self.start_epochs, self.end_epochs = 0, end_epoch_num
            # self.start_batch = 0

        logging.info(f'All setting done, train from epoch {self.start_epochs} to epoch {self.end_epochs}')

        logging.info(
            pformat(f"self.amp:{self.amp}," +
                    f"self.criterion:{self.criterion}," +
                    f"self.optimizer:{self.optimizer}," +
                    f"self.scheduler:{self.scheduler.state_dict() if self.scheduler is not None else None}," +
                    f"self.scaler:{self.scaler.state_dict() if self.scaler is not None else None})")
        )

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def save_all_state_to_path(self,
                               path: str,
                               epoch: Optional[int] = None,
                               batch: Optional[int] = None,
                               only_model_state_dict: bool = False) -> None:
        '''
        save all information needed to continue training, include 3 random state in random, numpy and torch
        :param path: where to save
        :param epoch: which epoch when save
        :param batch: which batch index when save
        :param only_model_state_dict: only save the model, drop all other information
        '''

        save_dict = {
            'epoch_num_when_save': epoch,
            'batch_num_when_save': batch,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.random.get_rng_state(),
            'model_state_dict': self.get_model_params(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'criterion_state_dict': self.criterion.state_dict(),
            "scaler": self.scaler.state_dict(),
        } \
            if only_model_state_dict == False else self.get_model_params()

        torch.save(
            save_dict,
            path,
        )

    def load_from_path(self,
                       path: str,
                       device,
                       only_load_model: bool = False
                       ) -> [Optional[int], Optional[int]]:
        '''

        :param path:
        :param device: map model to which device
        :param only_load_model: only_load_model or not?
        '''

        self.model = self.model.to(device)

        load_dict = torch.load(
            path, map_location=device
        )

        logging.info(
            f"loading... keys:{load_dict.keys()}, only_load_model:{only_load_model}")

        attr_list = [
            'epoch_num_when_save',
            'batch_num_when_save',
            'random_state',
            'np_random_state',
            'torch_random_state',
            'model_state_dict',
            'optimizer_state_dict',
            'scheduler_state_dict',
            'criterion_state_dict',
        ]

        if all([key_name in load_dict for key_name in attr_list]):
            # all required key can find in load dict
            # AND only_load_model == False
            if only_load_model == False:
                random.setstate(load_dict['random_state'])
                np.random.set_state(load_dict['np_random_state'])
                torch.random.set_rng_state(
                    load_dict['torch_random_state'].cpu())  # since may map to cuda

                self.model.load_state_dict(
                    load_dict['model_state_dict']
                )
                self.optimizer.load_state_dict(
                    load_dict['optimizer_state_dict']
                )
                if self.scheduler is not None:
                    self.scheduler.load_state_dict(
                        load_dict['scheduler_state_dict']
                    )
                self.criterion.load_state_dict(
                    load_dict['criterion_state_dict']
                )
                if 'scaler' in load_dict:
                    self.scaler.load_state_dict(
                        load_dict["scaler"]
                    )
                    logging.info(
                        f'load scaler done. scaler={load_dict["scaler"]}')
                logging.info('all state load successful')
                return load_dict['epoch_num_when_save'], load_dict['batch_num_when_save']
            else:
                self.model.load_state_dict(
                    load_dict['model_state_dict'],
                )
                logging.info('only model state_dict load')
                return None, None

        else:  # only state_dict

            if 'model_state_dict' in load_dict:
                self.model.load_state_dict(
                    load_dict['model_state_dict'],
                )
                logging.info('only model state_dict load')
                return None, None
            else:
                self.model.load_state_dict(
                    load_dict,
                )
                logging.info('only model state_dict load')
                return None, None

    # def grad_info_op(self, test_data, device):
    #     model = self.model
    #     model.to(device)
    #     model.eval()

    #     metrics = {
    #         'GSNR': 0,
    #         'test_total': 0,
    #         'grad_mean': 0,
    #         'grad_var': 0,
    #         'grad_norm': 0,
    #     }

    #     criterion = self.criterion.to(device)
    #     epoch_grad_sum = 0
    #     epoch_grad_sum_squared = 0
    #     test_total = 0
    #     print('Collecting gradients info: GSNR')
    #     for batch_idx, (x, target, *additional_info) in tqdm(enumerate(test_data)):
    #         x = x.to(device)
    #         target = target.to(device)
    #         batch_grad_sum, batch_grad_sum_squared = compute_sample_grads_sums_op(model, criterion, x, target)
    #         epoch_grad_sum += batch_grad_sum
    #         epoch_grad_sum_squared += batch_grad_sum_squared
    #         test_total += target.size(0)

    #     grad_var = epoch_grad_sum_squared / test_total - (epoch_grad_sum / test_total) ** 2
    #     metrics['GSNR'] = np.mean((epoch_grad_sum / test_total)**2 / grad_var)
    #     metrics['test_total'] = test_total
    #     metrics['grad_mean'] = np.mean(epoch_grad_sum / test_total)
    #     metrics['grad_var'] = np.mean(grad_var)
    #     metrics['grad_norm'] = np.linalg.norm(epoch_grad_sum / test_total)

    #     return metrics

    def grad_info(self, test_data, device, epoch, save_folder_path):
        model = self.model
        model.to(device)
        model.eval()

        metrics = {
            'GSNR': 0,
            'test_total': 0,
            'grad_mean': 0,
            'grad_var': 0,
            'grad_norm': 0,
            'GSNR_clean': 0,
            'clean_total': 0,
            'clean_grad_mean': 0,
            'clean_grad_var': 0,
            'clean_grad_norm': 0,
            'GSNR_bd': 0,
            'bd_total': 0,
            'bd_grad_mean': 0,
            'bd_grad_var': 0,
            'bd_grad_norm': 0,
            'cosine_tot_clean': 0,
            'cosine_tot_bd': 0,
            'cosine_clean_bd': 0,
        }
        num_class = 10

        criterion = self.criterion.to(device)
        epoch_grad_sum = 0
        epoch_grad_sum_squared = 0

        epoch_grad_sum_clean = 0
        epoch_grad_sum_squared_clean = 0

        epoch_grad_sum_bd = 0
        epoch_grad_sum_squared_bd = 0

        epoch_grad_sum_class = [0 for _ in range(num_class)]
        epoch_grad_sum_squared_class = [0 for _ in range(num_class)]

        test_total = 0
        clean_total = 1e-12  # avoid zero division
        bd_total = 1e-12  # avoid zero division
        class_total = [1e-12 for _ in range(10)]  # avoid zero division
        print('Collecting gradients info: GSNR')
        grad_dis_clean_total = []
        grad_dis_bd_total = []

        grad_dis_class_total = [[] for _ in range(num_class)]

        grad_cos_clean_total = []
        grad_cos_bd_total = []

        grad_cos_class_total = [[] for _ in range(num_class)]
        # collect mean grad vector
        grad_epoch = 0
        for batch_idx, (x, target, *additional_info) in tqdm(enumerate(test_data)):
            x = x.to(device)
            target = target.to(device)
            grad_batch_i = compute_grad_batch(model, self.criterion, x, target)
            grad_epoch+=grad_batch_i
        grad_epoch = grad_epoch/(batch_idx+1)
            

        for batch_idx, (x, target, *additional_info) in tqdm(enumerate(test_data)):
            x = x.to(device)
            target = target.to(device)

            # batch_grad_sum, batch_grad_sum_squared, batch_grad_sum_clean, batch_grad_sum_squared_clean, batch_grad_sum_bd, batch_grad_sum_squared_bd = compute_sample_grads_sums_vmap(model, criterion, x, target, additional_info)
            batch_grad_sum, batch_grad_sum_squared, batch_grad_sum_clean, batch_grad_sum_squared_clean, batch_grad_sum_bd, batch_grad_sum_squared_bd, batch_grad_dis_clean, batch_grad_dis_bd, batch_grad_sum_class, batch_grad_sum_squared_class, batch_grad_dis_class, batch_grad_cos_clean, batch_grad_cos_bd, batch_grad_cos_class = compute_sample_grads_sums(
                model, criterion, x, target, additional_info, num_class, grad_epoch)

            epoch_grad_sum += batch_grad_sum
            epoch_grad_sum_squared += batch_grad_sum_squared

            epoch_grad_sum_clean += batch_grad_sum_clean
            epoch_grad_sum_squared_clean += batch_grad_sum_squared_clean

            epoch_grad_sum_bd += batch_grad_sum_bd
            epoch_grad_sum_squared_bd += batch_grad_sum_squared_bd

            test_total += target.size(0)
            bd_total += torch.sum(additional_info[1]).item()
            clean_total += target.size(0) - \
                torch.sum(additional_info[1]).item()

            grad_dis_clean_total += batch_grad_dis_clean
            grad_dis_bd_total += batch_grad_dis_bd

            grad_cos_clean_total += batch_grad_cos_clean
            grad_cos_bd_total += batch_grad_cos_bd

            for c_i in range(num_class):
                epoch_grad_sum_class[c_i] += batch_grad_sum_class[c_i]
                epoch_grad_sum_squared_class[c_i] += batch_grad_sum_squared_class[c_i]
                grad_dis_class_total[c_i]+=batch_grad_dis_class[c_i]
                grad_cos_class_total[c_i]+=batch_grad_cos_class[c_i]
                class_total[c_i] += len(batch_grad_dis_class[c_i])

        grad_dis_clean_total_numpy = np.array(
            [i.cpu().numpy() for i in grad_dis_clean_total])
        grad_dis_bd_total_numpy = np.array(
            [i.cpu().numpy() for i in grad_dis_bd_total])
        
        grad_cos_clean_total_numpy = np.array(
            [i.cpu().numpy() for i in grad_cos_clean_total])
        grad_cos_bd_total_numpy = np.array(
            [i.cpu().numpy() for i in grad_cos_bd_total])

        grad_dis_class_numpy = []
        grad_cos_class_numpy = []
        for c_i in range(num_class):
            temp_grad_class = grad_dis_class_total[c_i]
            grad_dis_class_numpy.append(np.array([i.cpu().numpy() for i in temp_grad_class]))

            temp_grad_class = grad_cos_class_total[c_i]
            grad_cos_class_numpy.append(np.array([i.cpu().numpy() for i in temp_grad_class]))

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        np.save(f'./{save_folder_path}/{epoch}_grad_dis_clean_total.npy',
                grad_dis_clean_total_numpy)
        np.save(f'./{save_folder_path}/{epoch}_grad_dis_bd_total.npy',
                grad_dis_bd_total_numpy)
        np.save(f'./{save_folder_path}/{epoch}_grad_cos_clean_total.npy',
                grad_cos_clean_total_numpy)
        np.save(f'./{save_folder_path}/{epoch}_grad_cos_bd_total.npy',
                grad_cos_bd_total_numpy)
        
        for c_i in range(num_class):
            np.save(f'./{save_folder_path}/{epoch}_grad_dis_class_{c_i}.npy',
                    grad_dis_class_numpy[c_i])
            np.save(f'./{save_folder_path}/{epoch}_grad_cos_class_{c_i}.npy',
                    grad_cos_class_numpy[c_i])
                

        mean_grad = epoch_grad_sum / test_total
        mean_grad_clean = epoch_grad_sum_clean / clean_total
        mean_grad_bd = epoch_grad_sum_bd / bd_total
        mean_grad_class = [epoch_grad_sum_class[c_i] /
                           class_total[c_i] for c_i in range(num_class)]

        grad_var = epoch_grad_sum_squared / test_total - mean_grad.square()+1e-16
        grad_var_clean = epoch_grad_sum_squared_clean / \
            clean_total - mean_grad_clean.square()+1e-16
        grad_var_bd = epoch_grad_sum_squared_bd / \
            bd_total - mean_grad_bd.square()+1e-16
        grad_var_class = [epoch_grad_sum_squared_class[c_i] / class_total[c_i] -
                          mean_grad_class[c_i].square()+1e-16 for c_i in range(num_class)]

        metrics['GSNR'] = torch.mean(mean_grad.square() / grad_var).item()
        print(torch.mean(mean_grad.square()).item())
        print(torch.mean(grad_var).item())
        print(torch.min(mean_grad.square()).item())
        print(torch.min(grad_var).item())
        metrics['test_total'] = test_total
        metrics['grad_mean'] = torch.mean(mean_grad).item()
        metrics['grad_var'] = torch.mean(grad_var).item()
        metrics['grad_norm'] = torch.linalg.norm(mean_grad).item()

        metrics['GSNR_clean'] = torch.mean(
            mean_grad_clean.square() / grad_var_clean).item()
        metrics['clean_total'] = clean_total
        metrics['clean_grad_mean'] = torch.mean(mean_grad_clean).item()
        metrics['clean_grad_var'] = torch.mean(grad_var_clean).item()
        metrics['clean_grad_norm'] = torch.linalg.norm(mean_grad_clean).item()

        metrics['GSNR_bd'] = torch.mean(
            mean_grad_bd.square() / grad_var_bd).item()
        metrics['bd_total'] = bd_total
        metrics['bd_grad_mean'] = torch.mean(mean_grad_bd).item()
        metrics['bd_grad_var'] = torch.mean(grad_var_bd).item()
        metrics['bd_grad_norm'] = torch.linalg.norm(mean_grad_bd).item()
        


        # compute the cosine similarity between the clean and bd gradients
        metrics['cosine_tot_clean'] = F.cosine_similarity(
            mean_grad_clean, mean_grad, dim=0).item()
        metrics['cosine_tot_bd'] = F.cosine_similarity(
            mean_grad_bd, mean_grad, dim=0).item()
        metrics['cosine_clean_bd'] = F.cosine_similarity(
            mean_grad_clean, mean_grad_bd, dim=0).item()


        for c_i in range(num_class):
            metrics[f'GSNR_class_{c_i}'] = torch.mean(mean_grad_class[c_i].square()/grad_var_class[c_i]).item()
            metrics[f'class_{c_i}_total'] =class_total[c_i]
            metrics[f'class_{c_i}_grad_mean'] = torch.mean(mean_grad_class[c_i]).item()
            metrics[f'class_{c_i}_grad_var'] = torch.mean(grad_var_class[c_i]).item()
            metrics[f'class_{c_i}_grad_norm'] = torch.linalg.norm(mean_grad_class[c_i]).item()
            metrics[f'cosine_{c_i}_total'] = F.cosine_similarity(mean_grad_class[c_i], mean_grad, dim=0).item()



        # To avoid GPU memory overflow, use numpy instead of tensor

        # mean_grad = mean_grad.detach().cpu().numpy()
        # mean_grad_clean = mean_grad_clean.detach().cpu().numpy()
        # mean_grad_bd = mean_grad_bd.detach().cpu().numpy()

        # grad_var = grad_var.detach().cpu().numpy()
        # grad_var_clean = grad_var_clean.detach().cpu().numpy()
        # grad_var_bd = grad_var_bd.detach().cpu().numpy()
        # metrics['GSNR'] = np.mean((mean_grad)**2 / grad_var)
        # metrics['test_total'] = test_total
        # metrics['grad_mean'] = np.mean(mean_grad)
        # metrics['grad_var'] = np.mean(grad_var)
        # metrics['grad_norm'] = np.linalg.norm(test_total)

        # metrics['GSNR_clean'] = np.mean((mean_grad_clean)**2 / grad_var_clean)
        # metrics['clean_total'] = clean_total
        # metrics['clean_grad_mean'] = np.mean(mean_grad_clean)
        # metrics['clean_grad_var'] = np.mean(grad_var_clean)
        # metrics['clean_grad_norm'] = np.linalg.norm(mean_grad_clean)

        # metrics['GSNR_bd'] = np.mean((mean_grad_bd)**2 / grad_var_bd)
        # metrics['bd_total'] = bd_total
        # metrics['bd_grad_mean'] = np.mean(mean_grad_bd)
        # metrics['bd_grad_var'] = np.mean(grad_var_bd)
        # metrics['bd_grad_norm'] = np.linalg.norm(mean_grad_bd)
        # # compute the cosine similarity between the clean and bd gradients
        # metrics['cosine_tot_clean'] = scipy.spatial.distance.cosine(mean_grad_clean, mean_grad)
        # metrics['cosine_tot_bd'] = scipy.spatial.distance.cosine(mean_grad_bd, mean_grad)
        # metrics['cosine_clean_bd'] = scipy.spatial.distance.cosine(mean_grad_clean, mean_grad_bd)

        return metrics
    def test_train(self, test_data, device):
        model = self.model
        model.to(device)
        model.eval()

        metrics = {
            'clean_correct': 0,
            'clean_loss': 0,
            'clean_total': 0,
            'clean_acc': 0,
            'bd_correct': 0,
            'bd_loss': 0,
            'bd_total': 0,
            'bd_acc': 0,
        }
        criterion_sep = torch.nn.CrossEntropyLoss(reduction='none')
        criterion = self.criterion.to(device)

        with torch.no_grad():
            for batch_idx, (x, target, *additional_info) in enumerate(test_data):
                ori_idx, poi_indicator, ori_target = additional_info
                poi_indicator = poi_indicator.to(device)
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion_sep(pred, target.long())
                loss_clean = torch.sum(loss*(1-poi_indicator))
                loss_bd = torch.sum(loss*poi_indicator)

                # logging.info(list(zip(additional_info[0].cpu().numpy(), pred.detach().cpu().numpy(),
                #                target.detach().cpu().numpy(), )))

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target)
                correct_clean = torch.sum(correct*(1-poi_indicator))
                correct_bd = torch.sum(correct*poi_indicator)

                metrics['clean_correct'] += correct_clean.item()
                metrics['clean_loss'] += loss_clean.item()
                metrics['clean_total'] += torch.sum(1-poi_indicator).item()
                metrics['bd_correct'] += correct_bd.item()
                metrics['bd_loss'] += loss_bd.item()
                metrics['bd_total'] += torch.sum(poi_indicator).item()
                
        metrics['clean_acc'] += metrics['clean_correct']/metrics['clean_total']
        metrics['bd_acc'] += metrics['bd_correct']/metrics['bd_total']

        return metrics

    def test(self, test_data, device):
        model = self.model
        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
            # 'detail_list' : [],
        }

        criterion = self.criterion.to(device)

        with torch.no_grad():
            for batch_idx, (x, target, *additional_info) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                # logging.info(list(zip(additional_info[0].cpu().numpy(), pred.detach().cpu().numpy(),
                #                target.detach().cpu().numpy(), )))

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        return metrics

    # @resource_check
    def train_one_batch(self, x, labels, device):
        clip_weight = True
        eps = self.args.weight_eps
        self.model.train()
        self.model.to(device)

        x, labels = x.to(device), labels.to(device)

        with torch.cuda.amp.autocast(enabled=self.amp):
            log_probs = self.model(x)
            loss = self.criterion(log_probs, labels.long())
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        batch_loss = loss.item() * labels.size(0)
        
        if clip_weight:
            # print('Do Clipping...')
            with torch.no_grad():
                for idx, param in enumerate(self.model.parameters()):
                    param.clamp_(self.init_weights[idx]-eps, self.init_weights[idx]+eps)
        

        return batch_loss

    def train_one_epoch(self, train_data, device):
        startTime = time()
        batch_loss = []
        total_batch = len(train_data)
        for batch_idx, (x, labels, *additional_info) in enumerate(train_data):
            # if batch_idx % 1 == 0:
            #     for possi_i in range(1000000):
            #         import os
            #         if os.path.exists(f'/workspace/weishaokui/BackdoorBench/record/model_{possi_i}.pt'):
            #             pass
            #         else:
            #             torch.save(self.model.state_dict(), f'/workspace/weishaokui/BackdoorBench/record/model_{possi_i}.pt')
            #             break
            batch_loss.append(self.train_one_batch(x, labels, device))
        one_epoch_loss = sum(batch_loss)
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # here since ReduceLROnPlateau need the train loss to decide next step setting.
                self.scheduler.step(one_epoch_loss)
            else:
                self.scheduler.step()

        endTime = time()

        logging.info(
            f"one epoch training part done, use time = {endTime - startTime} s")

        return one_epoch_loss

    def train(self, train_data, end_epoch_num,
              criterion,
              optimizer,
              scheduler, device,  frequency_save, save_folder_path,
              save_prefix,
              continue_training_path: Optional[str] = None,
              only_load_model: bool = False, ):
        '''

        simplest train algorithm with init function put inside.

        :param train_data: train_data_loader
        :param end_epoch_num: end training epoch number, if not continue training process, then equal to total training epoch
        :param criterion: loss function used
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param device: device
        :param frequency_save: how many epoch to save model and random states information once
        :param save_folder_path: folder path to save files
        :param save_prefix: for saved files, the prefix of file name
        :param continue_training_path: where to load files for continue training process
        :param only_load_model: only load the model, do not load other settings and random state.
        '''

        self.init_or_continue_train(
            train_data,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):
            one_epoch_loss = self.train_one_epoch(train_data, device)
            epoch_loss.append(one_epoch_loss)
            logging.info(f'train, epoch_loss: {epoch_loss[-1]}')
            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")

    def train_with_test_each_epoch(self,
                                   train_data,
                                   test_data,
                                   adv_test_data,
                                   end_epoch_num,
                                   criterion,
                                   optimizer,
                                   scheduler,
                                   device,
                                   frequency_save,
                                   save_folder_path,
                                   save_prefix,
                                   continue_training_path: Optional[str] = None,
                                   only_load_model: bool = False,
                                   ):
        '''
        train with test on benign and backdoor dataloader for each epoch

        :param train_data: train_data_loader
        :param test_data: benign test data
        :param adv_test_data: backdoor poisoned test data (for ASR)
        :param end_epoch_num: end training epoch number, if not continue training process, then equal to total training epoch
        :param criterion: loss function used
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param device: device
        :param frequency_save: how many epoch to save model and random states information once
        :param save_folder_path: folder path to save files
        :param save_prefix: for saved files, the prefix of file name
        :param continue_training_path: where to load files for continue training process
        :param only_load_model: only load the model, do not load other settings and random state.
        '''
        collect_grad = False
        agg = Metric_Aggregator()
        self.init_or_continue_train(
            train_data,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):
            one_epoch_loss = self.train_one_epoch(train_data, device)
            epoch_loss.append(one_epoch_loss)
            logging.info(
                f'train_with_test_each_epoch, epoch:{epoch} ,epoch_loss: {epoch_loss[-1]}')

            train_metrics = self.test(train_data, device)
            # metric_info = {
            #     'epoch': epoch,
            #     'train acc': train_metrics['test_correct'] / train_metrics['test_total'],
            #     'train loss': train_metrics['test_loss'],
            # }
            # agg(metric_info)

            metrics = self.test(test_data, device)
            # metric_info = {
            #     'epoch': epoch,
            #     'benign acc': metrics['test_correct'] / metrics['test_total'],
            #     'benign loss': metrics['test_loss'],
            # }
            # agg(metric_info)

            adv_metrics = self.test(adv_test_data, device)
            # adv_metric_info = {
            #     'epoch': epoch,
            #     'ASR': adv_metrics['test_correct'] / adv_metrics['test_total'],
            #     'backdoor loss': adv_metrics['test_loss'],
            # }
            # agg(adv_metric_info)

#            grad_metric = self.grad_info(train_data, device)
            if collect_grad:
                grad_metric = self.grad_info(
                    train_data, device, epoch, save_folder_path)

            # metric_info = {
            #     'epoch': epoch,
            #     'train acc': train_metrics['test_correct'] / train_metrics['test_total'],
            #     'train loss': train_metrics['test_loss'],
            #     'test benign acc': metrics['test_correct'] / metrics['test_total'],
            #     'test benign loss': metrics['test_loss'],
            #     'test ASR': adv_metrics['test_correct'] / adv_metrics['test_total'],
            #     'test backdoor loss': adv_metrics['test_loss'],
            #     'GSNR': grad_metric['GSNR'],
            #     'grad_mean': grad_metric['grad_mean'],
            #     'grad_var': grad_metric['grad_var'],
            #     'grad_norm': grad_metric['grad_norm'],
            #     'GSNR_clean': grad_metric['GSNR_clean'],
            #     'clean_grad_mean': grad_metric['clean_grad_mean'],
            #     'clean_grad_var': grad_metric['clean_grad_var'],
            #     'clean_grad_norm': grad_metric['clean_grad_norm'],
            #     'GSNR_bd': grad_metric['GSNR_bd'],
            #     'bd_grad_mean': grad_metric['bd_grad_mean'],
            #     'bd_grad_var': grad_metric['bd_grad_var'],
            #     'bd_grad_norm': grad_metric['bd_grad_norm'],
            #     'cosine_tot_clean': grad_metric['cosine_tot_clean'],
            #     'cosine_tot_bd': grad_metric['cosine_tot_bd'],
            #     'cosine_clean_bd': grad_metric['cosine_clean_bd'],
            # }

            metric_info = {
                'epoch': epoch,
                'train acc': train_metrics['test_correct'] / train_metrics['test_total'],
                'train loss': train_metrics['test_loss'],
                'test benign acc': metrics['test_correct'] / metrics['test_total'],
                'test benign loss': metrics['test_loss'],
                'test ASR': adv_metrics['test_correct'] / adv_metrics['test_total'],
                'test backdoor loss': adv_metrics['test_loss']
            }

            if collect_grad:
                # combine all metrics
                metric_info.update(grad_metric)
            
            agg(metric_info)

            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")
            # logging.info(f"training, epoch:{epoch}, batch:{batch_idx},batch_loss:{loss.item()}")
            agg.to_dataframe().to_csv(
                f"{save_folder_path}/{save_prefix}_df.csv")
        agg.summary().to_csv(
            f"{save_folder_path}/{save_prefix}_df_summary.csv")

    def train_with_test_each_epoch_v2(self,
                                      train_data,
                                      test_dataloader_dict,
                                      end_epoch_num,
                                      criterion,
                                      optimizer,
                                      scheduler,
                                      device,
                                      frequency_save,
                                      save_folder_path,
                                      save_prefix,
                                      continue_training_path: Optional[str] = None,
                                      only_load_model: bool = False,
                                      ):
        '''
        v2 can feed many test_dataloader, so easier for test with multiple dataloader.

        only change the test data part, instead of predetermined 2 dataloader, you can input any number of dataloader to test
        with {
            test_name (will show in log): test dataloader
        }
        in log you will see acc and loss for each test dataloader

        :param test_dataloader_dict: { name : dataloader }

        :param train_data: train_data_loader
        :param end_epoch_num: end training epoch number, if not continue training process, then equal to total training epoch
        :param criterion: loss function used
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param device: device
        :param frequency_save: how many epoch to save model and random states information once
        :param save_folder_path: folder path to save files
        :param save_prefix: for saved files, the prefix of file name
        :param continue_training_path: where to load files for continue training process
        :param only_load_model: only load the model, do not load other settings and random state.
        '''
        agg = Metric_Aggregator()
        self.init_or_continue_train(
            train_data,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):
            one_epoch_loss = self.train_one_epoch(train_data, device)
            epoch_loss.append(one_epoch_loss)
            logging.info(
                f'train_with_test_each_epoch, epoch:{epoch} ,epoch_loss: {epoch_loss[-1]}')

            for dl_name, test_dataloader in test_dataloader_dict.items():
                metrics = self.test(test_dataloader, device)
                metric_info = {
                    'epoch': epoch,
                    f'{dl_name} acc': metrics['test_correct'] / metrics['test_total'],
                    f'{dl_name} loss': metrics['test_loss'],
                }


                metrics_train = self.test_train(train_data, device)
                metric_info.update(metrics_train)
                agg(metric_info)

            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")
            # logging.info(f"training, epoch:{epoch}, batch:{batch_idx},batch_loss:{loss.item()}")
            agg.to_dataframe().to_csv(f"{save_folder_path}/{save_prefix}_df.csv")
        agg.summary().to_csv(f"{save_folder_path}/{save_prefix}_df_summary.csv")

    def train_with_test_each_epoch_v2_sp(self,
                                      batch_size,
                                      train_dataset,
                                      test_dataset_dict,
                                      end_epoch_num,
                                      criterion,
                                      optimizer,
                                      scheduler,
                                      device,
                                      frequency_save,
                                      save_folder_path,
                                      save_prefix,
                                      continue_training_path: Optional[str] = None,
                                      only_load_model: bool = False,
                                      ):

        '''
        Nothing different, just be simplified to accept dataset instead.
        '''
        train_data = DataLoader(
            dataset = train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        test_dataloader_dict = {
            name : DataLoader(
                    dataset = test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                )
            for name, test_dataset in test_dataset_dict.items()
        }

        self.train_with_test_each_epoch_v2(
            train_data,
            test_dataloader_dict,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            frequency_save,
            save_folder_path,
            save_prefix,
            continue_training_path,
            only_load_model,
        )
