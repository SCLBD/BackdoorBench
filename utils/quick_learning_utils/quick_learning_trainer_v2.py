import sys, logging
sys.path.append('../')
import random
from pprint import pformat
from typing import *
import numpy as np
import torch
import pandas as pd
from time import time
from copy import deepcopy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.prefetch import PrefetchLoader, prefetch_transform
from utils.bd_dataset import prepro_cls_DatasetBD


import torch.nn.functional as F
from utils.trainer_cls import BackdoorModelTrainer, all_acc

import datetime
import enum
from tqdm import tqdm

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


class QuickLearningBackdoorModelTrainer(BackdoorModelTrainer):
    def __init__(self, model):
        super().__init__(model)
        logging.debug("This class REQUIRE bd dataset to implement overwrite methods. This is NOT a general class for all cls task.")

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
        train_asr_list = []
        train_ra_list = []
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

            clean_metrics, \
            clean_test_epoch_predict_list, \
            clean_test_epoch_label_list, \
             = self.test_given_dataloader(self.test_dataloader_dict["clean_test_dataloader"], verbose=1)

            clean_test_loss_avg_over_batch = clean_metrics["test_loss_avg_over_batch"]
            test_acc = clean_metrics["test_acc"]

            bd_metrics, \
            bd_test_epoch_predict_list, \
            bd_test_epoch_label_list, \
            bd_test_epoch_original_index_list, \
            bd_test_epoch_poison_indicator_list, \
            bd_test_epoch_original_targets_list = self.test_given_dataloader_on_mix(self.test_dataloader_dict["bd_test_dataloader"], verbose=1)

            bd_test_loss_avg_over_batch = bd_metrics["test_loss_avg_over_batch"]
            test_asr = all_acc(bd_test_epoch_predict_list, bd_test_epoch_label_list)
            test_ra = all_acc(bd_test_epoch_predict_list, bd_test_epoch_original_targets_list)

            grad_metric = self.grad_info(
                    self.train_dataloader, device, epoch, save_folder_path)
            
            info = {
                    "train_epoch_loss_avg_over_batch": train_epoch_loss_avg_over_batch,
                    "train_acc": train_mix_acc,
                    "train_acc_clean_only": train_clean_acc,
                    "train_asr_bd_only": train_asr,
                    "train_ra_bd_only": train_ra,

                    "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                    "bd_test_loss_avg_over_batch" : bd_test_loss_avg_over_batch,
                    "test_acc" : test_acc,
                    "test_asr" : test_asr,
                    "test_ra" : test_ra,
                }
            info.update(grad_metric)

            self.agg(
                info,
            )

            train_loss_list.append(train_epoch_loss_avg_over_batch)
            train_mix_acc_list.append(train_mix_acc)
            train_asr_list.append(train_asr)
            train_ra_list.append(train_ra)

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
                train_asr_list,
                train_ra_list,
                test_acc_list,
                test_asr_list,
                test_ra_list,
            )

            self.agg_save_dataframe()

        self.agg_save_summary()

        return train_loss_list, \
                train_mix_acc_list, \
                train_asr_list, \
                train_ra_list, \
                clean_test_loss_list, \
                bd_test_loss_list, \
                test_acc_list, \
                test_asr_list, \
                test_ra_list

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

        return metrics