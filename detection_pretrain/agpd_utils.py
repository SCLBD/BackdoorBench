import os,sys
import numpy as np
import torch
import torch.nn as nn
sys.path.append('../')
sys.path.append(os.getcwd())

from utils.nCHW_nHWC import *

import heapq
from utils.bd_dataset_v2 import dataset_wrapper_with_transform,xy_iter, prepro_cls_DatasetBD_v2
import time
from scipy.spatial.distance import pdist

def get_gradient(name,model,x_batch,y_batch, cross_entropy, layer, num_classes, device):
    model.eval()
    assert name in ['preactresnet18', 'vgg19','vgg19_bn', 'vgg11_bn','resnet18', 'mobilenet_v3_large', 'densenet161', 'efficientnet_b3','convnext_tiny','vit_b_16', 'resnet183']
    if name in ['preactresnet18','resnet18', 'resnet183']:
        gins = []
        gouts = []
        def layer_hook(module, grad_input, grad_output):
            gins.append(grad_input[0])
            gouts.append(grad_output[0])

        module_dict = dict(model.named_modules())
        target_layer = module_dict[layer]
        hook = target_layer.register_backward_hook(layer_hook)
        pred = model(x_batch)
        model.zero_grad()
        if cross_entropy:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(pred, y_batch)
        else:
            new_labels = torch.nn.functional.one_hot(y_batch, num_classes=num_classes)
            new_labels = new_labels.to(device)
            loss = torch.sum(pred*new_labels)
        loss.backward()
        gradients_in = gins[0]
        gradients_out = gouts[0]
        hook.remove()
        gradients = gradients_out
    if name in ['vgg19_bn', 'vgg11_bn']:
        gins = []
        gouts = []
        def layer_hook(module, grad_input, grad_output):
            gins.append(grad_input[0])
            gouts.append(grad_output[0])
        hook = model.features[int(layer)].register_backward_hook(layer_hook)
        pred = model(x_batch)
        new_labels = torch.nn.functional.one_hot(y_batch, num_classes=num_classes)
        new_labels = new_labels.to(device)
        loss = torch.sum(pred*new_labels)
        loss.backward()
        gradients_in = gins[0]
        gradients_out = gouts[0]
        hook.remove()
        gradients = gradients_out
    return gradients

def get_all_gradient(data_loader, device, model_name, model, cross_entropy, layer, num_classes):
    gradients_mean_all = []
    for i, (x_batch,y_batch) in enumerate(data_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        batch_gradients = get_gradient(model_name,model,x_batch,y_batch, cross_entropy, layer, num_classes, device)
        batch_gradients_mean = []
        for i, gradient in enumerate(batch_gradients):
            gradient_mean = torch.mean(gradient, dim=(1,2)).cpu().numpy()
            batch_gradients_mean.append(gradient_mean)
        gradients_mean_all.append(batch_gradients_mean)
    gradients_mean_all = np.concatenate(gradients_mean_all, axis=0)
    return gradients_mean_all

def compute_change(gradient_all, clean_gradient_s1, poison_list):
    known_idx3 = np.array(poison_list)
    poison_gradient_avg = np.mean(gradient_all[poison_list],axis=0)
    similar_clean = []
    similar_poison = []
    for i in range(len(gradient_all)):
        if i not in known_idx3:
            dis_clean = compute_distance(gradient_all[i], clean_gradient_s1, 'cosin')
            dis_poison = compute_distance(gradient_all[i], poison_gradient_avg, 'cosin')
            similar_poison.append(dis_poison)
            similar_clean.append(dis_clean)
    l2_dis = np.linalg.norm(np.array(similar_clean)-np.array(similar_poison),2)
    return l2_dis

def compute_distance(sample_test, template, distance):
    if distance == 'norm':
        dis = np.linalg.norm(sample_test-template, ord=1)
    elif distance == 'cosin':
        dis = 1- pdist(np.vstack([sample_test, template]),'cosine')[0]
    return dis

###for ctrl attack
def remove_backbone_prefix(state_dict):
    return {k.replace('backbone.', ''): v for k, v in state_dict.items()}

def get_unsimilar_sample_idx(gradient_all, clean_gradient_s1, poison_list, sus_num):
    known_idx3 = poison_list
    similar = []
    dis_id = []
    for i in range(len(gradient_all)):
        if i not in known_idx3:
            dis = compute_distance(gradient_all[i], clean_gradient_s1, 'cosin')
            similar.append(dis)
            dis_id.append(i)

    similar_min = heapq.nsmallest(sus_num, similar)
    unsimilar_idx = [id for s, id in zip(similar, dis_id) if s in similar_min]
    return unsimilar_idx

def test_clean_samples(dataset, num, tran, device, model_name, model, cross_entrop, layer, num_class, batch_size,num_workers):
    x = []
    y = []
    for i, j in dataset:
        x.append(i)
        y.append(j)
        
    test_dataset = xy_iter(x, y, transform=tran)
    data_clean_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    result = []
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(data_clean_loader):
            input, label = input.to(device), label.to(device)
            outputs = model(input)
            _, predicted = outputs.max(1)
            result.append(predicted.cpu().numpy())
    result = np.concatenate(result, axis=0)

    test_reserved = []
    for i,j in zip(y, result):
        if i == j:
            test_reserved.append(i)
        else:
            test_reserved.append(-1)
            
    class_idx_whole = []
    class_idx_num = [0]
    for i in range(num_class):
        class_correct_num = len(np.where(np.array(test_reserved) == i)[0])
        if class_correct_num >= num:
            class_idx_whole.append(np.random.choice(np.where(np.array(test_reserved) == i)[0], num, replace=False))
            class_idx_num.append(num)
        elif class_correct_num == 0:
            raise ValueError("The clean sample is 0!")
        else:
            class_idx_whole.append(np.where(np.array(test_reserved) == i)[0])
            class_idx_num.append(class_correct_num)

    class_idx_whole = np.concatenate(class_idx_whole, axis=0)
    x_select = [x[i] for i in class_idx_whole]
    y_select = [y[i] for i in class_idx_whole]

    class_idx_num_accumulate = np.cumsum(class_idx_num)

    data_set_o = xy_iter(x_select, y_select, tran)
    data_loader = torch.utils.data.DataLoader(
        data_set_o, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    gradients_mean_all = get_all_gradient(data_loader, device, model_name, model, cross_entrop, layer, num_class)
    gradients_mean_class_avg = []
    for i in range(len(class_idx_num) - 1):
        mean = np.mean(gradients_mean_all[class_idx_num_accumulate[i]:class_idx_num_accumulate[i + 1]], axis=0)
        gradients_mean_class_avg.append(mean)
    return gradients_mean_class_avg

def mad(gap_list):
    J_t_median = np.median(gap_list)
    J_MAD = np.median(np.abs(gap_list - J_t_median))
    J_star = np.abs(gap_list - J_t_median)/1.4826/(J_MAD+1e-6)
    J_star[(gap_list - J_t_median) < 0] = 0
    return J_star


def stage1_new(gradients_mean_all, clean_mean_list, poison_list, thresh, distance, poison_rest_num):
    if len(poison_list) == poison_rest_num:
        return [], [],[]

    known_idx2 = np.array(poison_list)
    poison_mean_list = np.mean(gradients_mean_all[poison_list], axis=0)
    rates = []
    idx = []

    for i in range(len(gradients_mean_all)):
        if i not in known_idx2:
            dis_p = compute_distance(gradients_mean_all[i], poison_mean_list, distance)
            dis_c = compute_distance(gradients_mean_all[i], clean_mean_list, distance)
            rate = (1 - dis_p) / ((1 - dis_p) + (1 - dis_c))
            rates.append(rate)
            idx.append(i)

    if not rates:
        return [], rates,[]

    rates_range = np.max(rates) - np.min(rates)
    if rates_range == 0:
        if np.min(rates) > thresh:
            filtered_idx = idx
        else:
            filtered_idx = [i for rate, i in zip(rates, idx) if rate <= thresh]
    else:
        rates_norm = (np.array(rates) - np.min(rates)) / rates_range
        filtered_idx = [i for rate_norm, i in zip(rates_norm, idx) if rate_norm <= thresh]

    rates_new = [r for i, r in zip(idx, rates) if i not in filtered_idx]
    idx_new = [i for i in idx if i not in filtered_idx]

    return filtered_idx, rates_new, idx_new


def find_small_and_smooth_window_start(js_info, window_size=10):
    
    if len(js_info) < window_size:
        raise ValueError("js_info must contain at least 'window_size' number of elements.")
    
    min_score = float('inf')
    optimal_window_start = None

    scores = []

    for start in range(0, len(js_info) - window_size + 1):
        window = js_info[start:start + window_size]
        window_avg = sum(window) / window_size
        window_std = (sum((x - window_avg) ** 2 for x in window) / window_size) ** 0.5
        score = 1*window_avg + 5*window_std
        scores.append(score)
    
    scores_order = np.argsort(scores)
    optimal_window_start = scores_order[0]
    return optimal_window_start