import sys
import os
from tqdm import tqdm
import numpy as np
import argparse
import torch
from torch import nn
sys.path.append("./")
sys.path.append(os.getcwd())

from utils.defense.utils_dbr.dataloader_bd import normalization

def calculate_consistency(args, dataloader, model):
    model.eval()

    for i, (inputs, labels, _, isCleans, gt_labels) in enumerate(dataloader):
        inputs1, inputs2 = inputs[0], inputs[2]
        inputs1, inputs2 = normalization(args, inputs1), normalization(args, inputs2)  # Normalize
        inputs1, inputs2, labels, gt_labels = inputs1.to(args.device), inputs2.to(args.device), labels.to(args.device), gt_labels.to(args.device)
        clean_idx, poison_idx = torch.where(isCleans == True), torch.where(isCleans == False)

        ### Feature ###
        if hasattr(model, "module"):  # abandon FC layer
            features_out = list(model.module.children())[:-1]
        else:
            features_out = list(model.children())[:-1]
        modelout = nn.Sequential(*features_out).to(args.device)
        features1, features2 = modelout(inputs1), modelout(inputs2)
        features1, features2 = features1.view(features1.size(0), -1), features2.view(features2.size(0), -1)

        ### Calculate consistency ###
        feature_consistency = torch.mean((features1 - features2)**2, dim=1)

        ### Save ###
        draw_features = feature_consistency.detach().cpu().numpy()
        draw_clean_features = feature_consistency[clean_idx].detach().cpu().numpy()
        draw_poison_features = feature_consistency[poison_idx].detach().cpu().numpy()
        
        
        f_path = os.path.join(args.save_path,'data_produce')
        if not os.path.exists(f_path):
            os.makedirs(f_path)
        f_all = os.path.join(f_path,'all.txt')
        f_clean = os.path.join(f_path,'clean.txt')
        f_poison = os.path.join(f_path,'poison.txt')
        with open(f_all, 'ab') as f:
            np.savetxt(f, draw_features, delimiter=" ")
        with open(f_clean, 'ab') as f:
            np.savetxt(f, draw_clean_features, delimiter=" ")
        with open(f_poison, 'ab') as f:
            np.savetxt(f, draw_poison_features, delimiter=" ")
    return

def calculate_gamma(args):
    args.clean_ratio = 0.20
    args.poison_ratio = 0.05


    f_path = os.path.join(args.save_path,'data_produce')
    f_all = os.path.join(f_path,'all.txt')

    all_data = np.loadtxt(f_all)
    all_size = all_data.shape[0] # 50000

    clean_size = int(all_size * args.clean_ratio) # 10000
    poison_size = int(all_size * args.poison_ratio) # 2500

    new_data = np.sort(all_data) # in ascending order
    gamma_low = new_data[clean_size]
    gamma_high = new_data[all_size-poison_size]
    print("gamma_low: ", gamma_low)
    print("gamma_high: ", gamma_high)
    return gamma_low, gamma_high

def separate_samples(args, trainloader, model):
    gamma_low, gamma_high = args.gamma_low, args.gamma_high
    model.eval()
    clean_samples, poison_samples, suspicious_samples = [], [], []

    for i, (inputs, labels, _, _, gt_labels) in enumerate(trainloader):
        if i == 10001 and args.debug:
            break
        if i % 1000 == 0:
            print("Processing samples:", i)
        inputs1, inputs2 = inputs[0], inputs[2]

        ### Prepare for saved ###
        img = inputs1
        img = img.squeeze()
        target = labels.squeeze()
        img = np.transpose((img * 255).cpu().numpy(), (1, 2, 0)).astype('uint8')
        target = target.cpu().numpy()

        inputs1, inputs2 = normalization(args, inputs1), normalization(args, inputs2)  # Normalize
        inputs1, inputs2, labels, gt_labels = inputs1.to(args.device), inputs2.to(args.device), labels.to(args.device), gt_labels.to(args.device)

        ### Features ###
        if hasattr(model, "module"):  # abandon FC layer
            features_out = list(model.module.children())[:-1]
        else:
            features_out = list(model.children())[:-1]
        modelout = nn.Sequential(*features_out).to(args.device)
        features1, features2 = modelout(inputs1), modelout(inputs2)
        features1, features2 = features1.view(features1.size(0), -1), features2.view(features2.size(0), -1)

        ### Compare consistency ###
        feature_consistency = torch.mean((features1 - features2)**2, dim=1)
        # feature_consistency = feature_consistency.detach().cpu().numpy()

        ### Separate samples ###
        if feature_consistency.item() <= gamma_low:
            flag = 0
            clean_samples.append((img, target, flag))
        elif feature_consistency.item() >= gamma_high:
            flag = 2
            poison_samples.append((img, target, flag))
        else:
            flag = 1
            suspicious_samples.append((img, target, flag))

    ### Save samples ###

    folder_path = os.path.join(args.save_path,'data_produce')

    data_path_clean = os.path.join(folder_path, 'clean_samples.npy')
    data_path_poison = os.path.join(folder_path, 'poison_samples.npy')
    data_path_suspicious = os.path.join(folder_path, 'suspicious_samples.npy')
    np.save(data_path_clean, clean_samples)
    np.save(data_path_poison, poison_samples)
    np.save(data_path_suspicious, suspicious_samples)
