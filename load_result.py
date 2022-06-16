from __future__ import absolute_import, division, print_function

import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
from pyexpat import model

from utils.aggregate_block.dataset_and_transform_generate import (
    get_input_shape, get_num_classes, get_transform)
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import nCHW_to_nHWC
from utils.save_load_attack import load_attack_result


def load_result_rc(result_folder, defense):
    save_path = './record/' + result_folder
    result = torch.load(save_path + '/' + defense + '/defense_result.pt')
    print('asr:{} acc:{} ra:{}'.format(result['asr'], result['acc'], result['ra']))
    return result['acc'].cpu().item(), result['asr'].cpu().item(), result['ra'].cpu().item()


def load_result_no(result_folder, dataset):
    save_path = './record/' + result_folder
    num_classes = get_num_classes(dataset)
    input_height, input_width, input_channel = get_input_shape(dataset)

    print(save_path + '/attack_result.pt')
    data = load_attack_result(save_path + '/attack_result.pt')
    ori_label_un = data['clean_test']['y']
    ori_label = [i for i in ori_label_un if i != 0]

    try:
        model = generate_cls_model(data['model_name'], num_classes)
        model.load_state_dict(data['model'])
    except RuntimeError:
        print('ERROR in generate_cls_model')

    model.to('cuda')
    model.eval()

    tran = get_transform(dataset, *([input_height, input_width]), train=False)

    x = data['bd_test']['x']
    y = ori_label
    data_set = list(zip(x,y))

    # x = torch.tensor(nCHW_to_nHWC(data['bd_test']['x'].detach().numpy()))
    # y = torch.tensor(ori_label)
    # print('y:{}'.format(y.size(0)))
    # print('x:{}'.format(x.size(0)))
    # data_set = torch.utils.data.TensorDataset(x, y)

    data_set_o = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        clean_image_pre_transform  = None,
        bd_image_pre_transform = None,
        bd_label_pre_transform = None,
        end_pre_process = None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )

    data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=128, num_workers=4, shuffle=False)
    robust_acc = 0
    for i, (inputs, labels) in enumerate(data_loader):  # type: ignore
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        pre_label = torch.max(outputs, dim=1)[1]
        robust_acc += torch.sum(pre_label == labels) / len(data_set_o)

    print('rc{}'.format(robust_acc))

    file_name = os.path.basename(save_path)
    df = pd.read_csv(f'{save_path}/attack_df_summary.csv', index_col=0)
    if not ('wanet' in file_name or 'input' in file_name):
        acc, asr = (df.loc['last', ['benign acc', 'ASR']]).values
    elif 'wanet' in file_name:
        acc, asr = (df.loc['last', ['acc_clean', 'acc_bd']]).values
    elif 'input' in file_name:
        acc, asr = (df.loc['last', ['test_avg_acc_clean', 'test_avg_acc_bd']]).values
    else:
        raise SystemError('Cannot decide which attack')

    return acc, asr, robust_acc.cpu().item()


def load_results_pratio(dataset,poison_rate):
    models = ['preactresnet18', 'vgg19', "efficientnet_b3", "mobilenet_v3_large", "densenet161"]
    attacks = ['badnet', 'blended', 'sig', 'ssba', 'wanet', 'inputaware']
    defenses = ['no defense', 'ac', 'fp', 'ft', 'abl', 'nad', 'spectral','dbd','nc','anp']
    result_file = "{}_0_{}.csv".format(dataset, poison_rate)
    

    
    with open(result_file, "w") as csvfile:
        writer = csv.writer(csvfile)
        heads = []
        for defense in defenses:
            heads += [defense + ' ca', defense + ' asr', defense + ' rc']
        df_head = ['Attack', 'Model'] + heads
        writer.writerow(df_head)

    for model in models:
        for attack in attacks:
            results_row = []
            results_row += [model, attack]
            result_folder = '{}_{}_{}_0_{}'.format(dataset, model, attack, poison_rate)
            for defense in defenses:
                if defense == 'no defense':
                    result_path = 'record/' + result_folder + '/attack_result.pt'
                    if os.path.exists(result_path):
                        acc, asr, ra = load_result_no(result_folder, dataset)
                    else:
                        acc, asr, ra = 'NA', 'NA', 'NA'
                        
                else:
                    result_path = 'record/' + result_folder + '/' + defense + '/defense_result.pt'
                    if os.path.exists(result_path):
                        acc, asr, ra = load_result_rc(result_folder, defense)
                    else:
                        acc, asr, ra = 'NA', 'NA', 'NA'

                results_row += [acc, asr, ra]

            with open(result_file, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(results_row)



def load_results():
    parser = argparse.ArgumentParser(description='load results.')
    parser.add_argument('--dataset', type=str,default='cifar10')
    parser.add_argument('--pratio', type=str,default='all',help='the poison rate ')
    args = parser.parse_args()

    dataset = args.dataset

    pratio=args.pratio
    if pratio=='all':
        for pr in ["1","01","001","05","005"]:
            load_results_pratio(dataset,pr)
    else:
        load_results_pratio(dataset,pratio)
    
    
    

if __name__ == "__main__":
    load_results()


