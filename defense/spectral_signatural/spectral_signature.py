
'''
@article{tran2018spectral,
  title={Spectral signatures in backdoor attacks},
  author={Tran, Brandon and Li, Jerry and Madry, Aleksander},
  journal={Advances in neural information processing systems},
  volume={31},
  year={2018}
}

code : https://github.com/MadryLab/backdoor_data_poisoning
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
from datetime import datetime
import json
import math
from pyexpat import model
import shutil
import sys
import os
import time
sys.path.append('../')
sys.path.append(os.getcwd())
from timeit import default_timer as timer

import numpy as np
import torch
#import tensorflow as tf
#import tensorflow.contrib.slim as slim
from tqdm import trange

#import dataset_input
#from eval import evaluate 
#import resnet
import yaml
from utils.aggregate_block.dataset_and_transform_generate import get_transform

from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import nCHW_to_nHWC
from utils.save_load_attack import load_attack_result
#from utils.preact_resnet import get_activation
from pprint import pprint, pformat

def compute_corr_v1(arg,result,config):
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    # logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    if args.log is not None and args.log != '':
        fileHandler = logging.FileHandler(os.getcwd() + args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    else:
        fileHandler = logging.FileHandler(os.getcwd() + './log' + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))
    model = generate_cls_model(arg.model,arg.num_classes)
    model.load_state_dict(result['model'])
    model.to(arg.device)
    model.eval()

    # Setting up the data and the model
    target_label = arg.target_label
    #dataset = dataset_input.CIFAR10Data(config,
    #                                    seed=config.training.np_random_seed)
    tran = get_transform(arg.dataset, *([arg.input_height,arg.input_width]) , train = True)
    x = torch.tensor(nCHW_to_nHWC(result['bd_train']['x'].numpy()))
    y = result['bd_train']['y']
    data_set = torch.utils.data.TensorDataset(x,y)
    data_set_o = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    
    dataset = data_set_o
    ####为了看现在dataset里面还有多少剩余数据(在去除了backdoor数据，和预设的荼毒比例，为了之后测试用，我们用预设的poison_rate来测试)
    #num_poisoned_left = dataset.num_poisoned_left
    num_poisoned_left = int(len(dataset)*arg.poison_rate_test)
    print('Num poisoned left: ', num_poisoned_left)
    #num_training_examples = len(dataset[0])
    #global_step = tf.contrib.framework.get_or_create_global_step()
    
    ####model输入进来，但是没改representation后续可能会出现问题！！！！！！！！！！！
    #model = resnet.Model(config.model)

    # Setting up the Tensorboard and checkpoint outputs
    #model_dir = config.model.output_dir
    #saver = tf.train.Saver(max_to_keep=3)

    # initialize data augmentation
    print('Dataset Size: ', len(dataset))

    ####应该是提取最后的一部分的model，我们之前提取完了
    #sess.run(tf.global_variables_initializer())
    #latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    #if latest_checkpoint is not None:
    #    saver.restore(sess, latest_checkpoint)
    #    print('Restoring last saved checkpoint: ', latest_checkpoint)
    #else:
    #    print('Check model directory')
    #    exit()

    lbl = target_label
    dataset_y = []
    for i in range(len(dataset)):
        dataset_y.append(dataset[i][1])
    cur_indices = [i for i,v in enumerate(dataset_y) if v==lbl]
    cur_examples = len(cur_indices)
    print('Label, num ex: ', lbl, cur_examples)
    #########
    #cur_op = model.representation
    for iex in trange(cur_examples):
        cur_im = cur_indices[iex]
        x_batch = dataset[cur_im][0].unsqueeze(0).to(arg.device)
        y_batch = dataset[cur_im][1]

        #dict_nat = {model.x_input: x_batch,
        #            model.y_input: y_batch,
        #            model.is_training: False}
        #######得在原有模型基础上加入representation！！！！
        assert arg.model in ['resnet18']
        if arg.model == 'resnet18':
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            hook = model.layer4.register_forward_hook(layer_hook)
            _ = model(x_batch)
            batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
            hook.remove()

        #batch_grads = sess.run(cur_op, feed_dict=dict_nat)
        if iex==0:
            clean_cov = np.zeros(shape=(cur_examples-num_poisoned_left, len(batch_grads)))
            full_cov = np.zeros(shape=(cur_examples, len(batch_grads)))
        if iex < (cur_examples-num_poisoned_left):
            clean_cov[iex]= batch_grads.detach().cpu().numpy()
        full_cov[iex] = batch_grads.detach().cpu().numpy()

    #np.save(corr_dir+str(lbl)+'_full_cov.npy', full_cov)
    total_p = arg.percentile            
    clean_mean = np.mean(clean_cov, axis=0, keepdims=True)
    full_mean = np.mean(full_cov, axis=0, keepdims=True)            

    print('Norm of Difference in Mean: ', np.linalg.norm(clean_mean-full_mean))
    clean_centered_cov = clean_cov - clean_mean
    s_clean = np.linalg.svd(clean_centered_cov, full_matrices=False, compute_uv=False)
    print('Top 7 Clean SVs: ', s_clean[0:7])
    
    centered_cov = full_cov - full_mean
    u,s,v = np.linalg.svd(centered_cov, full_matrices=False)
    print('Top 7 Singular Values: ', s[0:7])
    eigs = v[0:1]  
    p = total_p
    corrs = np.matmul(eigs, np.transpose(full_cov)) #shape num_top, num_active_indices
    scores = np.linalg.norm(corrs, axis=0) #shape num_active_indices
    print('Length Scores: ', len(scores))
    p_score = np.percentile(scores, p)
    top_scores = np.where(scores>p_score)[0]
    print(top_scores)
    num_bad_removed = np.count_nonzero(top_scores>=(len(scores)-num_poisoned_left))
    print('Num Bad Removed: ', num_bad_removed)
    print('Num Good Rmoved: ', len(top_scores)-num_bad_removed)
    
    num_poisoned_after = num_poisoned_left - num_bad_removed
    removed_inds = np.copy(top_scores)
    re = [cur_indices[v] for i,v in enumerate(removed_inds)]
    left_inds = np.delete(range(len(dataset)), re)
    
    #####直接返回一个dataset就可以了，不需要记录
    #removed_inds_file = os.path.join(model_dir, 'removed_inds.npy')
    #np.save(removed_inds_file, cur_indices[removed_inds])        
    print('Num Poisoned Left: ', num_poisoned_after)   
    

    ######创建一个dataset
    dataset.subset(left_inds)
    dataset_left = dataset
    data_loader_sie = torch.utils.data.DataLoader(dataset_left, batch_size=arg.batch_size, num_workers=arg.num_workers, shuffle=True)
    
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = torch.tensor(nCHW_to_nHWC(result['bd_test']['x'].numpy()))
    y = result['bd_test']['y']
    data_bd_test = torch.utils.data.TensorDataset(x,y)
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = torch.tensor(nCHW_to_nHWC(result['clean_test']['x'].numpy()))
    y = result['clean_test']['y']
    data_clean_test = torch.utils.data.TensorDataset(x,y)
    data_clean_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_test,
        poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    best_acc = 0
    best_asr = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100) 
    criterion = torch.nn.CrossEntropyLoss() 
    for j in range(arg.epochs):
        model.train()
        for i, (inputs,labels) in enumerate(data_loader_sie):  # type: ignore
            inputs, labels = inputs.to(arg.device), labels.to(arg.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            model.eval()
            asr_acc = 0
            for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                asr_acc += torch.sum(pre_label == labels)/len(data_bd_test)

            
            clean_acc = 0
            for i, (inputs,labels) in enumerate(data_clean_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                clean_acc += torch.sum(pre_label == labels)/len(data_clean_test)
        
        
        if best_acc < clean_acc:
            best_acc = clean_acc
            best_asr = asr_acc
            torch.save(
            {
                'model_name':args.model,
                'model': model.cpu().state_dict(),
                'asr': asr_acc,
                'acc': clean_acc
            },
            os.getcwd() + f'{args.checkpoint_save}/defense_result.pt'
            )
            model.to(arg.device)
        logging.info(f'Epoch{j}: clean_acc:{clean_acc} asr:{asr_acc} best_acc:{best_acc} best_asr{best_asr}')

    result = {}
    result["dataset"] = dataset_left
    result['model'] = model
    return result

    #if os.path.exists('job_result.json'):
    #    with open('job_result.json') as result_file:
    #        result = json.load(result_file)
    #        result['num_poisoned_left'] = '{}'.format(num_poisoned_after)
    #else:
    #    result = {'num_poisoned_left': '{}'.format(num_poisoned_after)}
    #with open('job_result.json', 'w') as result_file:
    #    json.dump(result, result_file, sort_keys=True, indent=4) 

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str)
    parser.add_argument('--checkpoint_save', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument("--data_root", type=str)

    parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny') 
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    parser.add_argument("--input_channel", type=int)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--num_workers", type=float)
    parser.add_argument('--lr', type=float)

    parser.add_argument('--attack', type=str)
    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)
    parser.add_argument('--trigger_type', type=str, help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    ####添加额外
    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--result_file', type=str, help='the location of result')

    ####spectral
    parser.add_argument('--poison_rate_test', type=float)
    parser.add_argument('--percentile', type=int)
    

    arg = parser.parse_args()

    print(arg)
    return arg

if __name__ == '__main__':
    
    args = get_args()
    with open("./defense/spectral_signatural/config/config.yaml", 'r') as stream: 
        config = yaml.safe_load(stream) 
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    if args.dataset == "mnist":
        args.num_classes = 10
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
    elif args.dataset == "cifar10":
        args.num_classes = 10
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "gtsrb":
        args.num_classes = 43
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "celeba":
        args.num_classes = 8
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    elif args.dataset == "tiny":
        args.num_classes = 200
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    #args.checkpoint_save = os.getcwd() + '/record/defence/ac/' + args.dataset + '.tar'
    
    #args.log = 'saved/log/log_' + args.dataset + '.txt'

    ######为了测试临时写的代码
    save_path = '/record/' + args.result_file
    if args.checkpoint_save is None:
        args.checkpoint_save = save_path + '/record/defence/spectral/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = save_path + '/saved/spectral/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log)  
    args.save_path = save_path
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    
    if args.save_path is not None:
        print("Continue training...")
        result_defense = compute_corr_v1(args,result,config)

        tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        x = torch.tensor(nCHW_to_nHWC(result['bd_test']['x'].numpy()))
        y = result['bd_test']['y']
        data_bd_test = torch.utils.data.TensorDataset(x,y)
        data_bd_testset = prepro_cls_DatasetBD(
            full_dataset_without_transform=data_bd_test,
            poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=tran,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    
        asr_acc = 0
        for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = result_defense['model'](inputs)
            pre_label = torch.max(outputs,dim=1)[1]
            asr_acc += torch.sum(pre_label == labels)/len(data_bd_test)

        tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        x = torch.tensor(nCHW_to_nHWC(result['clean_test']['x'].numpy()))
        y = result['clean_test']['y']
        data_clean_test = torch.utils.data.TensorDataset(x,y)
        data_clean_testset = prepro_cls_DatasetBD(
            full_dataset_without_transform=data_clean_test,
            poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=tran,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    
        clean_acc = 0
        for i, (inputs,labels) in enumerate(data_clean_loader):  # type: ignore
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = result_defense['model'](inputs)
            pre_label = torch.max(outputs,dim=1)[1]
            clean_acc += torch.sum(pre_label == labels)/len(data_clean_test)

        if not (os.path.exists(os.getcwd() + f'{save_path}/spectral/')):
            os.makedirs(os.getcwd() + f'{save_path}/spectral/')
        torch.save(
        {
            'model_name':args.model,
            'model': result_defense['model'].cpu().state_dict(),
            'asr': asr_acc,
            'acc': clean_acc
        },
        os.getcwd() + f'{save_path}/spectral/defense_result.pt'
    )
    else:
        print("There is no target model")
