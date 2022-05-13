'''
This file implements the defense method called finetuning (ft), which is a standard fine-tuning that uses clean data to finetune the model.

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. ft defense:
        a. get some clean data
        b. retrain the backdoor model
    4. test the result and get ASR, ACC, RC 
'''

import argparse
import logging
import os
import random
import sys


sys.path.append('../')
sys.path.append(os.getcwd())
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler

from defense.anp import anp_model
#from utils import args
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random 
from utils.aggregate_block.dataset_and_transform_generate import get_transform
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
#from utils.input_aware_utils import progress_bar
from utils.nCHW_nHWC import nCHW_to_nHWC
from utils.save_load_attack import load_attack_result
import yaml
from pprint import pprint, pformat

# mask step

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


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, anp_model.NoisyBatchNorm2d) or isinstance(module, anp_model.NoisyBatchNorm1d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, anp_model.NoisyBatchNorm2d) or isinstance(module, anp_model.NoisyBatchNorm1d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, anp_model.NoisyBatchNorm2d) or isinstance(module, anp_model.NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=args.anp_eps)


def mask_train(args, model, criterion, mask_opt, noise_opt, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
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
        for i, (images, labels) in enumerate(data_loader):
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
        from anp_model.preact_anp import PreActResNet18
        net = PreActResNet18(num_classes = num_classes, **kwargs)
    elif model_name == 'vgg19':
        net = anp_model.vgg_anp.vgg19_bn(num_classes = num_classes,  **kwargs)
    elif model_name == 'densenet161':
        net = anp_model.den_anp.densenet161(num_classes= num_classes, **kwargs)
    elif model_name == 'mobilenet_v3_large':
        net = anp_model.mobilenet_anp.mobilenet_v3_large(num_classes= num_classes, **kwargs)
    elif model_name == 'efficientnet_b3':
        net = anp_model.eff_anp.efficientnet_b3(num_classes= num_classes, **kwargs)
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')

    return net

# prune

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


def evaluate_by_number(args, model, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader, best_dis):
    results = []
    nb_max = int(np.ceil(pruning_max))
    nb_step = int(np.ceil(pruning_step))
    model_best = copy.deepcopy(model)
    for start in range(0, nb_max + 1, nb_step):
        i = start
        for i in range(start, start + nb_step):
            pruning(model, mask_values[i])
        layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
        cl_loss, cl_acc = test(args, model=model, criterion=criterion, data_loader=clean_loader)
        po_loss, po_acc = test(args, model=model, criterion=criterion, data_loader=poison_loader)
        logging.info('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
        results.append('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
        if cl_acc - po_acc > best_dis:
            model_best = copy.deepcopy(model)
            best_dis = cl_acc - po_acc
            logging.info('update model best')
    return results, model_best


def evaluate_by_threshold(args, model, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader, best_dis):
    results = []
    thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
    start = 0
    model_best = copy.deepcopy(model)
    for threshold in thresholds:
        idx = start
        for idx in range(start, len(mask_values)):
            if float(mask_values[idx][2]) <= threshold:
                pruning(model, mask_values[idx])
                start += 1
            else:
                break
        layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
        cl_loss, cl_acc = test(args, model=model, criterion=criterion, data_loader=clean_loader)
        po_loss, po_acc = test(args, model=model, criterion=criterion, data_loader=poison_loader)
        logging.info('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
        results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
            start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
        if cl_acc - po_acc > best_dis:
            model_best = copy.deepcopy(model)
            best_dis = cl_acc - po_acc
            logging.info('update model best')
    return results, model_best



def get_args():
    #set the basic parameter
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

    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--seed', type=str, help='random seed')
    parser.add_argument('--index', type=str, help='index of clean data')
    parser.add_argument('--result_file', type=str, help='the location of result')

    #set the parameter for the anp defense
    parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
    parser.add_argument('--print_every', type=int, help='print results every few iterations')
    parser.add_argument('--nb_iter', type=int, help='the number of iterations for training')

    parser.add_argument('--anp_eps', type=float)
    parser.add_argument('--anp_steps', type=int)
    parser.add_argument('--anp_alpha', type=float)

    # parser.add_argument('--arch', type=str, default='resnet18',
    #                 choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn'])
    # parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be pruned')
    # parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
    # parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
    # parser.add_argument('--lr', type=float, default=0.2, help='the learning rate for mask optimization')
    # parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
    # parser.add_argument('--val-frac', type=float, default=0.01, help='The fraction of the validate set')
    # parser.add_argument('--output-dir', type=str, default='logs/models/')

    parser.add_argument('--pruning_by', type=str, default='threshold', choices=['number', 'threshold'])
    parser.add_argument('--pruning_max', type=float, default=0.90, help='the maximum number/threshold for pruning')
    parser.add_argument('--pruning_step', type=float, default=0.05, help='the step size for evaluating the pruning')
    # parser.add_argument('--mask-file', type=str, required=True, help='The text file containing the mask values')

    arg = parser.parse_args()

    print(arg)
    return arg

def anp(args,result,config):
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

    fix_random(args.seed)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    x = result['clean_train']['x']
    y = result['clean_train']['y']
    data_all_length = len(y)
    ran_idx = choose_index(args, data_all_length) 
    log_index = os.getcwd() + args.log + 'index.txt'
    np.savetxt(log_index, ran_idx, fmt='%d')
    data_set = list(zip([x[ii] for ii in ran_idx],[y[ii] for ii in ran_idx]))
    data_set_clean = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_set,
        poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    random_sampler = RandomSampler(data_source=data_set_clean, replacement=True,
                                   num_samples=args.print_every * args.batch_size)
    clean_val_loader = DataLoader(data_set_clean, batch_size=args.batch_size,
                                  shuffle=False, sampler=random_sampler, num_workers=0)
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    y = result['bd_test']['y']
    data_bd_test = list(zip(x,y))
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    poison_test_loader = DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_clean_test = list(zip(x,y))
    data_clean_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_test,
        poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    clean_test_loader = DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    # state_dict = torch.load(args.checkpoint, map_location=args.device)
    state_dict = result['model']
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
    save_mask_scores(net.state_dict(), os.path.join(os.getcwd() + args.checkpoint_save, 'mask_values.txt'))

    net_prune = generate_cls_model(args.model,args.num_classes)
    net_prune.load_state_dict(result['model'])
    net_prune.to(args.device)

    mask_values = read_data(os.getcwd() + args.checkpoint_save + 'mask_values.txt')
    mask_values = sorted(mask_values, key=lambda x: float(x[2]))
    logging.info('No. \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    cl_loss, cl_acc = test(args, model=net_prune, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(args, model=net_prune, criterion=criterion, data_loader=poison_test_loader)
    logging.info('0 \t None     \t None     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
    best_dis = cl_acc - po_acc
    if args.pruning_by == 'threshold':
        results, model_pru = evaluate_by_threshold(
            args, net_prune, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
            criterion=criterion, clean_loader=clean_test_loader, poison_loader=poison_test_loader, best_dis = best_dis
        )
    else:
        results, model_pru = evaluate_by_number(
            args, net_prune, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
            criterion=criterion, clean_loader=clean_test_loader, poison_loader=poison_test_loader, best_dis = best_dis
        )
    file_name = os.path.join(os.getcwd() + args.checkpoint_save, 'pruning_by_{}.txt'.format(args.pruning_by))
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
        f.writelines(results)

    result = {}
    result['model'] = model_pru
    return result


if __name__ == '__main__':
    ### 1. basic setting: args
    args = get_args()
    with open("./defense/anp/config.yaml", 'r') as stream: 
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
    elif args.dataset == "cifar100":
        args.num_classes = 100
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
    
    save_path = '/record/' + args.result_file
    if args.checkpoint_save is None:
        args.checkpoint_save = save_path + '/record/defence/anp/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = save_path + '/saved/anp/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log)  
    args.save_path = save_path

    ### 2. attack result(model, train data, test data)
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    
    print("Continue training...")
    ### 3. anp defense:
    result_defense = anp(args,result,config)

    ### 4. test the result and get ASR, ACC, RC 
    result_defense['model'].eval()
    result_defense['model'].to(args.device)
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    y = result['bd_test']['y']
    data_bd_test = list(zip(x,y))
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
        asr_acc += torch.sum(pre_label == labels)
    asr_acc = asr_acc/len(data_bd_test)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_clean_test = list(zip(x,y))
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
        clean_acc += torch.sum(pre_label == labels)
    clean_acc = clean_acc/len(data_clean_test)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    robust_acc = -1
    if 'original_targets' in result['bd_test']:
        y_ori = result['bd_test']['original_targets']
        if y_ori is not None:
            if len(y_ori) != len(x):
                y_idx = result['bd_test']['original_index']
                y = y_ori[y_idx]
            else :
                y = y_ori
            data_bd_test = list(zip(x,y))
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
        
            robust_acc = 0
            for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = result_defense['model'](inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                robust_acc += torch.sum(pre_label == labels)
            robust_acc = robust_acc/len(data_bd_test)

    if not (os.path.exists(os.getcwd() + f'{save_path}/anp/')):
        os.makedirs(os.getcwd() + f'{save_path}/anp/')
    torch.save(
    {
        'model_name':args.model,
        'model': result_defense['model'].cpu().state_dict(),
        'asr': asr_acc,
        'acc': clean_acc,
        'ra': robust_acc
    },
    os.getcwd() + f'{save_path}/anp/defense_result.pt'
    )
