

import argparse
import copy
import os,sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from tqdm import tqdm

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense
from torch.utils.data import DataLoader, RandomSampler, random_split
from collections import OrderedDict

from utils.aggregate_block.train_settings_generate import argparser_criterion
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, general_plot_for_epoch
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2
from utils.choose_index import choose_index
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler

method_name = 'ucn-br'
reinit_number_range = [450, 550, 650,750, 850, 950, 1050, 1150, 1250]

def zero_reinit(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    try:
        bias_name = '{}.{}'.format(neuron[0], 'bias')
        state_dict[bias_name][int(neuron[1])] = 0.0
    except:
        pass
    net.load_state_dict(state_dict)

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    values = list(zip(layer, idx, value))
    return values


def get_layerName_from_type(model, layer_type):
    if layer_type == 'conv':
        instance_name = nn.Conv2d
    elif layer_type == 'bn':
        instance_name = nn.BatchNorm2d
    else:
        raise SystemError('NO valid layer_type match!')
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, instance_name):
            layer_names.append(name)
    return layer_names

def name_drop_last(name):
    return '.'.join(name.split('.')[:-1])

class ucn_br(defense):

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

        self.policyLoss = nn.CrossEntropyLoss(reduction='none')

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr_un', type=float)
        parser.add_argument('--lr_ft', type=float)
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
        parser.add_argument('--yaml_path', type=str, default="./config/defense/ucn-br/config.yaml", help='the path of yaml')
        parser.add_argument('--layer_type', type=str, help='the type of layer for reinitialization')
        parser.add_argument('--reinit_num', type=int)
        parser.add_argument('--ft_epoch', type=int)

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + f'/defense/{method_name}/'
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
        self.device = self.args.device


    def train_unlearning(self, args, model, criterion, optimizer, data_loader):
        model.train()
        total_correct = 0
        total_loss = 0.0
        pbar = tqdm(data_loader)
        for i, (images, labels, *additional_info)in enumerate(pbar):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)

            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            (-loss).backward()
            optimizer.step()
            pbar.set_description("Loss: "+str(loss))

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc

    def train_finetuning(self, args, model, criterion, optimizer, data_loader):
        model.train()
        total_correct = 0
        total_loss = 0.0
        nb_samples = 0
        for i, (images, labels, *additional_info) in enumerate(data_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            nb_samples += images.size(0)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)

            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / nb_samples
        return loss, acc
        
    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        #load clean val set
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

        clean_val_loader = DataLoader(data_set_clean, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)

        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
        test_dataloader_dict["bd_test_dataloader"] = data_bd_loader

        criterion = argparser_criterion(args)
        
        model_ori = generate_cls_model(self.args.model,self.args.num_classes)
        model_ori.load_state_dict(self.result['model'])
        model_ori = model_ori.to(args.device)

        target_layers = get_layerName_from_type(model_ori, args.layer_type)

        parameters_o = list(model_ori.named_parameters())
        params_o = {'names':[n for n, v in parameters_o if name_drop_last(n) in target_layers and 'weight' in n],
                        'params':[v for n, v in parameters_o if name_drop_last(n) in target_layers and 'weight' in n]}
        
        do_unlearn = True
        if do_unlearn:
            model = copy.deepcopy(model_ori)

            parameters = list(model.named_parameters())
            unlearn_optimizer = torch.optim.SGD([v for n, v in parameters if name_drop_last(n) in target_layers and 'weight' in n], lr=args.lr_un, momentum=0.9)

            logging.info("Unlearning...")
            for i, epoch in enumerate(range(args.epochs)):
                train_loss, train_acc = self.train_unlearning(args, model, criterion, unlearn_optimizer, clean_val_loader)
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
                    save_prefix = 'unlearn',

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
                logging.info(f"{i} Test unlearned model: acc_{test_acc}, asr_{test_asr}, ra_{test_ra}")
                if test_acc < 0.10:
                    logging.info(f"Break unlearn.")
                    break

            parameters_u = list(model.named_parameters())
            params_u = {'names':[n for n, v in parameters_u if name_drop_last(n) in target_layers and 'weight' in n],
                            'params':[v for n, v in parameters_u if name_drop_last(n) in target_layers and 'weight' in n]}

            
            changed_values = []
            count = 0
            for layer_i in range(len(params_u['params'])):
                name_i  = params_u['names'][layer_i]
                changed_params_i = params_u['params'][layer_i] - params_o['params'][layer_i]
                changed_params_i = changed_params_i.view(changed_params_i.shape[0], -1).abs().sum(dim=-1)
                for idx in range(changed_params_i.size(0)):
                    neuron_name = '.'.join(name_i.split('.')[:-1])
                    changed_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, changed_params_i[idx].item()))
                    count += 1
            with open(os.path.join(args.checkpoint_save, f'ucn.txt'), "w") as f:
                f.write('No \t Layer_Name \t Neuron_Idx \t Score \n')
                f.writelines(changed_values)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_save, 'unlearned_model.pt'))

        
        # ==================
        max2min = True
        changed_values = read_data(args.checkpoint_save + f'ucn.txt')
        changed_values = sorted(changed_values, key=lambda x: float(x[2]), reverse=max2min)

        agg = Metric_Aggregator()
        ft_agg = Metric_Aggregator()
        logging.info("Reinitializing...")
        
        reinit_number_range.append(args.reinit_num)

        for top_num in reinit_number_range:
            model_copy = copy.deepcopy(model_ori)
            for i in range(top_num):
                zero_reinit(model_copy, changed_values[i])

            self.set_trainer(model_copy)
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
                save_prefix = 'reinitialize',

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
            logging.info(f"Test reinitialized model: acc_{test_acc}, asr_{test_asr}, ra_{test_ra}")

            agg({
                    "top_num": top_num,
                    "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                    "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                    "test_acc": test_acc,
                    "test_asr": test_asr,
                    "test_ra": test_ra,
                })
            agg.to_dataframe().to_csv(f"{args.save_path}unlearn_df.csv")

            is_finetune = True
            if is_finetune:
                logging.info("Fine Tuning...")

                update_neuron_params_optimizer = torch.optim.SGD(model_copy.parameters(), lr=args.lr_ft, momentum=0.9)

                for epoch in range(args.ft_epoch+1):
                    self.train_finetuning(args, model_copy, criterion, update_neuron_params_optimizer, clean_val_loader)
                    
                if top_num == args.reinit_num:
                    model_result = copy.deepcopy(model_copy)
                
                self.set_trainer(model_copy)
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
                    save_prefix = 'finetune',

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
                logging.info(f"Test finetuned model: acc_{test_acc}, asr_{test_asr}, ra_{test_ra}")
        
                ft_agg({
                        "finetune_epoch":epoch,
                        "prune_num": top_num,
                        "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                        "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                        "test_acc": test_acc,
                        "test_asr": test_asr,
                        "test_ra": test_ra,
                    })
                ft_agg.to_dataframe().to_csv(f"{args.save_path}finetune_df.csv")
        
        agg.summary().to_csv(f"{args.save_path}unlearn_df_summary.csv")
        ft_agg.summary().to_csv(f"{args.save_path}finetune_df_summary.csv")
        result = {}
        result['model'] = model_result
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model_result.cpu().state_dict(),
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
    ucn_br.add_arguments(parser)
    args = parser.parse_args()
    method = ucn_br(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = method.defense(args.result_file)