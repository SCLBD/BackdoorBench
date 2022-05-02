'''
This script is for normal training process, no any attack is applied
'''

import sys, yaml, os

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
from pprint import  pformat
import numpy as np
import torch
import time
import logging

from utils.aggregate_block.save_path_generate import generate_save_folder
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
from utils.bd_dataset import prepro_cls_DatasetBD
from torch.utils.data import DataLoader
from utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion



def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings

    parser.add_argument('--device', type = str)
    parser.add_argument('--yaml_path', type=str, default='./default.yaml',
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--lr_scheduler', type=str,
                        help='which lr_scheduler use for optimizer')
    # only all2one can be use for clean-label
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dataset', type=str,
                        help='which dataset to use'
                        )
    parser.add_argument('--dataset_path', type=str)

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--steplr_stepsize', type=int)
    parser.add_argument('--steplr_gamma', type=float)
    parser.add_argument('--sgd_momentum', type=float)
    parser.add_argument('--wd', type=float, help='weight decay of sgd')
    parser.add_argument('--steplr_milestones', type=list)
    parser.add_argument('--client_optimizer', type=int)
    parser.add_argument('--random_seed', type=int,
                        help='random_seed')
    parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')
    parser.add_argument('--model', type=str,
                        help='choose which kind of model')
    parser.add_argument('--save_folder_name', type=str,
                        help='(Optional) should be time str + given unique identification str')
    parser.add_argument('--git_hash', type=str,
                        help='git hash number, in order to find which version of code is used')
    return parser

def main():

    ### 1. config args, save_path, fix random seed
    parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)

    defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

    args.__dict__ = defaults

    args.terminal_info = sys.argv

    args.attack = 'None'

    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"

    ### save path
    if 'save_folder_name' not in args:
        save_path = generate_save_folder(
            run_info=('afterwards' if 'load_path' in args.__dict__ else 'attack') + '_' + args.attack,
            given_load_file_path=args.load_path if 'load_path' in args else None,
            all_record_folder_path='../record',
        )
    else:
        save_path = '../record/' + args.save_folder_name
        os.mkdir(save_path)

    args.save_path = save_path

    torch.save(args.__dict__, save_path + '/info.pickle')

    ### set the logger
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()

    fileHandler = logging.FileHandler(save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))

    ### set the random seed
    fix_random(int(args.random_seed))

    ### 2. set the clean train data and clean test data
    train_dataset_without_transform, \
                train_img_transform, \
                train_label_transfrom, \
    test_dataset_without_transform, \
                test_img_transform, \
                test_label_transform = dataset_and_transform_generate(args)

    benign_train_dl = DataLoader(
        prepro_cls_DatasetBD(
            full_dataset_without_transform=train_dataset_without_transform,
            poison_idx=np.zeros(len(train_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=train_img_transform,
            ori_label_transform_in_loading=train_label_transfrom,
            add_details_in_preprocess=True,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    benign_test_dl = DataLoader(
        prepro_cls_DatasetBD(
            test_dataset_without_transform,
            poison_idx=np.zeros(len(test_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=test_img_transform,
            ori_label_transform_in_loading=test_label_transform,
            add_details_in_preprocess=True,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    net  = generate_cls_model(
        model_name=args.model,
        num_classes=args.num_classes,
    )

    trainer = generate_cls_trainer(
        net,
        args.attack
    )

    criterion = argparser_criterion(args)

    optimizer, scheduler = argparser_opt_scheduler(net, args)

    trainer.train_with_test_each_epoch_v2(
        train_data=benign_train_dl,
        test_dataloader_dict={'benign_test_dl':benign_test_dl},
        end_epoch_num=args.epochs,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        frequency_save=args.frequency_save,
        save_folder_path=save_path,
        save_prefix='attack',
        continue_training_path=None,
    )

if __name__ == '__main__':
    main()