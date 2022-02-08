import sys
sys.path.append('../')
'''
logic of load:
1. yaml file, if yaml setting name is given then find the yaml setting
2.
3. argparse overwrite args from yaml file if any in args is not None
(so ANY params in add_args should have NO default value except yaml config and yaml setting name)
4. delete any params in args with value None
'''

import argparse
import logging
import os
import sys
from pprint import pprint, pformat
import random
import numpy as np
import torch

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    # parser.add_argument('--mode', type=str,
    #                     help='classification/detection/segmentation')

    parser.add_argument('--lambda_similar', type = float,
                        help = 'only use in contrastive case, the coef of similar term')
    parser.add_argument('--cos_abs', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help='only use in contrastive case, whether add a abs to cos similarity')
    parser.add_argument('--cos_final_lower_bound',type = float,
                        help = 'only use in contrastive case, whether add a lower bound to cos similarity after all operation'
                        )
    parser.add_argument(
        '--lower_bound_mode', type = str, # in or de for increase or decrease
        help = 'in case you have cos_final_lower_bound, then this will control the bound in each epoch'
    )
    parser.add_argument('--lr_scheduler', type = str,
                        help = 'which lr_scheduler use for optimizer')

    parser.add_argument('--yaml_path', type=str, default='../config/settings.yaml',
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--yaml_setting_name', type=str, default='default',
                        help='In case yaml file contains several groups of default settings, get the one with input name',
                        )

    parser.add_argument('--additional_yaml_path',  type = str, default = '../config/blocks.yaml',
                        help = 'this file should contrains additional blocks of params',
                        )

    parser.add_argument('--additional_yaml_blocks_names', nargs='*', type = str,
                        help = 'names of additional yaml blocks will be used')


    parser.add_argument('--attack_label_trans', type = str,
        help = 'which type of label modification in backdoor attack'
    )
    parser.add_argument('--pratio', type = float,
        help = 'the poison rate '
    )
    parser.add_argument('--dataset', type = str,
                        help = 'which dataset to use'
    )
    parser.add_argument('--attack', type=str,
                        help='which attack used')
    parser.add_argument('--attack_target', type=int,
                        help='target class in all2one attack')
    parser.add_argument('--blended_alpha', type=float,
                        help='alpha for blended')
    parser.add_argument('--random_seed', type=int,
                        help='random_seed')
    parser.add_argument('--load_path', type=str,
                        help='load_path used in load model')
    parser.add_argument('--recover',
                        default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help='when load_path is applied, 2 case, recover training or do finetune/...')
    parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')
    parser.add_argument('--model', type=str,
                        help='choose which kind of model')
    parser.add_argument('--save_folder_name', type=str,
                        help='(Optional) should be time str + given unique identification str')
    parser.add_argument('--git_hash', type=str,
                        help='git hash number, in order to find which version of code is used')
    parser.add_argument(
        '--flooding_scalar', type = float,
        help = 'flooding scalar used in the training process of flooding',
    )
    return parser

from utils.argparse_with_yaml import load_yamls_into_args
parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
args = parser.parse_args()

args = load_yamls_into_args(args)

args.terminal_info = sys.argv

from utils.aggregate_block.save_path_generate import generate_save_folder

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

import time
import logging
# logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logFormatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
)
logger = logging.getLogger()
# logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")

fileHandler = logging.FileHandler(save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

logger.setLevel(logging.INFO)
logging.info(pformat(args.__dict__))

try:
    import wandb
    wandb.init(
        project="bdzoo2",
        entity="chr",
        name=('afterwards' if 'load_path' in args.__dict__ else 'attack') + '_' + os.path.basename(save_path),
        config=args,
    )
    set_wandb = True
except:
    set_wandb = False
logging.info(f'set_wandb = {set_wandb}')

import torchvision.transforms as transforms
from utils.aggregate_block.fix_random import fix_random
fix_random(int(args.random_seed))

from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate

train_dataset_without_transform, \
            train_img_transform, \
            train_label_transfrom, \
test_dataset_without_transform, \
            test_img_transform, \
            test_label_transform = dataset_and_transform_generate(args)

from utils.bd_dataset import prepro_cls_DatasetBD
from torch.utils.data import DataLoader

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

from utils.backdoor_generate_pindex import generate_pidx_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate

train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)

bd_label_transform = bd_attack_label_trans_generate(args)

from copy import deepcopy

train_pidx = generate_pidx_from_label_transform(
    benign_train_dl.dataset.targets,
    label_transform=bd_label_transform,
    is_train=True,
    pratio= args.pratio if 'pratio' in args.__dict__ else None,
    p_num= args.p_num if 'p_num' in args.__dict__ else None,
)
torch.save(train_pidx,
    args.save_path + '/train_pidex_list.pickle',
)

adv_train_ds = prepro_cls_DatasetBD(
    deepcopy(train_dataset_without_transform),
    poison_idx= train_pidx,
    bd_image_pre_transform=train_bd_img_transform,
    bd_label_pre_transform=bd_label_transform,
    ori_image_transform_in_loading=train_img_transform,
    ori_label_transform_in_loading=train_label_transfrom,
    add_details_in_preprocess=True,
)

adv_train_dl = DataLoader(
    dataset = adv_train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)

test_pidx = generate_pidx_from_label_transform(
    benign_test_dl.dataset.targets,
    label_transform=bd_label_transform,
    is_train=False,
)

adv_test_dataset = prepro_cls_DatasetBD(
    deepcopy(test_dataset_without_transform),
    poison_idx=test_pidx,
    bd_image_pre_transform=test_bd_img_transform,
    bd_label_pre_transform=bd_label_transform,
    ori_image_transform_in_loading=test_img_transform,
    ori_label_transform_in_loading=test_label_transform,
    add_details_in_preprocess=True,
)

adv_test_dataset.subset(
    np.where(test_pidx == 1)[0]
)

adv_test_dl = DataLoader(
    dataset = adv_test_dataset,
    batch_size= args.batch_size,
    shuffle= False,
    drop_last= False,
)

from utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

net  = generate_cls_model(
    model_name=args.model,
    num_classes=args.num_classes,
)

trainer = generate_cls_trainer(
    net,
    args.attack
)

from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion

criterion = argparser_criterion(args)

optimizer, scheduler = argparser_opt_scheduler(net, args)


if __name__ == '__main__':

    if 'load_path' not in args.__dict__:

        trainer.train_with_test_each_epoch(
            train_data = adv_train_dl,
            test_data = benign_test_dl,
            adv_test_data = adv_test_dl,
            end_epoch_num = args.epochs,
            criterion = criterion,
            optimizer = optimizer,
            scheduler = scheduler,
            device = device,
            frequency_save = args.frequency_save,
            save_folder_path = save_path,
            save_prefix = 'attack',
            continue_training_path = None,
        )

    else:

        if 'recover' not in args.__dict__ or args.recover == False :

            print('finetune so use less data, 5% of benign train data')

            benign_train_dl.dataset.subset(
                np.random.choice(
                    np.arange(
                        len(benign_train_dl.dataset)),
                    size=round((len(benign_train_dl.dataset)) / 20),  # 0.05
                    replace=False,
                )
            )

            torch.save(
                list(benign_train_dl.dataset.original_index),
                args.save_path + '/finetune_idx_list.pt',
            )

            trainer.train_with_test_each_epoch(
                train_data=benign_train_dl,
                test_data=benign_test_dl,
                adv_test_data=adv_test_dl,
                end_epoch_num=args.epochs,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                frequency_save=args.frequency_save,
                save_folder_path=save_path,
                save_prefix='finetune',
                continue_training_path=args.load_path,
                only_load_model=True,
            )

        elif 'recover' in args.__dict__ and args.recover == True :

            trainer.train_with_test_each_epoch(
                train_data=adv_train_dl,
                test_data=benign_test_dl,
                adv_test_data=adv_test_dl,
                end_epoch_num=args.epochs,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                frequency_save=args.frequency_save,
                save_folder_path=save_path,
                save_prefix='attack',
                continue_training_path=args.load_path,
                only_load_model=False,
            )

adv_train_ds.save(save_path+'/adv_train_ds.pth', only_bd = True)
adv_test_dataset.save(save_path+'/adv_test_dataset.pth', only_bd = True)