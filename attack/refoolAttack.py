import sys
sys.path.append('../')
'''
rewrite from basicAttack since refool do not affect the training process.
some of settings in github of refool is different from the paper, we choose to use the settings in the paper.
'''

import argparse
import logging
import os
import sys
from pprint import pprint, pformat
import random
import numpy as np
import torch
import imageio
import yaml

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument('--img_r_seq_folder_path', type=str, #default='../resource/refool_reflections',
                        help='the folder path for reflection img (use in refool data poison)')
    parser.add_argument('--max_image_size', type=int, #default=560,
                        help = 'the max(height, width) of output')
    parser.add_argument('--ghost_rate', type=float, #default=0.49,
                        help='the probability that one img use ghost mode (other times may use out of focus mode) during data poison')
    parser.add_argument('--alpha_t', type=float, #default=-1.,
                        help='intensity number (ratio) of blend , when negative, pick from 1- U[0.05,0.45]')
    parser.add_argument('--offset', type=list, #default=(0,0),
                        help='padding to img in ghost mode. input (0,0) use default = (random.randint(3, 8), random.randint(3, 8))')
    parser.add_argument('--sigma', type=float, #default=-1. ,
                        help='sigma in 2-d gaussian of out of focus mode. input negative value then use default sigma = random.uniform(1, 5)')
    parser.add_argument('--ghost_alpha', type=float, #default=-1.,
                        help='the alpha in ghost mode, negative input may use default value, plz see refoolGhostEffectAttack')

    parser.add_argument('--yaml_path', type=str, default='../config/refoolAttack/default.yaml',
                        help='path for yaml file provide additional default attributes')

    parser.add_argument('--lr_scheduler', type=str,
                        help='which lr_scheduler use for optimizer')
    parser.add_argument('--attack_label_trans', type = str,
        help = 'which type of label modification in backdoor attack'
    )
    parser.add_argument('--pratio', type = float,
        help = 'the poison rate, Notice that here is poison rate inside the target class !!!!'
    )
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dataset', type = str,
                        help = 'which dataset to use'
    )
    parser.add_argument('--dataset_path', type = str)
    parser.add_argument('--attack_target', type=int,
                        help='target class in all2one attack')
    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--img_size', type=list)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--steplr_stepsize', type=int)
    parser.add_argument('--steplr_gamma', type=float)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--sgd_momentum', type=float)
    parser.add_argument('--wd', type=float, help = 'weight decay of sgd')

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


parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
args = parser.parse_args()

with open(args.yaml_path, 'r') as f:
    defaults = yaml.safe_load(f)

defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

args.__dict__ = defaults

args.attack = 'refool'

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

from utils.backdoor_generate_pindex import generate_pidx_from_label_transform, generate_single_target_attack_train_pidx
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate

# here we load the attack use reflections
args.img_r_seq = []
for reflection_img_name in os.listdir(args.img_r_seq_folder_path):
    reflection_img_path = f'{args.img_r_seq_folder_path}/{reflection_img_name}'
    args.img_r_seq.append(imageio.imread(reflection_img_path))

train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)

bd_test_bd_label_transform = bd_attack_label_trans_generate(args)

from copy import deepcopy

args.p_num = round((benign_train_dl.dataset.targets == args.attack_target).sum() * args.pratio)
print('Notice that here is the poison rate inside the target class ')

train_pidx = generate_single_target_attack_train_pidx(
    targets = benign_train_dl.dataset.targets,
    tlabel = int(args.attack_target),
    pratio= None,
    p_num=args.p_num,
    clean_label = True,
)

torch.save(train_pidx,
    args.save_path + '/train_pidex_list.pickle',
)

adv_train_ds = prepro_cls_DatasetBD(
    deepcopy(train_dataset_without_transform),
    poison_idx= train_pidx,
    bd_image_pre_transform=train_bd_img_transform,
    bd_label_pre_transform=None,
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
    label_transform=bd_test_bd_label_transform,
    is_train=False,
)

adv_test_dataset = prepro_cls_DatasetBD(
    deepcopy(test_dataset_without_transform),
    poison_idx=test_pidx,
    bd_image_pre_transform=test_bd_img_transform,
    bd_label_pre_transform=bd_test_bd_label_transform,
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

net = generate_cls_model(
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

adv_train_ds.save(save_path+'/adv_train_ds.pth', only_bd = True)
adv_test_dataset.save(save_path+'/adv_test_dataset.pth', only_bd = True)