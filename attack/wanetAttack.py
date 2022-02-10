'''
rewrite from
    @inproceedings{
    nguyen2021wanet,
    title={WaNet - Imperceptible Warping-based Backdoor Attack},
    author={Tuan Anh Nguyen and Anh Tuan Tran},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=eEn8KTtJOx}
    }
    code : https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release
'''

import sys, os, logging, yaml
os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

from utils.backdoor_generate_pindex import generate_single_target_attack_train_pidx

import torch
import numpy as np

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
    parser.add_argument('--warp_kernel_size', type = int, help = "warp kernel")
    parser.add_argument('--warping_strength', type = float, help = 'warping_strength')
    parser.add_argument('--grid_rescale', type = float, help = 'use to avoid goes out of (-1,1) for pixel value')
    parser.add_argument('--cross_ratio', type = float, help = 'noise sample num = total * pratio * cross_ratio')
    parser.add_argument('--random_crop', type = int, help = 'in train transform')
    parser.add_argument('--random_rotation', type=int, help = 'in train transform')

    parser.add_argument('--yaml_path', type=str, default='../config/wanetAttack/default.yaml',
                        help='path for yaml file provide additional default attributes')

    parser.add_argument('--lr_scheduler', type=str,
                        help='which lr_scheduler use for optimizer')
    parser.add_argument('--attack_label_trans', type = str,
        help = 'which type of label modification in backdoor attack'
    )
    parser.add_argument('--pratio', type = float,
        help = 'the poison rate '
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
    parser.add_argument('--steplr_milestones', type = list)
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

args.attack = 'wanet'

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
        ori_image_transform_in_loading=transforms.ToTensor(), # since we may inject  the poisiton during training
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

from utils.bd_img_transform.wanet import imageWarp

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

train_bd_transform = None,
#TODO
test_bd_transform = imageWarp(
    args.warp_kernel_size,
    args.img_size[0],
    args.warping_strength,
    args.grid_rescale,
    False,
    device=device,
)

bd_test_bd_label_transform = bd_attack_label_trans_generate(args)

test_pidx = generate_pidx_from_label_transform(
    benign_test_dl.dataset.targets,
    label_transform=bd_test_bd_label_transform,
    is_train=False,
)

from copy import deepcopy

adv_test_dataset = prepro_cls_DatasetBD(
    deepcopy(test_dataset_without_transform),
    poison_idx=test_pidx,
    bd_image_pre_transform=test_bd_transform,
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

net = generate_cls_model(
    model_name=args.model,
    num_classes=args.num_classes,
)

from utils.bd_trainer.wanet_trainer import wanetTrainerCLS

trainer = wanetTrainerCLS(net)

from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion

criterion = argparser_criterion(args)

optimizer, scheduler = argparser_opt_scheduler(net, args)

from PIL import Image
def npToPIL(array):
    return Image.fromarray(np.uint8((array)*255)).convert('RGB')

def wanet_batch_operation(
        x,
        labels,
        attack_mode,
        target_label,
        pratio,
        cross_ratio,
        bd_attack_trans,
        noise_trans,
        post_transform,
        num_classes,
        ):

    bs = x.shape[0]

    # Create backdoor data

    if attack_mode == "all2one": # rewrite, prevent the case that trigger put on target class
       # #TODO make sure whether use target class sample or not
       #  poison_position = list(np.random.choice(np.where(labels.numpy() != target_label)[0],
       #                   min(round(pratio * bs), (labels.numpy() != target_label).sum()),
       #                   replace = False))
       #  cross_position = list(np.random.choice(np.where(labels.numpy() == target_label)[0],
       #                   min(round(pratio * bs * cross_ratio), (labels.numpy() == target_label).sum()),
       #                   replace = False))
       #  clean_position = list((set(np.arange(bs)).difference(set(poison_position)).difference(set(cross_position))))
       #
       #  #TODO ugly implementation tensor to numpy to pil to np to tensor
       #
       #  x = torch.cat([
       #      post_transform(npToPIL(np_i.transpose((1,2,0))))[None,...] for np_i in torch.cat([
       #      x[clean_position],
       #      bd_attack_trans(x[poison_position]),
       #      noise_trans(x[cross_position]),
       #  ], dim = 0).numpy()
       #  ], dim = 0)
       #
       #
       #  labels = torch.cat([
       #      labels[clean_position],
       #      torch.ones_like(labels[poison_position]) * target_label,
       #      labels[cross_position],
       #  ], dim = 0)
       #
       #  return x, labels

       num_bd = int(bs * pratio)
       num_cross = int(num_bd * cross_ratio)

       inputs_bd = bd_attack_trans(x[: num_bd])

       inputs_cross = noise_trans(x[num_bd: (num_bd + num_cross)])

       total_inputs = torch.cat([inputs_bd, inputs_cross, x[(num_bd + num_cross):]], dim=0)

       total_inputs = torch.cat(
           [post_transform(npToPIL(np_i.transpose((1, 2, 0))))[None, ...] for np_i in total_inputs.numpy()], dim=0)

       targets_bd = torch.ones_like(labels[:num_bd]) * target_label
       total_targets = torch.cat([targets_bd, labels[num_bd:]], dim=0)

       return total_inputs, total_targets

    if attack_mode == "all2all": # from original code

        num_bd = int(bs * pratio)
        num_cross = int(num_bd * cross_ratio)

        inputs_bd = bd_attack_trans(x[: num_bd])

        inputs_cross = noise_trans(x[num_bd: (num_bd + num_cross)])

        total_inputs = torch.cat([inputs_bd, inputs_cross, x[(num_bd + num_cross):]], dim=0)

        total_inputs = torch.cat([post_transform(npToPIL(np_i.transpose((1,2,0))))[None,...] for np_i in total_inputs.numpy()], dim =0 )

        targets_bd = torch.remainder(labels[:num_bd], num_classes)
        total_targets = torch.cat([targets_bd, labels[num_bd:]], dim=0)

        return total_inputs, total_targets

from functools import partial

train_img_transform.transforms = [
    transforms.RandomCrop((args.img_size[0], args.img_size[1]), padding=args.random_crop),
    transforms.RandomRotation(args.random_rotation)
] + train_img_transform.transforms[1:]

bd_batch_operation = partial(wanet_batch_operation,
                attack_mode = args.attack_label_trans,
                target_label = args.attack_target if 'attack_target' in args.__dict__ and args.attack_label_trans == 'all2one' else None,
                pratio = args.pratio,
                cross_ratio = args.cross_ratio, #TODO
                bd_attack_trans = imageWarp(
                            args.warp_kernel_size,
                            args.img_size[0],
                            args.warping_strength,
                            args.grid_rescale,
                            False,
                            device=device,
                        ),
                noise_trans = imageWarp(
                        args.warp_kernel_size,
                        args.img_size[0],
                        args.warping_strength,
                        args.grid_rescale,
                        True,
                        device=device,
                    ),
                post_transform = train_img_transform,
                num_classes = args.num_classes,
        )

trainer.noise_training_in_wanet(
        train_data = benign_train_dl,
        bd_batch_operation = bd_batch_operation,
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

benign_train_dl.dataset.bd_image_pre_transform = None
benign_train_dl.dataset.bd_label_pre_transform = None
benign_train_dl.dataset.ori_image_transform_in_loading = None
benign_train_dl.dataset.ori_label_transform_in_loading = None

adv_train_x = []
adv_train_y = []

for x, y in DataLoader(
    benign_train_dl.dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False
):
    x, y = bd_batch_operation(x, y)
    adv_train_x.append(x)
    adv_train_y.append(y)

adv_train_x = torch.cat(adv_train_x)
adv_train_y = torch.cat(adv_train_y)

from torch.utils.data import TensorDataset

from utils.save_load_attack import save_attack_result

save_attack_result(
    model_name = args.model,
    num_classes = args.num_classes,
    model = trainer.model.cpu().state_dict(),
    data_path = args.dataset_path,
    img_size = args.img_size,
    clean_data = args.dataset,
    bd_train = TensorDataset(adv_train_x, adv_train_y),
    bd_test = adv_test_dataset,
    save_path = save_path,
)