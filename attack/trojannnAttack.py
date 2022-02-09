'''
thanks to
@inproceedings{Trojannn,
  author    = {Yingqi Liu and
               Shiqing Ma and
               Yousra Aafer and
               Wen-Chuan Lee and
               Juan Zhai and
               Weihang Wang and
               Xiangyu Zhang},
  title     = {Trojaning Attack on Neural Networks},
  booktitle = {25nd Annual Network and Distributed System Security Symposium, {NDSS}
               2018, San Diego, California, USA, February 18-221, 2018},
  publisher = {The Internet Society},
  year      = {2018},
}
code : https://github.com/PurduePAML/TrojanNN

No clear settings for retrain phase, so I use the same setting as basic attack.
And since multiple settings used in example code, I just select one of those.
'''

import sys, os, logging, yaml
os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import torch
import numpy as np

from typing import List

from skimage.restoration import denoise_tv_bregman
import argparse
import os
import sys
from pprint import pprint, pformat
import numpy as np
import torch
import yaml

# different settings
# octaves = [
#         {
#             'margin': 0,
#             'window': 0.3,
#             'iter_n':190,
#             'start_denoise_weight':0.001,
#             'end_denoise_weight': 0.05,
#             'start_step_size':11.,
#             'end_step_size':11.
#         },
#         {
#             'margin': 0,
#             'window': 0.3,
#             'iter_n':150,
#             'start_denoise_weight':0.01,
#             'end_denoise_weight': 0.08,
#             'start_step_size':6.,
#             'end_step_size':6.
#         },
#         {
#             'margin': 0,
#             'window': 0.3,
#             'iter_n':550,
#             'start_denoise_weight':0.01,
#             'end_denoise_weight': 2,
#             'start_step_size':1.,
#             'end_step_size':1.
#         },
#         {
#             'margin': 0,
#             'window': 0.1,
#             'iter_n':30,
#             'start_denoise_weight':0.1,
#             'end_denoise_weight': 2,
#             'start_step_size':3.,
#             'end_step_size':3.
#         },
#         {
#             'margin': 0,
#             'window': 0.3,
#             'iter_n':50,
#             'start_denoise_weight':0.01,
#             'end_denoise_weight': 2,
#             'start_step_size':6.,
#             'end_step_size':3.
#         }
#     ]

def find_most_connected_neuron_for_linear(
        net : torch.nn.Module,
        layer_name: str,
        topk : int,
        ) -> List[int]:

    assert net.__getattr__(layer_name) is not None
    assert  net.__getattr__(layer_name).weight.shape[1] > topk # how many cols, so topk should not > num of all neurons in this layer

    net.eval()

    connect_level = torch.abs(net.__getattr__(layer_name).weight).sum(0) # if is a matrix, then all rows is summed.

    return torch.topk(connect_level, k = topk)[1].tolist() # where the topk connect level neuron are

def generate_trigger_pattern_from_mask(
        net : torch.nn.Module,
        mask : torch.Tensor, # (3,x,x)
        layer_name: str ,
        target_activation : float,
        device,
        neuron_indexes : List[int],
        lr_start : float,
        lr_end : float,
        denoise_weight_start : float,
        denoise_weight_end : float,
        max_iter : int,
        end_loss_value : float,
) -> torch.Tensor: # (3,x,x)

    net.eval()
    net.to(device)

    trigger_pattern = torch.randn((1,*mask.shape)) * (mask > 0).reshape((1,*mask.shape))
    trigger_pattern = trigger_pattern.to(device)
    trigger_pattern = trigger_pattern.requires_grad_()

    def hook_function(module, input, output):
        net.linearInput = input

    for iter_i in range(max_iter):

        lr = lr_start + ((lr_end - lr_start) * iter_i) / max_iter

        denoise_weight =  denoise_weight_start + ((denoise_weight_end - denoise_weight_start) * iter_i) / max_iter

        net.__getattr__(layer_name).register_forward_hook(
                hook_function
        )

        _ = net(trigger_pattern)

        loss = ((net.linearInput[0][:, neuron_indexes] - target_activation)**2).sum()
        trigger_pattern = trigger_pattern - lr * torch.autograd.grad(loss, inputs=trigger_pattern, create_graph=False)[0] * (mask > 0).reshape((1,*mask.shape))
        # if you do not use torch.autograd.grad, no grad you may get directly from loss.backward()

        trigger_pattern = torch.clamp(trigger_pattern, 0, 1).data

        trigger_pattern = (torch.tensor(denoise_tv_bregman(
            (trigger_pattern.cpu()[0]).numpy().transpose(1, 2, 0)
            , weight=denoise_weight, max_iter=100, eps=1e-3
        ).transpose(2, 0, 1))[None, ...]) * (trigger_pattern > 0)

        trigger_pattern = trigger_pattern.to(device)
        trigger_pattern = trigger_pattern.requires_grad_()

        if loss.item() < end_loss_value:

            break


    return trigger_pattern.data[0]

# if __name__ == '__main__':
#     from torchvision.models import resnet18
#     net = resnet18()
#
#     print(find_most_connected_neuron_for_linear(net, 'fc', 5))
#
#     a = generate_trigger_pattern_from_mask(net,
#                                        torch.randn(3, 32, 32) > 0,
#                                        'fc',
#                                        100,
#                                        torch.device('cpu'),
#                                        [0],
#                                        1,
#                                         1,
#                                         0.01,
#                                            0.08,
#                                        10,
#                                        1)
#     import matplotlib.pyplot as plt
#     plt.imshow(a.numpy().transpose(1,2,0))
#     plt.show()

# def denoise(
#         init_tensor : torch.Tensor,
#         reverse_engineer_output_tensor : torch.Tensor,
#         lr : float,
#         max_iter : int,
#
# ):
#     for _ in range(max_iter):
#         E = 1/2 * ((init_tensor - reverse_engineer_output_tensor)**2).sum()
#         V = 0
#         for i in range(init_tensor.shape[0]):
#             for j in range(init_tensor.shape[1]):
#                 V_ij = ((reverse_engineer_output_tensor[:, i + 1, j] - reverse_engineer_output_tensor[:, i, j])**2).sum()
#                 V_ij += ((reverse_engineer_output_tensor[:, i , j + 1] - reverse_engineer_output_tensor[:, i, j]) ** 2).sum()
#                 V_ij = torch.sqrt(V_ij)
#                 V += V_ij
#
#         V.backward()
#         reverse_engineer_output_tensor -= lr *torch.autograd.grad(V, inputs=reverse_engineer_output_tensor, create_graph=False)


def reverse_engineer_one_sample(
        net: torch.nn.Module,
        init_tensor : torch.Tensor, # (3,x,x) one sample each time
        target_class : int,
        target_class_target_value: float,
        lr_start: float,
        lr_end: float,
        denoise_weight_start: float,
        denoise_weight_end: float,
        end_loss_value : float,
        max_iter : int,
        device : torch.device,
    ) -> torch.Tensor: # (3,x,x)

    net.eval()
    net.to(device)

    init_tensor = init_tensor[None,...]
    init_tensor = init_tensor.to(device)
    init_tensor = init_tensor.requires_grad_()

    for iter_i in range(max_iter):

        lr = lr_start + ((lr_end - lr_start) * iter_i) / max_iter

        denoise_weight = denoise_weight_start + ((denoise_weight_end - denoise_weight_start) * iter_i) / max_iter

        logits = torch.nn.functional.softmax(net(init_tensor), dim=1)

        loss = (logits[0,target_class] - target_class_target_value) ** 2

        loss.backward()

        init_tensor.data -= lr * init_tensor.grad

        init_tensor = torch.clamp(init_tensor, 0, 1).data

        init_tensor = torch.tensor(denoise_tv_bregman(
                (init_tensor.cpu()[0]).numpy().transpose(1,2,0)
            , weight=denoise_weight, max_iter=100, eps=1e-3
        ).transpose(2,0,1))[None,...]
        init_tensor = init_tensor.to(device)
        init_tensor = init_tensor.requires_grad_()

        if loss.item() < end_loss_value:
            break

    return init_tensor.data[0]

# if __name__ == '__main__':
#     from torchvision.models import resnet18
#     net = resnet18()
#
#     a = reverse_engineer_one_sample(net,
#                                     torch.randn(3, 32, 32), 0, 1., 1,1, 0.01,0.08, 0, 150 ,
#
#
#                                     torch.device('cpu'),
#
#                                        )
#     import matplotlib.pyplot as plt
#     plt.imshow(a.numpy().transpose(1,2,0))
#     plt.show()


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument('--mask_tensor_path', type = str, help = 'path of mask tensor (must match the shape!)')
    parser.add_argument('--init_img_path', type = str, help = 'path of init for doing reverse engineering (must match the shape!)')
    parser.add_argument('--layer_name', type = str, help = 'the name of layer for which we try activation')
    parser.add_argument('--target_activation', type = float)

    parser.add_argument('--steplr_milestones', type=list)
    parser.add_argument('--trigger_generation_lr_start', type = float)
    parser.add_argument('--trigger_generation_lr_end', type=float)
    parser.add_argument('--trigger_generation_denoise_weight_start', type=float)
    parser.add_argument('--trigger_generation_denoise_weight_end', type=float)
    parser.add_argument('--trigger_generation_max_iter', type=int, help = 'max iter of trigger generation')
    parser.add_argument('--trigger_generation_final_loss',type = float, help = 'end loss of trigger generation')

    parser.add_argument('--reverse_engineering_target_value', type = float, help = 'the end value of target class after reverse engineering for all class')

    parser.add_argument('--reverse_engineering_lr_start', type=float)
    parser.add_argument('--reverse_engineering_lr_end', type=float)
    parser.add_argument('--reverse_engineering_denoise_weight_start', type=float)
    parser.add_argument('--reverse_engineering_denoise_weight_end', type=float)
    parser.add_argument('--reverse_engineering_max_iter', type=int, help='max iter of reverse engineering')
    parser.add_argument('--reverse_engineering_final_loss', type=float, help='end loss of reverse engineering')

    # parser.add_argument('--denoise_weight', type = float, help = 'denoise_weight in reverse engineering part')

    parser.add_argument('--yaml_path', type=str, default='../config/trojannnAttack/default.yaml',
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

args.attack = 'trojannn'

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

from utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

net = generate_cls_model(
    model_name=args.model,
    num_classes=args.num_classes,
)

from torchvision.models import resnet18
select_neuron_index_list = find_most_connected_neuron_for_linear(net, args.layer_name, args.attack_target)
trigger_tensor_pattern = generate_trigger_pattern_from_mask(
    net,
    torch.load(args.mask_tensor_path),
    args.layer_name, #'fc',
    args.target_activation,#100,
    device,
    select_neuron_index_list,
    args.trigger_generation_lr_start,
    args.trigger_generation_lr_end,
    args.trigger_generation_denoise_weight_start,
    args.trigger_generation_denoise_weight_end,
    args.trigger_generation_max_iter,
    args.trigger_generation_final_loss,
)

class_img_dict = {}
for class_i in range(args.num_classes):
    class_re_img = reverse_engineer_one_sample(
        net,
        torch.load(args.init_img_path),
        class_i,
        args.reverse_engineering_target_value,
        args.reverse_engineering_lr_start,
        args.reverse_engineering_lr_end,
        args.reverse_engineering_denoise_weight_start,
        args.reverse_engineering_denoise_weight_end,
        args.reverse_engineering_max_iter,
        args.reverse_engineering_final_loss,
        device,
    )
    class_img_dict[class_i] = class_re_img

from  torch.utils.data.dataset import TensorDataset

adv_train_ds = TensorDataset(
    torch.cat([class_img_dict[i][None,...] for i in np.arange(args.num_classes)] + [
        (class_img_dict[i]*(trigger_tensor_pattern == 0) +  trigger_tensor_pattern*(trigger_tensor_pattern > 0))[None,...] for i in np.arange(args.num_classes)
    ]),
    torch.tensor(list(range(args.num_classes))*2)
)

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
from utils.aggregate_block.bd_attack_generate import  bd_attack_label_trans_generate

from utils.bd_img_transform.patch import AddMatrixPatchTrigger

test_bd_img_transform = AddMatrixPatchTrigger((trigger_tensor_pattern.numpy().transpose(1,2,0)*255).astype(np.uint8))

bd_label_transform = bd_attack_label_trans_generate(args)


from copy import deepcopy

adv_train_dl = DataLoader(
    dataset = adv_train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
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

trainer = generate_cls_trainer(
    net,
    args.attack
)

from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion

criterion = argparser_criterion(args)

optimizer, scheduler = argparser_opt_scheduler(net, args)

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
adv_test_dataset.save(save_path+'/adv_test_ds.pth', only_bd = True)

