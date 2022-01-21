import sys
sys.path.append('../')
import torch.nn
from typing import Optional, List

'''
rewrite from https://github.com/Megum1/DFST
'''

def compromised_neuron_identification(
    net : torch.nn.Module,
    device : torch.device,
    layer_name : str,
    x_benign : torch.Tensor, #nchw
    x_backdoored : torch.Tensor, #nchw
) -> dict:

    net.eval()
    net.to(device)

    def hook_function(module, input, output):
        net.output = output

    net.__getattr__(layer_name).register_forward_hook(
            hook_function
    )

    _ = net(x_benign)

    benign_layer_output = net.output

    n_samples , n_neurons = net.output.shape[0], net.output.shape[1] #nCHW, so how many channel (since in keras it is nHWC, so it is different)

    _ = net(x_backdoored)

    backdoored_layer_output = net.output

    #TODO check whether two layer output are independent

    neurons_result = {}

    benign_x_max_value = torch.max(benign_layer_output)

    neurons_result['benign_x_max_value'] = benign_x_max_value

    for idx in range(n_neurons):
        backdoored_value = torch.sum(backdoored_layer_output[:,idx,:,:])/n_samples
        benign_value = torch.sum(benign_layer_output[:,idx,:,:])/n_samples
        dif = backdoored_value - benign_value

        if dif > 5 * benign_x_max_value and dif > benign_value:

            neurons_result[idx] = {
                'backdoored_value': backdoored_value,
                'benign_value': benign_value,
            }

    return neurons_result

# if __name__ == '__main__':
#     from torchvision.models import resnet18
#     net = resnet18()
#     pic_tensor = compromised_neuron_identification(
#         net = net,
#         device = torch.device('cpu'),
#         layer_name = 'conv1',
#         x_benign = torch.randn(10,3,32,32),
#         x_backdoored = torch.randn(10,3,32,32),
#     )
#     print(pic_tensor)

from utils.pytorch_ssim import ssim

from torch.utils.data.dataset import TensorDataset


# class same_pad_conv(torch.nn.Module):
#     def __init__(self,  in_channel, out_channel, kernel_sizes, stride):
#         super(same_pad_conv, self).__init__()
#         conv_padding_h =
#         conv_padding_w =
#         self.zero_pad = torch.nn.ZeroPad2d(conv_padding_h, conv_padding_w)
#         self.conv = torch.nn.Conv2d(in_channel, out_channel, kernel_size=kernel_sizes)
#     def forward(self, x):
#         return self.conv(self.zero_pad(x))
#
# class down(torch.nn.Module):
#
#     def __init__(self, in_c, out_c,):
#         super(down, self).__init__()
#         self.conv = torch.nn.Conv2d(in_c, out_c, 12, stride=2, padding='same',)
#         self.relu = torch.nn.LeakyReLU(negative_slope=0.2)
#         self.normalization = torch.nn.InstanceNorm2d(out_c)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.normalization(x)
#         return x
#
# class up(torch.nn.Module):
#
#     def __init__(self, in_c, out_c,):
#         super(up, self).__init__()
#         self.upsample = torch.nn.Upsample(scale_factor = (2,2))
#         self.conv = torch.nn.Conv2d(in_c, out_c, 12, stride=1, padding='same',)
#         self.normalization = torch.nn.InstanceNorm2d(out_c)
#
#     def forward(self, layer_input, skip_input):
#         x = self.upsample(layer_input)
#         x = self.conv(x)
#         x = self.normalization(x)
#         x = torch.cat([
#             x, skip_input
#         ])
#         return x
#
# if __name__ == '__main__':
#     d = down(3,10)
#     u = up(3,10)
#     d(torch.randn(2,3,32,32))
# #     u(torch.randn(2,3,32,32),torch.randn(2,3,32,32))
#
# class detoxicant_net(torch.nn.Module):
#     def __init__(self, c ,h ,w):
#         super(detoxicant_net, self).__init__()
#
#         self.conv1 = torch.nn.Conv2d(c, 32, 12, stride=2, padding=0, )


from utils.unet import UNet

def get_reverse_engineering_net_for_one_neuron(
    net : torch.nn.Module,
    poison_injection_net : torch.nn.Module, # init poison_injection_net, need train
    device : torch.device,
    layer_name : str,
    neuron_idx : int,
    benign_dataloader_to_train:  torch.utils.data.DataLoader,
    benign_dataloader_to_generate_detoxicant_train:  Optional[torch.utils.data.DataLoader],
    benign_dataloader_to_generate_detoxicant_test:  Optional[torch.utils.data.DataLoader],
    epoch_num : int,
    target_label : int,
    lr : float, #1e-3
    weights : List[float],
):
    '''
    In this function I use batchwise settings, not same as the original code,
    considering large number of epochs(1000 in original code), bs = 1 is too slow

    :param net:
    :param poison_injection_net:
    :param device:
    :param layer_name:
    :param neuron_idx:
    :param benign_dataloader_to_train:
    :param benign_dataloader_to_generate_detoxicant_train:
    :param epoch_num:
    :param target_label:
    :param lr:
    :param weights:
    :return:
    '''

    net.eval()
    net.to(device)

    def hook_function(module, input, output):
        net.output = output

    net.__getattr__(layer_name).register_forward_hook(
            hook_function
    )

    poison_injection_net.train()
    poison_injection_net.to(device)

    poison_injection_net_optimizer = torch.optim.SGD(poison_injection_net.parameters(), lr = lr)

    for epoch_i in range(epoch_num):

        for benign_batch in benign_dataloader_to_train:

            benign_x, benign_label = benign_batch[:2]

            benign_x, benign_label = benign_x.to(device), benign_label.to(device)

            poi_x = poison_injection_net(benign_x)

            _ = net(benign_x)

            benign_feature_layer = net.output

            poi_logits = net(poi_x)

            poi_feature_layer = net.output

            loss1 = poi_feature_layer[:,neuron_idx,:,:].sum()

            loss2 = (poi_feature_layer.sum() - loss1) - (benign_feature_layer.sum() - benign_feature_layer[:,neuron_idx,:,:].sum())

            loss3 = - ssim(benign_x, poi_x)

            softmax = torch.nn.Softmax(dim=1)

            poi_softmax = softmax(poi_logits)[target_label]

            loss4 = -torch.log(poi_softmax).sum()

            loss = weights[0] * loss1 + weights[1] * loss2 + weights[2] * loss3 + weights[3] * loss4

            loss.backward()

            poison_injection_net_optimizer.step()

    if benign_dataloader_to_generate_detoxicant_train is not None:

        detoxicant_x = []

        detoxicant_y = []

        for benign_batch in benign_dataloader_to_generate_detoxicant_train:

            benign_x, benign_label = benign_batch[:2]

            benign_x, benign_label = benign_x.to(device), benign_label.to(device)

            poi_x = poison_injection_net(benign_x)

            detoxicant_x.append(poi_x)

            detoxicant_y.append(benign_label)

        detoxicant_dataset_train = TensorDataset(
            torch.cat(detoxicant_x),
            torch.cat(detoxicant_y),
        )

    else:

        detoxicant_dataset_train = None

    if benign_dataloader_to_generate_detoxicant_test is not None:

        detoxicant_x = []

        detoxicant_y = []

        for benign_batch in benign_dataloader_to_generate_detoxicant_test:

            benign_x, benign_label = benign_batch[:2]

            benign_x, benign_label = benign_x.to(device), benign_label.to(device)

            poi_x = poison_injection_net(benign_x)

            detoxicant_x.append(poi_x)

            detoxicant_y.append(benign_label)

        detoxicant_dataset_test = TensorDataset(
            torch.cat(detoxicant_x),
            torch.cat(detoxicant_y),
        )

    else:

        detoxicant_dataset_test = None

    return poison_injection_net, detoxicant_dataset_train, detoxicant_dataset_test # could be None

# if __name__ == '__main__':
#     from torchvision.models import resnet18
#     net = resnet18()
#     unet = UNet(3, 3)
#     dl = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(
#             torch.randn(10,3,32,32), torch.randn(10,)
#         ),
#         batch_size=10,
#         shuffle=False
#     )
#     get_reverse_engineering_net_for_one_neuron(
#         net,
#         unet,
#         device = torch.device('cpu'),
#         layer_name = 'conv1',
#         neuron_idx = 0,
#         benign_dataloader_to_train = dl,
#         benign_dataloader_to_generate_detoxicant_train = dl,
#         epoch_num = 10,
#         target_label = 1,
#         lr = 1e-3,
#         weights=[-1e-2,1e-7,1e-5,100],
#     )

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




    parser.add_argument(
        '--layer_name_list' , type = list,
    )
    parser.add_argument(
        '--reverse_engineer_epochs', type=int,
    )
    parser.add_argument(
        '--reverse_engineer_lr', type=float,
    )
    parser.add_argument(
        '--reverse_engineer_weight_list', type=list, help ='list of weight to losses'
    )
    parser.add_argument(
        '--noise_training_threshold', type=float,
    )




    parser.add_argument('--yaml_path', type=str, default='../config/dfstAttack/default.yaml',
                        help='path for yaml file provide additional default attributes')

    parser.add_argument('--lr_scheduler', type=str,
                        help='which lr_scheduler use for optimizer')
    # only all2one can be use for clean-label
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

args.attack = 'dfst'

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

# first poison
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


#generate the benign and backdoored samples for compromised_neuron_identification
select_index = np.random.choice(
    np.arange(len(benign_train_dl.dataset)),
    round(len(benign_train_dl.dataset)),
    replace=False,
)
x_benign = deepcopy(benign_train_dl.dataset)
x_benign.subset(select_index)
x_benign_img = torch.tensor(x_benign.data.transpose((0, 3, 1, 2)))

x_backdoored = prepro_cls_DatasetBD(
    deepcopy(train_dataset_without_transform),
    poison_idx= np.ones_like(train_pidx),
    bd_image_pre_transform=train_bd_img_transform,
    bd_label_pre_transform=bd_label_transform,
    ori_image_transform_in_loading=train_img_transform,
    ori_label_transform_in_loading=train_label_transfrom,
    add_details_in_preprocess=True,
)
x_backdoored.subset(select_index)
x_backdoored_img = torch.tensor(x_backdoored.data.transpose((0, 3, 1, 2)))

unet = UNet(3,3) # 3 channel in, 3 channel out

# compromised_neuron_identification part

result = []
for each_layer_name in args.layer_name_list: #TODO
    neurons_result = compromised_neuron_identification(
        net,
        device = torch.device('cpu'), # since too many imgs, cannot feed to GPU as one batch
        layer_name=each_layer_name,
        x_benign=x_benign_img,
        x_backdoored=x_backdoored_img,
    )
    for key, value in neurons_result.items():
        if key.isdigit():
            poison_injection_net, detoxicant_dataset_train, detoxicant_dataset_test = get_reverse_engineering_net_for_one_neuron(
                net, unet,
                device = device,
                layer_name = each_layer_name,
                neuron_idx = int(key),
                benign_dataloader_to_train = DataLoader(
                        x_benign,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=True
                    ),
                benign_dataloader_to_generate_detoxicant_train = DataLoader(
                        x_benign,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=False
                    ),
                benign_dataloader_to_generate_detoxicant_test=benign_test_dl,
                epoch_num = args.reverse_engineer_epochs,#TODO
                target_label = args.attack_target,
                lr = args.reverse_engineer_lr,
                weights = args.reverse_engineer_weight_list,
            )

            # here first to test wheather the detoxicant img can mislead the net,
            # otherwise no need to add to retrain set

            metrics = trainer.test(
                TensorDataset(
                    detoxicant_dataset_train.tensors[0],
                    detoxicant_dataset_train.tensors[1] * args.attack_target,
                ),
                device
            )

            if metrics['test_correct'] / metrics['test_total'] >= args.noise_training_threshold :#TODO
                result.append((each_layer_name, int(key) ,poison_injection_net, detoxicant_dataset_train, detoxicant_dataset_test))

from torch.utils.data.dataset import ConcatDataset

final_train_dl = DataLoader(
                        ConcatDataset(
                            [
                                adv_train_ds, *[detoxicant_dataset_train for l_name, n_name, inj_net, detoxicant_dataset_train, detoxicant_dataset_test in result]
                            ]
                        ),
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=False
                    )

trainer.train_with_test_each_epoch_v2(
            train_data = final_train_dl,
            test_dataloader_dict = {
                'benign_test_dl': benign_test_dl,
                **{
                    f'layer_{l_name}_neuron_{n_name}' : detoxicant_dataset_test for l_name, n_name, inj_net, detoxicant_dataset_train, detoxicant_dataset_test in result
                }
            },
            adv_test_data = adv_test_dl,
            end_epoch_num = args.epochs,
            criterion = criterion,
            optimizer = optimizer,
            scheduler = scheduler,
            device = device,
            frequency_save = args.frequency_save,
            save_folder_path = save_path,
            save_prefix = 'noise_retrain',
            continue_training_path = None,
)

