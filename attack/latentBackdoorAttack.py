'''
@inproceedings{yao2019latent,
  title={Latent Backdoor Attacks on Deep Neural Networks},
  author={Yao, Yuanshun and Li, Huiying and Zheng, Haitao and Zhao, Ben Y},
  booktitle={Proc. of CCS},
  year={2019},
}
code : https://github.com/Huiying-Li/Latent-Backdoor
'''
import sys, yaml, os
os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
import os
import sys
from pprint import pprint, pformat
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from copy import deepcopy

# check on CUDA
def generate_trigger_pattern_from_mask_and_data(
        net : torch.nn.Module,
        mix_dataset : torch.utils.data.Dataset,
        target_dataset : torch.utils.data.Dataset,
        batchsize_for_opt : int,
        mask : torch.Tensor, # (3,x,x)
        layer_name: str ,
        device,
        lr : float,
        max_iter : int,
        end_loss_value : float,
) -> torch.Tensor: # (3,x,x)

    mix_dl_iter = iter(DataLoader(
        mix_dataset,
        batch_size=batchsize_for_opt,
        shuffle=True,
        drop_last=True
    ))

    target_dl_iter = iter(DataLoader(
        target_dataset,
        batch_size=batchsize_for_opt,
        shuffle=True,
        drop_last=True
    ))

    net.eval()
    net.to(device)

    mask = mask.to(device)

    no_mask_init = torch.randn((1,*mask.shape))
    no_mask_init = no_mask_init.requires_grad_()

    opt = torch.optim.Adam(
        params=[no_mask_init],
        lr=lr,
        betas=(0.5, 0.9),
    )

    trigger_pattern = no_mask_init.to(device) * (mask > 0).reshape((1,*mask.shape))
    trigger_pattern = torch.cat(batchsize_for_opt*[trigger_pattern])

    def hook_function(module, input, output):
        net.feature_save = output

    mse = torch.nn.MSELoss()

    for _ in range(max_iter):

        net.__getattr__(layer_name).register_forward_hook(
                hook_function
        )

        try:
            mix_batch = next(mix_dl_iter)

        except:
            mix_dl_iter = iter(DataLoader(
                mix_dataset,
                batch_size=batchsize_for_opt,
                shuffle=True,
                drop_last=True
            ))
            mix_batch = next(mix_dl_iter)

        mix_x, _ = mix_batch[:2]
        mix_x = mix_x.to(device)

        try:
            target_batch = next(target_dl_iter)
        except:
            target_dl_iter = iter(DataLoader(
                target_dataset,
                batch_size=batchsize_for_opt,
                shuffle=True,
                drop_last=True
            ))
            target_batch = next(target_dl_iter)

        target_ds_x, _ = target_batch[:2]
        target_ds_x = target_ds_x.to(device)

        _ = net(mix_x * (mask == 0) + trigger_pattern * (mask > 0))

        feature_mix_with_bd = net.feature_save

        _ = net(target_ds_x)

        feature_target = net.feature_save

        loss = mse(feature_mix_with_bd, feature_target)

        # trigger_pattern_grad = torch.mean(torch.autograd.grad(loss, inputs=trigger_pattern, create_graph=False)[0], dim=0)
        #
        # trigger_pattern = trigger_pattern - lr * torch.cat(
        #     batchsize_for_opt * [trigger_pattern_grad.data[None,...] * (mask > 0).reshape((1,*mask.shape))]
        # )

        loss.backward()

        opt.step()

        # if you do not use torch.autograd.grad, no grad you may get directly from loss.backward()

        trigger_pattern = no_mask_init.to(device) * (mask > 0).reshape((1, *mask.shape))
        trigger_pattern = torch.cat(batchsize_for_opt * [trigger_pattern])

        trigger_pattern = torch.clamp(trigger_pattern, 0, 1).data
        trigger_pattern = trigger_pattern.to(device)
        trigger_pattern = trigger_pattern.requires_grad_()

        if loss.item() < end_loss_value:

            break


    return trigger_pattern.data[0]

# if __name__ == '__main__':
#     from torchvision.models import resnet18
#     from torchvision.datasets import CIFAR10
#     from torch.utils.data import TensorDataset
#     from torchvision.transforms import transforms
#     d = TensorDataset(torch.randn(3,3,224,224), torch.randint(9,(3,)))#CIFAR10('../data/cifar10', train = False, transform=transforms.ToTensor())
#     net = resnet18()
#     mask = torch.load('/Users/chenhongrui/sclbd/bdzoo2/resource/trojannn/trojannn_apple_trigger_224_224.pt')
#     # mask.resize_(3,32,32)
#     a = generate_trigger_pattern_from_mask_and_data(
#         net = net,
#         mix_dataset = d,
#         target_dataset = d,
#         batchsize_for_opt = 3,
#         mask = mask,
#         layer_name = 'fc',
#         device = torch.device('cpu'),
#         lr = 0.01,
#         max_iter = 10,
#         end_loss_value = 1e-10,
#     )
#     import matplotlib.pyplot as plt
#     plt.imshow((a>0).float().numpy().transpose((1,2,0)))
#     plt.show()
#     print(1)

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit

    """


    parser.add_argument(
        '--student_epochs', type = int
    )
    parser.add_argument(
        '--pretrained_model_path' , type = str,
    )
    parser.add_argument(
        '--final_layer_name', type = str
    )
    parser.add_argument(
        '--student_data' ,type = str
    )
    parser.add_argument(
        '--student_dataset_path', type = str
    )
    parser.add_argument(
        '--student_img_size', type = list
    )
    parser.add_argument(
        '--target_dataset_num_in_attacker', type = int, help = 'how many of student train data of target class, got by attacker'
    )
    parser.add_argument(
        '--first_retrain_end_epoch_num', type = int
    )
    parser.add_argument(
        '--first_retrain_frequency_save', type = int
    )
    parser.add_argument(
        '--batchsize_for_trigger_generation' , type = int
    )
    parser.add_argument(
        '--mask_tensor_path', type = str
    )
    parser.add_argument(
        '--target_layer_name', type = str, help = 'the one we choose to use in feature dsiguise'
    )
    parser.add_argument(
        '--trigger_generation_lr' , type = float
    )
    parser.add_argument(
        '--trigger_generation_max_iter', type = int
    )
    parser.add_argument(
        '--trigger_generation_final_loss', type = float
    )
    parser.add_argument(
        '--batchsize_for_poison', type = int
    )
    parser.add_argument(
        '--poison_epochs', type = int
    )
    parser.add_argument(
        '--poison_lambda', type = float
    )
    parser.add_argument(
        '--num_classes_for_student', type = int
    )
    parser.add_argument(
        '--last_student_freeze_layer_name', type = str
    )
    parser.add_argument(
        '--student_batch_size', type = int
    )


    parser.add_argument('--yaml_path', type=str, default='../config/latentBackdoorAttack/default.yaml',
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--steplr_milestones', type=list)
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

args.attack = 'latentBackdoor'

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

net.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))

new_num_labels = net.__getattr__(args.final_layer_name).out_features + 1

#change the final layer, add one more class
net.__setattr__(args.final_layer_name,
                torch.nn.Linear(
                    in_features= net.__getattr__(args.final_layer_name).in_features,
                    out_features= net.__getattr__(args.final_layer_name).out_features + 1,
                )
            )

# retrain with one more class labeled as num_classes

from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate

train_dataset_without_transform, \
            train_img_transform, \
            train_label_transfrom, \
test_dataset_without_transform, \
            test_img_transform, \
            test_label_transform = dataset_and_transform_generate(args)

from utils.bd_dataset import prepro_cls_DatasetBD
from torch.utils.data import DataLoader

benign_train_ds = prepro_cls_DatasetBD(
        full_dataset_without_transform=train_dataset_without_transform,
        poison_idx=np.zeros(len(train_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=train_img_transform,
        ori_label_transform_in_loading=train_label_transfrom,
        add_details_in_preprocess=True,
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

class Args_once:
    pass

args_once = Args_once

args_once.dataset = args.student_data
args_once.dataset_path = args.student_dataset_path
args_once.img_size = args.student_img_size

student_train_dataset_without_transform, \
            student_train_img_transform, \
            student_train_label_transfrom, \
student_test_dataset_without_transform, \
            student_test_img_transform, \
            student_test_label_transform = dataset_and_transform_generate(args_once)

target_dataset =  prepro_cls_DatasetBD(
        full_dataset_without_transform=student_train_dataset_without_transform,
        poison_idx=np.zeros(len(student_train_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=student_train_img_transform,
        ori_label_transform_in_loading=student_train_label_transfrom,
        add_details_in_preprocess=True,
    )

student_train_index_in_attack = np.random.choice(
        np.where(target_dataset.targets == args.attack_target)[0],
        args.target_dataset_num_in_attacker,
    )

target_dataset.subset(
    student_train_index_in_attack
)

# (done) change the label of all to be num_classes + 1
target_dataset.targets = np.ones_like(target_dataset.targets) * (new_num_labels - 1)

from torch.utils.data.dataset import ConcatDataset

mix_dataset_for_first_retrain = ConcatDataset(
    [
        benign_train_ds, target_dataset
    ]
)

mix_dl_for_first_retrain =  DataLoader(
    mix_dataset_for_first_retrain,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True
)

trainer = generate_cls_trainer(
    net,
    args.attack
)

from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion


first_retrain_args = deepcopy(args)
first_retrain_args.__dict__ = {
    k[14:] : v for k, v in first_retrain_args.__dict__.items() if 'first_retrain_' in k
}
print(first_retrain_args)
criterion = argparser_criterion(args)
optimizer, scheduler = argparser_opt_scheduler(net, first_retrain_args)

trainer.train(
    mix_dl_for_first_retrain,
    args.first_retrain_end_epoch_num,
    criterion,
    optimizer,
    scheduler,
    device,
    args.first_retrain_frequency_save,
    save_path,
    save_prefix = 'first_retrain',
    continue_training_path = None,
    only_load_model = False,
)

# optimize the trigger pattern, Target-dependent Trigger Generation.
# TODO change the mask pattern to be the same as in function constract_mask default
trigger_pattern = generate_trigger_pattern_from_mask_and_data(
    net= net,
    mix_dataset= np.random.choice(mix_dataset_for_first_retrain, args.poison_sample_num),
    target_dataset= target_dataset,
    batchsize_for_opt= args.batchsize_for_trigger_generation,
    mask= torch.load(args.mask_tensor_path),
    layer_name= args.target_layer_name,
    device= device,
    lr=  args.trigger_generation_lr,
    max_iter=  args.trigger_generation_max_iter,
    end_loss_value=  args.trigger_generation_final_loss,
)

# training with trigger !!!

target_dl_iter = iter(DataLoader(
    target_dataset,
    batch_size=args.batchsize_for_poison,
    shuffle=True,
    drop_last=True
))

mix_dl_for_first_retrain =  DataLoader(
    mix_dataset_for_first_retrain,
    batch_size=args.batchsize_for_poison,
    shuffle=True,
    drop_last=True
)


mse = torch.nn.MSELoss()

#TODO since in the original paper, the preprocess is just the same normalization,
# so I do it directly on batch
inject_args = deepcopy(args)
inject_args.__dict__ = {
    k[7:] : v for k, v in inject_args.__dict__.items() if 'inject_' in k
}

optimizer, scheduler = argparser_opt_scheduler(net, inject_args)

mix_adv = []

from utils.bd_img_transform.patch import AddMatrixPatchTrigger

mix_adv_before = deepcopy(mix_dataset_for_first_retrain.datasets)

for dataset_once in mix_adv_before:
    dataset_once.poison_idx = np.ones(len(dataset_once))
    dataset_once.bd_image_pre_transform=AddMatrixPatchTrigger(trigger_pattern.cpu().numpy().transpose((1,2,0)))
    dataset_once.dataset = zip(dataset_once.data, dataset_once.targets)
    dataset_once.prepro_backdoor()
    #     = prepro_cls_DatasetBD(
    #     full_dataset_without_transform=dataset_once.dataset,
    #     poison_idx=np.ones(len(dataset_once.dataset)),
    #     # one-hot to determine which image may take bd_transform
    #     bd_image_pre_transform=AddMatrixPatchTrigger(trigger_pattern.cpu().numpy().transpose((1,2,0))),
    #     bd_label_pre_transform=None, # none here, since we do not need
    #     ori_image_transform_in_loading=dataset_once.ori_image_transform_in_loading,
    #     ori_label_transform_in_loading=dataset_once.ori_label_transform_in_loading,
    #     add_details_in_preprocess=True,
    # )
    mix_adv.append(
        dataset_once
    )

mix_adv = ConcatDataset(mix_adv)

from utils.sync_dataset import syncDataset

sync_ds = syncDataset(mix_dataset_for_first_retrain, mix_adv)

sync_dl = DataLoader(
    sync_ds,
    batch_size=args.batchsize_for_poison,
    shuffle=True,
    drop_last=True
)

for epoch in range(args.poison_epochs):
    batch_loss = []
    for batch_idx, ((x, labels, _, _, _), (adv_x, _, _, _, _)),  in enumerate(sync_dl):

        net.train()
        net.to(device)

        trigger_pattern = trigger_pattern.to(device)
        trigger_pattern.requires_grad = False

        x, labels, adv_x = x.to(device), labels.to(device), adv_x.to(device)
        net.zero_grad()
        log_probs = net(x)

        _ = net(adv_x)

        def hook_function(module, input, output):
            net.feature_save = output

        net.__getattr__(args.target_layer_name).register_forward_hook(
            hook_function
        )
        feature_mix_with_bd = net.feature_save

        try:
            target_batch=next(target_dl_iter)
        except:
            target_dl_iter = iter(DataLoader(
                target_dataset,
                batch_size=args.batchsize_for_poison,
                shuffle=True,
                drop_last=True
            ))
            target_batch = next(target_dl_iter)

        target_x, target_y = target_batch[:2]
        target_x = target_x.to(device)

        _ = net(target_x)
        feature_target = net.feature_save

        loss = criterion(log_probs, labels.long()) + args.poison_lambda * mse(feature_mix_with_bd, feature_target)
        loss.backward()

        optimizer.step()

        batch_loss.append( (loss.item()))

    one_epoch_loss = sum(batch_loss) / len(batch_loss)

    if scheduler is not None:
        scheduler.step()

    logging.info(f'train, epoch_loss: {one_epoch_loss}')
    if args.frequency_save != 0 and epoch % args.frequency_save == args.frequency_save - 1:
        logging.info(f'saved. epoch:{epoch}')
        trainer.save_all_state_to_path(
            epoch=epoch,
            path=f"{save_path}/{'poison_retrain'}_epoch_{epoch}.pt")

# here no need to finetune as paper, since we can assume student drop it immediately in program

net.__setattr__(args.final_layer_name,
                torch.nn.Linear(
                    in_features= net.__getattr__(args.final_layer_name).in_features,
                    out_features= args.num_classes_for_student,
                )
            )

# fix layers before( including chosen layer)
for name, param in net.named_parameters():
    param.requires_grad = False
    if name.split('.')[0] == args.last_student_freeze_layer_name:
        break

# here should use the adv_test for first retrain, as how paper does
from utils.backdoor_generate_pindex import generate_pidx_from_label_transform
from utils.aggregate_block.bd_attack_generate import  bd_attack_label_trans_generate

from utils.bd_img_transform.patch import AddMatrixPatchTrigger

test_bd_img_transform = AddMatrixPatchTrigger((trigger_pattern.cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))

bd_label_transform = bd_attack_label_trans_generate(args)

from copy import deepcopy
from utils.backdoor_generate_pindex import generate_pidx_from_label_transform, generate_single_target_attack_train_pidx
test_pidx = generate_pidx_from_label_transform(
    benign_test_dl.dataset.targets,
    label_transform=bd_label_transform,
    is_train=False,
)

student_adv_test_dataset =  prepro_cls_DatasetBD(
    deepcopy(test_dataset_without_transform),
    poison_idx=test_pidx,
    bd_image_pre_transform=test_bd_img_transform,
    bd_label_pre_transform=bd_label_transform,
    ori_image_transform_in_loading=test_img_transform,
    ori_label_transform_in_loading=test_label_transform,
    add_details_in_preprocess=True,
)

student_adv_test_dataset.subset(
    np.where(test_pidx == 1)[0]
)

student_adv_test_dl = DataLoader(
    dataset = student_adv_test_dataset,
    batch_size= args.batch_size,
    shuffle= False,
    drop_last= False,
)

student_train_ds = prepro_cls_DatasetBD(
        full_dataset_without_transform=student_train_dataset_without_transform,
        poison_idx=np.zeros(len(student_train_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=student_train_img_transform,
        ori_label_transform_in_loading=student_train_label_transfrom,
        add_details_in_preprocess=True,
    )

# this parr used, attacker takes
student_train_ds.subset(
    np.where( [i for i in np.arange(len(student_train_ds)) if i not in student_train_index_in_attack])[0]
)

student_train_dl = DataLoader(
    student_train_ds,
    batch_size=args.student_batch_size,
    shuffle=True,
    drop_last=True
)

student_test_dl = DataLoader(
    prepro_cls_DatasetBD(
        student_test_dataset_without_transform,
        poison_idx=np.zeros(len(student_test_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=student_test_img_transform,
        ori_label_transform_in_loading=student_test_label_transform,
        add_details_in_preprocess=True,
    ),
    batch_size=args.student_batch_size,
    shuffle=False,
    drop_last=False,
)

student_transfer_args = deepcopy(args)
student_transfer_args.__dict__ = {
    k[17:] : v for k, v in student_transfer_args.__dict__.items() if 'student_transfer_' in k
}

optimizer, scheduler = argparser_opt_scheduler(net, student_transfer_args)

#TODO here I share the trainer settings
trainer.train_with_test_each_epoch(
            train_data = student_train_dl,
            test_data = student_test_dl,
            adv_test_data =  student_adv_test_dl,
            end_epoch_num = args.student_epochs,
            criterion = criterion,
            optimizer = optimizer,
            scheduler = scheduler,
            device = device,
            frequency_save = args.frequency_save,
            save_folder_path = save_path,
            save_prefix = 'student_training',
            continue_training_path = None,
)


from utils.save_load_attack import save_attack_result

save_attack_result(
    model_name = args.model,
    num_classes = args.num_classes,
    model = trainer.model.cpu().state_dict(),
    data_path = args.dataset_path,
    img_size = args.img_size,
    clean_data = args.dataset,
    bd_train = None,
    bd_test = student_adv_test_dl.dataset,
    save_path = save_path,
)