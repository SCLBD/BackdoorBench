import sys, yaml, os

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
from pprint import pformat
import numpy as np
import torch
import time
import logging
from torchvision import transforms
from typing import List,Union
from torch.nn.modules import ReLU,LeakyReLU, InstanceNorm2d, Upsample, Tanh

from utils.aggregate_block.save_path_generate import generate_save_folder
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape, dataset_and_transform_generate, get_dataset_denormalization
from utils.aggregate_block.fix_random import fix_random
from utils.bd_dataset import prepro_cls_DatasetBD, xy_iter
from torch.utils.data import DataLoader
from utils.backdoor_generate_pindex import generate_pidx_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from copy import deepcopy
from utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.save_load_attack import save_attack_result, sample_pil_imgs
from utils.log_assist import get_git_info
from utils.conv_pad_same import Conv2d # This is special implementation mimic the Conv2d in tf with padding = same
from utils.pytorch_ssim import ssim

to_pil = transforms.ToPILImage()

class syncDataset(torch.utils.data.Dataset):
    def __init__(self, *kwargs):
        self.kwargs = kwargs
        assert min([len(dataset) for dataset in self.kwargs]) == max([len(dataset) for dataset in self.kwargs])
    def __getitem__(self, index):
        return [dataset[index] for dataset in self.kwargs]
    def __len__(self):
        return len(self.kwargs[0])

def transform_to_denormalization(
        transform_given
):
    denormalizer = torch.nn.Identity() # avoid non-define case for dataset without a normalization
    for trans_t in deepcopy(transform_given.transforms):
        if isinstance(trans_t, transforms.Normalize):
            denormalizer = get_dataset_denormalization(trans_t)
    return denormalizer

class ge_conv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, num_features):
        super(ge_conv2d, self).__init__()
        self.conv2d = Conv2d(in_channels=in_channels,
                             out_channels = out_channels,
                             kernel_size=12,
                             stride=2,
                             padding='same')
        self.leakyrelu = LeakyReLU(negative_slope = 0.2)
        self.instancenorm = InstanceNorm2d(num_features=num_features)

    def forward(self,x):
        x = self.conv2d(x)
        x = self.leakyrelu(x)
        x = self.instancenorm(x)
        return x

class de_conv2d(torch.nn.Module):

    def __init__(self, figsize, in_channels, out_channels, num_features):
        super(de_conv2d, self).__init__()
        self.upsample = Upsample(size = figsize)
        self.conv2d = Conv2d(in_channels=in_channels,
                             out_channels = out_channels, kernel_size=12, stride=1, padding='same',)
        self.relu = ReLU()
        self.instancenorm = InstanceNorm2d(num_features=num_features)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.conv2d(x)
        x = self.instancenorm(x)
        x = torch.cat([x, skip],dim = 1)
        return x

class detoxicant_net(torch.nn.Module):

    def __init__(self, input_shape):
        super(detoxicant_net, self).__init__()
        self.conv1 = ge_conv2d(3,32,32)
        self.conv2 = ge_conv2d(32,64,64)
        self.conv3 = ge_conv2d(64,128,128)
        self.deconv1 = de_conv2d((int(input_shape[0]/4), int(input_shape[1]/4)), 128,64,64, )
        self.deconv2 = de_conv2d((int(input_shape[0]/2), int(input_shape[1]/2)), 128,32,32, )
        self.upsample = Upsample(size = (input_shape[0], input_shape[1]))
        self.conv4 = Conv2d(in_channels=64, out_channels =3, kernel_size=12, stride=1, padding='same', )
        self.tanh = Tanh()

    def forward(self,d0):
        # Downsampling
        d1 = self.conv1(d0) #32
        d2 = self.conv2(d1) #64
        d3 = self.conv3(d2) #128

        # Upsampling
        u1 = self.deconv1(d3, d2)
        u2 = self.deconv2(u1, d1)

        u3 = self.upsample(u2)
        output_img = self.tanh(self.conv4(u3))

        return output_img

def total_variation(
        torch_batch,
):
    '''
    rewrite from https://github.com/tensorflow/tensorflow/blob/v2.7.0/tensorflow/python/ops/image_ops_impl.py#L3213-L3282
    this is the torch implemment
    '''
    tf_batch = torch.permute(torch_batch, (0,2,3,1))

    pixel_dif1 = tf_batch[:, 1:, :, :] - tf_batch[:, :-1, :, :]
    pixel_dif2 = tf_batch[:, :, 1:, :] - tf_batch[:, :, :-1, :]

    sum_axis = [1, 2, 3]

    tot_var = (
        torch.sum(torch.abs(pixel_dif1), dim=sum_axis) +
        torch.sum(torch.abs(pixel_dif2), dim=sum_axis))

    return tot_var


def reverse_engineer_one_neuron(
        poison_model,
        benign_ds, # should have no transforms
        batch_size,
        input_shape,
        trojan_layer_name,
        torjan_neuron_idx,
        torjan_label,
        lr,
        device,
):
    name_module_dict = {name: module for name, module in poison_model.named_modules()}

    reverse_engineer_model = detoxicant_net(
        input_shape[:2],
        )
    optimizer = torch.optim.Adam(
        reverse_engineer_model.parameters(),
        lr,
    )

    def one_step(benign_batch, train=True):

        optimizer.zero_grad()
        if train:
            reverse_engineer_model.train()
        else:
            reverse_engineer_model.eval()
        poison_model.to(device)
        poison_model.eval()
        reverse_engineer_model.to(device)
        benign_batch = benign_batch.to(device)
        reverse_batch = reverse_engineer_model(benign_batch)

        feature_list = []
        def hook_function(module, input, output):
            # poison_model.benign_feature = output
            feature_list.append(output)
        handle = name_module_dict[trojan_layer_name].register_forward_hook(
            hook_function
        )
        benign_logits = poison_model(benign_batch)
        handle.remove()

        def hook_function(module, input, output):
            # poison_model.reverse_feature = output
            feature_list.append(output)
        handle = name_module_dict[trojan_layer_name].register_forward_hook(
            hook_function
        )
        reverse_logits = poison_model(reverse_batch)
        handle.remove()

        benign_feature, reverse_feature, *other = feature_list
        vloss1 = torch.sum(reverse_feature[:,torjan_neuron_idx,:,:])
        vloss2 = torch.sum(reverse_feature[:,:torjan_neuron_idx,:,:]) + torch.sum(reverse_feature[:,torjan_neuron_idx+1:,:,:])
        vloss_benign = torch.sum(benign_feature[:,torjan_neuron_idx,:,:])
        tvloss = torch.sum(total_variation(reverse_batch))

        loss = - vloss1 + 0.00001 * vloss2 + 0.001 * tvloss

        ssim_loss = -torch.mean(ssim(reverse_batch, benign_batch))
        if ssim_loss.item() > -0.6:
            loss = 0.01 * loss + 100000 * ssim_loss
        else:
            loss = 0.01 * loss + 100 * ssim_loss

        if train:
            loss.backward()
            optimizer.step()

        return vloss_benign.item(), reverse_logits.detach().clone().cpu(), loss.item(), vloss1.item(), vloss2.item(), tvloss.item(), ssim_loss.item(), reverse_batch.detach().clone().cpu()

    benign_dl = DataLoader(
        dataset=benign_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    result_max_ratio = 0
    result_ssim_loss = 0
    save_location = None
    result_parameter = None
    for e in range(1000):
        vloss_benign_list = []
        reverse_logits_list = []
        ssim_loss_list = []
        reverse_batch_list = []
        vloss1_list = []
        for batch_x, _, *_ in benign_dl:
            vloss_benign, reverse_logits, loss, vloss1, vloss2, tvloss, ssim_loss, reverse_batch = one_step(batch_x)
            vloss_benign_list.append(vloss_benign)
            reverse_logits_list.append(reverse_logits)
            vloss1_list.append(vloss1)
            ssim_loss_list.append(ssim_loss)
            reverse_batch_list.append(reverse_batch)
        vloss_benign = np.array(vloss_benign_list).mean()
        vloss1 = np.array(vloss1_list).mean()
        ssim_loss = np.array(ssim_loss_list).mean()
        _, predicted = torch.max(torch.cat(reverse_logits_list), -1)
        ratio = predicted.eq(torjan_label).float().mean()
        dif = vloss1 - vloss_benign
        if ratio > result_max_ratio and \
            ssim_loss < -0.6 and \
            dif > 30 and \
            dif > 0.5 * vloss_benign :
            result_max_ratio = ratio
            result_ssim_loss = ssim_loss
            result_parameter = deepcopy(reverse_engineer_model.cpu().state_dict())
            logging.info(f"update epoch {e}, result_max_ratio:{result_max_ratio}  result_ssim_loss:{result_ssim_loss}")
        # logging.info(f"epoch :{e}, vloss_benign:{vloss_benign}, loss:{loss}, vloss1:{vloss1}, vloss2:{vloss2}, tvloss:{tvloss}, ssim_loss:{ssim_loss}, ratio:{ratio}, dif:{dif}")

    return result_max_ratio, result_ssim_loss, result_parameter






def neuron_detection_for_layers(
    net: torch.nn.Module,
    layer_name_list : List[str],
    benign_dataset,
    batch_size,
    adv_dataset,
    device,
):
    '''
    save the activation for CONV layer. (in original code it only suppport layer with 4 dim output)
    each layer we get the largest benign value
    then sum over (all dim except 0,1). take mean over dim 0
    '''
    sync_dl = DataLoader(
        dataset=syncDataset(benign_dataset, adv_dataset),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # generate {name:module} dict
    name_module_dict = {name: module for name, module in net.named_modules()}

    benign_layer_extract_list_dict = {layer_name: [] for layer_name in layer_name_list}
    adv_layer_extract_list_dict = {layer_name: [] for layer_name in layer_name_list}

    # get all intermediate activation for benign and adv data
    for name in layer_name_list:
        for (benign_x, _, *other), (adv_x, _, *other) in sync_dl:
            net.eval()
            net.to(device)
            benign_x, adv_x = benign_x.to(device), adv_x.to(device)

            def hook_function(module, input, output):
                benign_layer_extract_list_dict[name].append(
                    output.detach().clone().cpu()
                )

            handle = (name_module_dict[name].register_forward_hook(
                hook_function
            ))

            _ = net(benign_x)

            handle.remove()

            def hook_function(module, input, output):
                adv_layer_extract_list_dict[name].append(
                    output.detach().clone().cpu()
                )

            handle = (name_module_dict[name].register_forward_hook(
                hook_function
            ))
            _ = net(adv_x)

            handle.remove()

    benign_layer_extract_list_dict = {
        layer_name: torch.cat(layer_activation_list, dim=0) for layer_name, layer_activation_list in
        benign_layer_extract_list_dict.items()
    }
    adv_layer_extract_list_dict = {
        layer_name: torch.cat(layer_activation_list, dim=0) for layer_name, layer_activation_list in
        adv_layer_extract_list_dict.items()
    }

    selection_layer_neuron_dict = {}
    for layer_name in layer_name_list:
        benign_activation = benign_layer_extract_list_dict[layer_name]
        benign_max_value = benign_activation.max()
        benign_value_per_channel = torch.mean(
            torch.sum(benign_activation, dim=[i for i in np.arange(len(benign_activation.shape)) if i != 0 and i != 1]),
            dim=0)

        adv_activation = adv_layer_extract_list_dict[layer_name]
        adv_value_per_channel = torch.mean(
            torch.sum(adv_activation, dim=[i for i in np.arange(len(benign_activation.shape)) if i != 0 and i != 1]),
            dim=0)

        dif = adv_value_per_channel - benign_value_per_channel

        selection_pos = (dif > 5 * benign_max_value) & (dif > benign_value_per_channel)
        selection_layer_neuron_dict[layer_name] = np.where(selection_pos.detach().clone().cpu().numpy())[0]
        logging.info(
            f"layer:{layer_name},benign_max:{benign_max_value}, selection_pos_count:{selection_pos.sum().item()} \n selection_pos:{selection_pos},\n  benign_value_per_channel:{benign_value_per_channel},\n adv_value_per_channel:{adv_value_per_channel}")

    return selection_layer_neuron_dict

def keep_normalization_resize_totensor_only(
        given_transform,
):
    return transforms.Compose(
        list(
            filter(
                lambda x: isinstance(x,
                                     (transforms.Normalize, transforms.Resize, transforms.ToTensor)
                                     ),
                    given_transform.transforms
            )
        )
    )

def get_part_for_each_label(
        y: np.ndarray,
        percent_or_num: Union[int, float],
):
    '''
    use in generate sunrise set, each label take a percentage or num
    if take
    '''
    unique_label_values = np.unique(y)
    select_pos = []
    if percent_or_num >= 1 :
        for one_label_value in unique_label_values:
            label_value_pos = np.where(y == one_label_value)[0]
            select_pos += np.random.choice(label_value_pos,
                                           size=int(
                                               min(
                                                   percent_or_num,
                                                   len(label_value_pos)
                                               )
                                           ),
                                           replace=False,
                                           ).tolist()
    else:
        for one_label_value in unique_label_values:
            label_value_pos = np.where(y == one_label_value)[0]
            select_pos += np.random.choice(label_value_pos,
                                                size = int(
                                                        min(
                                                            np.ceil(percent_or_num*len(label_value_pos)), # ceil to make sure that at least one sample each label
                                                            len(label_value_pos)
                                                        )
                                                ),
                                                replace=False,
                                                ).tolist()
    return select_pos

def generate_with_reverse_network(model_path, input_shape, ds :prepro_cls_DatasetBD, batch_size, device):
    reverse_generator = detoxicant_net(input_shape)
    reverse_generator.load_state_dict(torch.load(model_path, map_location='cpu'))
    reverse_generator.eval()

    denormalizer = transform_to_denormalization(ds.ori_image_transform_in_loading)
    dl = DataLoader(ds, batch_size = batch_size, shuffle = False, drop_last=False)
    reversed_batch_list = []
    target_list = []
    with torch.no_grad():
        for batch_x, batch_y in dl:
            reverse_generator.to(device)
            batch_x = batch_x.to(device)
            reversed_batch_x = reverse_generator(batch_x)
            reversed_batch_list.append(denormalizer(reversed_batch_x.detach().clone().cpu()))
            target_list.append(batch_y)
        reverse_tensor = torch.cat(reversed_batch_list)
        target = torch.cat(target_list)
    pil_imgs = [to_pil(t_img) for t_img in reverse_tensor]
    return pil_imgs, target



def test_with_reverse_network(model, criterion, test_dataloader, input_shape, reverse_engineer_network_params, device):
    model.to(device)
    model.eval()

    metrics = {
        'test_correct': 0,
        'test_loss': 0,
        'test_total': 0,
        # 'detail_list' : [],
    }

    criterion = criterion.to(device)

    net = detoxicant_net(input_shape)
    net.load_state_dict(reverse_engineer_network_params)
    net.to(device)
    net.eval()

    with torch.no_grad():
        for batch_idx, (x, target, *additional_info) in enumerate(test_dataloader):
            x = x.to(device)
            target = target.to(device)
            rv_batch = net(x)
            pred = model(rv_batch)
            loss = criterion(pred, target.long())

            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            metrics['test_correct'] += correct.item()
            metrics['test_loss'] += loss.item() * target.size(0)
            metrics['test_total'] += target.size(0)

    if metrics['test_total'] != 0:
        ratio = metrics['test_correct']/metrics['test_total']
    else:
        ratio = None
    return ratio, metrics['test_loss'], rv_batch.detach().clone().cpu()


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    # parser.add_argument('--mode', type=str,
    #                     help='classification/detection/segmentation')
    parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])
    parser.add_argument('--device', type=str)
    parser.add_argument('--attack', type=str, )
    parser.add_argument('--yaml_path', type=str, default='../config/dfstAttack/default.yaml',
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--lr_scheduler', type=str,
                        help='which lr_scheduler use for optimizer')
    # only all2one can be use for clean-label
    parser.add_argument('--attack_label_trans', type=str,
                        help='which type of label modification in backdoor attack'
                        )
    parser.add_argument('--pratio', type=float,
                        help='the poison rate '
                        )
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dataset', type=str,
                        help='which dataset to use'
                        )
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--attack_target', type=int,
                        help='target class in all2one attack')
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
    parser.add_argument('--attack_train_replace_imgs_path', type=str)
    parser.add_argument('--attack_test_replace_imgs_path', type=str)
    return parser


def main():
## 1. config args, save_path, fix random seed
    parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)

    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})

    args.__dict__ = defaults

    args.terminal_info = sys.argv

    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"

    if ('attack_train_replace_imgs_path' not in args.__dict__) or (args.attack_train_replace_imgs_path is None):
        args.attack_train_replace_imgs_path = f"../resource/dfst/{args.dataset}_sunrise_x_train.npy"
        print(f"args.attack_train_replace_imgs_path does not found, so = {args.attack_train_replace_imgs_path}")

    if ('attack_test_replace_imgs_path' not in args.__dict__) or (args.attack_test_replace_imgs_path is None):
        args.attack_test_replace_imgs_path = f"../resource/dfst/{args.dataset}_sunrise_x_test.npy"
        print(f"args.attack_test_replace_imgs_path does not found, so = {args.attack_test_replace_imgs_path}")

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

    try:
        logging.info(pformat(get_git_info()))
    except:
        logging.info('Getting git info fails.')

    ### set the random seed
    fix_random(int(args.random_seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net = generate_cls_model(
        model_name=args.model,
        num_classes=args.num_classes,
    )
    net.load_state_dict(torch.load(args.pretrained_model_path, map_location=device)['model_state_dict'])
    logging.info(f'load args.pretrained_model_path:{args.pretrained_model_path}')

    # as data_poisoning, make the data set

    ### 2. set the clean train data and clean test data
    train_dataset_without_transform, \
    train_img_transform, \
    train_label_transfrom, \
    test_dataset_without_transform, \
    test_img_transform, \
    test_label_transform = dataset_and_transform_generate(args)

    benign_train_ds = prepro_cls_DatasetBD(
            full_dataset_without_transform=train_dataset_without_transform,
            poison_idx=np.zeros(len(train_dataset_without_transform)),
            # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=train_img_transform,
            ori_label_transform_in_loading=train_label_transfrom,
            add_details_in_preprocess=True,
        )

    benign_test_dl = DataLoader(
        prepro_cls_DatasetBD(
            test_dataset_without_transform,
            poison_idx=np.zeros(len(test_dataset_without_transform)),
            # one-hot to determine which image may take bd_transform
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

    train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)
    bd_label_transform = bd_attack_label_trans_generate(args)

    train_pidx = get_part_for_each_label(benign_train_ds.targets, args.pratio)
    train_pidx_zero_one = np.zeros(len(train_dataset_without_transform))
    train_pidx_zero_one[train_pidx] = 1

    torch.save(train_pidx,
               args.save_path + '/train_pidex_list.pickle',
               )

    adv_train_part_ds = prepro_cls_DatasetBD(
        deepcopy(train_dataset_without_transform),
        poison_idx=train_pidx_zero_one,
        bd_image_pre_transform=train_bd_img_transform,
        bd_label_pre_transform=bd_label_transform,
        ori_image_transform_in_loading=keep_normalization_resize_totensor_only(train_img_transform), # only be used in neuron selection
        ori_label_transform_in_loading=(train_label_transfrom),
        add_details_in_preprocess=True,
    )

    adv_train_part_ds.subset(train_pidx)

    adv_retrain_ds = xy_iter(
        benign_train_ds.data + adv_train_part_ds.data,
        benign_train_ds.targets.tolist() + adv_train_part_ds.targets.tolist(),
        transform=train_img_transform,
    )
    adv_retrain_dl = DataLoader(
        adv_retrain_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    ### decide which img to poison in ASR Test
    test_pidx = generate_pidx_from_label_transform(
        benign_test_dl.dataset.targets,
        label_transform=bd_label_transform,
        train=False,
    )

    ### generate test dataset for ASR
    adv_test_dataset = prepro_cls_DatasetBD(
        deepcopy(test_dataset_without_transform),
        poison_idx=test_pidx,
        bd_image_pre_transform=test_bd_img_transform,
        bd_label_pre_transform=bd_label_transform,
        ori_image_transform_in_loading=test_img_transform,
        ori_label_transform_in_loading=test_label_transform,
        add_details_in_preprocess=True,

    )

    # delete the samples that do not used for ASR test (those non-poisoned samples)
    adv_test_dataset.subset(
        np.where(test_pidx == 1)[0]
    )

    adv_test_dl = DataLoader(
        dataset=adv_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # start retrain with poisoned

    trainer = generate_cls_trainer(
        net,
        args.attack,
        args.amp,
    )
    criterion = argparser_criterion(args)
    optimizer, scheduler = argparser_opt_scheduler(net, args)
    if 'poison_model_save_path_for_test' in args and os.path.exists(f"{args.poison_model_save_path_for_test}"):
        logging.info(f'load the poison_model, test mode. path : {args.poison_model_save_path_for_test}')
        net.load_state_dict(
            torch.load(
                args.poison_model_save_path_for_test,
                map_location='cpu',
            )['model_state_dict']
        )
    else:
        logging.info('No poison_model find.')
        trainer.train_with_test_each_epoch_v2(
            train_data=adv_retrain_dl,
            test_dataloader_dict={
                'adv_retrain_dl':adv_retrain_dl,
                'benign_test':benign_test_dl,
                'adv_test':adv_test_dl,
            },
            end_epoch_num=args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            frequency_save=args.frequency_save,
            save_folder_path=save_path,
            save_prefix='retrain_with_poison',
            continue_training_path=None,
        )

    #neuron detection part
    benign_train_for_neuron_detection_ds = prepro_cls_DatasetBD(
            full_dataset_without_transform=train_dataset_without_transform,
            poison_idx=np.zeros(len(train_dataset_without_transform)),
            # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=keep_normalization_resize_totensor_only(train_img_transform),
            ori_label_transform_in_loading=(train_label_transfrom),
            add_details_in_preprocess=False,
        ).subset(
        train_pidx,
        inplace = False,
        memorize_original=False
    )

    # for each layer get a list of channels
    selection_layer_neuron_dict = neuron_detection_for_layers(
        net, args.layer_name_list, benign_train_for_neuron_detection_ds, args.batch_size, adv_train_part_ds, device
    )
    logging.info(f"selection_layer_neuron_dict:{selection_layer_neuron_dict}")
    # the reverse engineering part

    each_label_selected = get_part_for_each_label(benign_train_ds.targets, 0.002)
    logging.info(f"reverse_engineer_select_idx:{each_label_selected}")
    torch.save(each_label_selected, f"{save_path}/reverse_engineer_select_idx.pth")
    reverse_engineer_ds = prepro_cls_DatasetBD(
            full_dataset_without_transform=train_dataset_without_transform,
            poison_idx=np.zeros(len(train_dataset_without_transform)),
            # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=keep_normalization_resize_totensor_only(train_img_transform),
            ori_label_transform_in_loading=(train_label_transfrom),
            add_details_in_preprocess=False,
        ).subset(
        each_label_selected,
        inplace = False,
        memorize_original=False
    )

    each_label_selected = get_part_for_each_label(benign_test_dl.dataset.targets, 0.1)
    logging.info(f"linear_test_select_idx:{each_label_selected}")
    torch.save(each_label_selected, f"{save_path}/linear_test_select_idx.pth")
    linear_test_prototype_ds = prepro_cls_DatasetBD(
            full_dataset_without_transform=test_dataset_without_transform,
            poison_idx=np.zeros(len(test_dataset_without_transform)),
            # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=keep_normalization_resize_totensor_only(test_img_transform),
            ori_label_transform_in_loading=(test_label_transform),
            add_details_in_preprocess=False,
        ).subset(
        each_label_selected,
        inplace = False,
        memorize_original=False
    )
    linear_test_prototype_ds = xy_iter(linear_test_prototype_ds.data,
                                       np.ones_like(linear_test_prototype_ds.targets)*args.attack_target,
                                       transform=linear_test_prototype_ds.ori_image_transform_in_loading)
    linear_test_denormalization = transform_to_denormalization(linear_test_prototype_ds.transform)
    linear_test_prototype_dl = DataLoader(linear_test_prototype_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,)
    trainer = generate_cls_trainer(net)
    trainer.criterion = argparser_criterion(args)

    detoxicant_dict = {}
    for layer_name, neuron_list in selection_layer_neuron_dict.items():
        for neuron_idx in neuron_list:
            result_max_ratio, result_ssim_loss, result_parameter = reverse_engineer_one_neuron(
                net,
                reverse_engineer_ds,  # should have no transforms
                args.batch_size,
                (args.input_width, args.input_height),
                layer_name,
                neuron_idx,
                args.attack_target,
                args.reverse_engineer_lr,
                device,
            )
            if result_max_ratio > args.detoxicant_ratio_require:
                test_ratio, test_loss, one_batch = test_with_reverse_network(
                    net,
                    criterion,
                    linear_test_prototype_dl,
                    (args.input_height, args.input_width),
                    result_parameter,
                    device,
                )
                logging.info(f"detoxicant test, ASR:{test_ratio}, tset_loss:{test_loss}")
                if test_ratio > args.detoxicant_ratio_require:
                    save_location = f"{save_path}/layer_{layer_name}_neuron_{neuron_idx}.pth"

                    pil_image_list = [to_pil(tensor_img) for tensor_img in linear_test_denormalization(one_batch)]
                    sample_pil_imgs(pil_image_list, f"{save_path}/layer_{layer_name}_neuron_{neuron_idx}_samples", num=5,)
                    torch.save(result_parameter, save_location)
                    logging.info(f'One detoxicant added. layer:{layer_name}, neuron_idx:{neuron_idx},\n    ASR:{test_ratio}, tset_loss:{test_loss}, result_max_ratio:{result_max_ratio}, result_ssim_loss:{result_ssim_loss}, \n    save_location:{save_location}')
                    detoxicant_dict[(layer_name, neuron_idx)] = (result_max_ratio, result_ssim_loss, save_location)

    logging.info(f"All detoxicant_dict:{detoxicant_dict}")

    # the final retrain part, for train
    each_label_selected = get_part_for_each_label(benign_train_ds.targets, 0.01)
    logging.info(f"final_denoise_select_idx:{each_label_selected}")
    torch.save(each_label_selected, f"{save_path}/final_denoise_select_idx.pth")
    final_denoise_preprocess_ds = prepro_cls_DatasetBD(
            full_dataset_without_transform=train_dataset_without_transform,
            poison_idx=np.zeros(len(train_dataset_without_transform)),
            # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=keep_normalization_resize_totensor_only(train_img_transform),
            ori_label_transform_in_loading=(train_label_transfrom),
            add_details_in_preprocess=False,
        ).subset(
        each_label_selected,
        inplace = False,
        memorize_original=False
    )
    # generate denoise img from above
    all_reverse_generate_pil_imgs = []
    all_target = []
    for (layer_name, neuron_idx), (result_max_ratio, result_ssim_loss, save_location) in detoxicant_dict.items():
        pil_imgs, target = generate_with_reverse_network(save_location, (args.input_height,args.input_width), final_denoise_preprocess_ds, args.batch_size, device)
        all_reverse_generate_pil_imgs += pil_imgs
        all_target+=target.numpy().tolist()
        sample_pil_imgs(pil_imgs, f"{save_path}/{layer_name}_{neuron_idx}_final_denoise_samples")
    logging.info(f"final denoise train dataset len = retrain {len(adv_retrain_ds.targets)} + denoise {len(all_reverse_generate_pil_imgs)}")

    denoise_train_pre_ds = xy_iter(
        adv_retrain_ds.data + all_reverse_generate_pil_imgs,
        adv_retrain_ds.targets + all_target,
        None,
    )
    poison_idx_for_denoise_train = np.zeros(len(adv_retrain_ds.data + all_reverse_generate_pil_imgs))
    # To indicate all added data samples as 'poison', for final save_result.
    poison_idx_for_denoise_train[
        np.arange(len(benign_train_ds), len(adv_retrain_ds.data + all_reverse_generate_pil_imgs))
        ] = 1
    denoise_train_ds = prepro_cls_DatasetBD(
            full_dataset_without_transform=denoise_train_pre_ds,
            poison_idx=poison_idx_for_denoise_train,
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=train_img_transform,
            ori_label_transform_in_loading=train_label_transfrom,
            add_details_in_preprocess=True,
            clean_image_pre_transform = None,
            end_pre_process= None,
    )
    denoise_train_ds.original_targets =  np.array(benign_train_ds.targets.tolist() + adv_train_part_ds.original_targets.tolist() + all_target)
    denoise_train_dl = DataLoader(denoise_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,)

    each_label_selected = get_part_for_each_label(benign_test_dl.dataset.targets, 0.05)
    logging.info(f"final_denoise_test_select_idx:{each_label_selected}")
    torch.save(each_label_selected, f"{save_path}/final_denoise_test_select_idx.pth")
    final_denoise_test_preprocess_ds = prepro_cls_DatasetBD(
            full_dataset_without_transform=test_dataset_without_transform,
            poison_idx=np.zeros(len(test_dataset_without_transform)),
            # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=keep_normalization_resize_totensor_only(test_img_transform),
            ori_label_transform_in_loading=(test_label_transform),
            add_details_in_preprocess=False,
        ).subset(
        each_label_selected,
        inplace = False,
        memorize_original=False
    )
    all_test_reverse_generate_pil_imgs = []
    all_test_target = []
    for (layer_name, neuron_idx), (result_max_ratio, result_ssim_loss, save_location) in detoxicant_dict.items():
        pil_imgs, target = generate_with_reverse_network(save_location, (args.input_height,args.input_width), final_denoise_test_preprocess_ds, args.batch_size, device)
        all_test_reverse_generate_pil_imgs += pil_imgs
        all_test_target+=target.numpy().tolist()
        sample_pil_imgs(pil_imgs, f"{save_path}/{layer_name}_{neuron_idx}_final_denoise_test_samples")

    denoise_test_ds = xy_iter(
        all_test_reverse_generate_pil_imgs,
        all_test_target,
        test_img_transform,
    )
    denoise_test_dl = DataLoader(denoise_test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,)

    trainer = generate_cls_trainer(
        net,
        args.attack,
        args.amp,
    )
    criterion = argparser_criterion(args)
    optimizer, scheduler = argparser_opt_scheduler(net, args)
    trainer.train_with_test_each_epoch_v2(
        train_data=denoise_train_dl,
        test_dataloader_dict={
            'final_denoise_train_dl':denoise_train_dl,
            'benign_test':benign_test_dl,
            'adv_test':adv_test_dl,
            'denoise_test_dl':denoise_test_dl,
        },
        end_epoch_num= int(args.epochs * 1.2),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        frequency_save=args.frequency_save,
        save_folder_path=save_path,
        save_prefix='final_denoise',
        continue_training_path=None,
    )

    save_attack_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=trainer.model.cpu().state_dict(),
            data_path=args.dataset_path,
            img_size=args.img_size,
            clean_data=args.dataset,
            bd_train=denoise_train_ds,
            bd_test=adv_test_dataset,
            save_path=save_path,
        )

if __name__ == '__main__':
    main()