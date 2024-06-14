'''
This script is for normal training process, no any attack is applied
'''

import sys, yaml, os

os.chdir(sys.path[0])
sys.path.append('../../')
os.getcwd()

from copy import deepcopy

import argparse
from pprint import  pformat
import numpy as np
import torch
import time
import logging
from PIL import Image
from typing import Union
from utils.aggregate_block.save_path_generate import generate_save_folder
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
from utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer
from universal_pert import universal_perturbation
from torchvision.transforms import Resize
from torchvision import transforms
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform, y_iter

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

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])
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
    parser.add_argument('--attack_target')
    parser.add_argument('--clean_model_path', type = str)

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
            all_record_folder_path='../../record',
        )
    else:
        save_path = '../../record/' + args.save_folder_name
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
                train_label_transform, \
    test_dataset_without_transform, \
                test_img_transform, \
                test_label_transform = dataset_and_transform_generate(args)

    benign_train_ds = train_dataset_without_transform

    eval_ds = prepro_cls_DatasetBD_v2(
        deepcopy(test_dataset_without_transform),
        poison_indicator=None,
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        save_folder_path=f"{args.save_path}/bd_test_dataset",
    )

    eval_ds_target = np.array(i for i in y_iter(eval_ds))

    eval_ds.subset(
        get_part_for_each_label(eval_ds_target, 10)
    )

    eval_ds = dataset_wrapper_with_transform(
        eval_ds,
        test_img_transform,
        test_label_transform,
    )

    net  = generate_cls_model(
        model_name=args.model,
        num_classes=args.num_classes,
    )

    try:
        net.load_state_dict(
            torch.load(
                args.clean_model_path,
                map_location='cpu'
            )
        )
    except:
        net.load_state_dict(
            torch.load(
                args.clean_model_path,
                map_location='cpu'
            )['model']
        )

    # just get 100 pil from benign train data
    random100 = np.random.choice(benign_train_ds.__len__(), 100, replace=False) #1, replace=False)
    # benign_train_ds.subset(random100)
    # dataset_pil = benign_train_ds.data

    dataset_pil = []
    for selected_img_idx in random100:
        pil_img, *other = benign_train_ds[selected_img_idx] # the img must be the first element.
        dataset_pil.append(pil_img)

    r = Resize((args.input_height, args.input_width))
    dataset_npy = np.concatenate(
        [
            np.array(
                r(pil_img)
            )[None,...].astype(np.float32)/255
            for pil_img in dataset_pil]
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    save_path_prefix = f'{save_path}/{args.dataset}_{args.model}'
    max_iter_uni = 50
    v = universal_perturbation(
        dataset_npy,
        eval_ds,
        net,
        target = args.attack_target,
        # delta=0.2,
        max_iter_uni=max_iter_uni, # 50 default 1 just for test speed
        num_classes=args.num_classes,
        overshoot=0.02,
        max_iter_df=200,
        device = device,
        save_path_prefix = save_path_prefix,
    )
    logging.info(f"max_iter_uni={max_iter_uni}")

    v_lossy_image = np.clip(deepcopy(v) * 255 + 255 / 2, 0, 255).squeeze()  # since v is [0,1]
    np.save(f'{save_path_prefix}.npy', v_lossy_image.astype(np.uint8))
    Image.fromarray(v_lossy_image.astype(np.uint8)).save(f'{save_path_prefix}_lossy.jpg')

    Image.fromarray(v_lossy_image.astype(np.uint8)).save(f'{save_path_prefix}_lossy.jpg')

    logging.info('end')



if __name__ == '__main__':
    main()
