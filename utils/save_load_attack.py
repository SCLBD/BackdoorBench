'''
This script aims to save and load the attack result as a bridge between attack and defense files.

Model, clean data, backdoor data and all infomation needed to reconstruct will be saved.

Note that in default, only the poisoned part of backdoor dataset will be saved to save space.

'''
import logging


import torch, os
from utils.bd_dataset import prepro_cls_DatasetBD
import numpy as np
from copy import deepcopy
from torchvision.transforms import ToTensor, Resize,Compose
from pprint import pformat
from typing import Union
from tqdm import tqdm
from PIL import Image

def summary_dict(input_dict):
    '''
    Input a dict, this func will do summary for it.
    :return:
    '''
    summary_dict_return = dict()
    for k,v in input_dict.items():
        if isinstance(v, dict):
            summary_dict_return[k] = summary_dict(v)
        elif isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            summary_dict_return[k] = {
                'shape':v.shape,
                'min':v.min(),
                'max':v.max(),
            }
        elif isinstance(v, list):
            summary_dict_return[k] = {
                'len':v.__len__(),
                'first ten':v[:10],
                'last ten':v[-10:],
            }
        else:
            summary_dict_return[k] = v
    return  summary_dict_return

def add_resize_and_subset_for_prepro_cls_DatasetBD(
        given_data: prepro_cls_DatasetBD,
        resize_list: list,
        only_bd: bool = False,
):
    resize_list = resize_list[:2]

    resize_bd_totensor = Compose([
        Resize(resize_list),
        # ToTensor(),
        # lambda x: torch.clamp(x, min=0, max=1),
        # lambda x: (x * 255).to(torch.long),
    ])

    all_img_r_t = []

    if only_bd:
        given_data.subset(np.where(np.array(given_data.poison_indicator) == 1)[0]) # only bd samples remain

    for img in tqdm(given_data.data, desc=f'resize'):
        img_r_t = resize_bd_totensor(
            img#Image.fromarray(img.astype(np.uint8))
        )
        all_img_r_t.append(img_r_t)
        #all_img_r_t.append(img_r_t[None, ...])
    # all_img_r_t = torch.cat(all_img_r_t, dim=0)

    return all_img_r_t, \
           given_data.targets, \
           given_data.original_index, \
           given_data.poison_indicator, \
           given_data.original_targets

def sample_pil_imgs(pil_image_list, save_folder, num = 5,):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    select_index = np.random.choice(
        len(pil_image_list),
        num,
    ).tolist() + np.arange(num).tolist() + np.arange(len(pil_image_list) - num, len(pil_image_list)).tolist()

    for ii in select_index:
        pil_image_list[ii].save(f"{save_folder}/{ii}.png")

def save_attack_result(
    model_name : str,
    num_classes : int,
    model : dict, # the state_dict
    data_path : str,
    img_size : Union[list, tuple],
    clean_data : str,
    bd_train : prepro_cls_DatasetBD, # dataset without transform
    bd_test : prepro_cls_DatasetBD, # dataset without transform
    save_path : str,
):
    '''

    main idea is to loop through the backdoor train and test dataset, and match with the clean dataset
    by remove replicated parts, this function can save the space.

    WARNING: keep all dataset with shuffle = False, same order of data samples is the basic of this function !!!!

    :param model_name : str,
    :param num_classes : int,
    :param model : dict, # the state_dict
    :param data_path : str,
    :param img_size : list, like [32,32,3]
    :param clean_data : str, clean dataset name
    :param bd_train : torch.utils.data.Dataset, # dataset without transform !!
    :param bd_test : torch.utils.data.Dataset, # dataset without transform
    :param save_path : str,
    '''

    if bd_train is not None:
        bd_train_x, bd_train_y, bd_train_original_index, bd_train_poison_indicator, bd_train_original_targets  = add_resize_and_subset_for_prepro_cls_DatasetBD(bd_train, img_size, only_bd=True)
    else:
        logging.info('bd_train is set to be None in saving process!')

    bd_test_x, bd_test_y, bd_test_original_index, bd_test_poison_indicator, bd_test_original_targets  = add_resize_and_subset_for_prepro_cls_DatasetBD(bd_test, img_size, only_bd=True)
    
    save_dict = {
            'model_name': model_name,
            'num_classes' : num_classes,
            'model': model,

            'data_path': data_path,
            'img_size' : img_size,


            'clean_data': clean_data,

            'bd_train': ({
                'x': bd_train_x,
                'y': bd_train_y,
                'original_index' : bd_train_original_index,
            } if bd_train is not None else {
                'x': None,
                'y': None,
                'original_index': None,
            }) ,

            'bd_test': {
                'x': bd_test_x,
                'y': bd_test_y,
                'original_index': bd_test_original_index,
                'original_targets': bd_test_original_targets,
            },
        }
    
    logging.info(f"saving...")
    logging.info(f"location : {save_path}/attack_result.pt, content summary :{pformat(summary_dict(save_dict))}")

    sample_pil_imgs(bd_train_x, f"{save_path}/save_bd_train_samples")
    sample_pil_imgs(bd_test_x, f"{save_path}/save_bd_test_samples")
    
    torch.save(
        save_dict,
        f'{save_path}/attack_result.pt',
    )

from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate

class Args:
    pass

def load_attack_result(
    save_path : str,
):
    '''
    This function first replicate the basic steps of generate models and clean train and test datasets
    then use the index given in files to replace the samples should be poisoned to re-create the backdoor train and test dataset

    save_path MUST have 'record' in its abspath, and data_path in attack result MUST have 'data' in its path!!!
    save_path : the path of "attack_result.pt"
    '''
    load_file = torch.load(save_path)

    if all(key in load_file for key in ['model_name',
        'num_classes',
        'model',
        'data_path',
        'img_size',
        'clean_data',
        'bd_train',
        'bd_test',
        ]):

        logging.info('key match for attack_result, processing...')

        # model = generate_cls_model(load_file['model_name'], load_file['num_classes'])
        # model.load_state_dict(load_file['model'])

        clean_setting = Args()

        clean_setting.dataset = load_file['clean_data']

        # convert the relative/abs path in attack result to abs path for defense
        clean_setting.dataset_path = load_file['data_path']
        logging.warning("save_path MUST have 'record' in its abspath, and data_path in attack result MUST have 'data' in its path")
        clean_setting.dataset_path = save_path[:save_path.index('record')] + clean_setting.dataset_path[clean_setting.dataset_path.index('data'):]

        clean_setting.img_size = load_file['img_size']

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transfrom, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform = dataset_and_transform_generate(clean_setting)

        clean_train_ds = prepro_cls_DatasetBD(
            full_dataset_without_transform=train_dataset_without_transform,
            poison_idx=np.zeros(len(train_dataset_without_transform)),
            # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=train_img_transform,
            ori_label_transform_in_loading=train_label_transfrom,
            add_details_in_preprocess=True,
        )

        clean_train_x, clean_train_y, _, _, _ = add_resize_and_subset_for_prepro_cls_DatasetBD(clean_train_ds,  clean_setting.img_size)

        clean_test_ds = prepro_cls_DatasetBD(
            test_dataset_without_transform,
            poison_idx=np.zeros(len(test_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=test_img_transform,
            ori_label_transform_in_loading=test_label_transform,
            add_details_in_preprocess=True,
        )

        clean_test_x, clean_test_y, _, _, _ = add_resize_and_subset_for_prepro_cls_DatasetBD(clean_test_ds,  clean_setting.img_size)

        if (load_file['bd_train']['x'] is not None) and (load_file['bd_train']['y'] is not None) and (load_file['bd_train']['original_index'] is not None):
                bd_train_x = deepcopy(clean_train_x)
                bd_train_y = deepcopy(clean_train_y)
                for ii, original_index_i in enumerate(load_file['bd_train']['original_index']):
                    bd_train_x[original_index_i] = load_file['bd_train']['x'][ii]
                    bd_train_y[original_index_i] = load_file['bd_train']['y'][ii]
        else:
            bd_train_x = None
            bd_train_y = None
            logging.info('bd_train is None !')
            
        load_dict = {
                'model_name': load_file['model_name'],
                'model': load_file['model'],

                'clean_train': {
                    'x' : clean_train_x,
                    'y' : clean_train_y,
                },

                'clean_test' : {
                    'x' : clean_test_x,
                    'y' : clean_test_y,
                },

                'bd_train': {
                    'x': bd_train_x,
                    'y': bd_train_y,
                    'original_index': load_file['bd_train'].get('original_index'), #could be None
                },

                'bd_test': {
                    'x': load_file['bd_test']['x'],
                    'y': load_file['bd_test']['y'],
                    'original_index': load_file['bd_test'].get('original_index'),
                    'original_targets':load_file['bd_test'].get('original_targets'),
                },
            }
        logging.info(f"loading...")
        logging.info(f"location : {save_path}, content summary :{pformat(summary_dict(load_dict))}")
        return load_dict
    
    else:
        logging.info(f"loading...")
        logging.info(f"location : {save_path}, content summary :{pformat(summary_dict(load_file))}")
        return load_file