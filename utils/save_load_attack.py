'''
This script aims to save and load the attack result as a bridge between attack and defense files.

Model, clean data, backdoor data and all infomation needed to reconstruct will be saved.

Note that in default, only the poisoned part of backdoor dataset will be saved to save space.

Jun 12th update:
    change save_load to adapt to alternative save method.
    But notice that this method assume the bd_train after reconstruct MUST have the SAME length with clean_train.

'''
import copy
import logging, time

from typing import Optional
import torch, os
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
import numpy as np
from copy import deepcopy
from torchvision.transforms import ToTensor, Resize,Compose
from pprint import pformat
from typing import Union
from tqdm import tqdm
from PIL import Image
from utils.aggregate_block.model_trainer_generate import generate_cls_model

from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate

def summary_dict(input_dict):
    '''
    Input a dict, this func will do summary for it.
    deepcopy to make sure no influence for summary
    :return:
    '''
    input_dict = deepcopy(input_dict)
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
    ])

    all_img_r_t = []

    if only_bd:
        given_data.subset(np.where(np.array(given_data.poison_indicator) == 1)[0]) # only bd samples remain

    for img in tqdm(given_data.data, desc=f'resize'):
        img_r_t = resize_bd_totensor(
            img
        )
        all_img_r_t.append(img_r_t)

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

    for ii in select_index :
        if 0 <= ii < len(pil_image_list):
            pil_image_list[ii].save(f"{save_folder}/{ii}.png")

def save_attack_result(
    model_name : str,
    num_classes : int,
    model : dict, # the state_dict
    data_path : str,
    img_size : Union[list, tuple],
    clean_data : str,
    bd_test : prepro_cls_DatasetBD_v2, # MUST be dataset without transform
    save_path : str,
    bd_train : Optional[prepro_cls_DatasetBD_v2] = None, # MUST be dataset without transform
    **kwargs,
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

    save_dict = {
            'model_name': model_name,
            'num_classes' : num_classes,
            'model': model,
            'data_path': data_path,
            'img_size' : img_size,
            'clean_data': clean_data,
            'bd_train': bd_train.retrieve_state() if bd_train is not None else None,
            'bd_test': bd_test.retrieve_state(),
            **kwargs,
        }

    logging.info(f"saving...")
    logging.debug(f"location : {save_path}/attack_result.pt") #, content summary :{pformat(summary_dict(save_dict))}")

    torch.save(
        save_dict,
        f'{save_path}/attack_result.pt',
    )

    logging.info("Saved, folder path: {}".format(save_path))

def save_defense_result(
    model_name : str,
    num_classes : int,
    model : dict, # the state_dict
    save_path : str,
):
    '''

    main idea is to loop through the backdoor train and test dataset, and match with the clean dataset
    by remove replicated parts, this function can save the space.

    WARNING: keep all dataset with shuffle = False, same order of data samples is the basic of this function !!!!

    :param model_name : str,
    :param num_classes : int,
    :param model : dict, # the state_dict
    :param save_path : str,
    '''

    save_dict = {
            'model_name': model_name,
            'num_classes' : num_classes,
            'model': model,
        }

    logging.info(f"saving...")
    logging.debug(f"location : {save_path}/defense_result.pt") #, content summary :{pformat(summary_dict(save_dict))}")

    torch.save(
        save_dict,
        f'{save_path}/defense_result.pt',
    )


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
        save_path = os.path.realpath(save_path)
        clean_setting.dataset_path = save_path[:save_path.index('record')] + clean_setting.dataset_path[clean_setting.dataset_path.index('data'):]

        clean_setting.img_size = load_file['img_size']

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform = dataset_and_transform_generate(clean_setting)

        clean_train_dataset_with_transform = dataset_wrapper_with_transform(
            train_dataset_without_transform,
            train_img_transform,
            train_label_transform,
        )

        clean_test_dataset_with_transform = dataset_wrapper_with_transform(
            test_dataset_without_transform,
            test_img_transform,
            test_label_transform,
        )

        if load_file['bd_train'] is not None:
            bd_train_dataset = prepro_cls_DatasetBD_v2(train_dataset_without_transform)
            bd_train_dataset.set_state(
                load_file['bd_train']
            )
            bd_train_dataset_with_transform = dataset_wrapper_with_transform(
                bd_train_dataset,
                train_img_transform,
                train_label_transform,
            )
        else:
            logging.info("No bd_train info found.")
            bd_train_dataset_with_transform = None


        bd_test_dataset = prepro_cls_DatasetBD_v2(test_dataset_without_transform)
        bd_test_dataset.set_state(
            load_file['bd_test']
        )
        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_img_transform,
            test_label_transform,
        )

        new_dict = copy.deepcopy(load_file['model'])
        for k, v in load_file['model'].items():
            if k.startswith('module.'):
                del new_dict[k]
                new_dict[k[7:]] = v

        load_file['model'] = new_dict

        # change the key name of model to match the state_dict of model
        model = generate_cls_model(load_file['model_name'], load_file['num_classes'], )
        old_keys = list(load_file['model'].keys())
        assert len(old_keys) == len(model.state_dict().keys()), "state_dict key length not match"
        for key_idx, model_key in enumerate(model.state_dict().keys()):
            if model_key != old_keys[key_idx]:
                logging.info(f"change key name from {old_keys[key_idx]} to {model_key}")
                load_file['model'][model_key] = load_file['model'].pop(old_keys[key_idx])

        load_dict = {
                'model_name': load_file['model_name'],
                'model': load_file['model'],
                'clean_train': clean_train_dataset_with_transform,
                'clean_test' : clean_test_dataset_with_transform,
                'bd_train': bd_train_dataset_with_transform,
                'bd_test': bd_test_dataset_with_transform,
            }

        print(f"loading...")

        return load_dict

    else:
        logging.info(f"loading...")
        logging.debug(f"location : {save_path}, content summary :{pformat(summary_dict(load_file))}")
        return load_file
