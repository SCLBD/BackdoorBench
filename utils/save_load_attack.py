'''
This script aims to save and load the attack result as a bridge between attack and defense files.

Model, clean data, backdoor data and all infomation needed to reconstruct will be saved.

Note that in default, only the poisoned part of backdoor dataset will be saved to save space.

Jun 12th update:
    change save_load to adapt to alternative save method.
    But notice that this method assume the bd_train after reconstruct MUST have the SAME length with clean_train.

'''
import logging, time


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

    # no dependence to later parts., just save debug info at the save folder of attack_result
    debug_info_folder_path = f"{os.path.dirname(save_path)}/debug"
    # if does not have folder or have file with same name but not a folder
    if (not os.path.exists(debug_info_folder_path)) or (not os.path.isdir(debug_info_folder_path)):
        os.makedirs(debug_info_folder_path)
    debug_file_path_for_load = debug_info_folder_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '_load_debug.log'

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
        train_label_transform, \
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
            ori_label_transform_in_loading=train_label_transform,
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

            # in case use the new alternative saving,
            # 1. cause the length different.
            # 2. and have poison_indicator with same length as original_index
            # so we can cut out origianl_index of poison samples to turn back to the oldest saving method
            if load_file['bd_train']['y'].__len__() != load_file['bd_train']['original_index'].__len__() and load_file['bd_train'].get('poison_indicator') is not None:

                train_original_index = load_file['bd_train']['original_index']
                train_poison_indicator = load_file['bd_train'].get('poison_indicator')
                where_use = np.where(train_poison_indicator==1)[0]
                print(train_original_index,
                    train_poison_indicator,
                    where_use,)
                load_file['bd_train']['original_index'] = np.array([train_original_index[pos_i] for pos_i in where_use])

            # check if the length match for old reconstruction by replacement
            # print(
            #     load_file['bd_train']['x'].__len__(),
            #     load_file['bd_train']['y'].__len__(),
            #     load_file['bd_train']['original_index'].__len__(),
            # )
            assert min(
                load_file['bd_train']['x'].__len__(),
                load_file['bd_train']['y'].__len__(),
                load_file['bd_train']['original_index'].__len__(),
            ) == max(
                load_file['bd_train']['x'].__len__(),
                load_file['bd_train']['y'].__len__(),
                load_file['bd_train']['original_index'].__len__(),
            )

            bd_train_x = deepcopy(clean_train_x)
            bd_train_y = deepcopy(clean_train_y)
            for ii, original_index_i in enumerate(load_file['bd_train']['original_index']):
                bd_train_x[original_index_i] = load_file['bd_train']['x'][ii]
                bd_train_y[original_index_i] = load_file['bd_train']['y'][ii]

        else:
            bd_train_x = None
            bd_train_y = None
            logging.info('bd_train is None !')

        # assume that bd_train after reconstruction must have same number of samples as clean_train
        # print(bd_train_x.__len__(),
        #     bd_train_y.__len__(),
        #     clean_train_x.__len__(),
        #     clean_train_y.__len__(),)
        assert min(
            bd_train_x.__len__(),
            bd_train_y.__len__(),
            clean_train_x.__len__(),
            clean_train_y.__len__(),
        ) == max(
            bd_train_x.__len__(),
            bd_train_y.__len__(),
            clean_train_x.__len__(),
            clean_train_y.__len__(),
        )

        # assume that all vec in bd_test must have the same length
        assert min(
            load_file['bd_test']['x'].__len__(),
            load_file['bd_test']['y'].__len__(),
            load_file['bd_test']['original_index'].__len__(),
            load_file['bd_test']['original_targets'].__len__(),
        ) == max(
            load_file['bd_test']['x'].__len__(),
            load_file['bd_test']['y'].__len__(),
            load_file['bd_test']['original_index'].__len__(),
            load_file['bd_test']['original_targets'].__len__(),
        )

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

        print(f"loading...")

        summary_for_load = summary_dict(load_dict)
        with open(debug_file_path_for_load, 'w') as f:
            f.write(pformat(summary_for_load))
        summary_for_load_without_model = {x : y for x,y in summary_for_load.items() if x != 'model'}
        print(f"location : {save_path}, content summary :{pformat(summary_for_load_without_model)}") # ignore model info for print only

        return load_dict

    else:
        logging.info(f"loading...")
        logging.info(f"location : {save_path}, content summary :{pformat(summary_dict(load_file))}")
        return load_file