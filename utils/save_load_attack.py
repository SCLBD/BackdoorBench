'''torch.save(
        {
            'model_name':args.model,
            'model': trainer.model.cpu().state_dict(),
            'clean_train': {
                'x' : torch.tensor(nHWC_to_nCHW(benign_train_dl.dataset.data)).float().cpu(),
                'y' : torch.tensor(benign_train_dl.dataset.targets).long().cpu(),
            },

            'clean_test' : {
                'x' : torch.tensor(nHWC_to_nCHW(benign_test_dl.dataset.data)).float().cpu(),
                'y' : torch.tensor(benign_test_dl.dataset.targets).long().cpu(),
            },

            'bd_train': {
                'x' : torch.tensor(nHWC_to_nCHW(adv_train_ds.data)).float().cpu(),
                'y' : torch.tensor(adv_train_ds.targets).long().cpu(),
            },

            'bd_test': {
                'x': torch.tensor(nHWC_to_nCHW(adv_test_dataset.data)).float().cpu(),
                'y' : torch.tensor(adv_test_dataset.targets).long().cpu(),
            },
        },
    f'{save_path}/attack_result.pt'
)'''
import logging
from pprint import pformat
import numpy as np
import torch
from utils.nCHW_nHWC import *
from utils.bd_dataset import prepro_cls_DatasetBD
import numpy as np
from copy import deepcopy

def summary_dict(input_dict):
    '''
    Input a dict, this func will do summary for it.
    :param dict:
    :return:
    '''
    summary_dict_return = dict()
    for k,v in input_dict.items():
        if isinstance(v, dict):
            summary_dict_return[k] = summary_dict(v)
        elif isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            summary_dict_return[k] = v.shape
        elif isinstance(v, list):
            summary_dict_return[k] = v.__len__()
        else:
            summary_dict_return[k] = v
    return  summary_dict_return

def test_summary_dict():
    print(pformat(summary_dict(torch.load('/Users/chenhongrui/sclbd/bdzoo2/record/ssba_0_1/attack_result.pt'))))

def save_attack_result(
    model_name : str,
    num_classes : int,
    model : dict, # the state_dict
    data_path : str,
    img_size : list,
    clean_data : str,
    bd_train : torch.utils.data.Dataset, # dataset without transform
    bd_test : torch.utils.data.Dataset, # dataset without transform
    save_path : str,
):

    def loop_through_cls_ds_without_transform(dataset_without_transform):
        if isinstance(dataset_without_transform, prepro_cls_DatasetBD):
            return torch.tensor(nHWC_to_nCHW(dataset_without_transform.data)).float().cpu(), \
                torch.tensor(dataset_without_transform.targets).long().cpu(), \
                   dataset_without_transform.original_index, \
                   dataset_without_transform.poison_indicator, \
                    dataset_without_transform.original_targets
        else:
            all_x = []
            all_y = []
            for x, y, *addition in dataset_without_transform:
                all_x.append(torch.from_numpy(nHWC_to_nCHW(x[None,...])) if isinstance(x, np.ndarray) else x)
                all_y.append(int(y) if isinstance(y, np.ndarray) else y.item())
            all_x = torch.cat(all_x).float().cpu()
            all_y = torch.tensor(all_y).long().cpu()
            return all_x, all_y, None, None, None

    if bd_train is not None:
        bd_train_x, bd_train_y, _, bd_train_poison_indicator, _  = loop_through_cls_ds_without_transform(bd_train)
        if bd_train_poison_indicator is not None:
            bd_train_x, bd_train_y = \
                bd_train_x[np.where(bd_train_poison_indicator == 1)[0]], \
                bd_train_y[np.where(bd_train_poison_indicator == 1)[0]],

    else:
        logging.info('bd_train is set to be None in saving process!')

    bd_test_x, bd_test_y, bd_test_original_index, _, bd_test_original_targets  = loop_through_cls_ds_without_transform(bd_test)
    bd_test_original_targets = torch.from_numpy(bd_test_original_targets) if bd_test_original_targets is not None else None
    
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
                'original_index' : np.where(bd_train_poison_indicator == 1)[0] if bd_train_poison_indicator is not None else None,
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
    logging.info(f"location : {save_path}/attack_result.pt, content summary :{summary_dict(save_dict)}")
    
    torch.save(
        save_dict,
        f'{save_path}/attack_result.pt',
    )

from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
from utils.aggregate_block.model_trainer_generate import generate_cls_model

class Args:
    pass

def load_attack_result(
    save_path : str,
):
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

        clean_train_x = torch.tensor(nHWC_to_nCHW(clean_train_ds.data)).float().cpu()
        clean_train_y = torch.tensor(clean_train_ds.targets).long().cpu()

        clean_test_ds = prepro_cls_DatasetBD(
            test_dataset_without_transform,
            poison_idx=np.zeros(len(test_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=test_img_transform,
            ori_label_transform_in_loading=test_label_transform,
            add_details_in_preprocess=True,
        )

        clean_test_x = torch.tensor(nHWC_to_nCHW(clean_test_ds.data)).float().cpu()
        clean_test_y = torch.tensor(clean_test_ds.targets).long().cpu()

        if load_file['bd_train']['x'] is not None and load_file['bd_train']['y'] is not None:

            if load_file['bd_train']['original_index'] is not None:
                bd_train_x = deepcopy(clean_train_x)
                bd_train_x[load_file['bd_train']['original_index']] = load_file['bd_train']['x']

                bd_train_y = deepcopy(clean_train_y)
                bd_train_y[load_file['bd_train']['original_index']] = load_file['bd_train']['y']
            else:
                bd_train_x = load_file['bd_train']['x']
                bd_train_y = load_file['bd_train']['y']
                logging.warning(f"load_file['bd_train']['original_index'] is None")
        else:
            bd_train_x = None
            bd_train_y = None
            logging.info('bd_train is None !')

        if load_file['bd_test'].get('original_index') is None:
            logging.warning(f"load_file['bd_test'] has no original_index, return None instead")
        if load_file['bd_test'].get('original_targets') is None:
            logging.warning(f"load_file['bd_test'].get('original_targets') is None")
            
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
        logging.info(f"location : {save_path}, content summary :{summary_dict(load_dict)}")
        return load_dict
    
    else:
        logging.info(f"loading...")
        logging.info(f"location : {save_path}, content summary :{summary_dict(load_file)}")
        return load_file