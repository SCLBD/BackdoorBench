
# 这个dataset可以是通用的，不过只能做到在某一个时间点对一部分数据集施加同一种扰动
# 比如说如果是每一个epoch都要update的autoencoder，那么ok，只要我修改callable的那个bd_image_pre_transform 和bd_label_pre_transform就行
# 但是一旦是需要实时update一小部分的数据集，那么就比较头疼了，因为每一次都需要对全部数据做self.prepro_backdoor()，但是这种情况也可以不通过原本的数据集完成就是，放在trainer内部就行。
# 这里我默认不移除原本的数据集，只是为了更加general的情况。如果希望回收内存，那么可以手动删除。

import sys, logging
sys.path.append('../')

import numpy as np
import torch

from PIL import Image

from tqdm import tqdm
from typing import *

from copy import deepcopy

class prepro_cls_DatasetBD(torch.utils.data.dataset.Dataset):

    def __init__(self,
                    full_dataset_without_transform : torch.utils.data.dataset.Dataset,
                    poison_idx : Sequence, # one-hot to determine which image may take bd_transform
                    bd_image_pre_transform : Optional[Callable] = None,
                    bd_label_pre_transform : Optional[Callable] = None,
                    ori_image_transform_in_loading : Optional[Callable] = None,
                    ori_label_transform_in_loading : Optional[Callable] = None,
                    add_details_in_preprocess: Optional[bool] = True,
                    init_with_prepro_backdoor: Optional[bool] = True,
                 ):

        logging.info('dataset must have NO transform in BOTH image and label !')

        self.dataset = full_dataset_without_transform
        self.ori_image_transform_in_loading = ori_image_transform_in_loading
        self.ori_label_transform_in_loading = ori_label_transform_in_loading

        self.poison_idx = poison_idx #actually poison indicator
        self.bd_image_pre_transform = bd_image_pre_transform
        self.bd_label_pre_transform = bd_label_pre_transform

        self.add_details_in_preprocess = add_details_in_preprocess

        if init_with_prepro_backdoor:
            assert len(poison_idx) == len(full_dataset_without_transform)
            self.prepro_backdoor()

    def prepro_backdoor(self):

        self.data = []
        self.targets = []
        if self.add_details_in_preprocess:
            self.original_index = []
            self.poison_indicator = deepcopy(self.poison_idx)
            self.original_targets = []

        for original_idx, content in enumerate(tqdm(self.dataset, desc=f'pre-process bd dataset')):

            img, label = content

            img = np.array(img)

            '''
            img = self.bd_transform(img, target, original_index) if self.bd_transform is not None else img

            target = self.bd_label_transform(target, original_index, img,) if self.bd_label_transform is not None else target'''

            if self.bd_image_pre_transform is not None and self.poison_idx[original_idx] == 1:

                img = self.bd_image_pre_transform(img, label, original_idx)

            original_label = deepcopy(label)

            if self.bd_label_pre_transform is not None and self.poison_idx[original_idx] == 1:

                label = self.bd_label_pre_transform(
                    label, original_idx, img)

            if self.add_details_in_preprocess:

                # data_list_after_process.append([
                #     img, label, original_idx, self.poison_idx[original_idx], original_label
                # ])
                self.data.append(img)
                self.targets.append(label)
                self.original_index.append(original_idx)
                # self.poison_indicator.append(self.poison_idx[original_idx]) no need.
                self.original_targets.append(original_label)
            else:
                self.data.append(img)
                self.targets.append(label)

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        if self.add_details_in_preprocess:
            self.original_index = np.array(self.original_index)
            self.poison_indicator = np.array(self.poison_indicator)
            self.original_targets = np.array(self.original_targets)

        # self.dataset = None # save the memory

    def __getitem__(self, item):

        img  = self.data[item]
        label  = self.targets[item]

        img = Image.fromarray(img.astype(np.uint8))

        if self.ori_image_transform_in_loading is not None:
            img = self.ori_image_transform_in_loading(img)
        if self.ori_label_transform_in_loading is not None:
            label = self.ori_label_transform_in_loading(label)

        if self.add_details_in_preprocess:
            return img, label, self.original_index[item],self.poison_indicator[item],self.original_targets[item],
        else:
            return img, label,

    def __len__(self):
        all_length = (len(self.data),len(self.targets),len(self.original_index),len(self.poison_indicator),len(self.original_targets),)
        assert max(all_length) == min(all_length)
        return len(self.targets)

    def forget_original(self):
        self.dataset, self.poison_idx = None, None

    def subset(self, 
               chosen_index_array,
               inplace = True,
               memorize_original = True,
               ):
        if inplace:
            self.data = self.data[chosen_index_array]
            self.targets = self.targets[chosen_index_array]
            self.original_index = self.original_index[chosen_index_array]
            self.poison_indicator = self.poison_indicator[chosen_index_array]
            self.original_targets = self.original_targets[chosen_index_array]
            if not memorize_original:
                self.dataset, self.poison_idx = None, None
        else:
            set_without_pre = prepro_cls_DatasetBD(
                full_dataset_without_transform = self.dataset if memorize_original else None,
                poison_idx = self.poison_idx if memorize_original else None,
                bd_image_pre_transform = self.bd_image_pre_transform,
                bd_label_pre_transform = self.bd_label_pre_transform,
                ori_image_transform_in_loading = self.ori_image_transform_in_loading,
                ori_label_transform_in_loading = self.ori_label_transform_in_loading,
                add_details_in_preprocess = False,
                init_with_prepro_backdoor = False,
            )
            set_without_pre.data = self.data[chosen_index_array]
            set_without_pre.targets = self.targets[chosen_index_array]
            set_without_pre.original_index = self.original_index[chosen_index_array]
            set_without_pre.poison_indicator = self.poison_indicator[chosen_index_array]
            set_without_pre.original_targets = self.original_targets[chosen_index_array]
            return set_without_pre

    def save(self, save_path, only_bd = False, additional_info = None):
        
        if only_bd:
            poison_position = np.where(self.poison_indicator == 1)[0]
            return_subset_bd = self.subset(poison_position, inplace=False)
            torch.save(
                {'add_details_in_preprocess': return_subset_bd.add_details_in_preprocess,
                 'bd_image_pre_transform': return_subset_bd.bd_image_pre_transform,
                 'bd_label_pre_transform': return_subset_bd.bd_label_pre_transform,
                 'data': return_subset_bd.data,
                 'dataset': return_subset_bd.dataset,
                 'ori_image_transform_in_loading': return_subset_bd.ori_image_transform_in_loading,
                 'ori_label_transform_in_loading': return_subset_bd.ori_label_transform_in_loading,
                 'original_index': return_subset_bd.original_index,
                 'original_targets': return_subset_bd.original_targets,
                 'poison_idx': return_subset_bd.poison_idx,
                 'poison_indicator': return_subset_bd.poison_indicator,
                 'targets': return_subset_bd.targets,
                 'additional_info': additional_info,
                 'only_bd': only_bd,
                 },
                save_path,
            )
        else:
            torch.save(
                {'add_details_in_preprocess': self.add_details_in_preprocess,
                 'bd_image_pre_transform': self.bd_image_pre_transform,
                 'bd_label_pre_transform': self.bd_label_pre_transform,
                 'data': self.data,
                 'dataset': self.dataset,
                 'ori_image_transform_in_loading': self.ori_image_transform_in_loading,
                 'ori_label_transform_in_loading': self.ori_label_transform_in_loading,
                 'original_index': self.original_index,
                 'original_targets': self.original_targets,
                 'poison_idx': self.poison_idx,
                 'poison_indicator': self.poison_indicator,
                 'targets': self.targets,
                 'additional_info' : additional_info,
                 'only_bd': only_bd,
                 },
                save_path
            )
    
    @classmethod       
    def load(self, load_path, ):
        
        load_file = torch.load(load_path)

        load_dataset = prepro_cls_DatasetBD(
            full_dataset_without_transform=load_file['dataset'],
            poison_idx=load_file['poison_idx'],
            bd_image_pre_transform=load_file['bd_image_pre_transform'],
            bd_label_pre_transform=load_file['bd_label_pre_transform'],
            ori_image_transform_in_loading=load_file['ori_image_transform_in_loading'],
            ori_label_transform_in_loading=load_file['ori_label_transform_in_loading'],
            add_details_in_preprocess=False,
            init_with_prepro_backdoor=False,
        )

        load_dataset.data = load_file['data']
        load_dataset.targets = load_file['targets']
        load_dataset.original_index = load_file['original_index']
        load_dataset.poison_indicator = load_file['poison_indicator']
        load_dataset.original_targets = load_file['original_targets']

        return load_dataset,  load_file['additional_info'], load_file['only_bd']
