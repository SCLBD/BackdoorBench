'''
 This script is for prepro_cls_DatasetBD (preprocess classifcation dataset of backdoor, warpper)
    the main idea is to separate the transform from original clean dataset, do the backdoor modification first on raw image.

    default for img:
        [preprocess]:
            raw img [-> np.array] [-> (add trigger)] -> big np.array
        __getitem__:
            img in self.data [-> PIL image] [-> transform] -> batch

        "[...]" means optional
'''

import sys, logging
sys.path.append('../')

import numpy as np
import torch

from PIL import Image

from tqdm import tqdm
from typing import *

from copy import deepcopy

class prepro_cls_DatasetBD(torch.utils.data.dataset.Dataset):

    '''
    This is a warpper object for clean dataset
    '''

    def __init__(self,
                    full_dataset_without_transform : torch.utils.data.dataset.Dataset,
                    poison_idx : Sequence, # one-hot to determine which image may take bd_transform
                    bd_image_pre_transform : Optional[Callable] = None,
                    bd_label_pre_transform : Optional[Callable] = None,
                    ori_image_transform_in_loading : Optional[Callable] = None,
                    ori_label_transform_in_loading : Optional[Callable] = None,
                    add_details_in_preprocess: bool = True,
                    init_with_prepro_backdoor: bool = True,
                    to_np_array_before_poison : bool = True,
                    to_PIL_before_get_item : bool = True,
                 ):
        '''
        For analysis consideration, the original dataset will not be delete,
            you can assign the self.dataset = None to free up momery if no further need.

        :param full_dataset_without_transform:
        :param poison_idx: one-hot array of indicate whether poison or not
        :param bd_image_pre_transform: backdoor attack img transform
        :param bd_label_pre_transform: backdoor attack label transform
        :param ori_image_transform_in_loading: original clean data transform
        :param ori_label_transform_in_loading: original clean data transform
        :param add_details_in_preprocess: whether add details to dataset (index, original label, and poison indicator)
        :param init_with_prepro_backdoor: whether to do the backdoor modification at the initialization,
            you can do it after init by manually call prepro_backdoor method
        '''

        logging.info('dataset must have NO transform in BOTH image and label !')

        self.dataset = full_dataset_without_transform
        self.ori_image_transform_in_loading = ori_image_transform_in_loading
        self.ori_label_transform_in_loading = ori_label_transform_in_loading

        self.poison_idx = poison_idx #actually poison indicator
        self.bd_image_pre_transform = bd_image_pre_transform
        self.bd_label_pre_transform = bd_label_pre_transform

        self.add_details_in_preprocess = add_details_in_preprocess

        self.to_np_array_before_poison = to_np_array_before_poison
        self.to_PIL_before_get_item = to_PIL_before_get_item

        if init_with_prepro_backdoor:
            assert len(poison_idx) == len(full_dataset_without_transform)
            self.prepro_backdoor()

    def prepro_backdoor(self):
        '''
        do the backdoor modification by poison indicator, (and add details)
        '''

        self.data = []
        self.targets = []
        if self.add_details_in_preprocess:
            self.original_index = []
            self.poison_indicator = deepcopy(self.poison_idx)
            self.original_targets = []

        for original_idx, content in enumerate(tqdm(self.dataset, desc=f'pre-process bd dataset')):

            img, label = content

            if self.to_np_array_before_poison:
                img = np.array(img)

            '''
            img = self.bd_transform(img, target, original_index) if self.bd_transform is not None else img

            target = self.bd_label_transform(target, original_index, img,) if self.bd_label_transform is not None else target'''

            if self.bd_image_pre_transform is not None and self.poison_idx[original_idx] == 1:

                img = self.bd_image_pre_transform(img, label, original_idx)

            img = np.array(img) # make sure both clean and bd converted to array

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

        # if the data is all of the same size, then can stack to array.
        if all(
                [issubclass(img.dtype.type, np.integer) for img in self.data]
        ) and min(img.shape for img in self.data) == max(img.shape for img in self.data):
            # all dtype be int, and all shape same
            logging.info('found all img same shape and dtype are int, so stack.')
            self.data = np.stack(self.data)
        else:
            self.data = np.array(self.data)

        logging.info(f'self.data after preprocess, shape: {self.data.shape}')

        self.targets = np.array(self.targets)

        if self.add_details_in_preprocess:
            self.original_index = np.array(self.original_index)
            self.poison_indicator = np.array(self.poison_indicator)
            self.original_targets = np.array(self.original_targets)

        # self.dataset = None # save the memory

    def __getitem__(self, item):

        img  = self.data[item]
        label  = self.targets[item]

        if self.to_PIL_before_get_item:
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
        if  self.add_details_in_preprocess:
            all_length = (len(self.data),len(self.targets),len(self.original_index),len(self.poison_indicator),len(self.original_targets),)
            assert max(all_length) == min(all_length)
            return len(self.targets)
        else:
            all_length = (len(self.data), len(self.targets),)
            assert max(all_length) == min(all_length)
            return len(self.targets)


    def subset(self, 
               chosen_index_array,
               inplace = True,
               memorize_original = True,
               **kwargs, #allow more args, if inplace = False, then **kwargs will be passed to new prepro_cls_DatasetBD object.
               ):
        '''

        :param chosen_index_array: array of position to keep in subset
            Note that here the array is NOT one-hot !!!
            eg. [5,10,31,651....]
        :param inplace:
            inplace = True will modify attr of obj itself
             inplace = False will return a new obj with subseted attr
        :param memorize_original: delete the original dataset or not
        :return:
        '''
        if inplace:
            self.data = self.data[chosen_index_array]
            self.targets = self.targets[chosen_index_array]
            if self.add_details_in_preprocess:
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
                **kwargs,
            )
            set_without_pre.data = self.data[chosen_index_array]
            set_without_pre.targets = self.targets[chosen_index_array]
            if self.add_details_in_preprocess:
                set_without_pre.original_index = self.original_index[chosen_index_array]
                set_without_pre.poison_indicator = self.poison_indicator[chosen_index_array]
                set_without_pre.original_targets = self.original_targets[chosen_index_array]
            return set_without_pre

