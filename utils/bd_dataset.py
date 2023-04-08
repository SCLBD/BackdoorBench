
import sys, logging
sys.path.append('../')

import numpy as np
import torch

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
from typing import *

from copy import deepcopy

class xy_iter(torch.utils.data.dataset.Dataset):
    def __init__(self,
             x : Sequence,
             y : Sequence,
             transform
         ):
        assert len(x) == len(y)
        self.data = x
        self.targets = y
        self.transform = transform
    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.targets)

class prepro_cls_DatasetBD(torch.utils.data.dataset.Dataset):

    def __init__(self,
                 full_dataset_without_transform: Sequence,
                 poison_idx: Sequence,  # one-hot to determine which image may take bd_transform
                 add_details_in_preprocess: Optional[bool] = True,

                 clean_image_pre_transform : Optional[Callable] = np.array,
                 bd_image_pre_transform: Optional[Callable] = None,
                 bd_label_pre_transform: Optional[Callable] = None,
                 end_pre_process : Optional[Callable] = lambda img_list : [Image.fromarray(img.astype(np.uint8)) for img in img_list],

                 ori_image_transform_in_loading: Optional[Callable] = None,
                 ori_label_transform_in_loading: Optional[Callable] = None,
             ):
        logging.debug('dataset must have NO transform in BOTH image and label !')

        self.dataset = full_dataset_without_transform
        self.ori_image_transform_in_loading = ori_image_transform_in_loading
        self.ori_label_transform_in_loading = ori_label_transform_in_loading

        self.poison_idx = poison_idx  # actually poison indicator
        self.clean_image_pre_transform = clean_image_pre_transform
        self.bd_image_pre_transform = bd_image_pre_transform
        self.bd_label_pre_transform = bd_label_pre_transform
        self.end_pre_process = end_pre_process

        self.add_details_in_preprocess = add_details_in_preprocess

        assert len(poison_idx) == len(full_dataset_without_transform)
        self.prepro_backdoor()

        self.getitem_all_switch = False

    def prepro_backdoor(self):

        self.data = []
        self.targets = []
        if self.add_details_in_preprocess:
            self.original_index = []
            self.poison_indicator = deepcopy(self.poison_idx)
            self.original_targets = []

        for original_idx, content in enumerate(tqdm(self.dataset, desc=f'pre-process bd dataset')):

            img, label = content

            if self.clean_image_pre_transform is not None and self.poison_idx[original_idx] == 0:

                img = self.clean_image_pre_transform(img)

            if self.bd_image_pre_transform is not None and self.poison_idx[original_idx] == 1:

                img = self.bd_image_pre_transform(img, label, original_idx)

            original_label = deepcopy(label)

            if self.bd_label_pre_transform is not None and self.poison_idx[original_idx] == 1:

                label = self.bd_label_pre_transform(label, original_idx, img)

            if self.add_details_in_preprocess:
                self.data.append(img)
                self.targets.append(label)
                self.original_index.append(original_idx)
                self.original_targets.append(original_label)
            else:
                self.data.append(img)
                self.targets.append(label)

        self.targets = np.array(self.targets)

        if self.add_details_in_preprocess:
            self.original_index = np.array(self.original_index)
            self.poison_indicator = np.array(self.poison_indicator)
            self.original_targets = np.array(self.original_targets)

        if self.end_pre_process is not None:
            self.data = self.end_pre_process(self.data)

    def __getitem__(self, item):

        img  = self.data[item]
        label  = self.targets[item]

        if self.ori_image_transform_in_loading is not None:
            img = self.ori_image_transform_in_loading(img)
        if self.ori_label_transform_in_loading is not None:
            label = self.ori_label_transform_in_loading(label)

        if self.add_details_in_preprocess:
            if self.getitem_all_switch:
                return img, \
                       self.original_targets[item], \
                       self.original_index[item], \
                       self.poison_indicator[item], \
                       label
            else:
                return img, \
                       label, \
                       self.original_index[item], \
                       self.poison_indicator[item], \
                       self.original_targets[item]
        else:
            return img, label

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
               chosen_index_list,
               inplace = True,
               memorize_original = True,
               ):
        if inplace:
            self.data = [self.data[ii] for ii in chosen_index_list] #self.data[chosen_index_array]
            self.targets = self.targets[chosen_index_list]
            if self.add_details_in_preprocess:
                self.original_index = self.original_index[chosen_index_list]
                self.poison_indicator = self.poison_indicator[chosen_index_list]
                self.original_targets = self.original_targets[chosen_index_list]
            if not memorize_original:
                self.dataset, self.poison_idx = None, None
        else:
            new_obj = deepcopy(self)
            new_obj.subset(
                chosen_index_list,
                inplace=True,
                memorize_original=memorize_original,
            )
            return new_obj

