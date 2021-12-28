
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
                 ):

        logging.info('dataset must have NO transform in BOTH image and label !')

        assert len(poison_idx) == len(full_dataset_without_transform)

        self.dataset = full_dataset_without_transform
        self.ori_image_transform_in_loading = ori_image_transform_in_loading
        self.ori_label_transform_in_loading = ori_label_transform_in_loading

        self.poison_idx = poison_idx #actually poison indicator
        self.bd_image_pre_transform = bd_image_pre_transform
        self.bd_label_pre_transform = bd_label_pre_transform

        self.add_details_in_preprocess = add_details_in_preprocess

        self.prepro_backdoor()

    def prepro_backdoor(self):

        self.data = []
        self.targets = []
        if self.add_details_in_preprocess:
            self.original_index = []
            self.poison_indicator = self.poison_idx
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

    def subset(self, chosen_index_array):
        self.data = self.data[chosen_index_array]
        self.targets = self.targets[chosen_index_array]
        self.original_index = self.original_index[chosen_index_array]
        self.poison_indicator = self.poison_indicator[chosen_index_array]
        self.original_targets = self.original_targets[chosen_index_array]

