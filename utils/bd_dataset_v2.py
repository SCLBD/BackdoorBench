import os.path
import sys, logging
sys.path.append('../')

import numpy as np
import torch
import copy
from copy import deepcopy
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
from typing import *

from torchvision.transforms import ToPILImage
from torchvision.datasets import DatasetFolder, ImageFolder

class slice_iter(torch.utils.data.dataset.Dataset):
    '''iterate over a slice of the dataset'''
    def __init__(self,
             dataset,
             axis = 0
         ):
        self.data = dataset
        self.axis = axis

    def __getitem__(self, item):
        return self.data[item][self.axis]

    def __len__(self):
        return len(self.data)



class x_iter(torch.utils.data.dataset.Dataset):
    def __init__(self,
             dataset
         ):
        self.data = dataset

    def __getitem__(self, item):
        img = self.data[item][0]
        return img

    def __len__(self):
        return len(self.data)

class y_iter(torch.utils.data.dataset.Dataset):
    def __init__(self,
             dataset
         ):
        self.data = dataset

    def __getitem__(self, item):
        target = self.data[item][1]
        return target

    def __len__(self):
        return len(self.data)


def get_labels(given_dataset):
    if isinstance(given_dataset, DatasetFolder) or isinstance(given_dataset, ImageFolder):
        logging.debug("get .targets")
        return given_dataset.targets
    else:
        logging.debug("Not DatasetFolder or ImageFolder, so iter through")
        return [label for img, label, *other_info in given_dataset]

class dataset_wrapper_with_transform(torch.utils.data.Dataset):
    '''
    idea from https://stackoverflow.com/questions/1443129/completely-wrap-an-object-in-python
    '''

    def __init__(self, obj, wrap_img_transform=None, wrap_label_transform=None):

        # this warpper should NEVER be warp twice.
        # Since the attr name may cause trouble.
        assert not "wrap_img_transform" in obj.__dict__
        assert not "wrap_label_transform" in obj.__dict__

        self.wrapped_dataset = obj
        self.wrap_img_transform = wrap_img_transform
        self.wrap_label_transform = wrap_label_transform

    def __getattr__(self, attr):
        # # https://github.com/python-babel/flask-babel/commit/8319a7f44f4a0b97298d20ad702f7618e6bdab6a
        # # https://stackoverflow.com/questions/47299243/recursionerror-when-python-copy-deepcopy
        # if attr == "__setstate__":
        #     raise AttributeError(attr)
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.wrapped_dataset, attr)

    def __getitem__(self, index):
        img, label, *other_info = self.wrapped_dataset[index]
        if self.wrap_img_transform is not None:
            img = self.wrap_img_transform(img)
        if self.wrap_label_transform is not None:
            label = self.wrap_label_transform(label)
        return (img, label, *other_info)

    def __len__(self):
        return len(self.wrapped_dataset)
    
    def __deepcopy__(self, memo):
        # In copy.deepcopy, init() will not be called and some attr will not be initialized. 
        # The getattr will be infinitely called in deepcopy process.
        # So, we need to manually deepcopy the wrapped dataset or raise error when "__setstate__" us called. Here we choose the first solution.
        return dataset_wrapper_with_transform(copy.deepcopy(self.wrapped_dataset), copy.deepcopy(self.wrap_img_transform), copy.deepcopy(self.wrap_label_transform))


class poisonedCLSDataContainer:
    '''
    Two mode:
        in RAM / disk
        if in RAM
            save {key : value}
        elif in disk:
            save {
                key : {
                    "path":path, (must take a PIL image and save to .png)
                    "other_info": other_info, (Non-img)
                    }
            }
            where img, *other_info = value
    '''
    def __init__(self, save_folder_path=None, save_file_format = ".png"):
        self.save_folder_path = save_folder_path
        self.data_dict = {}
        self.save_file_format = save_file_format
        logging.info(f"save file format is {save_file_format}")

    def retrieve_state(self):
        return {
            "save_folder_path":self.save_folder_path,
            "data_dict":self.data_dict,
            "save_file_format":self.save_file_format,
        }

    def set_state(self, state_file):
        self.save_folder_path = state_file["save_folder_path"]
        self.data_dict = state_file["data_dict"]
        self.save_file_format = state_file["save_file_format"]

    def setitem(self, key, value, relative_loc_to_save_folder_name=None):

        if self.save_folder_path is None:
            self.data_dict[key] = value
        else:
            img, *other_info = value

            save_subfolder_path = f"{self.save_folder_path}/{relative_loc_to_save_folder_name}"
            if not (
                os.path.exists(save_subfolder_path)
                and
                os.path.isdir(save_subfolder_path)
            ):
                os.makedirs(save_subfolder_path)

            file_path = f"{save_subfolder_path}/{key}{self.save_file_format}"
            img.save(file_path)

            self.data_dict[key] = {
                    "path": file_path,
                    "other_info": other_info,
            }

    def __getitem__(self, key):
        if self.save_folder_path is None:
            return self.data_dict[key]
        else:
            file_path = self.data_dict[key]["path"]
            other_info = self.data_dict[key]["other_info"]
            img =  Image.open(file_path)
            im = deepcopy(img)
            img.close()
            return (im, *other_info)

    def __len__(self):
        return len(self.data_dict)

class prepro_cls_DatasetBD_v2(torch.utils.data.Dataset):

    def __init__(
            self,
            full_dataset_without_transform,
            poison_indicator: Optional[Sequence] = None,  # one-hot to determine which image may take bd_transform

            bd_image_pre_transform: Optional[Callable] = None,
            bd_label_pre_transform: Optional[Callable] = None,
            save_folder_path = None,

            mode = 'attack',
        ):
        '''
        This class require poisonedCLSDataContainer

        :param full_dataset_without_transform: dataset without any transform. (just raw data)

        :param poison_indicator:
            array with 0 or 1 at each position corresponding to clean/poisoned
            Must have the same len as given full_dataset_without_transform (default None, regarded as all 0s)

        :param bd_image_pre_transform:
        :param bd_label_pre_transform:
        ( if your backdoor method is really complicated, then do not set these two params. These are for simplicity.
        You can inherit the class and rewrite method preprocess part as you like)

        :param save_folder_path:
            This is for the case to save the poisoned imgs on disk.
            (In case, your RAM may not be able to hold all poisoned imgs.)
            If you do not want this feature for small dataset, then just left it as default, None.

        '''

        self.dataset = full_dataset_without_transform

        if poison_indicator is None:
            poison_indicator = np.zeros(len(full_dataset_without_transform))
        self.poison_indicator = poison_indicator

        assert len(full_dataset_without_transform) == len(poison_indicator)

        self.bd_image_pre_transform = bd_image_pre_transform
        self.bd_label_pre_transform = bd_label_pre_transform

        self.save_folder_path = save_folder_path # since when we want to save this dataset, this may cause problem

        self.original_index_array = np.arange(len(full_dataset_without_transform))

        self.bd_data_container = poisonedCLSDataContainer(self.save_folder_path, ".png")

        if sum(self.poison_indicator) >= 1:
            self.prepro_backdoor()

        self.getitem_all = True
        self.getitem_all_switch = False

        self.mode = mode

    def prepro_backdoor(self):
        for selected_index in tqdm(self.original_index_array, desc="prepro_backdoor"):
            if self.poison_indicator[selected_index] == 1:
                img, label = self.dataset[selected_index]
                img = self.bd_image_pre_transform(img, target=label, image_serial_id=selected_index)
                bd_label = self.bd_label_pre_transform(label)
                self.set_one_bd_sample(
                    selected_index, img, bd_label, label
                )

    def set_one_bd_sample(self, selected_index, img, bd_label, label):

        '''
        1. To pil image
        2. set the image to container
        3. change the poison_index.

        logic is that no matter by the prepro_backdoor or not, after we set the bd sample,
        This method will automatically change the poison index to 1.

        :param selected_index: The index of bd sample
        :param img: The converted img that want to put in the bd_container
        :param bd_label: The label bd_sample has
        :param label: The original label bd_sample has

        '''

        # we need to save the bd img, so we turn it into PIL
        if (not isinstance(img, Image.Image)) :
            if isinstance(img, np.ndarray):
                img = img.astype(np.uint8)
            img = ToPILImage()(img)
        self.bd_data_container.setitem(
            key=selected_index,
            value=(img, bd_label, label),
            relative_loc_to_save_folder_name=f"{label}",
        )
        self.poison_indicator[selected_index] = 1

    def __len__(self):
        return len(self.original_index_array)

    def __getitem__(self, index):

        original_index = self.original_index_array[index]
        if self.poison_indicator[original_index] == 0:
            # clean
            img, label = self.dataset[original_index]
            original_target = label
            poison_or_not = 0
        else:
            # bd
            img, label, original_target = self.bd_data_container[original_index]
            poison_or_not = 1

        if not isinstance(img, Image.Image):
            img = ToPILImage()(img)

        if self.getitem_all:
            if self.getitem_all_switch:
                # this is for the case that you want original targets, but you do not want change your testing process
                return img, \
                       original_target, \
                       original_index, \
                       poison_or_not, \
                       label

            else: # here should corresponding to the order in the bd trainer
                return img, \
                       label, \
                       original_index, \
                       poison_or_not, \
                       original_target
        else:
            return img, label

    def subset(self, chosen_index_list):
        self.original_index_array = self.original_index_array[chosen_index_list]

    def retrieve_state(self):
        return {
            "bd_data_container" : self.bd_data_container.retrieve_state(),
            "getitem_all":self.getitem_all,
            "getitem_all_switch":self.getitem_all_switch,
            "original_index_array": self.original_index_array,
            "poison_indicator": self.poison_indicator,
            "save_folder_path": self.save_folder_path,
        }

    def copy(self):
        bd_train_dataset = prepro_cls_DatasetBD_v2(self.dataset)
        copy_state = copy.deepcopy(self.retrieve_state())
        bd_train_dataset.set_state(
            copy_state
        )
        return bd_train_dataset

    def set_state(self, state_file):
        self.bd_data_container = poisonedCLSDataContainer()
        self.bd_data_container.set_state(
            state_file['bd_data_container']
        )
        self.getitem_all = state_file['getitem_all']
        self.getitem_all_switch = state_file['getitem_all_switch']
        self.original_index_array = state_file["original_index_array"]
        self.poison_indicator = state_file["poison_indicator"]
        self.save_folder_path = state_file["save_folder_path"]


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


