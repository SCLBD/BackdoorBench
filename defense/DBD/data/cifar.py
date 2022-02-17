import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from .prefetch import prefetch_transform


class CIFAR10(Dataset):
    """CIFAR-10 Dataset.

    Args:
        root (string): Root directory of dataset.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set (default: True).
        prefetch (bool, optional): If True, remove `ToTensor` and `Normalize` in
            `transform["remaining"]`, and turn on prefetch mode (default: False).
    """

    def __init__(self, root, transform=None, train=True, prefetch=False):
        self.train = train
        self.pre_transform = transform["pre"]
        self.primary_transform = transform["primary"]
        if prefetch:
            self.remaining_transform, self.mean, self.std = prefetch_transform(
                transform["remaining"]
            )
        else:
            self.remaining_transform = transform["remaining"]
        if train:
            data_list = [
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5",
            ]
        else:
            data_list = ["test_batch"]
        self.prefetch = prefetch
        data = []
        targets = []
        if root[0] == "~":
            # interprete `~` as the home directory.
            root = os.path.expanduser(root)
        for file_name in data_list:
            file_path = os.path.join(root, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            data.append(entry["data"])
            targets.extend(entry["labels"])
        # Convert data (List) to NHWC (np.ndarray) works with PIL Image.
        data = np.vstack(data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        self.data = data
        self.targets = np.asarray(targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)  ## HWC ndarray->HWC Image.
        # Pre-processing transformations (HWC Image->HWC Image).
        if self.pre_transform is not None:
            img = self.pre_transform(img)
        # Primary transformations (HWC Image->HWC Image).
        img = self.primary_transform(img)
        # The remaining transformations (HWC Image->CHW tensor).
        img = self.remaining_transform(img)
        if self.prefetch:
            # HWC ndarray->CHW tensor with C=3.
            img = np.rollaxis(np.array(img, dtype=np.uint8), 2)
            img = torch.from_numpy(img)
        item = {"img": img, "target": target}

        return item

    def __len__(self):
        return len(self.data)
