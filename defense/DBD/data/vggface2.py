"""VGGFace2 Dataset in Pytorch.
"""
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from .prefetch import prefetch_transform


def find_classes(dir):
    """Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.
    
    Returns:
        (classes, class_to_idx) (tuple): classes are relative to (dir),
            and class_to_idx is a dictionary.
    """
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    dataset = []
    for target_class in sorted(class_to_idx.keys()):
        target = class_to_idx[target_class]
        target_dir = os.path.join(dir, target_class)
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                dataset.append((path, target))
    return dataset


class VGGFace2(Dataset):
    """VGGFace2 Dataset.

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
        if root[0] == "~":
            # interprete `~` as the home directory.
            root = os.path.expanduser(root)
        pickle_file_dict = {"train": "train.pickle", "test": "test.pickle"}
        self.pre_transform = transform["pre"]
        self.primary_transform = transform["primary"]
        if prefetch:
            self.remaining_transform, self.mean, self.std = prefetch_transform(
                transform["remaining"]
            )
        else:
            self.remaining_transform = transform["remaining"]
        self.prefetch = prefetch
        self.classes, self.class_to_idx = find_classes(root)
        self.train = train
        if self.train:
            pickle_file_path = os.path.join(root, pickle_file_dict["train"])
        else:
            pickle_file_path = os.path.join(root, pickle_file_dict["test"])
        with open(pickle_file_path, "rb") as f:
            data_target = pickle.load(f)
        self.data = np.array([os.path.join(root, s[0]) for s in data_target])
        self.targets = np.array([s[1] for s in data_target])

    def __getitem__(self, index):
        img_path = self.data[index]
        # Open path as file to avoid ResourceWarning.
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        # Pre-processing, primary and the remaining transformations.
        img = self.pre_transform(img)
        img = self.primary_transform(img)
        img = self.remaining_transform(img)
        if self.prefetch:
            # HWC ndarray->CHW tensor with C=3.
            img = np.rollaxis(np.array(img, dtype=np.uint8), 2)
            img = torch.from_numpy(img)
        item = {"img": img, "target": self.targets[index]}

        return item

    def __len__(self):
        return len(self.data)
