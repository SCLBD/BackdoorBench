import copy
import numpy as np

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class PoisonLabelDataset(Dataset):
    """Poison-Label dataset wrapper.

    Args:
        dataset (Dataset): The dataset to be wrapped.
        transform (callable): The backdoor transformations.
        poison_idx (np.array): An 0/1 (clean/poisoned) array with
            shape `(len(dataset), )`.
        target_label (int): The target label.
    """

    def __init__(self, dataset, transform, poison_idx, target_label):
        super(PoisonLabelDataset, self).__init__()
        self.dataset = copy.deepcopy(dataset)
        self.train = self.dataset.train
        if self.train:
            self.data = self.dataset.data
            self.targets = self.dataset.targets
            self.poison_idx = poison_idx
        else:
            # Only fetch poison data when testing.
            self.data = self.dataset.data[np.nonzero(poison_idx)[0]]
            self.targets = self.dataset.targets[np.nonzero(poison_idx)[0]]
            self.poison_idx = poison_idx[poison_idx == 1]
        self.pre_transform = self.dataset.pre_transform
        self.primary_transform = self.dataset.primary_transform
        self.remaining_transform = self.dataset.remaining_transform
        self.prefetch = self.dataset.prefetch
        if self.prefetch:
            self.mean, self.std = self.dataset.mean, self.dataset.std

        self.bd_transform = transform
        self.target_label = target_label

    def __getitem__(self, index):
        if isinstance(self.data[index], str):
            with open(self.data[index], "rb") as f:
                img = np.array(Image.open(f).convert("RGB"))
        else:
            img = self.data[index]
        target = self.targets[index]
        poison = 0
        origin = target  # original target

        if self.poison_idx[index] == 1:
            img = self.bd_first_augment(img, bd_transform=self.bd_transform)
            target = self.target_label
            poison = 1
        else:
            img = self.bd_first_augment(img, bd_transform=None)
        item = {"img": img, "target": target, "poison": poison, "origin": origin}

        return item

    def __len__(self):
        return len(self.data)

    def bd_first_augment(self, img, bd_transform=None):
        # Pre-processing transformation (HWC ndarray->HWC ndarray).
        img = Image.fromarray(img)
        img = self.pre_transform(img)
        img = np.array(img)
        # Backdoor transformation (HWC ndarray->HWC ndarray).
        if bd_transform is not None:
            img = bd_transform(img)
        # Primary and the remaining transformations (HWC ndarray->CHW tensor).
        img = Image.fromarray(img)
        img = self.primary_transform(img)
        img = self.remaining_transform(img)

        if self.prefetch:
            # HWC ndarray->CHW tensor with C=3.
            img = np.rollaxis(np.array(img, dtype=np.uint8), 2)
            img = torch.from_numpy(img)

        return img


class MixMatchDataset(Dataset):
    """Semi-supervised MixMatch dataset.

    Args:
        dataset (Dataset): The dataset to be wrapped.
        semi_idx (np.array): An 0/1 (labeled/unlabeled) array with
            shape `(len(dataset), )`.
        labeled (bool, optional): If True, creates dataset from labeled set, otherwise
            creates from unlabeled set (default: True).
    """

    def __init__(self, dataset, semi_idx, labeled=True):
        super(MixMatchDataset, self).__init__()
        self.dataset = copy.deepcopy(dataset)
        if labeled:
            self.semi_indice = np.nonzero(semi_idx == 1)[0]
        else:
            self.semi_indice = np.nonzero(semi_idx == 0)[0]
        self.labeled = labeled
        self.prefetch = self.dataset.prefetch
        self.mean, self.std = self.dataset.mean, self.dataset.std

    def __getitem__(self, index):
        if self.labeled:
            item = self.dataset[self.semi_indice[index]]
            item["labeled"] = True
        else:
            item1 = self.dataset[self.semi_indice[index]]
            item2 = self.dataset[self.semi_indice[index]]
            img1, img2 = item1.pop("img"), item2.pop("img")
            item1.update({"img1": img1, "img2": img2})
            item = item1
            item["labeled"] = False

        return item

    def __len__(self):
        return len(self.semi_indice)


class SelfPoisonDataset(Dataset):
    """Self-supervised poison-label contrastive dataset.

    Args:
        dataset (PoisonLabelDataset): The poison-label dataset to be wrapped.
        transform (dict): Augmented transformation dict has three keys `pre`, `primary`
            and `remaining` which corresponds to pre-processing, primary and the
            remaining transformations.
    """

    def __init__(self, x,y, transform):
        super(SelfPoisonDataset, self).__init__()
        self.dataset = torch.utils.data.TensorDataset(x,y)
        self.data = x
        self.targets = y
        # self.poison_idx = self.dataset.poison_idx
        # self.bd_transform = self.dataset.bd_transform
        # self.target_label = self.dataset.target_label

        self.pre_transform = transform["pre"]
        self.primary_transform = transform["primary"]
        self.remaining_transform = transform["remaining"]
        # self.remaining_transform = self.dataset.remaining_transform
        # self.prefetch = self.dataset.prefetch
        # if self.prefetch:
        #     self.mean, self.std = self.dataset.mean, self.dataset.std

    def __getitem__(self, index):
        if isinstance(self.data[index], str):
            with open(self.data[index], "rb") as f:
                img = np.array(Image.open(f).convert("RGB"))
        else:
            img = self.data[index]
        target = self.targets[index]
        # poison = 0
        # origin = target  # original target
        # if self.poison_idx[index] == 1:
        #     img1 = self.bd_first_augment(img, bd_transform=self.bd_transform)
        #     img2 = self.bd_first_augment(img, bd_transform=self.bd_transform)
        #     target = self.target_label
        #     poison = 1
        # else:
        img1 = self.bd_first_augment(img)
        img2 = self.bd_first_augment(img)
        item = {
            "img1": img1,
            "img2": img2,
            "target": target,
            # "poison": poison,
            # "origin": origin,
        }
        # item = {
        #     "img1": img1,
        #     "img2": img2,
        #     "target": target,
        #     "poison": poison,
        #     "origin": origin,
        # }

        return item

    def __len__(self):
        return len(self.data)

    def bd_first_augment(self, img):
        # Pre-processing transformations (HWC ndarray->HWC ndarray).
        img = Image.fromarray(img)
        img = self.pre_transform(img)
        img = np.array(img)
        # # Backdoor transformationss (HWC ndarray->HWC ndarray).
        # if bd_transform is not None:
        #     img = bd_transform(img)
        # Primary and the remaining transformations (HWC ndarray->CHW tensor).
        img = Image.fromarray(img)
        img = self.primary_transform(img)
        img = self.remaining_transform(img)

        # if self.prefetch:
        #     # HWC ndarray->CHW tensor with C=3.
        #     img = np.rollaxis(np.array(img, dtype=np.uint8), 2)
        #     img = torch.from_numpy(img)

        return img
