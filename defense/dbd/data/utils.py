import random

import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFilter
from torch.utils.data import DataLoader

from .backdoor import BadNets, Blend
from .cifar import CIFAR10
from .imagenet import ImageNet
from .prefetch import PrefetchLoader
from .vggface2 import VGGFace2


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR.
    
    Borrowed from https://github.com/facebookresearch/moco/blob/master/moco/loader.py.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))

        return x


def query_transform(name, kwargs):
    if name == "random_crop":
        return transforms.RandomCrop(**kwargs)
    elif name == "random_resize_crop":
        return transforms.RandomResizedCrop(**kwargs)
    elif name == "resize":
        return transforms.Resize(**kwargs)
    elif name == "center_crop":
        return transforms.CenterCrop(**kwargs)
    elif name == "random_horizontal_flip":
        return transforms.RandomHorizontalFlip(**kwargs)
    elif name == "random_color_jitter":
        # In-place!
        p = kwargs.pop("p")
        return transforms.RandomApply([transforms.ColorJitter(**kwargs)], p=p)
    elif name == "random_grayscale":
        return transforms.RandomGrayscale(**kwargs)
    elif name == "gaussian_blur":
        # In-place!
        p = kwargs.pop("p")
        return transforms.RandomApply([GaussianBlur(**kwargs)], p=p)
    elif name == "to_tensor":
        if kwargs:
            return transforms.ToTensor()
    elif name == "normalize":
        return transforms.Normalize(**kwargs)
    else:
        raise ValueError("Transformation {} is not supported!".format(name))


def get_transform(transform_config):
    transform = []
    if transform_config is not None:
        for (k, v) in transform_config.items():
            if v is not None:
                transform.append(query_transform(k, v))
    transform = transforms.Compose(transform)

    return transform


def get_dataset(dataset_dir, transform, train=True, prefetch=False):
    if "cifar" in dataset_dir:
        dataset = CIFAR10(
            dataset_dir, transform=transform, train=train, prefetch=prefetch
        )
    elif "imagenet" in dataset_dir:
        dataset = ImageNet(
            dataset_dir, transform=transform, train=train, prefetch=prefetch
        )
    elif "vggface2" in dataset_dir:
        dataset = VGGFace2(
            dataset_dir, transform=transform, train=train, prefetch=prefetch
        )
    else:
        raise NotImplementedError("Dataset in {} is not supported.".format(dataset_dir))

    return dataset


def get_loader(dataset, loader_config=None, **kwargs):
    if loader_config is None:
        loader = DataLoader(dataset, **kwargs)
    else:
        loader = DataLoader(dataset, **loader_config, **kwargs)
    if dataset.prefetch:
        loader = PrefetchLoader(loader, dataset.mean, dataset.std)

    return loader


def gen_poison_idx(dataset, target_label, poison_ratio=None):
    poison_idx = np.zeros(len(dataset))
    train = dataset.train
    for (i, t) in enumerate(dataset.targets):
        if train and poison_ratio is not None:
            if random.random() < poison_ratio and t != target_label:
                poison_idx[i] = 1
        else:
            if t != target_label:
                poison_idx[i] = 1

    return poison_idx


def get_bd_transform(bd_config):
    if "badnets" in bd_config:
        bd_transform = BadNets(bd_config["badnets"]["trigger_path"])
    elif "blend" in bd_config:
        bd_transform = Blend(**bd_config["blend"])
    else:
        raise NotImplementedError("Backdoor {} is not supported.".format(bd_config))

    return bd_transform


def get_semi_idx(record_list, ratio, logger):
    """Get labeled and unlabeled index.
    """
    keys = [r.name for r in record_list]
    loss = record_list[keys.index("loss")].data.numpy()
    poison = record_list[keys.index("poison")].data.numpy()
    semi_idx = np.zeros(len(loss))
    # Sort loss and fetch `ratio` of the smallest indices.
    indice = loss.argsort()[: int(len(loss) * ratio)]
    logger.info(
        "{}/{} poisoned samples in semi_idx".format(poison[indice].sum(), len(indice))
    )
    semi_idx[indice] = 1

    return semi_idx
