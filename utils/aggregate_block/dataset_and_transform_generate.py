import sys, logging
from typing import Tuple
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms


#TODO just copy from wanet, now finished
def get_num_classes(dataset_name : str) -> int:
    if dataset_name in ["mnist", "cifar10"]:
        num_classes = 10
    elif dataset_name == "gtsrb":
        num_classes = 43
    elif dataset_name == "celeba":
        num_classes = 8
    else:
        raise Exception("Invalid Dataset")
    return num_classes

#TODO just copy from wanet, now finished
def get_input_shape(dataset_name : str) -> Tuple[int, int, int]:
    if dataset_name == "cifar10":
        input_height = 32
        input_width = 32
        input_channel = 3
    elif dataset_name == "gtsrb":
        input_height = 32
        input_width = 32
        input_channel = 3
    elif dataset_name == "mnist":
        input_height = 28
        input_width = 28
        input_channel = 1
    elif dataset_name == "celeba":
        input_height = 64
        input_width = 64
        input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    return input_height, input_width, input_channel

#TODO get this function combined into get_transform (lines are copied)
def get_dataset_normalization(dataset_name):
    if dataset_name == "cifar10":
        #from wanet
        dataset_normalization = (transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif dataset_name == 'cifar100':
        '''get from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151'''
        dataset_normalization = (transforms.Normalize([0.5071, 0.4865, 0.4409],[0.2673, 0.2564, 0.2762]))
    elif dataset_name == "mnist":
        dataset_normalization = (transforms.Normalize([0.5], [0.5]))
    elif dataset_name == 'tiny':
        dataset_normalization = (transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]))
    elif dataset_name == "gtsrb" or dataset_name == "celeba":
        dataset_normalization = lambda x : x
    elif dataset_name == 'imagenet':
        dataset_normalization = (
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    else:
        raise Exception("Invalid Dataset")
    return dataset_normalization

# this if from bd_zoo
def get_transform(dataset_name, input_height, input_width,train=True):
    transforms_list = []
    transforms_list.append(transforms.Resize((input_height, input_width)))
    if train:
        transforms_list.append(transforms.RandomCrop((input_height, input_width), padding=4))
        # transforms_list.append(transforms.RandomRotation(10))
        if dataset_name == "cifar10":
            transforms_list.append(transforms.RandomHorizontalFlip())

    transforms_list.append(transforms.ToTensor())
    if dataset_name == "cifar10":
        #from wanet
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif dataset_name == 'cifar100':
        '''get from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151'''
        transforms_list.append(transforms.Normalize([0.5071, 0.4865, 0.4409],[0.2673, 0.2564, 0.2762]))
    elif dataset_name == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif dataset_name == 'tiny':
        transforms_list.append(transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]))
    elif dataset_name == "gtsrb" or dataset_name == "celeba":
        pass
    elif dataset_name == 'imagenet':
        transforms_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)

def dataset_and_transform_generate(args):

    if args.dataset.startswith('test'): # for test only

        import torchvision.transforms as transforms
        from torchvision.datasets import ImageFolder

        train_dataset_without_transform = ImageFolder('../data/test')
        train_img_transform = get_transform('mnist', *(args.img_size[:2]) , train = True)
        train_label_transfrom = None
        test_dataset_without_transform = ImageFolder('../data/test')
        test_img_transform = get_transform('mnist', *(args.img_size[:2]) , train = False)
        test_label_transform = None

    elif args.dataset == 'mnist':

        from torchvision.datasets import MNIST

        train_dataset_without_transform = MNIST(
            args.dataset_path,
            train=True,
            transform=None,
            download=True,
        )

        train_img_transform = get_transform('mnist', *(args.img_size[:2]), train=True)
        train_label_transfrom = None

        test_dataset_without_transform = MNIST(
            args.dataset_path,
            train=False,
            transform=None,
            download=True,
        )

        test_img_transform = get_transform('mnist', *(args.img_size[:2]), train=False)
        test_label_transform = None

    elif args.dataset == 'cifar10':

        from torchvision.datasets import CIFAR10

        from utils.dataset_preprocess.cifar10_preprocess import data_transforms_cifar10

        train_dataset_without_transform = CIFAR10(
            args.dataset_path,
            train=True,
            transform=None,
            download=True,
        )

        train_img_transform = get_transform('cifar10', *(args.img_size[:2]) , train = True)
        train_label_transfrom = None

        test_dataset_without_transform = CIFAR10(
            args.dataset_path,
            train=False,
            transform=None,
            download=True,
        )

        test_img_transform = get_transform('cifar10', *(args.img_size[:2]) , train = False)
        test_label_transform = None

    elif args.dataset == 'cifar100':
        from torchvision.datasets import CIFAR100
        train_dataset_without_transform = CIFAR100(
            root = args.dataset_path,
            train = True,
            download = True,
        )
        train_img_transform = get_transform('cifar100', *(args.img_size[:2]) , train = True)
        train_label_transfrom = None
        test_dataset_without_transform = CIFAR100(
            root = args.dataset_path,
            train = False,
            download = True,
        )
        test_img_transform = get_transform('cifar100', *(args.img_size[:2]) , train = False)
        test_label_transform = None

    elif args.dataset == 'gtsrb':

        from utils.dataset.GTSRB import GTSRB

        train_dataset_without_transform = GTSRB(args.dataset_path,
                                                  train=True,
                                                )

        train_img_transform = get_transform('gtsrb', *(args.img_size[:2]) , train = True)
        train_label_transfrom = None
        test_dataset_without_transform =  GTSRB(args.dataset_path,
                                                  train=False,
                                                )
        test_img_transform = get_transform('gtsrb', *(args.img_size[:2]) , train = True)
        test_label_transform = None

    elif args.dataset == "celeba":

        from utils.dataset.CelebA import CelebA_attr

        train_dataset_without_transform = CelebA_attr(args.dataset_path,
                                                      split='train')
        train_img_transform = get_transform('celeba', *(args.img_size[:2]) , train = True)
        train_label_transfrom = None
        test_dataset_without_transform = CelebA_attr(args.dataset_path,
                                                     split = 'test')
        test_img_transform = get_transform('celeba', *(args.img_size[:2]) , train = False)
        test_label_transform = None

    elif args.dataset == "tiny":

        from utils.dataset.Tiny import Tiny

        train_dataset_without_transform = Tiny(args.dataset_path,
                                                  train=True,
                                                )
        train_img_transform = get_transform('tiny', *(args.img_size[:2]) , train = True)
        train_label_transfrom = None
        test_dataset_without_transform = Tiny(args.dataset_path,
                                                  train=False,
                                                )
        test_img_transform = get_transform('tiny', *(args.img_size[:2]) , train = False)
        test_label_transform = None

    elif args.dataset == "imagenet":

        from torchvision.datasets import ImageNet

        train_dataset_without_transform = ImageNet(
            root = args.dataset_path,
            split = 'train',
        )

        train_img_transform = get_transform('imagenet', *(args.img_size[:2]) , train = True)

        train_label_transfrom = None

        test_dataset_without_transform = ImageNet(
            root = args.dataset_path,
            split = 'val',
        )

        test_img_transform = get_transform('imagenet', *(args.img_size[:2]) , train = False)

        test_label_transform = None

    return train_dataset_without_transform, \
            train_img_transform, \
            train_label_transfrom, \
            test_dataset_without_transform, \
            test_img_transform, \
            test_label_transform