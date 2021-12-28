import sys, logging

import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms

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
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]))
    elif dataset_name == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif dataset_name == 'tiny':
        transforms_list.append(transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]))
    elif dataset_name == "gtsrb" or dataset_name == "celeba":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)

def dataset_and_transform_generate(args):

    if args.dataset == 'cifar10':

        from torchvision.datasets import CIFAR10

        from tool.dataset_preprocess.cifar10_preprocess import data_transforms_cifar10

        train_dataset_without_transform = CIFAR10(args.dataset_path,
                        train=True,
                        transform=None, )

        train_img_transform = get_transform('cifar10', *(args.img_size[:2]) , train = True)
        train_label_transfrom = None

        test_dataset_without_transform = CIFAR10(args.dataset_path,
                        train=False,
                        transform=None, )

        test_img_transform = get_transform('cifar10', *(args.img_size[:2]) , train = False)
        test_label_transform = None

    elif args.dataset == 'gtsrb':

        from tool.dataset.GTSRB import GTSRB

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

        from tool.dataset.CelebA import CelebA_attr

        train_dataset_without_transform = CelebA_attr(args.dataset_path,
                                                      split='train')
        train_img_transform = get_transform('celeba', *(args.img_size[:2]) , train = True)
        train_label_transfrom = None
        test_dataset_without_transform = CelebA_attr(args.dataset_path,
                                                     split = 'test')
        test_img_transform = get_transform('celeba', *(args.img_size[:2]) , train = False)
        test_label_transform = None

    elif args.dataset == "tiny":

        from tool.dataset.Tiny import Tiny

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


    return train_dataset_without_transform, \
            train_img_transform, \
            train_label_transfrom, \
            test_dataset_without_transform, \
            test_img_transform, \
            test_label_transform