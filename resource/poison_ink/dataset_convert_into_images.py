'''
This file is to convert given dataset into images.
eg.
    image_folder/train/img1.png
    image_folder/train/img2.png
    image_folder/train/img3.png
    ...
        and
    image_folder/test/img1.png
    image_folder/test/img2.png
    image_folder/test/img3.png
    ...
'''
import sys, yaml, os
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse


os.chdir(sys.path[0])
sys.path.append('../../')
os.getcwd()

from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate, get_input_shape


class Args:
    pass

def dataset_convert_into_images(dataset_name, dataset_path, image_folder):

    args = Args()
    args.dataset = dataset_name
    args.dataset_path = os.path.join(dataset_path,dataset_name)
    args.img_size = get_input_shape(args.dataset)

    train_dataset_without_transform, \
    _, \
    _, \
    test_dataset_without_transform, \
    _, \
    _ = dataset_and_transform_generate(args)

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    train_image_folder = os.path.join(image_folder, 'train')

    for img_idx, (img, label, *other) in tqdm(enumerate(train_dataset_without_transform)):
        if not os.path.exists(os.path.join(train_image_folder, str(label))):
            os.makedirs(os.path.join(train_image_folder, str(label)))
        img_path = os.path.join(train_image_folder, str(label), f'img_{img_idx}.png')
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img.save(img_path)

    test_image_folder = os.path.join(image_folder, 'test')

    for img_idx, (img, label, *other) in tqdm(enumerate(test_dataset_without_transform)):
        if not os.path.exists(os.path.join(test_image_folder, str(label))):
            os.makedirs(os.path.join(test_image_folder, str(label)))
        img_path = os.path.join(test_image_folder, str(label), f'img_{img_idx}.png')
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img.save(img_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-d",'--dataset', type=str, default='cifar10')
    args.add_argument("-dp",'--dataset_path', type=str, default='../../data')
    args.add_argument("-i",'--image_folder', type=str,)
    args = args.parse_args()
    if args.image_folder is None:
        args.image_folder = f'../../data/{args.dataset}_seperate_images'
    dataset_convert_into_images(
        dataset_name = args.dataset,
        dataset_path = args.dataset_path,
        image_folder = args.image_folder,
    )