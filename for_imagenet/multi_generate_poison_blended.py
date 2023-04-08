'''
Generation of Blended training data, with multiprocessing to speed up.
'''

class Args:
    pass
args = Args()
args.__dict__ = {
    "attack": "blended",
    "attack_trigger_img_path" : "../resource/blended/hello_kitty.jpeg",
    "attack_train_blended_alpha": 0.2,
    "attack_test_blended_alpha": 0.2,
    "img_size" : [224,224,3],
}


pratio = 0.001
attack = args.__dict__['attack']
imagenet_path = "../data/imagenet/train"
target_path = f"../imagenet_poison/{attack}/train"
target_class_folder_name = "n01440764" # None then do not filt

from multiprocessing import Pool
import tqdm
import os, glob, random, re
import sys, yaml, os
import numpy as np

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
MIN_VALID_IMG_DIM = 32

from utils.aggregate_block.bd_attack_generate import *
from des_stats import stats

train_bd_transform,test_bd_transform = bd_attack_img_trans_generate(args)

def is_valid_file(path):
    try:
        img = Image.open(path)
        img.verify()
    except:
        return False
    if not (img.height >= MIN_VALID_IMG_DIM and img.width >= MIN_VALID_IMG_DIM):
        return False
    return True

def do_work(filepath):
    img = Image.open(filepath)

    #  target path
    target_filepath = filepath.replace(
        imagenet_path,
        target_path
    )

    img = np.asarray(img).astype('uint8')
    if len(img.shape) == 2:
        img = np.concatenate(3 * [img[..., None]], axis=2)
    if img.shape[2] != 3:
        img = img[:, :, :3]
    img = Image.fromarray(img)

    # select
    if random.uniform(0, 1) < pratio:
        # do poison
        img = Image.fromarray(
            np.clip(
                train_bd_transform(img), 0, 255).astype(np.uint8)
        )

        p = re.compile(r'/n(\d)+/')
        target_filepath = p.sub(f"/{target_class_folder_name}/", target_filepath)

        print(target_filepath)

    # save to
    img.save(target_filepath)
    img.close()

if __name__ == '__main__':

    # copy the whole class folder structure

    originalClassFolderList = filter(os.path.isdir, [f"{imagenet_path}/{subfolder_name}" for subfolder_name in
                                                     os.listdir(imagenet_path)])
    for folderPath in originalClassFolderList:
        folderPath = folderPath.replace(
            imagenet_path,
            target_path
        )
        if not os.path.exists(
                folderPath
        ):
            os.makedirs(
                folderPath
            )

    # valid list for img
    filePathList = [
        filepath for filepath in tqdm.tqdm(glob.iglob(imagenet_path + '**/**', recursive=True),desc="valid list")
        if os.path.isfile(filepath) and is_valid_file(filepath)
    ]

    tasks = filePathList

    pool = Pool()
    for _ in tqdm.tqdm(pool.imap_unordered(do_work, tasks), total=len(tasks)):
        pass

    stats(imagenet_path)
    stats(target_path)
