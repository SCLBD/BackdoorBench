'''
Generation of BadNets validation data (for ASR and RA)
'''

class Args:
    pass
args = Args()
args.__dict__ = {
    'attack':"badnet",
    "patch_mask_path" : "../resource/badnet/bottom_right_3by3_white.npy",
    "img_size" : [224,224,3],
}


pratio = 1
attack = args.__dict__['attack']
imagenet_path = "../data/imagenet/val"
target_path = f"../imagenet_poison/{attack}/val"
ra_path = f"../imagenet_poison/{attack}/ra"

target_class_folder_name = "n01440764" # None then do not filt


import os, glob, random, re
import sys, yaml, os
import numpy as np

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

from tqdm import tqdm
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

# train_dataset_without_transform = ImageFolder(
#     root = f"{args.dataset_path}/train",
#     is_valid_file=is_valid_file,
# )

# valid list
filePathList = [
    filepath for filepath in tqdm(glob.iglob(imagenet_path + '**/**', recursive=True),desc="valid list")
    if os.path.isfile(filepath) and is_valid_file(filepath)
]

# filter target class for test
if target_class_folder_name is not None:
    filePathList = filter(lambda filepath: target_class_folder_name not in filepath , filePathList)

# poison_filelist = random.sample(filePathList, int(len(filePathList) * pratio))
poison_filelist = []

for filepath in tqdm(filePathList, desc="process bd"):

    img = Image.open(filepath)

    ra_filepath = filepath

    #  target path
    target_filepath = filepath.replace(
        imagenet_path,
        target_path
    )

    p = re.compile(r'/n(\d)+/')
    target_filepath = p.sub(f"/{target_class_folder_name}/", target_filepath)

    # check folder
    if not os.path.exists(
            os.path.dirname(
                target_filepath
            )
    ):
        os.makedirs(
            os.path.dirname(
                target_filepath
            )
        )

    ra_filepath = ra_filepath.replace(
        imagenet_path,
        ra_path,
    )

    if not os.path.exists(
            os.path.dirname(
                ra_filepath
            )
    ):
        os.makedirs(
            os.path.dirname(
                ra_filepath
            )
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

        poison_filelist.append(target_filepath)

    #save to
    img.save(target_filepath)
    img.save(ra_filepath)
    img.close()

with open(f'{attack}_val.txt', 'w') as f:
    for line in poison_filelist:
        f.write(f"{line}\n")

stats(imagenet_path)
stats(target_path)
stats(ra_path)