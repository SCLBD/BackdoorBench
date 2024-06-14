import os,re
from tqdm import tqdm
from PIL import Image
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, required=True)
parser.add_argument('--save_file_path', type = str, required=True)
args = parser.parse_args()

path = args.path #'/mnts2d/sec_data1/ChenHongrui/cifar10_stegastamp_b1/test/hidden'
save_file_path = args.save_file_path #'/mnts2d/sec_data1/ChenHongrui/cifar10_stegastamp_b1/test_b1.npy'
img_list = []
for file in tqdm(
    sorted(os.listdir(path), key=lambda x: [int(d) if d.isdigit() else d for d in re.split('(\d+)', x)])
    ):
    #if file.endswith('.png'):
    img = np.array(Image.open(path +'/'+file))
    img_list.append(img)

np.save(save_file_path,np.array(img_list))