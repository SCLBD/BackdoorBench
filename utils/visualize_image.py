import sys, logging
sys.path.append('../')

import torch, random
import numpy as np

from torchvision.utils import make_grid

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# just show image

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def all_to_torch(array):
    if isinstance(array, np.ndarray):
        if array.shape.__len__() == 2:
            return torch.tensor(array)
        elif array.shape.__len__() == 3:
            return torch.tensor(array.transpose((2,0,1)))
        elif array.shape.__len__() == 4:
            return torch.tensor(array.transpose((0, 3, 1, 2)))
    else:
        return array.detach().clone().cpu()

from typing import Tuple
import time

def image_show_for_all(
    img,
    title :str = None,
    figsize : Tuple[int,int] = None,
    s : bool = False,
):

    np_img = make_grid(all_to_torch(img))

    fig = plt.figure(
        figsize=(15, 15) if figsize is None else figsize
    )

    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.title(title)
    plt.show()

    if s :
        plt.imsave(f'{time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())}.png', np_img)

def test_image_show_for_all():

    i1 = np.random.randn(32,32)
    image_show_for_all(i1)

    i2 = np.random.randn(32,32,3)
    image_show_for_all(i2)

    i3 = np.random.randn(3,32,32,3)
    image_show_for_all(i3)

    t1 = torch.from_numpy(i1)
    image_show_for_all(t1)

    t2 = torch.from_numpy(i2.transpose((2,0,1)))
    image_show_for_all(t2)

    t3 = torch.from_numpy(i3.transpose((0, 3,1,2)))
    image_show_for_all(t3)




def imshow(torch_batch_images, title = None):
    npimages = make_grid(torch_batch_images.detach().cpu())
    fig = plt.figure(figsize = (15, 15))
    plt.imshow(np.transpose(npimages,(1,2,0)))
    plt.title(torch_batch_images.__str__ if title is None else title)
    plt.show()

# if __name__ == '__main__':

