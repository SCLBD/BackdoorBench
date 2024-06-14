import argparse
import os
import sys
import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import skimage
import torchvision.utils as vutils  
import numpy
from skimage import filters
sys.path.append("..") 
from models.HidingRes import HidingRes
from models.ReflectionUNet import UnetGenerator2
from models.up_UNet import UNet
from models.up_UNet_rp import UNet_rp
from models.Huang_UNet import UnetGenerator_H
from tqdm import tqdm

import utils.transformed as transforms

import math
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--Rnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--ngpu', type=int, default=1,  
                    help='number of GPUs to use')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')

# parser.add_argument('--test', action='store_true', help=' make data for test set ')
parser.add_argument('--rp', action='store_true', help='reflection padding in Unet ')
# parser.add_argument('--train', action='store_true', help='same ratio between trigger and clean example')
parser.add_argument('--uh', action='store_true', help='reflection padding in Huang_Unet ')
parser.add_argument('--num_downs', type=int, default=5, help='nums of  Unet downsample')

parser.add_argument('--Hnet', type = str)

# parser.add_argument("--data_dir", type = str)
parser.add_argument("--from_path", type = str, )
parser.add_argument("--to_path", type = str, )



def main():
    
    opt = parser.parse_args()
    cudnn.benchmark = True
    random_R = 80
    random_G = 160
    random_B = 80

    input_folder = opt.from_path
    output_folder = opt.to_path

    # if opt.test:
    #     input_folder = f"{opt.data_dir}/test"
    #     output_folder = "./test"
    #
    # if opt.train:
    #     input_folder = f"{opt.data_dir}/train"
    #     output_folder = "./train"

    mask_th = 28

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok = True)

    ###############################################    ##################################################################

    if opt.rp:
        Hnet = UNet_rp(n_channels=6, n_classes=3,  requires_grad=True)
    elif opt.uh:
        Hnet = UnetGenerator_H(input_nc=6, output_nc=3, num_downs=opt.num_downs, output_function=nn.Sigmoid, requires_grad=True)        
    else:
        Hnet = UNet(n_channels=6, n_classes=3,  requires_grad=True)

    Hnet.cuda()

    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()

    Hnet.load_state_dict(torch.load(opt.Hnet))
    Hnet.eval()

    ############################################### embedding step   ##################################################################
    for target in sorted(os.listdir(input_folder)):
        # each label

        input_folder_child = os.path.join(input_folder, target)
        output_folder_child = os.path.join(output_folder, target)

        if not os.path.exists(output_folder_child):
            os.makedirs(output_folder_child, exist_ok = True)

        clean_imgs_from_class = os.listdir(input_folder_child)
        clean_imgs_num_from_class = len(clean_imgs_from_class)

        with torch.no_grad():
            for i in tqdm(range(clean_imgs_num_from_class), desc=str(target)):
                cover_img = Image.open(input_folder_child + "/" + clean_imgs_from_class[i]).convert("RGB")

                loader = transforms.Compose([transforms.ToTensor()])
                cover_img = loader(cover_img)
                cover_img  = cover_img.cuda()
                cover_img = cover_img.unsqueeze(0)
                clean_img = ToPILImage()(torch.ones((3,32,32)))
                clean_img = loader(clean_img)
                clean_img = clean_img.unsqueeze(0).cuda()
                clean_img=clean_img[:,:,0:32,0:32].cuda()

                cover_B = cover_img

                cover_B_np = cover_B.squeeze(0).mul(255).cpu().numpy().transpose((1,2,0))  #记得乘255
                cover_B_gray_np = 0.299 * cover_B_np[:,:,0] +  0.587 * cover_B_np[:,:,1] + 0.114 * cover_B_np[:,:,2]
                mask = filters.sobel(cover_B_gray_np)
                mask = mask > mask_th
                mask = mask.astype(numpy.uint8)    #edge 作为水印和原图进行拼接   uint 8  (256,768) 值为0，1
                mask = mask[:,:,numpy.newaxis]  #(256,256,1)
                mask = mask
                mask_np = mask * 255
                mask_t = torch.from_numpy(mask_np.transpose((2, 0, 1)))
                mask_t = mask_t.float().div(255)
                mask_t = mask_t.repeat(3,1,1).unsqueeze(0).cuda()

                edge_np = numpy.ones((32,32,3))
                edge_np[:,:,0:1] = random_R * mask
                edge_np[:,:,1:2] = random_G * mask
                edge_np[:,:,2:3] = random_B * mask
                edge_t = torch.from_numpy(edge_np.transpose((2, 0, 1)))
                edge_t = edge_t.float().div(255).unsqueeze(0).cuda()

                watermark = edge_t * mask_t + clean_img * (1 - mask_t)

                concat_img = torch.cat([cover_B, watermark], dim=1)
                stego_img = Hnet(concat_img)
                stego_img = torch.clamp(stego_img, 0, 1)

                ToPILImage()(stego_img.squeeze(0).cpu()).save(output_folder_child + "/" + clean_imgs_from_class[i])



#
#
#
#
# def creatdir(path):
#     folders = []
#     while not os.path.isdir(path):
#         path, suffix = os.path.split(path)
#         folders.append(suffix)
#     for folder in folders[::-1]:
#         path = os.path.join(path, folder)
#         os.mkdir(path)


if __name__ == '__main__':
    main()

# ['n01883070', 'n02190166', 'n02361337', 'n02445715', 'n02526121', 'n02804414', 'n03208938', 'n03476684', 'n03935335', 'n04442312']
# ['n01558993', 'n01692333', 'n01729322', 'n01735189', 'n01749939', 'n01773797', 'n01820546', 'n01855672',
#  'n01978455', 'n01980166', 'n01983481', 'n02009229', 'n02018207', 'n02085620', 'n02086240', 'n02086910',
#  'n02087046', 'n02089867', 'n02089973', 'n02090622', 'n02091831', 'n02093428', 'n02099849', 'n02100583',
#  'n02104029', 'n02105505', 'n02106550', 'n02107142', 'n02108089', 'n02109047', 'n02113799', 'n02113978',
#  'n02114855', 'n02116738', 'n02119022', 'n02123045', 'n02138441', 'n02172182', 'n02231487', 'n02259212',
#  'n02326432', 'n02396427', 'n02483362', 'n02488291', 'n02701002', 'n02788148', 'n02804414', 'n02859443',
#  'n02869837', 'n02877765', 'n02974003', 'n03017168', 'n03032252', 'n03062245', 'n03085013', 'n03259280',
#  'n03379051', 'n03424325', 'n03492542', 'n03494278', 'n03530642', 'n03584829', 'n03594734', 'n03637318',
#  'n03642806', 'n03764736', 'n03775546', 'n03777754', 'n03785016', 'n03787032', 'n03794056', 'n03837869',
#  'n03891251', 'n03903868', 'n03930630', 'n03947888', 'n04026417', 'n04067472', 'n04099969', 'n04111531',
#  'n04127249', 'n04136333', 'n04229816', 'n04238763', 'n04336792', 'n04418357', 'n04429376', 'n04435653',
#  'n04485082', 'n04493381', 'n04517823', 'n04589890', 'n04592741', 'n07714571', 'n07715103', 'n07753275',
#  'n07831146', 'n07836838', 'n13037406', 'n13040303']