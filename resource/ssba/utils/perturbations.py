
import cv2
import argparse
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms

from PIL import Image, ImageOps
from calcu_metrics import avg_psnr
import diff_utils

parser = argparse.ArgumentParser(description='Perturbations on Images')

parser.add_argument('--original_dir', type=str, help='path to the nature image dataset')
parser.add_argument('--input_dir', type=str, help='path to the original dataset')
parser.add_argument('--output_dir', type=str, default=None, help='path to the output dataset')
parser.add_argument('--method', type=str, help='which type perturbation to add?')

parser.add_argument('--std', type=float, default=0.1, help='the std of gaussian noise')
parser.add_argument('--kernel_size', type=int, default=7, help='the kernel size for gaussian blur')
parser.add_argument('--crop_size', type=int, default=128, help='the size for center crop')

parser.add_argument('--quality', type=int, default=80, help='quality of jpeg compression')
parser.add_argument('--diff', action="store_true", help='use methods that are diffenrentiable or not')

parser.add_argument('--size', type=int, help='num of pairs of imgs to compute PSNR & SSIM')

args = parser.parse_args()








def custom_jpeg(input_dir, output_dir, quality):
    if not output_dir:    
        output_dir = input_dir.strip('/') + '_jpeg_'+ str(quality)
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fileList = os.listdir(input_dir) 
    size = len(fileList)
    
    #bar= tqdm(range(size))
    #for _, i in enumerate(bar):
    for i in range(size):
        main = fileList[i].split('.')[0]+'.jpg' 
        full_path = os.path.join(input_dir, fileList[i]) 
        try:
            img = cv2.imread(full_path, cv2.IMREAD_COLOR) 
            dest_path = os.path.join(output_dir, main)
            cv2.imwrite(dest_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        except:
            print('Something Wrong while loading images!')
            return
        
    return output_dir









def diff_jpeg(input_dir, output_dir, quality):
    if not output_dir:    
        output_dir = input_dir.strip('/') + '_jpeg_'+ str(quality)
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fileList = os.listdir(input_dir) 
    size = len(fileList)    
    bar = tqdm(range(size))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    
    for _, i in enumerate(bar):
        main = fileList[i].split('.')[0]+'.png' 
        full_path = os.path.join(input_dir, fileList[i]) 
        try:
            img = cv2.imread(full_path, cv2.IMREAD_COLOR)
            #img = ImageOps.fit(img, (128, 128))#裁剪图片至指定大小
            img = img[:, :, [2,1,0]] 
            img = np.array(img) / 255. 
            img = np.transpose(img, [2, 0, 1]) 
            img_tensor = torch.from_numpy(img).unsqueeze(0).float()
            
            recover = diff_utils.jpeg_compress_decompress(img_tensor, quality=quality)
            recover = recover.detach().squeeze(0).numpy()
            recover = np.transpose(recover, [1, 2, 0]) 
            recover = transform(recover)
            
            dest_path = os.path.join(output_dir, main)
            save_image(recover, dest_path, padding=0)
        except:
            print('Something Wrong while loading images!')
            return
        
    return output_dir








def custom_guassian_noise(input_dir, output_dir, std):
    if not output_dir:    
        output_dir = input_dir.strip('/') + '_gaussian_noise_'+ str(std)
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fileList = os.listdir(input_dir) 
    size = len(fileList)
    assert size != 0, 'The input image dir is empty!'
    bar= tqdm(range(size))

    transform = transforms.Compose([transforms.ToTensor()])
        
    for _, i in enumerate(bar):
        full_path = os.path.join(input_dir, fileList[i]) 
        try:
            img = cv2.imread(full_path, cv2.IMREAD_COLOR)
            img = img[:, :, [2,1,0]] 
            img = transform(img)
            noise = torch.normal(mean=0, std=std, size=img.size(), dtype=torch.float32)
            img = img + noise
            img = torch.clamp(img, 0, 1)
            dest_path = os.path.join(output_dir, fileList[i])
            save_image(img, dest_path, padding=0)
        except:
            print('Something Wrong while loading images!')
            return

    return output_dir








def custom_centercrop(input_dir, output_dir, crop_size):
    if not output_dir:    
        output_dir = input_dir.strip('/') + '_centercrop_'+ str(crop_size)
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fileList = os.listdir(input_dir) 
    size = len(fileList)
    assert size != 0, 'The input image dir is empty!'
    bar= tqdm(range(size))
    
    trial_path = os.path.join(input_dir, fileList[0]) 
    img = cv2.imread(trial_path, cv2.IMREAD_COLOR)
    img_height, img_width = img.shape[0], img.shape[1]
    assert img_height == img_width, 'The height and width of an image should be the same!'
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(crop_size),
            transforms.Resize(img_height),
        ]
    )
        
    for _, i in enumerate(bar):
        full_path = os.path.join(input_dir, fileList[i]) 
        try:
            img = cv2.imread(full_path, cv2.IMREAD_COLOR)
            img = img[:, :, [2,1,0]] 
            img = transform(img)
            img = torch.clamp(img, 0, 1)
            dest_path = os.path.join(output_dir, fileList[i])
            save_image(img, dest_path, padding=0)
        except:
            print('Something Wrong while loading images!')
            return

    return output_dir








def custom_gaussian_blur(input_dir, output_dir, kernel_size):
    if not output_dir:    
        output_dir = input_dir.strip('/') + '_gaussian_blur_'+ str(kernel_size)
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fileList = os.listdir(input_dir) 
    size = len(fileList)
    assert size != 0, 'The input image dir is empty!'
    bar= tqdm(range(size))
    sigma = 0.8 + 0.3 * ((kernel_size - 1) * 0.5 -1)
    kernel_size = (kernel_size, kernel_size)
    transform = transforms.Compose([transforms.ToTensor(),])
    
    for _, i in enumerate(bar):
        full_path = os.path.join(input_dir, fileList[i]) 
        try:
            img = cv2.imread(full_path, cv2.IMREAD_COLOR)
            #img = img[:, :, [2,1,0]] 
            img = cv2.GaussianBlur(img, ksize = kernel_size, sigmaX=-1)
            dest_path = os.path.join(output_dir, fileList[i])
            #img = transform(img)
            #save_image(img, dest_path, padding=0)
            cv2.imwrite(dest_path, img)
        except:
            print('Something Wrong while loading images!')
            return

    return output_dir








def diff_gaussian_blur(input_dir, output_dir, kernel_size):
    if not output_dir:    
        output_dir = input_dir.strip('/') + '_gaussian_blur_'+ str(kernel_size)
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fileList = os.listdir(input_dir) 
    size = len(fileList)
    assert size != 0, 'The input image dir is empty!'
    bar= tqdm(range(size))
    
    trial_path = os.path.join(input_dir, fileList[0]) 
    img = cv2.imread(trial_path, cv2.IMREAD_COLOR)
    img_height, img_width = img.shape[0], img.shape[1]
    assert img_height == img_width, 'The height and width of an image should be the same!'
    transform = transforms.Compose([transforms.ToTensor(),])
        
    for _, i in enumerate(bar):
        full_path = os.path.join(input_dir, fileList[i]) 
        try:
            img = cv2.imread(full_path, cv2.IMREAD_COLOR)
            img = img[:, :, [2,1,0]] 
            img = transform(img).unsqueeze(0)
            
            f = diff_utils.random_blur_kernel(probs=[.25, .25], N_blur=kernel_size, sigrange_gauss=[1., 3.], 
                                              sigrange_line=[.25, 1.], wmin_line=3)
            img = F.conv2d(img, f, bias=None, padding=int((kernel_size - 1) / 2)) 
            img = img.squeeze(0)
            dest_path = os.path.join(output_dir, fileList[i])
            save_image(img, dest_path, padding=0)
        except:
            print('Something Wrong while loading images!')
            return

    return output_dir
    
if __name__=="__main__":
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args,arg))), '<')
        
    ori_path = args.original_dir
    input_path = args.input_dir
    output_path = args.output_dir
    method = args.method
    
    std = args.std
    kernel_size = args.kernel_size
    crop_size = args.crop_size
    size = args.size
    quality = args.quality
    
    assert method in ['jpeg', 'gaussian_noise', 'gaussian_blur', 'center_crop'], \
        'Currently only support jpeg, gaussian_noise, gaussian_blur, center_crop'
        
    if method == 'jpeg':
        if not args.diff:
            print('Using JPEG from Opencv-Python with quality', quality)
            out_dir = custom_jpeg(input_path, output_path, quality)
        else:
            print('Using DiffJPEG with quality', quality)
            out_dir = diff_jpeg(input_path, output_path, quality)
    elif method == 'gaussian_noise':
        print('Using gaussain noise with std', std)
        out_dir = custom_guassian_noise(input_path, output_path, std)
    elif method == 'center_crop':
        print('Using center crop with size', crop_size)
        out_dir = custom_centercrop(input_path, output_path, crop_size)
    elif method == 'gaussian_blur':
        if not args.diff:
            print('Using gaussian blur from Opencv-Python with kernel size', kernel_size)
            out_dir = custom_gaussian_blur(input_path, output_path, kernel_size)
        else:
            print('Using diff gaussian blur with kernel size', kernel_size)
            out_dir = diff_gaussian_blur(input_path, output_path, kernel_size)
    
    avg_psnr(ori_path, out_dir, suffix='_hidden', size=size, sort=True)
