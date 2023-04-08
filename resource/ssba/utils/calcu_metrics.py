
import numpy
import math
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import os
from tqdm import tqdm
import argparse
import pandas as pd








def avg_psnr(img_dir1, img_dir2, suffix='_hidden', size=None, save_path=None, further=0, sort=True):
    fileList1 = os.listdir(img_dir1) 
    fileList2 = os.listdir(img_dir2) 
    if sort:
        fileList1.sort() 
        fileList2.sort()
    size1, size2 = len(fileList1), len(fileList2)
    assert size1 == size2, 'The number of images in the two dirs given should be the same!'
    
    if not size: size = size1
    
    total_psnr, total_ssim = 0.0, 0.0
    #bar= tqdm(range(size))
    for i in range(size):
    #for _, i in enumerate(bar):
        
        tmp1 = fileList1[i].split('.')[0].strip(suffix) 
        tmp2 = fileList2[i].split('.')[0].strip(suffix)
        if tmp1 != tmp2:
            print("{} and {} do not match!".format(fileList1[i], fileList2[i]))
            continue
        
        
        full_path1 = os.path.join(img_dir1, fileList1[i])     
        full_path2 = os.path.join(img_dir2, fileList2[i])            
        try:
            img1 = cv2.imread(full_path1, cv2.IMREAD_COLOR) 
            img1 = img1[:, :, [2,1,0]] 
            
            img2 = cv2.imread(full_path2, cv2.IMREAD_COLOR) 
            img2 = img2[:, :, [2,1,0]] 
        except:
            print('Something Wrong while loading images!')
            return
    
        assert img1.shape[0] == img2.shape[0], 'The resolution of two images must be the same!'
        
        psnr = compare_psnr(img1, img2, data_range=255)
        ssim = compare_ssim(img1, img2, win_size=11, data_range=255, multichannel=True) 

        total_psnr += psnr
        total_ssim += ssim

    avg_psnr = total_psnr / size
    avg_ssim = total_ssim / size
    print("The average PSNR between {} and {} is: {}".format(img_dir1, img_dir2, avg_psnr))
    print("The average SSIM between {} and {} is: {}".format(img_dir1, img_dir2, avg_ssim))
    if save_path:
        with open(args.test_save_file,'a') as f:
            f.write("The average PSNR between {} and {} is: {}".format(img_dir1, img_dir2, avg_psnr) + '\n')
            f.write("The average SSIM between {} and {} is: {}".format(img_dir1, img_dir2, avg_ssim) + '\n')
        f.close()

    if further:
        prefix_test = save_path.split('.')[0]
        test_save_path_csv = prefix_test + '.csv'
        data_frame = pd.read_csv(test_save_path_csv, index_col=False)
        alist = [img_dir1, avg_psnr, avg_ssim]
        data_frame.loc[len(data_frame)]=alist
        data_frame.to_csv(test_save_path_csv, index=False)

    return avg_psnr, avg_ssim

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Calculate Image Metrics')

    parser.add_argument('--input_dir1', type=str, help='path to the image dataset1')
    parser.add_argument('--input_dir2', type=str, help='path to the image dataset2')
    parser.add_argument('--size', type=int, default=None, help='num of pairs of imgs to compute PSNR & SSIM')
    
    parser.add_argument("--test_save_file", type=str, default=None, help="where to save test file")
    parser.add_argument("--further", type=int, default=0, help="futher save in .csv file nor not")

    args = parser.parse_args()

    avg_psnr(args.input_dir1, args.input_dir2, suffix='_hidden', size=args.size, save_path = args.test_save_file, 
            further = args.further)