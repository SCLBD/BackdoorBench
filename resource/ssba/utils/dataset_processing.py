
import os
import shutil
import random
from tqdm import tqdm
import argparse
from torchvision import transforms
import PIL
import cv2
from torchvision.utils import save_image
import numpy as np

parser = argparse.ArgumentParser(description='Dataset Preparation')

parser.add_argument('--input_dir', type=str, help='path to the original dataset')
parser.add_argument('--output_dir', type=str, help='path to the output dataset')
parser.add_argument('-train_size', type=int, help='size of the splited training set')
parser.add_argument('-test_size', type=int, help='size of the splited test set')
parser.add_argument('--shuffling', type=bool, help='shuffling or not', default=True)
parser.add_argument('--copy_only', type=bool, help='copy or cut', default=True)
parser.add_argument('--total_cnt', type=int, help='num of images for backdoor for each class', default=500)

args = parser.parse_args()






def rename_files(input_dir): 
    fileList = os.listdir(input_dir)
    bar = tqdm(fileList)
    for ind, filename in enumerate(bar): 
        oldname = input_dir + filename 
        main = fileList[ind].split('.')[0] 
        
        newname = input_dir + main +'.jpg' 
        os.rename(oldname,newname)   
        bar.set_description("Renaming {} / {}".format(ind+1, len(fileList)))










def split_dataset(input_dir, output_dir, train_size, test_size, shuffling=False, copy_only=True): 
    
    if output_dir is not None:
        output_train = output_dir+'train/'
        output_test = output_dir+'test/'
    else:
        output_train = input_dir+'train/'
        output_test = input_dir+'test/'
    
    if not os.path.exists(output_train):
        os.makedirs(output_train)
    if not os.path.exists(output_test):
        os.makedirs(output_test)  
      
    fileList = os.listdir(input_dir)
    if shuffling:
        random.shuffle(fileList)
  
    bar_train = tqdm(range(train_size))
    for _, i in enumerate(bar_train):
        full_path = os.path.join(input_dir, fileList[i])
        despath = os.path.join(output_train, fileList[i])
        if copy_only:
            shutil.copy(full_path, despath)
        else:
            shutil.move(full_path, despath)
        bar_train.set_description("Forming Training Set: {} / {}".format(i+1, train_size))

    bar_test = tqdm(range(train_size, train_size + test_size))
    for _, i in enumerate(bar_test):
        full_path = os.path.join(input_dir, fileList[i])
        despath = os.path.join(output_test, fileList[i])
        if copy_only:
            shutil.copy(full_path, despath)
        else:
            shutil.move(full_path, despath)   
        bar_test.set_description("Forming Test Set: {} / {}".format(i+1, train_size + test_size))











def transform_split(input_dir, output_dir, train_size, test_size, resize_resolution, crop_resolution, shuffling=False): 
    if output_dir is not None:
        output_train = output_dir+'train/'
        output_test = output_dir+'test/'
    else:
        output_train = input_dir+'train/'
        output_test = input_dir+'test/'
    
    if not os.path.exists(output_train):
        os.makedirs(output_train)
    if not os.path.exists(output_test):
        os.makedirs(output_test) 

    fileList = os.listdir(input_dir)
    if shuffling:
        random.shuffle(fileList)

    transform = transforms.Compose(
        [
            transforms.Resize(resize_resolution),
            transforms.CenterCrop(crop_resolution),
            transforms.ToTensor(),
        ]
    )

    bar_train = tqdm(range(train_size))
    for _, i in enumerate(bar_train):
        full_path = os.path.join(input_dir, fileList[i]) #获得完整的图像路径
        try:
            temp = cv2.imread(full_path, cv2.IMREAD_COLOR) 
            temp = temp[:, :, [2,1,0]] 
            image = PIL.Image.fromarray(temp)
            image = transform(image) #进行resize和crop操作改变图像大小
                        
            despath = os.path.join(output_train, fileList[i]) #输出路径
            save_image(image, despath, padding=0)
            
            bar_train.set_description("Resizing Training Set: {} / {}".format(i+1, train_size))
        except:
            print(full_path)

    bar_test = tqdm(range(train_size, train_size + test_size))
    for _, i in enumerate(bar_test):
        full_path = os.path.join(input_dir, fileList[i])
        try:
            temp = cv2.imread(full_path, cv2.IMREAD_COLOR) 
            temp = temp[:, :, [2,1,0]] 
            image = PIL.Image.fromarray(temp)
            image = transform(image) #进行resize和crop操作改变图像大小        
            
            despath = os.path.join(output_test, fileList[i])
            save_image(image, despath, padding=0)
            
            bar_test.set_description("Resizing Test Set: {} / {}".format(i+1, train_size + test_size))
        except:
            print(full_path)
        
    return









def resize_crop(input_dir, output_dir, resize_resolution, crop_resolution, shuffling=False): 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fileList = os.listdir(input_dir)
    if shuffling:
        random.shuffle(fileList)

    transform = transforms.Compose(
        [
            transforms.Resize(resize_resolution),
            transforms.CenterCrop(crop_resolution),
            transforms.ToTensor(),
        ]
    )
    size = len(fileList)
    bar_train = tqdm(range(size))
    for _, i in enumerate(bar_train):
        full_path = os.path.join(input_dir, fileList[i]) #获得完整的图像路径
        try:
            temp = cv2.imread(full_path, cv2.IMREAD_COLOR) 
            temp = temp[:, :, [2,1,0]] 
            image = PIL.Image.fromarray(temp)
            image = transform(image) #进行resize和crop操作改变图像大小
                        
            despath = os.path.join(output_dir, fileList[i]) #输出路径
            save_image(image, despath, padding=0)
            
            bar_train.set_description("Resizing Training Set: {} / {}".format(i+1, size))
        except:
            print(full_path)






def detect_grayscale(input_dir): 
    grayscale=[]
    fileList = os.listdir(input_dir)
    size = len(fileList)
    bar_train = tqdm(range(size))
    for _, i in enumerate(bar_train):
        full_path = os.path.join(input_dir, fileList[i]) #获得完整的图像路径
        image = PIL.Image.open(full_path)
        image = np.array(image)
        print(image.shape)
        if len(image.shape) == 2:
            grayscale.append(full_path)

    for i in range(len(grayscale)):
        print(grayscale[i])
    return









def merge_all(input_dir, output_dir, copy_only=True, rename=False, total_cnt=1e9, prefix='cifar', suffix='_hidden'):     
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dirList = os.listdir(input_dir) 
    dirSize = len(dirList)
    if dirSize == 0:
        print("The input directory must contain at least one sub-directory!")
        return
    
    bar = tqdm(range(dirSize))
    for _, i in enumerate(bar):
        filepath = os.path.join(input_dir, dirList[i])
        fileList = os.listdir(filepath)
        cnt = 0
        for j in fileList:
            full_path = os.path.join(filepath, j) #获得完整的图像路径  
            if rename:
                main = j.split('.')[0] 
                main = main.strip(prefix)
                main = main.strip(suffix)
                newname = prefix + main + suffix + '.png' 
                new_full_path = os.path.join(filepath, newname) #获得完整的图像路径  
                os.rename(full_path, new_full_path)   
                full_path = new_full_path
                
            if copy_only:
                shutil.copy(full_path, output_dir)
            else:
                shutil.move(full_path, output_dir) 
            
            cnt += 1
            if cnt >= total_cnt: break
        bar.set_description("Merging: {} / {}".format(i+1, dirSize))
    return








def rename_all(input_dir, prefix='cifar', suffix=''):     
    dirList = os.listdir(input_dir) 
    dirSize = len(dirList) 
    if dirSize == 0:
        print("The input directory must contain at least one sub-directory!")
        return
    
    bar = tqdm(range(dirSize))
    for _, i in enumerate(bar):
        filepath = os.path.join(input_dir, dirList[i])

        fileList = os.listdir(filepath)
        for j in fileList:
            full_path = os.path.join(filepath, j)
            main = j.split('.')[0] 
            main = main.strip(prefix)
            main = main.strip(suffix)
            newname = prefix + main + suffix + '.png' 
            new_full_path = os.path.join(filepath, newname)
            os.rename(full_path, new_full_path)   
            full_path = new_full_path

        bar.set_description("Renaming: {} / {}".format(i+1, dirSize))
    return
    
if __name__=="__main__":
    input_dir = '/workspace/getianshuo/data/ISSBA_dataset/sub-imagenet/sub-imagenet-200/val'
    output_dir = '/workspace/getianshuo/data/ISSBA_dataset/sub-imagenet/sub-imagenet-200/val_all'
    train_size = 50000
    test_size = 50000
    resize_resolution = 32
    crop_resolution = 32
    #rename_files(args.input_dir)
    #transform_split(input_dir, output_dir, train_size, test_size, resolution)
    #resize_crop(input_dir, output_dir, resize_resolution, crop_resolution)
    #detect_grayscale('E:/horse2zebra/trainB')
    #resize_crop(args.input_dir, args.output_dir, resize_resolution, crop_resolution, shuffling=False)
    #rename_all('E:/CIFAR-10/CIFAR10_Image')
    #split_dataset(args.input_dir, args.output_dir, args.train_size, args.test_size, args.shuffling)
    merge_all(input_dir, output_dir)
    
    