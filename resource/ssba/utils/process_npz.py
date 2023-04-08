
import numpy as np
import os
from tqdm import tqdm
import PIL
from torchvision import transforms
from torchvision.utils import save_image







def biggan_npz2images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    file = np.load(input_dir) 
    print('The columns in the .npz file is', file.files)
    data, label = file['x'], file['y']
    print('The shape of data is', data.shape)
    print('The shape of label is', label.shape)
    
    data_size = data.shape[0] 
    bar_data = tqdm(range(data_size)) 
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    
    for _, i in enumerate(bar_data):
        img = data[i].transpose(1,2,0) 
        img = PIL.Image.fromarray(img) 
        img = transform(img) 
        img_path = str(i)+"_"+str(label[i])+".png" 
        despath = os.path.join(output_dir, img_path)
        save_image(img, despath, padding=0)
        bar_data.set_description("Processing: {} / {}".format(i+1, data_size))
    
    print('Done!')
    
if __name__=="__main__":
    input_path = 'E:/samples.npz'
    output_path = 'E:/CUT_images'
    biggan_npz2images(input_path, output_path)