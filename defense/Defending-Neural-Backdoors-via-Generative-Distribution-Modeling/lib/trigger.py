import numpy as np
import matplotlib.pyplot as plt
import torch
import copy

class Trigger:
    def __init__(self, dataset_name, content, target, type_=0, args={}):
        self.name = dataset_name
        self.content = content # CHW float numpy array, 0. to 1.
        self.content_img = (self.content.transpose(1,2,0) * 255).astype(int)
        self.content_batch = torch.tensor(self.content, dtype=torch.float).unsqueeze(0) # placeholder
        self.clip_and_sync()
        self.target = target   # int
        self.type_ = type_     
        self.args = args       
        self.height = self.content.shape[1]
        self.width = self.content.shape[2]
        self.optimizer = torch.optim.SGD([self.content_batch], lr=0.1, momentum=0.9, weight_decay=0)
        self.accuracy = 0.
        self.apply_loc = (0,0)
    
    def sync_content(self, source):
        if self.name == 'cifar':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
            std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1)
        elif self.name == 'cifar100':
            mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1,3,1,1)
            std  = torch.tensor([0.2675, 0.2565, 0.2761]).view(1,3,1,1)
        else: # imagenet
            mean = torch.tensor([[0.485, 0.456, 0.406]]).view(1,3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        
        if source == "content":
            self.content_img = (self.content.transpose(1,2,0) * 255).astype(int)
            self.content_batch.data = (torch.tensor(self.content, dtype=torch.float).unsqueeze(0) - mean) / std
        elif source == "content_batch":
            self.content = (self.content_batch * std + mean).squeeze(0).numpy()
            self.content_img = (self.content.transpose(1,2,0) * 255).astype(int)
        return self
        
    def display(self):
        fig = plt.figure(figsize=(1,1), dpi=80)
        plt.axis('off')
        plt.imshow(self.content_img)
        plt.show()
        return self

    def duplicate(self):
        return copy.deepcopy(self)
    
    def clip_and_sync(self):
        self.content = np.clip(self.content, 0., 1.)
        self.sync_content("content")
        return self
    
    def add_noise(self, type_, args):
        if type_ == "Gaussian":
            mean, std = 0, args["std"]
            self.content += np.random.normal(mean, std, self.content.shape)
            self.clip_and_sync()
        elif type_ == "Const":
            const_noise = args
            self.content += const_noise
            self.clip_and_sync()
        return self
    
    def apply_img(self, img):
        """apply trigger to one image(HWC int array, 0-255)"""
        img_height = img_width = 32 if (self.name == "cifar" or self.name == 'cifar100') else 224
        if self.type_ == 0:
            # - bottom-right corner, covers image
            img[-self.height:,-self.width:,:] = self.content_img
        elif self.type_ == 1:
            # - location at (args['x'], args['y']), covers image 
            x, y = self.args['x'], self.args['y']
            if x>=0 and y>=0 and x+self.height<img_height and y+self.width<img_width:
                img[x:x+self.height,y:y+self.width,:] = self.content_img
            else:
                print("ERROR: Trigger location out of range.")
        elif self.type_ == 2:
            # - random location, convers image
            x = int(np.random.random() * (img_height - self.height))
            y = int(np.random.random() * (img_width - self.width))
            img[x:x+self.height,y:y+self.width,:] = self.content_img
        elif self.type_ == 99:
            rand_img = (np.random.random((self.height,self.width,3))*self.args['range']+self.args['base']).astype(int)
            img[-self.height:,-self.width:,:] = rand_img
        else:
            print("ERROR: Trigger type not implemented.")
        return img
    
    def apply_batch(self, inputs, targets, ratio=1):
        """apply trigger to input tensor(BCHW float tensor)"""
        img_height = img_width = 32 if (self.name == "cifar" or self.name == "cifar100") else 224
        if self.type_ == 0:
            if ratio == 1:
                inputs[:,:,-self.height:,-self.width:] = self.content_batch
                targets[:] = self.target
            else:
                mask = torch.rand(inputs.size(0))
                inputs[mask<ratio,:,-self.height:,-self.width:] = self.content_batch
                targets[mask<ratio] = self.target
        elif self.type_ == 2:
            # - random location, convers image
            x = int(np.random.random() * (img_height - self.height))
            y = int(np.random.random() * (img_width - self.width))
            self.apply_loc = (x, y)
            if ratio == 1:
                inputs[:,:,x:x+self.height,y:y+self.width] = self.content_batch
                targets[:] = self.target
            else:
                mask = torch.rand(inputs.size(0))
                inputs[mask<ratio,:,x:x+self.height,y:y+self.width] = self.content_batch
                targets[mask<ratio] = self.target
        else:
            print("ERROR: Trigger type not implemented.")
        return inputs, targets