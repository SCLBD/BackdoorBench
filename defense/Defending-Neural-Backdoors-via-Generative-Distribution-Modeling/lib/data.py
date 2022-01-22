import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import copy

class Data:
    def __init__(self, dataset_name, dataset, dataloader):
        self.dataset = dataset
        self.dataloader = dataloader
        self.name = dataset_name
        
        if self.name == 'cifar':
#             self.size = len(dataset.train_labels if dataset.train else 
#                         dataset.test_labels) # number of images
            self.size = 50000 if dataset.train else 10000
            self.num_poisoned = 0 # number of poisioned images
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#             if self.dataset.train == True:
#                 self.data = self.dataset.train_data     # N*32*32*3 array, 0 to 255
#                 self.labels = self.dataset.train_labels # N list, 0 to 9
#             else:
#                 self.data = self.dataset.test_data
#                 self.labels = self.dataset.test_labels
            self.data = self.dataset.data
            self.labels = self.dataset.targets
        
    def duplicate(self): # return a copy
        return copy.deepcopy(self)
    
    def reset(self):
        self.num_poisoned = 0
        return self
    
    def poison(self, trigger, ratio):
        if self.name == 'cifar':
            # return if not enough clean images are left
            N = int(self.size * ratio)
            if self.num_poisoned + N > self.size:
                print("WARNING: Not enough clean images.")
            else:
                # poison the images in sequence. dataloader will shuffle them when training.
                start, end = self.num_poisoned, self.num_poisoned + N
                # expecting more efficient implementation
                for i in range(start, end):
                    self.data[i] = trigger.apply_img(self.data[i])
                    self.labels[i] = trigger.target
                # update `num_poisoned`
                self.num_poisoned += N
        return self
            
    def display(self, ratio=0.002, width=20, zoom=2):
        if self.name == 'cifar':
            # pick 1/100 to display, 20 images per line
            stride = int(1./ratio)
            # try to plot it pixel-perfect, but not very accurate
            dpi = 80 
            margin = 0.05
            xpixels, ypixels = (1+2*margin)*32*width, (1+2*margin)*32*self.size//stride//width
            fig = plt.figure(figsize=(zoom*xpixels/dpi, zoom*ypixels/dpi), dpi=dpi)
            ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
            # rearrange images with split&stack trick
            output = np.vstack(np.hsplit(np.hstack(self.data[::stride][:self.size//stride//width*width]),self.size//stride//width))
            ax.imshow(output)
            plt.show()
            return self
        
    def sample(self):
        for i in self.dataloader:
            ret = i
            break
        return i