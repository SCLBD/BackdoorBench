import torch
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):
    def __init__(self,nofchannels=3):
        super(STN,self).__init__()         
        self.nofchannels=nofchannels
        self.localization= nn.Sequential(nn.Conv2d(3,16,kernel_size=7),
                           nn.MaxPool2d(2,stride=2),
                           nn.ReLU(True), 
                           nn.Conv2d(16,32,kernel_size=5),
                           nn.MaxPool2d(2,stride=2),
                           nn.ReLU(True),
                           nn.Conv2d(32,64,kernel_size=3),
                           nn.MaxPool2d(2,stride=2),
                           nn.ReLU(True))
        
        self.fc_loc=  nn.Sequential(nn.Linear(64,128),
                      nn.ReLU(True),
                      nn.Linear(128,64),
                      nn.ReLU(True),
                      nn.Linear(64,6))
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1,0,0,0,1,0]))

        
    def forward(self,x):
        xs=self.localization(x)
        xs=xs.view(-1,64)
        theta=self.fc_loc(xs)
        theta=theta.view(-1,2,3)
        grid=F.affine_grid(theta,x.size())
        x=F.grid_sample(x,grid)
        return x