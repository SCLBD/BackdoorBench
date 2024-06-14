# encoding: utf-8
"""
@author: yongzhi li
@contact: yongzhili@vip.qq.com

@version: 1.0
@file: Reveal.py
@time: 2018/3/20

"""
import torch 
from torch.autograd import Variable
import torch.nn as nn

# def gaussian_noise(tensor, mean=0, stddev=0.1):  
#     noise = torch.nn.init.normal_(tensor.new(tensor.size()), 0, 0.1)
#     return Variable(noise)

def gaussian_noise(tensor, mean=0, stddev=0.01):  
    noise = torch.nn.init.normal_(tensor.new(tensor.size()), 0, 0.01)
    noise_tensor = tensor + noise
    real_noise = torch.clamp(noise_tensor, 0, 1) - tensor
    return Variable(real_noise)

class RevealNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, nhf=64, output_function=nn.Sigmoid,requires_grad=True):
        super(RevealNet, self).__init__()
        # input is (3) x 256 x 256
        self.main = nn.Sequential(
            nn.Conv2d(in_c, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf*4),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, out_c, 3, 1, 1),
            output_function()
        )

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        output=self.main(input)
        return output



class Reveal_noise_Net(nn.Module):
    def __init__(self, nc=3, nhf=64, output_function=nn.Sigmoid):
        super(Reveal_noise_Net, self).__init__()
        # input is (3) x 256 x 256
        self.main = nn.Sequential(
            nn.Conv2d(nc, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf*4),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nc, 3, 1, 1),
            output_function()
        )

    def forward(self, input):
        x_n = gaussian_noise(input.data, 0, 0.01)
        output=self.main(x_n+input)
        # output = torch.clamp(output, min = 0,max = 1)
        return output