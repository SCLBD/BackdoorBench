# ResNet generator and discriminator
import torch
from torch import nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm
import numpy as np


channels = 3

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

GEN_SIZE=128
DISC_SIZE=128

class Generator(nn.Module):
    def __init__(self, name, param):
        super(Generator, self).__init__()
        self.name = name
        if self.name == "cifar10" or self.name == "cifar100":
            self.in_size     = in_size     = param['in_size']
            self.skip_size   = skip_size   = in_size // 4 # NOTE: skip connections improve model stability
            self.out_size    = out_size    = param['out_size']
            self.hidden_size = hidden_size = param['hidden_size']
            self.fc1 = nn.Linear(skip_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size + skip_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size + skip_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size + skip_size, out_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
            self.bn3 = nn.BatchNorm1d(hidden_size)
        else: # imagenet
            self.in_size     = in_size     = param['in_size']
            self.out_size    = out_size    = param['out_size']
            self.hidden_size = hidden_size = param['hidden_size']
            
            self.dense = nn.Linear(in_size, 2 * 2 * hidden_size)
            self.final = nn.Conv2d(hidden_size, 3, 3, stride=1, padding=1)
            self.model = nn.Sequential(
            ResBlockGenerator(hidden_size, hidden_size, stride=2),
            ResBlockGenerator(hidden_size, hidden_size, stride=2),
            ResBlockGenerator(hidden_size, hidden_size, stride=2),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            self.final,
            nn.Sigmoid())
            
        
    def forward(self, z):
        if self.name == "cifar10" or self.name == "cifar100":
            h = self.skip_size
            x = self.fc1(z[:,:h])
            x = self.bn1(x)
            x = F.leaky_relu(x, 0.2)
            x = self.fc2(torch.cat([x,z[:,h:2*h]],dim=1))
            x = self.bn2(x)
            x = F.leaky_relu(x, 0.2)
            x = self.fc3(torch.cat([x,z[:,2*h:3*h]],dim=1))
            x = self.bn3(x)
            x = F.leaky_relu(x, 0.2)
            x = self.fc4(torch.cat([x,z[:,3*h:4*h]],dim=1))
            x = torch.sigmoid(x)
            return x
        else: # imagenet
            output = self.model(self.dense(z).view(-1,self.hidden_size,2,2)).view(-1, self.out_size)
            return output

    def gen_noise(self, num):
        return torch.rand(num, self.in_size)

class Mine(nn.Module):
    def __init__(self, name, param):
        super().__init__()
        x_size      = param['in_size']
        y_size      = param['out_size']
        self.hidden_size = hidden_size = param['hidden_size']
        
        self.name = name
        self.fc1_x = nn.Linear(x_size, hidden_size, bias=False)
        self.fc1_y = nn.Linear(y_size, hidden_size, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # moving average
        self.ma_et = None
        self.ma_rate = 0.001
        self.conv = nn.Sequential(
                nn.Conv2d(3, hidden_size, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(),
                
                nn.Conv2d(hidden_size, 2* hidden_size, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_size * 2),
                nn.ReLU(),
                
                nn.Conv2d(hidden_size * 2, hidden_size, 4, 1, 0, bias=False),
            )
        self.fc1_y_after_conv = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.model = nn.Sequential(
#                 resNetG.FirstResBlockDiscriminator(3, hidden_size, stride=2),
#                 resNetG.ResBlockDiscriminator(hidden_size, hidden_size, stride=2),
#                 resNetG.ResBlockDiscriminator(hidden_size, hidden_size),
#                 resNetG.ResBlockDiscriminator(hidden_size, hidden_size),
#                 nn.ReLU(),
#                 nn.AvgPool2d(8),
#             )
        
    def forward(self, x, y):
        
        if self.name == "cifar10" or self.name == "cifar100":
            x = self.fc1_x(x)
            y = self.fc1_y(y)
            x = F.leaky_relu(x + y + self.fc1_bias, 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            x = F.leaky_relu(self.fc3(x), 0.2)
        else:            
            y = self.fc1_y_after_conv(self.model(y.view(-1,3,16,16)).view(-1,self.hidden_size))
            x = self.fc1_x(x)
            x = F.leaky_relu(x + y + self.fc1_bias, 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            x = F.leaky_relu(self.fc3(x), 0.2)
            
        return x

    def mi(self, x, x1, y):
        x = self.forward(x, y)
        x1 = self.forward(x1, y)
        return x.mean() - torch.log(torch.exp(x1).mean() + 1e-8)

    def mi_loss(self, x, x1, y):
        x = self.forward(x, y)
        x1 = self.forward(x1, y)
        et = torch.exp(x1).mean()
        if self.ma_et is None:
            self.ma_et = et.detach().item()
        self.ma_et += self.ma_rate * (et.detach().item() - self.ma_et)
        return x.mean() - torch.log(et + 1e-8) * et.detach() / self.ma_et

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        return self.fc(self.model(x).view(-1,DISC_SIZE))
