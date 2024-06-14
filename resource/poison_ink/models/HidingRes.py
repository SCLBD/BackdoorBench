import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)


class HidingRes(nn.Module):
    def __init__(self, in_c=4, out_c=3, only_residual=False, requires_grad=True):
        super(HidingRes, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 128, 3, 1, 1, bias=False)  #(3,1,1) keep size
        self.norm1 = nn.InstanceNorm2d(128, affine=True)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        self.conv3 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)  #(3,2,1) half size

        self.res1 = ResidualBlock(128, dilation=2)  #(3,1,1,2,2) keep size
        self.res2 = ResidualBlock(128, dilation=2)
        self.res3 = ResidualBlock(128, dilation=2)
        self.res4 = ResidualBlock(128, dilation=4)  #(3,1,1,4,4) keep size
        self.res5 = ResidualBlock(128, dilation=4)
        self.res6 = ResidualBlock(128, dilation=4)
        self.res7 = ResidualBlock(128, dilation=1) #(3,1,1,1,1) keep size  #kernel:3 pad=dilation,stride=1

        self.deconv3 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        #output = (input-1)*stride + outputpadding(default 0) -2*padding + kernel size
        self.norm4 = nn.InstanceNorm2d(128, affine=True)
        self.deconv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.norm5 = nn.InstanceNorm2d(128, affine=True)
        self.deconv1 = nn.Conv2d(128, out_c, 1)
        self.only_residual = only_residual

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y = F.relu(self.norm3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)

        y = F.relu(self.norm4(self.deconv3(y)))
        y = F.relu(self.norm5(self.deconv2(y)))
        if self.only_residual:
            y = self.deconv1(y)
        else:
            y = F.relu(self.deconv1(y))

        return y

class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)


class HidingRes_down4(nn.Module):
    def __init__(self, in_c=4, out_c=3, only_residual=False, requires_grad=True):
        super(HidingRes_down4, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 128, 3, 1, 1, bias=False)  #(3,1,1) keep size
        self.norm1 = nn.InstanceNorm2d(128, affine=True)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        self.conv3 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        self.norm4 = nn.InstanceNorm2d(128, affine=True)

        self.res1 = ResidualBlock(128, dilation=2)  #(3,1,1,2,2) keep size
        self.res2 = ResidualBlock(128, dilation=2)
        self.res3 = ResidualBlock(128, dilation=2)
        self.res4 = ResidualBlock(128, dilation=2)
        self.res5 = ResidualBlock(128, dilation=4)  #(3,1,1,4,4) keep size
        self.res6 = ResidualBlock(128, dilation=4)
        self.res7 = ResidualBlock(128, dilation=4)
        self.res8 = ResidualBlock(128, dilation=4)
        self.res9 = ResidualBlock(128, dilation=1) #(3,1,1,1,1) keep size  #kernel:3 pad=dilation,stride=1

        self.deconv4 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.norm5 = nn.InstanceNorm2d(128, affine=True)
        self.deconv3 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.norm6 = nn.InstanceNorm2d(128, affine=True)
        self.deconv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.norm7 = nn.InstanceNorm2d(128, affine=True)
        self.deconv1 = nn.Conv2d(128, out_c, 1)
        self.only_residual = only_residual

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y = F.relu(self.norm3(self.conv3(y)))
        y = F.relu(self.norm4(self.conv4(y)))

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)
        y = self.res8(y)
        y = self.res9(y)

        y = F.relu(self.norm5(self.deconv4(y)))
        y = F.relu(self.norm6(self.deconv3(y)))
        y = F.relu(self.norm7(self.deconv2(y)))
        if self.only_residual:
            y = self.deconv1(y)
        else:
            y = F.relu(self.deconv1(y))

        return y

class HidingRes11(nn.Module):
    def __init__(self, in_c=4, out_c=3, only_residual=False):
        super(HidingRes11, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 128, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(128, affine=True)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        self.conv3 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)

        self.res1 = ResidualBlock(128, dilation=2)
        self.res2 = ResidualBlock(128, dilation=2)
        self.res3 = ResidualBlock(128, dilation=2)
        self.res4 = ResidualBlock(128, dilation=2)
        self.res5 = ResidualBlock(128, dilation=2)
        self.res6 = ResidualBlock(128, dilation=4)
        self.res7 = ResidualBlock(128, dilation=4)
        self.res8 = ResidualBlock(128, dilation=4)
        self.res9 = ResidualBlock(128, dilation=4)
        self.res10 = ResidualBlock(128, dilation=4)
        self.res11 = ResidualBlock(128, dilation=1)

        self.deconv3 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.norm4 = nn.InstanceNorm2d(128, affine=True)
        self.deconv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.norm5 = nn.InstanceNorm2d(128, affine=True)
        self.deconv1 = nn.Conv2d(128, out_c, 1)
        self.only_residual = only_residual

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y = F.relu(self.norm3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)
        y = self.res8(y)
        y = self.res9(y)
        y = self.res10(y)
        y = self.res11(y)



        y = F.relu(self.norm4(self.deconv3(y)))
        y = F.relu(self.norm5(self.deconv2(y)))
        if self.only_residual:
            y = self.deconv1(y)
        else:
            y = F.relu(self.deconv1(y))

        return y

class HidingRes_S(nn.Module):
    def __init__(self, in_c=4, out_c=3, only_residual=False, requires_grad=True):
        super(HidingRes_S, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 128, 3, 1, 1, bias=False)  #(3,1,1) keep size
        self.norm1 = nn.InstanceNorm2d(128, affine=True)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        self.conv3 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)  #(3,2,1) half size

        self.res1 = ResidualBlock(128, dilation=2)  #(3,1,1,2,2) keep size
        self.res2 = ResidualBlock(128, dilation=2)
        self.res3 = ResidualBlock(128, dilation=2)
        self.res4 = ResidualBlock(128, dilation=4)  #(3,1,1,4,4) keep size
        self.res5 = ResidualBlock(128, dilation=4)
        self.res6 = ResidualBlock(128, dilation=4)
        self.res7 = ResidualBlock(128, dilation=1) #(3,1,1,1,1) keep size  #kernel:3 pad=dilation,stride=1

        self.deconv3 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        #output = (input-1)*stride + outputpadding(default 0) -2*padding + kernel size
        self.norm4 = nn.InstanceNorm2d(128, affine=True)
        self.deconv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.norm5 = nn.InstanceNorm2d(128, affine=True)
        self.deconv1 = nn.Conv2d(128, out_c, 1)
        self.only_residual = only_residual

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y = F.relu(self.norm3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)

        y = F.relu(self.norm4(self.deconv3(y)))
        y = F.relu(self.norm5(self.deconv2(y)))
        if self.only_residual:
            y = self.deconv1(y)
        else:
            y = F.sigmoid(self.deconv1(y))

        return y
