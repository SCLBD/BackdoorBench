import copy
import math
import torch
from torch import nn
from torch.nn.functional import relu
import torch.nn.functional as F
import numpy as np
from torch_utils import misc
from custom_modules import G_mapping, modulated_conv2d
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

class StegaStampEncoder(nn.Module):
    def __init__(
        self,
        resolution=32,
        IMAGE_CHANNELS=1,
        fingerprint_size=100,
        return_residual=0,
        bias_init=None,
        fused_modconv=1,
        demodulate=1,
        fc_layers=0
    ):
        super(StegaStampEncoder, self).__init__()
        if not fused_modconv:
            print('----------Not Using fused modconv!----------')
        else:
            print('----------Using fused modconv!----------')
        
        if return_residual: print("----------Defining the output of encoder as residual!----------")
        if not demodulate: print("----------Not using demodulation!----------")
        
        self.fingerprint_size = fingerprint_size 
        self.IMAGE_CHANNELS = IMAGE_CHANNELS 
        self.return_residual = return_residual
        self.secret_fixsize = 16 * 16 * IMAGE_CHANNELS  
        self.secret_dense = nn.Linear(self.fingerprint_size, self.secret_fixsize) 

        
        log_resolution = int(resolution // 16) 
        

        self.fc_layers=fc_layers
        if not self.fc_layers:
            print('----------Not Using FC layers!----------')
            self.secret_outsize = self.secret_fixsize
        else:
            print('----------Using FC layers!----------')
            self.secret_mapping = G_mapping(mapping_fmaps=fingerprint_size, dlatent_size=512) 
            self.secret_outsize = 512

        
        self.fingerprint_upsample = nn.Upsample(scale_factor=(log_resolution, log_resolution)) 
        self.conv1 = modulated_conv2d(2 * IMAGE_CHANNELS, 32, 3, 1, 1, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)
        self.conv2 = modulated_conv2d(32, 32, 3, 2, 1, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)
        self.conv3 = modulated_conv2d(32, 64, 3, 2, 1, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)
        self.conv4 = modulated_conv2d(64, 128, 3, 2, 1, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)
        self.conv5 = modulated_conv2d(128, 256, 3, 2, 1, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)
        self.pad6 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up6 = modulated_conv2d(256, 128, 2, 1, 0, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)
        self.upsample6 = nn.Upsample(scale_factor=(2, 2))
        self.conv6 = modulated_conv2d(128 + 128, 128, 3, 1, 1, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)
        self.pad7 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up7 = modulated_conv2d(128, 64, 2, 1, 0, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)
        self.upsample7 = nn.Upsample(scale_factor=(2, 2))
        self.conv7 = modulated_conv2d(64 + 64, 64, 3, 1, 1, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)
        self.pad8 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up8 = modulated_conv2d(64, 32, 2, 1, 0, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)
        self.upsample8 = nn.Upsample(scale_factor=(2, 2))
        self.conv8 = modulated_conv2d(32 + 32, 32, 3, 1, 1, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)
        self.pad9 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up9 = modulated_conv2d(32, 32, 2, 1, 0, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)
        self.upsample9 = nn.Upsample(scale_factor=(2, 2))
        self.conv9 = modulated_conv2d(32 + 32 + 2 * IMAGE_CHANNELS, 32, 3, 1, 1, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)
        self.conv10 = modulated_conv2d(32, 32, 3, 1, 1, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)
        
        self.residual = modulated_conv2d(32, IMAGE_CHANNELS, 1, 1, 0, self.secret_outsize, bias_init=bias_init, demodulate=demodulate, fused_modconv=fused_modconv)

    def forward(self, fingerprint, image):
        fp_mapping = relu(self.secret_dense(fingerprint)) 
        
        if self.fc_layers: 
            fp = self.secret_mapping(fingerprint) 
        else:
            fp = fp_mapping

        fingerprint = fp_mapping.view((-1, self.IMAGE_CHANNELS, 16, 16)) 
        fingerprint_enlarged = self.fingerprint_upsample(fingerprint) 
        inputs = torch.cat([fingerprint_enlarged, image], dim=1) 
        conv1 = relu(self.conv1(fp, inputs)) 
        conv2 = relu(self.conv2(fp, conv1)) 
        conv3 = relu(self.conv3(fp, conv2)) 
        conv4 = relu(self.conv4(fp, conv3)) 
        conv5 = relu(self.conv5(fp, conv4)) 

        up6 = relu(self.up6(fp, self.pad6(self.upsample6(conv5)))) 
        merge6 = torch.cat([conv4, up6], dim=1) 
        conv6 = relu(self.conv6(fp, merge6)) 

        up7 = relu(self.up7(fp, self.pad7(self.upsample7(conv6)))) 
        merge7 = torch.cat([conv3, up7], dim=1) 
        conv7 = relu(self.conv7(fp, merge7)) 

        up8 = relu(self.up8(fp, self.pad8(self.upsample8(conv7)))) 
        merge8 = torch.cat([conv2, up8], dim=1) 
        conv8 = relu(self.conv8(fp, merge8)) 

        up9 = relu(self.up9(fp, self.pad9(self.upsample9(conv8)))) 
        merge9 = torch.cat([conv1, up9, inputs], dim=1) 
        conv9 = relu(self.conv9(fp, merge9)) 

        conv10 = relu(self.conv10(fp, conv9)) 
        residual = self.residual(fp, conv10) 
        if not self.return_residual:
            residual = torch.sigmoid(residual)
        return residual 


class StegaStampDecoder(nn.Module):
    def __init__(self, resolution=32, IMAGE_CHANNELS=1, fingerprint_size=1):
        super(StegaStampDecoder, self).__init__()
        self.resolution = resolution 
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.decoder = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, (3, 3), 2, 1),  
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), 2, 1), 
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(resolution * resolution * 128 // 32 // 32, 512), 
            nn.ReLU(),
            nn.Linear(512, fingerprint_size),
        )

    def forward(self, image):
        x = self.decoder(image) 
        x = x.view(-1, self.resolution * self.resolution * 128 // 32 // 32) 
        return self.dense(x) 

