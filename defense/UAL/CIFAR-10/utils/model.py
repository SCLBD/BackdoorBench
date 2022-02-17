import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.stn import STN

class CNN_classifier(nn.Module):
    """ MNIST Encoder from Original Paper's Keras based Implementation.
        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
            embedding_dim (int): embedding dimensionality
    """
    def __init__(self, init_num_filters=32, lrelu_slope=0.2, inter_fc_dim=128, nofclasses=10,nofchannels=3,use_stn=True):
        super(CNN_classifier, self).__init__()
        self.use_stn=use_stn
        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.inter_fc_dim_ = inter_fc_dim
        self.nofclasses_ = nofclasses
        if self.use_stn:
            self.stn = STN()

        self.features = nn.Sequential(
            nn.Conv2d(nofchannels, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.MaxPool2d(2,2),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.init_num_filters_ *4*4, self.inter_fc_dim_),
            nn.BatchNorm1d(self.inter_fc_dim_),
            nn.ReLU(True),
            nn.Dropout(p=.2),

            nn.Linear(self.inter_fc_dim_, int(self.inter_fc_dim_/2)),
            nn.BatchNorm1d(int(self.inter_fc_dim_/2)),
            nn.ReLU(True),
            nn.Dropout(p=.2),

            nn.Linear(int(self.inter_fc_dim_/2), self.nofclasses_)
        )

    def forward(self, x):
        if self.use_stn:
            x = self.stn(x)
        x = self.features(x)
#         print(x.shape)
        x = x.view(-1, self.init_num_filters_ *4*4)
        x = self.fc(x)
        return x
