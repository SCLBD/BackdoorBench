import torch.nn as nn
import torch.nn.functional as F


class SelfModel(nn.Module):
    def __init__(self, backbone, head="mlp", proj_dim=128):
        super(SelfModel, self).__init__()
        self.backbone = backbone
        self.head = head

        if head == "linear":
            self.proj_head = nn.Linear(self.backbone.feature_dim, proj_dim)
        elif head == "mlp":
            self.proj_head = nn.Sequential(
                nn.Linear(self.backbone.feature_dim, self.backbone.feature_dim),
                nn.BatchNorm1d(self.backbone.feature_dim),
                nn.ReLU(),
                nn.Linear(self.backbone.feature_dim, proj_dim),
            )
        else:
            raise ValueError("Invalid head {}".format(head))

    def forward(self, x):
        feature = self.proj_head(self.backbone(x))
        feature = F.normalize(feature, dim=1)

        return feature


class LinearModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super(LinearModel, self).__init__()
        self.backbone = backbone
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feature = self.backbone(x)
        out = self.linear(feature)

        return out

    def update_encoder(self, backbone):
        self.backbone = backbone
