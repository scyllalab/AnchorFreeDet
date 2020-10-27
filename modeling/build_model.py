import torch
import torch.nn as nn
from modeling.backbone.hrnet import HRNet
from modeling.head.centernet import centernet
from modeling.head.centerloss import CenterLoss

class CenterNet(nn.Module):
    def __init__(self, cfg):
        super(CenterNet, self).__init__()
        self.backbone = HRNet()
        self.head = centernet(self.backbone.return_features_num_channels, cfg)
        self.loss = CenterLoss(cfg)

    def forward(self, x, target=None):
        x = self.backbone(x)
        out = self.head(x)
        
        if target is not None:
            loss = self.loss(out, target)
            return loss
        else:
            return out
    



