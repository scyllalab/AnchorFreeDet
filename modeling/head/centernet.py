import torch
import torch.nn as nn
import sys
sys.path[0] = '/workspace/mnt/storage/wangerwei/wew_filestroage/Code/Detection/AnchorFreeDet'
from layers.base import Conv_bn_relu
from modeling.backbone.resnest import SplAtConv2d



class centernet(nn.Module):
    def __init__(self, in_channels, cfg):
        super(centernet, self).__init__()
        self.num_class = cfg.MODEL.NUMCLASS
        self.head_net = nn.Sequential(
            Conv_bn_relu(in_channels, in_channels*2),
            Conv_bn_relu(in_channels*2, in_channels, kernel_size=1, use_relu=False))
        self.pre_cls = SplAtConv2d(in_channels, self.num_class, 3, norm_layer=nn.BatchNorm2d, bias= False)
        self.pre_offset = SplAtConv2d(in_channels, 2, 3, norm_layer=nn.BatchNorm2d, bias= False)
        self.pre_wh = SplAtConv2d(in_channels, 2, 3, norm_layer=nn.BatchNorm2d, bias= False)

    def forward(self, x):
        x = self.head_net(x)
        heatmap = self.pre_cls(x)
        offset = self.pre_offset(x)
        wh = self.pre_wh(x)

        return [heatmap, offset, wh]



if __name__ == "__main__":
    model = centernet(in_channels=144)
    print(model)
    data = torch.randn(8, 144, 128, 128)

    out = model(data)
    





