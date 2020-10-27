import torch.nn as nn

class Conv_bn_relu(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 kernel_size=3,
                 stride=1,
                 pad=1,
                 use_relu=True,
                 relu_name='ReLU'):
        super(Conv_bn_relu, self).__init__()
        self.use_relu = use_relu
        if self.use_relu and relu_name == 'ReLU':
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=False),
            )
        elif self.use_relu and relu_name == 'LeakyReLU':
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
                nn.LeakyReLU(inplace=False),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.convs(x)
        return out