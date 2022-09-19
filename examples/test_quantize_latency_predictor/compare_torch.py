# SE & Hswish
import torch
from torch import nn
from .nas_models.common import make_divisible


class SE_NNMETER(nn.Module):
    def __init__(self, num_channels, se_ratio=0.25):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.conv1 = nn.Conv2d(num_channels, num_channels // 4, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(num_channels // 4, num_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.hswish = nn.Hardswish()

    def forward(self, inputs):
        x = self.avgpool(inputs)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hswish(x)
        return x * inputs


class SE_OFA(nn.Module):
    def __init__(self, num_channels, se_ratio=0.25):
        super().__init__()
        mid_channels = make_divisible(num_channels * se_ratio)
        self.squeeze = nn.Conv2d(num_channels, mid_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.excite = nn.Conv2d(mid_channels, num_channels, kernel_size=1, padding=0)
        self.hsigmoid = nn.Hardswish()

    def _scale(self, x):
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)
        x = self.squeeze(x)
        x = self.relu(x)
        x = self.excite(x)
        x = self.hsigmoid(x)
        return x

    def forward(self, x: torch.Tensor):
        scale = self._scale(x)
        return scale * x


Hswish_OFA = nn.Hardswish()


class Hsiwsh_NNMETER(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU(6)

    def forward(self, x):
        return x * self.relu6(x + 3.) * (1. / 6.)