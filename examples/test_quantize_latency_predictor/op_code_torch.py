from turtle import forward
import torch
from torch import nn

from nas_models.networks.torch.mobilenetv3 import MobileNetV3Net
from nas_models.blocks.torch.mobilenetv3_block import SE
from nas_models.common import make_divisible


class Hswish(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU(6)

    def forward(self, x):
        return x * self.relu6(x + 3.) * (1. / 6.)


class SE_xudong(nn.Module):

    def __init__(self, num_channels, se_ratio=0.25):
        super().__init__()
        mid_channels = make_divisible(num_channels * se_ratio)
        self.squeeze = nn.Conv2d(num_channels, mid_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.excite = nn.Conv2d(mid_channels, num_channels, kernel_size=1, padding=0)
        self.hswish = Hswish() # ONLY FOR PARSE!!!!

    def _scale(self, x):
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)
        x = self.squeeze(x)
        x = self.relu(x)
        x = self.excite(x)
        x = self.hswish(x)
        return x

    def forward(self, x: torch.Tensor):
        scale = self._scale(x)
        return scale * x
