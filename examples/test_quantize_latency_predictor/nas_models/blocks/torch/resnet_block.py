from typing import Dict, List

import torch
from torch import nn 

from ...common import make_divisible
from ...search_space.resnet_space import ResNetSpace


class BasicBlock(nn.Module):

    def __init__(self, hwin, cin) -> None:
        super().__init__()
        self.hwin = hwin
        self.cin = cin

    @property
    def config_str(self) -> str:
        return ResNetSpace.config2str(self.config)

    def input_shape(self, batch_size=1) -> List:
        return [batch_size, self.cin, self.hwin, self.hwin]

    @property
    def config(self) -> dict:
        raise NotImplementedError()

    @classmethod
    def build_from_config(cls, config: Dict):
        name = config['name']
        if name == 'input_stem':
            return InputStem.build_from_config(config)
        if name == 'bconv':
            return BConv.build_from_config(config)
        if name == 'logits':
            return Logits.build_from_config(config)
        raise ValueError(f'{name} not recognized.')


class _ConvBnRelu(nn.Module):
    
    def __init__(self, cin, cout, kernel_size, strides) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=cin, out_channels=cout, 
            kernel_size=kernel_size, stride=strides, 
            padding=kernel_size//2, bias=False
        )
        self.bn = nn.BatchNorm2d(num_features=cout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class InputStem(BasicBlock):
    
    def __init__(self, hwin, cout, midc, skipping=False):
        super().__init__(hwin, 3)
        self.skipping = skipping
        self.conv0 = _ConvBnRelu(3, midc, kernel_size=3, strides=2)
        if not self.skipping:
            self.conv1 = _ConvBnRelu(midc, midc, kernel_size=3, strides=1)
        self.conv2 = _ConvBnRelu(midc, cout, kernel_size=3, strides=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        x = self.conv0(x)
        if not self.skipping:
            x = self.conv1(x) + x
        x = self.conv2(x)
        x = self.pool(x)
        return x

    @property
    def config(self):
        return dict(
            name = 'input_stem', 
            hwio = (self.hwin, self.hwin // 4),
            cio = (3, self.conv2.conv.filters),
            e = 0,
            midc = self.conv0.conv.filters,
            skipping = int(self.skipping)
        )

    @classmethod
    def build_from_config(cls, config):
        hwin, hwout = config['hwio']
        cin, cout = config['cio']
        return cls(hwin=hwin, cout=cout, midc=config['midc'], skipping=config['skipping'])


class BConv(BasicBlock):

    def __init__(self, hwin, cin, cout, expand_ratio, strides, k=3):
        super().__init__(hwin, cin)
        self.cout = cout
        self.expand_ratio = expand_ratio
        self.feature_size = make_divisible(cout * self.expand_ratio)
        self.strides = strides
        self.kernel_size =  k

        self.conv0 = _ConvBnRelu(cin, self.feature_size, kernel_size=1, strides=1)
        self.conv1 = _ConvBnRelu(self.feature_size, self.feature_size, kernel_size=self.kernel_size, strides=strides)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.feature_size, self.cout, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(cout)
        
        )

        if strides == 1 and cin == cout:
            self.down_sample = nn.Identity()
        else:
            self.down_sample = nn.Sequential(
                nn.AvgPool2d(kernel_size=strides, stride=strides, padding=0, ceil_mode=True),
                nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(cout)
            )

        self.final_act = nn.ReLU()

    def forward(self, x):
        residual = self.down_sample(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final_act(x + residual)
        return x

    @property
    def config(self):
        return dict(
            name = 'bconv',
            hwio = (self.hwin, self.hwin // self.strides),
            cio = (self.cin, self.cout),
            e = self.expand_ratio,
            midc = 0
        )
    
    @classmethod
    def build_from_config(cls, config):
        hwin, hwout = config['hwio']
        cin, cout = config['cio']
        strides = hwin // hwout

        return cls(hwin=hwin, cin=cin, cout=cout, expand_ratio=config['e'], strides=strides)


class Logits(BasicBlock):

    def __init__(self, hwin, cin, cout) -> None:
        super().__init__(hwin, cin)
        self.linear = nn.Linear(cin, cout)

    def forward(self, x):
        x = x.mean([2, 3])
        x = self.linear(x)
        return x

    @property
    def config(self):
        return dict(
            name = 'logits',
            hwio = (self.hwin, 1),
            cio = (self.cin, self.linear.units),
            e = 0,
            midc = 0
        )

    @classmethod
    def build_from_config(cls, config):
        hwin, _ = config['hwio']
        cin, cout = config['cio']
        return cls(hwin=hwin, cin=cin, cout=cout)