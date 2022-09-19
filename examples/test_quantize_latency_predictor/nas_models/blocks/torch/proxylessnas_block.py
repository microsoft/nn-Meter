from turtle import forward
from typing import Dict, List
import torch
from torch import nn

from ...search_space.proxylessnas_space import ProxylessNASSpace


class BasicBlock(nn.Module):

    def __init__(self, hwin, cin) -> None:
        super().__init__()
        self.hwin = hwin
        self.cin = cin

    @property
    def config_str(self) -> str:
        return ProxylessNASSpace.config2str(self.config)

    def input_shape(self, batch_size=1) -> List:
        return [batch_size, self.cin, self.hwin, self.hwin]

    @property
    def config(self) -> dict:
        raise NotImplementedError()

    @classmethod
    def build_from_config(cls, config: Dict):
        name = config['name']
        if name == 'first_conv':
            return FirstConv.build_from_config(config)
        elif name == 'first_mbconv':
            return FirstMBConv.build_from_config(config)
        elif name == 'mbconv':
            return MBConv.build_from_config(config)
        elif name == 'feature_mix':
            return FeatureMix.build_from_config(config)
        elif name == 'logits':
            return Logits.build_from_config(config)
        else:
            raise ValueError(f'{name} not recognized.')

        

class _ConvBnRelu(BasicBlock):
    
    def __init__(self, hwin, cin, cout, kernel_size, strides, name='convbnrelu') -> None:
        super().__init__(hwin, cin)
        self.conv = nn.Conv2d(
            in_channels=cin, out_channels=cout, 
            kernel_size=kernel_size, stride=strides, 
            padding=kernel_size//2, bias=False
        )
        self.bn = nn.BatchNorm2d(num_features=cout)
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

    # only called by FirstConv and FeatureMix
    @classmethod
    def build_from_config(cls, config: Dict) -> nn.Module:
        hwin, _ = config['hwio']
        cin, cout = config['cio']
        return cls(hwin=hwin, cin=cin, cout=cout)

    @property
    def config(self):
        return dict(
            name = self.block_name,
            hwio = (self.hwin, self.hwin // self.conv.strides[0]),
            cio = (self.cin, self.conv.filters),
            k = 0,
            e = 0,
        )


class FirstConv(_ConvBnRelu):

    def __init__(self, hwin, cin, cout) -> None:
        super().__init__(hwin, cin, cout, kernel_size=3, strides=2, name='first_conv')


class FeatureMix(_ConvBnRelu):

    def __init__(self, hwin, cin, cout) -> None:
        super().__init__(hwin, cin, cout, kernel_size=1, strides=1, name='feature_mix')


class Logits(BasicBlock):

    def __init__(self, hwin, cin, cout) -> None:
        super().__init__(hwin, cin)
        self.linear = nn.Linear(cin, cout)

    def forward(self, x):
        x = x.mean([2, 3], keepdim=False)
        x = self.linear(x)
        return x

    @property
    def config(self):
        return dict(
            name = 'logits',
            hwio = (self.hwin, 1),
            cio = (self.cin, self.linear.units),
            k = 0,
            e = 0
        )

    @classmethod
    def build_from_config(cls, config):
        hwin, _ = config['hwio']
        cin, cout = config['cio']
        return cls(hwin=hwin, cin=cin, cout=cout)


class MBConv(BasicBlock):

    def __init__(self, hwin, cin, cout, kernel_size, expand_ratio, strides, name='mbconv') -> None:
        super().__init__(hwin=hwin, cin=cin)
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.feature_size = round(cin * expand_ratio)
        self.strides = strides
        self.is_skip = strides == 1 and cin == cout
        self.block_name = name
        self.cout = cout

        if self.expand_ratio > 1:
            self.inverted_bottleneck = nn.Sequential(
                nn.Conv2d(cin, self.feature_size, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(self.feature_size),
                nn.ReLU6(inplace=True)
            )
        else:
            self.inverted_bottleneck = None
        self.depth_conv = nn.Sequential(
            nn.Conv2d(self.feature_size, self.feature_size, kernel_size=kernel_size, 
                stride=strides, padding=kernel_size//2, bias=False, groups=self.feature_size),
            nn.BatchNorm2d(self.feature_size),
            nn.ReLU6(inplace=True)
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(self.feature_size, cout, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        if self.is_skip:
            x0 = x
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)

        if self.is_skip:
            x = x + x0
        return x 

    @property
    def config(self):
        return dict(
            name = self.block_name,
            hwio = (self.hwin, self.hwin // self.strides),
            cio = (self.cin, self.cout),
            k = self.kernel_size,
            e = self.expand_ratio
        )

    @classmethod
    def build_from_config(cls, config: Dict):
        hwin, hwout = config['hwio']
        cin, cout = config['cio']

        return cls(hwin=hwin, cin=cin, cout=cout, kernel_size=config['k'], 
            expand_ratio=config['e'], strides=hwin//hwout)


class FirstMBConv(MBConv):
    def __init__(self, hwin, cin, cout) -> None:
        super().__init__(hwin, cin, cout, kernel_size=3, expand_ratio=1, strides=1, name='first_mbconv')

    @classmethod
    def build_from_config(cls, config: Dict):
        hwin, hwout = config['hwio']
        cin, cout = config['cio']

        return cls(hwin=hwin, cin=cin, cout=cout)