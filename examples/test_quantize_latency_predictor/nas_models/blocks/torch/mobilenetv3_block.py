from typing import Dict, List

import torch
from torch import nn

from ...search_space.mobilenetv3_space import MobileNetV3Space
from ...common import make_divisible


def build_act(act: str):
    if act == 'relu':
        return nn.ReLU()
    if act == 'h_swish':
        return nn.Hardswish()


class SE(nn.Module):

    def __init__(self, num_channels, se_ratio=0.25):
        super().__init__()
        mid_channels = make_divisible(num_channels * se_ratio)
        self.squeeze = nn.Conv2d(num_channels, mid_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.excite = nn.Conv2d(mid_channels, num_channels, kernel_size=1, padding=0)
        self.hswish = nn.Hardswish()

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


class BasicBlock(nn.Module):

    def __init__(self, hwin, cin) -> None:
        super().__init__()
        self.hwin = hwin
        self.cin = cin

    @property
    def config_str(self) -> str:
        return MobileNetV3Space.config2str(self.config)

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
        elif name == 'final_expand':
            return FinalExpand.build_from_config(config)
        elif name == 'logits':
            return Logits.build_from_config(config)
        else:
            raise ValueError(f'{name} not recognized.')


class _ConvBnAct(BasicBlock):
   
    def __init__(self, hwin, cin, cout, kernel_size, strides, name='convbnact', act='relu', use_bn=True) -> None:
        super().__init__(hwin, cin)
        self.conv = nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=strides, padding=strides//2, bias=False)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(cout)
        self.activation = build_act(act)
        self.block_name = name
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        return x

    @classmethod
    def build_from_config(cls, config):
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
            act = 'relu',
            se = 0
        )
    

class FirstConv(_ConvBnAct):

    def __init__(self, hwin, cin, cout) -> None:
        super().__init__(hwin, cin, cout, kernel_size=3, strides=2, name='first_conv', act='relu')


class FinalExpand(_ConvBnAct):

    def __init__(self, hwin, cin, cout) -> None:
        super().__init__(hwin, cin, cout, kernel_size=1, strides=1, name='final_expand', act='relu')


class FeatureMix(_ConvBnAct):

    def __init__(self, hwin, cin, cout) -> None:
        super().__init__(hwin, cin, cout, kernel_size=1, strides=1, name='feature_mix', act='relu', use_bn=False)

    def forward(self, x: torch.Tensor):
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)
        x = super().forward(x)
        return x


class Logits(BasicBlock):

    def __init__(self, hwin, cin, cout) -> None:
        super().__init__(hwin, cin)
        self.linear = nn.Linear(cin, cout)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

    @property
    def config(self):
        return dict(
            name = 'logits',
            hwio = (self.hwin, 1),
            cio = (self.cin, self.linear.units),
            k = 0,
            e = 0,
            act = 'relu',
            se = 0,
        )

    @classmethod
    def build_from_config(cls, config):
        hwin, _ = config['hwio']
        cin, cout = config['cio']
        return cls(hwin=hwin, cin=cin, cout=cout)


class MBConv(BasicBlock):

    def __init__(self, hwin, cin, cout, kernel_size, expand_ratio, strides, act, se, name='mbconv', se_module=SE) -> None:
        super().__init__(hwin=hwin, cin=cin)
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.feature_size = round(cin * expand_ratio)
        self.act = act
        self.use_se = se

        self.is_skip = strides == 1 and cin == cout
        self.block_name = name

        if self.expand_ratio > 1:
            self.inverted_bottleneck = nn.Sequential(
                nn.Conv2d(cin, self.feature_size, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(self.feature_size),
                build_act(self.act)
            )
        else:
            self.inverted_bottleneck = None
        self.depth_conv = nn.Sequential(
            nn.Conv2d(self.feature_size, self.feature_size, kernel_size=kernel_size, 
                stride=strides, padding=kernel_size//2, bias=False, groups=self.feature_size),
            nn.BatchNorm2d(self.feature_size),
            build_act(self.act)
        )

        if self.use_se:
            self.se = se_module(self.feature_size)

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
        if self.use_se:
            x = self.se(x)
        x = self.point_conv(x)

        if self.is_skip:
            x = x + x0
        return x 

    @property
    def config(self):
        return dict(
            name = self.block_name,
            hwio = (self.hwin, self.hwin // self.depth_conv.layers[0].strides[0]),
            cio = (self.cin, self.point_conv.layers[0].filters),
            k = self.kernel_size,
            e = self.expand_ratio,
            act = self.act,
            se = self.use_se
        )

    @classmethod
    def build_from_config(cls, config: Dict):
        hwin, hwout = config['hwio']
        cin, cout = config['cio']

        return cls(hwin=hwin, cin=cin, cout=cout, kernel_size=config['k'], 
            expand_ratio=config['e'], strides=hwin//hwout, act=config['act'], se=config['se'])


class FirstMBConv(MBConv):
    def __init__(self, hwin, cin, cout) -> None:
        super().__init__(hwin, cin, cout, kernel_size=3, expand_ratio=1, strides=1, name='first_mbconv',
            act='relu', se=0)

    @classmethod
    def build_from_config(cls, config: Dict):
        hwin, hwout = config['hwio']
        cin, cout = config['cio']

        return cls(hwin=hwin, cin=cin, cout=cout)


block_dict =  {
    'first_conv': FirstConv,
    'first_mbconv': FirstMBConv,
    'mbconv': MBConv,
    'feature_mix': FeatureMix,
    'final_expand': FinalExpand,
    'logits': Logits
}