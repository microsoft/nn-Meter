# Quantized MobileNetV3 block

import torch
from torch import nn
from torch.quantization import fuse_modules

from .mobilenetv3_block import SE_for_run as SE, MBConv, FirstConv, Logits, FeatureMix, FinalExpand


class QSE(SE):

    def __init__(self, num_channels, se_ratio=0.25):
        super().__init__(num_channels, se_ratio)
        self.skip_mul = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.skip_mul.mul(self._scale(x), x)

    def fuse_model(self):
        fuse_modules(self, ['squeeze', 'relu'], inplace=True)


class QMBConv(MBConv):

    def __init__(self, hwin, cin, cout, kernel_size, expand_ratio, strides, act, se, name='mbconv', se_module=QSE) -> None:
        super().__init__(hwin, cin, cout, kernel_size, expand_ratio, strides, act, se, name, se_module)
        self.skip_add = nn.quantized.FloatFunctional()

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
            x = self.skip_add.add(x0, x)
        return x 

    def fuse_model(self):
        modules_to_fuse = ['0', '1'] + (['2'] if self.act == 'relu' else [])
        if self.expand_ratio > 1:
            fuse_modules(self.inverted_bottleneck, modules_to_fuse, inplace=True)
        fuse_modules(self.depth_conv, modules_to_fuse, inplace=True)
        fuse_modules(self.point_conv, modules_to_fuse[:2], inplace=True)
        if self.use_se:
            self.se.fuse_model()


class QFirstMBConv(QMBConv):
    def __init__(self, hwin, cin, cout) -> None:
        super().__init__(hwin, cin, cout, kernel_size=3, expand_ratio=1, strides=1, name='first_mbconv',
            act='relu', se=0)

    @classmethod
    def build_from_config(cls, config):
        hwin, hwout = config['hwio']
        cin, cout = config['cio']

        return cls(hwin=hwin, cin=cin, cout=cout)


qblock_dict = {
    'first_conv': FirstConv,
    'first_mbconv': QFirstMBConv,
    'mbconv': QMBConv,
    'feature_mix': FeatureMix,
    'final_expand': FinalExpand,
    'logits': Logits
}