from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torch import nn 

from ...blocks.torch.qmobilenetv3_block import QMBConv, qblock_dict
from ...blocks.torch.mobilenetv3_block import _ConvBnAct
from .mobilenetv3 import MobileNetV3Net


class QMobileNetV3Net(MobileNetV3Net):

    def __init__(self, sample_str: str, hw=224, width_mult=1, num_classes=1000, block_dict=qblock_dict) -> None:
        super().__init__(sample_str, hw, width_mult, num_classes, block_dict)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == _ConvBnAct:
                modules_to_fuse = ['0', '1']
                if len(m) == 3 and type(m[2]) == nn.ReLU:
                    modules_to_fuse.append('2')
                    fuse_modules(m, modules_to_fuse, inplace=True)
            elif type(m) == QMBConv:
                m.fuse_model()

    def load_weights_from_ofa(self, ofa_model):
        ofa_state_dict = ofa_model.state_dict()
        my_keys = self.state_dict().keys()
        target_state_dict = {}
        for v, my_key in zip(ofa_state_dict.values(), my_keys):
            target_state_dict[my_key] = v

        self.load_state_dict(target_state_dict)

        return ofa_model