from typing import Dict, List
from numpy import block

from torch import nn

from ...blocks.torch.resnet_block import InputStem, BConv, Logits
from ...search_space.resnet_space import ResNetSpace
from ...common import parse_sample_str


class ResNetNet(nn.Module):

    def __init__(self, sample_str: str, hw=224, num_classes=1000):
        super().__init__()
        self.space = ResNetSpace(num_classes, hw)
        self.sample_str = sample_str
        self.hw = hw

        input_stem_w0, input_stem_w1, *bconv_w_list  = self.w_list
        input_stem_d, *bconv_d_list = self.d_list

        # add input_stem
        input_stem_skipping = input_stem_d != max(self.space.depth_list)
        midc = self.space.mid_input_channel[input_stem_w0]
        cin = self.space.input_channel[input_stem_w1]
        self.input_stem = InputStem(hw, cin, midc, input_stem_skipping)

        # add bottleneck blocks
        block_idx = 0
        blocks = []
        hwin = hw // 4
        for w_idx, stage_width_list, d, base_depth, strides in zip(bconv_w_list, self.space.stage_width_list,
            bconv_d_list, ResNetSpace.BASE_DEPTH_LIST, self.space.stride_list):
            width = stage_width_list[w_idx]
            for i in range(base_depth + d):
                s = 1 if i > 0 else strides
                expand_ratio = self.e_list[block_idx + i]
                blocks.append(BConv(hwin, cin, width, expand_ratio=expand_ratio, strides=s))
                hwin //= s
                cin = width
            block_idx += base_depth + max(self.space.depth_list)
        self.blocks = nn.Sequential(*blocks)

        # add classifier
        self.logits = Logits(hwin, cin, num_classes)

    def forward(self, x):
        x = self.input_stem(x)
        x = self.blocks(x)
        x = self.logits(x)
        return x

    @property
    def sample_config(self) -> Dict:
        try:
            return self._sample_config
        except:
            self._sample_config = parse_sample_str(self.sample_str)
            return self._sample_config

    def input_shape(self, batch_size=1):
        return [batch_size, 3, self.hw, self.hw]

    @property
    def model_config_str_list(self):
        rv = []
        rv.append(self.input_stem.config_str)
        for block in self.blocks.layers:
            rv.append(block.config_str)
        rv.append(self.logits.config_str)
        return rv

    @property
    def d_list(self) -> List:
        return self.sample_config['d']
    
    @property
    def e_list(self) -> List:
        return self.sample_config['e']
    
    @property
    def w_list(self) -> List:
        return self.sample_config['w']