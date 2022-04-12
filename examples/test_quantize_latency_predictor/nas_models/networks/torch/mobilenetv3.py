from typing import Dict

from torch import nn

from ...blocks.torch.mobilenetv3_block import block_dict
from ...search_space.mobilenetv3_space import MobileNetV3Space
from ...common import parse_sample_str


class MobileNetV3Net(nn.Module):

    def __init__(self, sample_str: str, hw=224, width_mult=1.0, num_classes=1000, block_dict=block_dict) -> None:
        super().__init__()
        self.space = MobileNetV3Space(width_mult=width_mult, num_classes=num_classes, hw=hw)
        self.sample_str = sample_str
        self.hw = hw

        self.first_conv = block_dict['first_conv'](hwin=hw, cin=3, cout=self.space.stage_width[0])
        self.first_mbconv = block_dict['first_mbconv'](
            hwin=hw//2,
            cin=self.space.stage_width[0],
            cout=self.space.stage_width[1]
        )

        hwin = hw // 2
        cin = self.space.stage_width[1]
        blocks = []
        block_idx = 0
        for strides, cout, max_depth, depth, act, se in zip(
            self.space.stride_stages[1:], self.space.stage_width[2:], 
            self.space.num_block_stages[1:], self.sample_config['d'],
            self.space.act_stages[1:], self.space.se_stages[1:]
        ):
            for i in range(depth):
                k = self.sample_config['ks'][block_idx + i]
                e = self.sample_config['e'][block_idx + i]
                strides = 1 if i > 0 else strides
                blocks.append(block_dict['mbconv'](hwin, cin, cout, kernel_size=k, expand_ratio=e, strides=strides,
                    act=act, se=int(se)))
                cin = cout 
                hwin //= strides
            block_idx += max_depth
        self.blocks = nn.Sequential(*blocks)

        self.final_expand = block_dict['final_expand'].build_from_config(self.space.block_configs[-3])
        self.feature_mix = block_dict['feature_mix'].build_from_config(self.space.block_configs[-2])
        self.logits = block_dict['logits'].build_from_config(self.space.block_configs[-1])

    def _forward_impl(self, x):
        x = self.first_conv(x)
        x = self.first_mbconv(x)
        x = self.blocks(x)
        x = self.final_expand(x)
        x = self.feature_mix(x)
        x = self.logits(x)
        return x
    
    def forward(self, x):
        return self._forward_impl(x)

    def input_shape(self, batch_size=1):
        return [batch_size, 3, self.hw, self.hw]

    @property
    def model_config_str_list(self):
        rv = []
        rv.append(self.first_conv.config_str)
        rv.append(self.first_mbconv.config_str)
        for block in self.blocks.layers:
            rv.append(block.config_str)
        rv.append(self.final_expand.config_str)
        rv.append(self.feature_mix.config_str)
        rv.append(self.logits.config_str)
        return rv

    @property
    def sample_config(self) -> Dict:
        try:
            return self._sample_config
        except:
            self._sample_config = parse_sample_str(self.sample_str)
            return self._sample_config