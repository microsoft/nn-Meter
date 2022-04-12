from typing import Dict

import tensorflow as tf

from ...blocks.tf.mobilenetv3_block import FirstConv, FirstMBConv, MBConv, FinalExpand, FeatureMix, Logits
from ...search_space.mobilenetv3_space import MobileNetV3Space
from ...common import parse_sample_str


class MobileNetV3Net(tf.keras.Model):

    def __init__(self, sample_str: str, hw=224, width_mult=1.0, num_classes=1000) -> None:
        super().__init__()
        self.space = MobileNetV3Space(width_mult=width_mult, num_classes=num_classes, hw=hw)
        self.sample_str = sample_str
        self.hw = hw

        self.first_conv = FirstConv(hwin=hw, cin=3, cout=self.space.stage_width[0])
        self.first_mbconv = FirstMBConv(
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
                blocks.append(MBConv(hwin, cin, cout, kernel_size=k, expand_ratio=e, strides=strides,
                    act=act, se=int(se)))
                cin = cout 
                hwin //= strides
            block_idx += max_depth
        self.blocks = tf.keras.Sequential(blocks)

        self.final_expand = FinalExpand.build_from_config(self.space.block_configs[-3])
        self.feature_mix = FeatureMix.build_from_config(self.space.block_configs[-2])
        self.logits = Logits.build_from_config(self.space.block_configs[-1])

    def call(self, x):
        x = self.first_conv(x)
        x = self.first_mbconv(x)
        x = self.blocks(x)
        x = self.final_expand(x)
        x = self.feature_mix(x)
        x = self.logits(x)
        return x

    def get_model_plus_input(self, batch_size=1) -> tf.keras.Model:
        x = tf.keras.Input([self.hw, self.hw, 3], batch_size)
        y = self(x)
        return tf.keras.Model(x, y)

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