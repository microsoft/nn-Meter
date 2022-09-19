from typing import Dict

import tensorflow as tf
from tensorflow.keras import layers

from ...search_space.proxylessnas_space import ProxylessNASSpace


class BasicBlock(tf.keras.Model):

    def __init__(self, hwin, cin) -> None:
        super().__init__()
        self.hwin = hwin
        self.cin = cin

    @property
    def config_str(self) -> str:
        return ProxylessNASSpace.config2str(self.config)

    def get_model_plus_input(self, batch_size=1) -> tf.keras.Model:
        x = tf.keras.Input([self.hwin, self.hwin, self.cin], batch_size)
        y = self(x)
        return tf.keras.Model(x, y)

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
        self.conv = layers.Conv2D(filters=cout, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)
        self.bn = layers.BatchNormalization()
        self.activation = layers.ReLU(6)
        self.block_name = name

    def call(self, x):
        x = self.conv(x)
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
        self.pool = layers.GlobalAveragePooling2D()
        self.linear = layers.Dense(cout)

    def call(self, x):
        x = self.pool(x)
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

        self.is_skip = strides == 1 and cin == cout
        self.block_name = name

        if self.expand_ratio > 1:
            self.inverted_bottleneck = tf.keras.Sequential([
                layers.Conv2D(filters=self.feature_size, kernel_size=1, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(6)
            ])
        else:
            self.inverted_bottleneck = None
        self.depth_conv = tf.keras.Sequential([
            layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(6)
        ])
        self.point_conv = tf.keras.Sequential([
            layers.Conv2D(filters=cout, kernel_size=1, padding='same', use_bias=False),
            layers.BatchNormalization()
        ])

    def call(self, x):
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
            hwio = (self.hwin, self.hwin // self.depth_conv.layers[0].strides[0]),
            cio = (self.cin, self.point_conv.layers[0].filters),
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