from typing import Dict

import tensorflow as tf
from tensorflow.keras import layers

from ...common import make_divisible
from ...search_space.resnet_space import ResNetSpace


class _Identity(tf.keras.Model):

    def __init__(self):
        super().__init__()

    def call(self, x):
        return x


class BasicBlock(tf.keras.Model):

    def __init__(self, hwin, cin) -> None:
        super().__init__()
        self.hwin = hwin
        self.cin = cin

    @property
    def config_str(self) -> str:
        return ResNetSpace.config2str(self.config)

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
        if name == 'input_stem':
            return InputStem.build_from_config(config)
        if name == 'bconv':
            return BConv.build_from_config(config)
        if name == 'logits':
            return Logits.build_from_config(config)
        raise ValueError(f'{name} not recognized.')


class _ConvBnRelu(tf.keras.Model):
   
    def __init__(self, cout, kernel_size, strides) -> None:
        super().__init__()
        self.conv = layers.Conv2D(filters=cout, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)
        self.bn = layers.BatchNormalization()
        self.activation = layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class InputStem(BasicBlock):
    
    def __init__(self, hwin, cout, midc, skipping=False):
        super().__init__(hwin, 3)
        self.skipping = skipping
        self.conv0 = _ConvBnRelu(midc, kernel_size=3, strides=2)
        if not self.skipping:
            self.conv1 = _ConvBnRelu(midc, kernel_size=3, strides=1)
        self.conv2 = _ConvBnRelu(cout, kernel_size=3, strides=1)
        self.pool = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')

    def call(self, x):
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

        self.conv0 = _ConvBnRelu(self.feature_size, kernel_size=1, strides=1)
        self.conv1 = _ConvBnRelu(self.feature_size, kernel_size=self.kernel_size, strides=strides)
        self.conv2 = tf.keras.Sequential([
            layers.Conv2D(cout, kernel_size=1, padding='same', use_bias=False),
            layers.BatchNormalization(),
        ])

        if strides == 1 and cin == cout:
            self.down_sample = _Identity()
        else:
            self.down_sample = tf.keras.Sequential([
                layers.AveragePooling2D(pool_size=strides, strides=strides, padding='same'),
                layers.Conv2D(cout, kernel_size=1, strides=1, padding='same', use_bias=False),
                layers.BatchNormalization()
            ])

        self.final_act = layers.ReLU()

    def call(self, x):
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
            e = 0,
            midc = 0
        )

    @classmethod
    def build_from_config(cls, config):
        hwin, _ = config['hwio']
        cin, cout = config['cio']
        return cls(hwin=hwin, cin=cin, cout=cout)