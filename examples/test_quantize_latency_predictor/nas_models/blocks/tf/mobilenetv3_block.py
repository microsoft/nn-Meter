from typing import Dict

import tensorflow as tf
from tensorflow.keras import layers

from ...search_space.mobilenetv3_space import MobileNetV3Space
from ...common import make_divisible


# class HSwish(tf.keras.Model):
    
#     def __init__(self):
#         super().__init__()
#         self.relu6 = layers.ReLU(6)

#     def call(self, x):
#         return x * self.relu6(x + 3.) * (1. / 6.)


# implementation in nn-meter
class HSwish(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.relu6 = layers.ReLU(6)

    def call(self, x):
        return tf.nn.relu6(tf.math.add(x, 3)) * 0.16667


class HSigmoid(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.relu6 = layers.ReLU(6)

    def call(self, x):
        return self.relu6(x + 3.) * (1. / 6.)


def build_act(act: str):
    if act == 'relu':
        return layers.ReLU()
    if act == 'h_swish':
        return HSwish()


# class SE(tf.keras.Model):
#     def __init__(self, num_channels, se_ratio=0.25):
#         super().__init__()
#         self.pool = layers.GlobalAveragePooling2D()
#         self.squeeze = layers.Conv2D(filters=make_divisible(num_channels * se_ratio), kernel_size=1, padding='same')
#         self.relu = layers.ReLU()
#         self.excite = layers.Conv2D(filters=num_channels, kernel_size=1, padding='same')
#         self.hsigmoid = HSigmoid()

#     def call(self, x):
#         x0 = x
#         x = self.pool(x)
#         x = tf.reshape(x, [-1, 1, 1, x.shape[-1]])
#         x = self.squeeze(x)
#         x = self.relu(x)
#         x = self.excite(x)
#         x = self.hsigmoid(x)
#         return x * x0

# implementation in nn-meter
class SE(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_channels // 4,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="same",
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=num_channels,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="same",
        )

    def call(self, inputs):
        hw = inputs.shape[1]
        x = tf.nn.avg_pool(
            inputs,
            ksize=[1, hw, hw, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
        )
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu6(tf.math.add(x, 3)) * 0.16667
        return x * inputs
    

class BasicBlock(tf.keras.Model):

    def __init__(self, hwin, cin) -> None:
        super().__init__()
        self.hwin = hwin
        self.cin = cin

    @property
    def config_str(self) -> str:
        return MobileNetV3Space.config2str(self.config)

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
        elif name == 'final_expand':
            return FinalExpand.build_from_config(config)
        elif name == 'logits':
            return Logits.build_from_config(config)
        else:
            raise ValueError(f'{name} not recognized.')


class _ConvBnAct(BasicBlock):
   
    def __init__(self, hwin, cin, cout, kernel_size, strides, name='convbnact', act='relu', use_bn=True) -> None:
        super().__init__(hwin, cin)
        self.conv = layers.Conv2D(filters=cout, kernel_size=kernel_size, strides=strides, padding='same')
        self.use_bn = use_bn
        if use_bn:
            self.bn = layers.BatchNormalization()
        self.activation = build_act(act)
        self.block_name = name
        self.act = act

    def call(self, x):
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

    def call(self, x):
        x = tf.reduce_mean(x, [1, 2], keepdims=True)
        x = super().call(x)
        return x

    @property
    def config(self):
        return dict(
            name = self.block_name,
            hwio = (self.hwin, 1),
            cio = (self.cin, self.conv.filters),
            k = 0,
            e = 0,
            act = 'relu',
            se = 0
        )

class Logits(BasicBlock):

    def __init__(self, hwin, cin, cout) -> None:
        super().__init__(hwin, cin)
        self.linear = layers.Dense(cout)

    def call(self, x):
        x = tf.reshape(x, [-1, x.shape[-1]])
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

    def __init__(self, hwin, cin, cout, kernel_size, expand_ratio, strides, act, se, name='mbconv') -> None:
        super().__init__(hwin=hwin, cin=cin)
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.feature_size = round(cin * expand_ratio)
        self.act = act
        self.use_se = se

        self.is_skip = strides == 1 and cin == cout
        self.block_name = name

        if self.expand_ratio > 1:
            self.inverted_bottleneck = tf.keras.Sequential([
                layers.Conv2D(filters=self.feature_size, kernel_size=1, padding='same'),
                layers.BatchNormalization(),
                build_act(self.act)
            ])
        else:
            self.inverted_bottleneck = None
        self.depth_conv = tf.keras.Sequential([
            layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same'),
            layers.BatchNormalization(),
            build_act(self.act)
        ])

        if self.use_se:
            self.se = SE(self.feature_size)

        self.point_conv = tf.keras.Sequential([
            layers.Conv2D(filters=cout, kernel_size=1, padding='same'),
            layers.BatchNormalization()
        ])


    def call(self, x):
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