import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers

from nas_models.blocks.tf.mobilenetv3_block import HSigmoid
from nas_models.common import make_divisible


class HSwish_NNMETER(tf.keras.Model):
    
    def __init__(self):
        super().__init__()

    def call(self, x):
        return tf.nn.relu6(tf.math.add(x, 3)) * 0.16667


class HSwish_OFA(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.relu6 = layers.ReLU(6)

    def call(self, x):
        return x * self.relu6(x + 3.) * (1. / 6.)


class SE_OFA(tf.keras.Model):

    def __init__(self, num_channels, se_ratio=0.25):
        super().__init__()
        self.pool = layers.GlobalAveragePooling2D()
        self.squeeze = layers.Conv2D(filters=make_divisible(num_channels * se_ratio), kernel_size=1, padding='same')
        self.relu = layers.ReLU()
        self.excite = layers.Conv2D(filters=num_channels, kernel_size=1, padding='same')
        self.hsigmoid = HSigmoid()

    def call(self, x):
        x0 = x
        x = self.pool(x)
        x = tf.reshape(x, [-1, 1, 1, x.shape[-1]])
        x = self.squeeze(x)
        x = self.relu(x)
        x = self.excite(x)
        x = self.hsigmoid(x)
        return x * x0


class HSwishBlock_xudong(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.relu = layers.ReLU(6)
    def call(self, x):
        return x * self.relu(x + 3.) * (1. / 6.)


class SE_xudong(tf.keras.Model):
    
    def __init__(self, num_channels, se_ratio=0.25):
        super().__init__()
        # open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/op_result_se.txt", "a").write(f'{num_channels}, {make_divisible(num_channels * se_ratio)}\n')
        self.pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.squeeze = layers.Conv2D(filters=make_divisible(num_channels * se_ratio), kernel_size=1, padding='same')
        self.relu = layers.ReLU()
        self.excite = layers.Conv2D(filters=num_channels, kernel_size=1, padding='same')
        self.hswish = HSwishBlock_xudong()

    def call(self, x):
        x0 = x
        x = self.pool(x)
        # x = tf.reshape(x, [-1, 1, 1, x.shape[-1]])
        x = self.squeeze(x)
        x = self.relu(x)
        x = self.excite(x)
        x = self.hswish(x)
        return x * x0
    

class SE_NNMETER(tf.keras.Model):
    def __init__(self, cin, hw):
        super().__init__()
        self.cin = cin
        self.hw = hw
        self.conv1 = tf.keras.layers.Conv2D(
            filters=cin // 4,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="same",
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=cin,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="same",
        )

    def call(self, inputs):
        # hw = inputs.shape[1]
        x = tf.nn.avg_pool(
            inputs,
            ksize=[1, self.hw, self.hw, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
        )
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu6(tf.math.add(x, 3)) * 0.16667
        return x * inputs





# test cascade of two mobilenetv1 blocks
import sys

from nas_models.blocks.tf.mobilenetv3_block import MBConv
# MBConv(hwin=56, cin=32, cout=32, kernel_size=5, expand_ratio=1, strides=1, act='relu', se=None, name='mbconv')

def seq_block(hw, cin1, cout1, ks1, s1, cin2, cout2, ks2, s2):
    hw2 = hw // s1
    class SeqBlock(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.mb1 = MBConv(hw, cin1, cout1, kernel_size=ks1, expand_ratio=1, strides=s1, act='relu', se=False, name='mbconv')
            self.mb2 = MBConv(hw2, cin2, cout2, kernel_size=ks2, expand_ratio=1, strides=s2, act='relu', se=False, name='mbconv')
        def call(self, inputs):
            x = self.mb1(inputs)
            x = self.mb2(x)
            return x
    return SeqBlock()
    
def res_block(hw, cin, cout, ks, s):
    class ResBlock(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.mb1 = MBConv(hw, cin, cout, ks, expand_ratio=1, strides=s, act='relu', se=False, name='mbconv')
            self.mb2 = MBConv(hw, cin, cout, ks, expand_ratio=1, strides=s, act='relu', se=False, name='mbconv')
            
        def call(self, inputs):
            x = self.mb1(inputs)
            x = self.mb2(x)
            return inputs + x
    return ResBlock()

