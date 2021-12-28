# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf
import tensorflow.keras as keras
from .operators import *
from nn_meter.builder.utils.utils import get_tensor_by_shapes 


def conv_bn_relu(input_shape, config):
    conv_op, out_shape = conv(input_shape, config)
    bn_op, out_shape = batch_norm(out_shape, config)
    relu_op, _ = relu(out_shape, config)
    
    class ConvBnRelu(keras.Model):
        def __init__(self, conv_op, bn_op, relu_op):
            super().__init__()
            self.conv = conv_op
            self.bn = bn_op
            self.relu = relu_op

        def call(self, inputs):
            x = self.conv(inputs)
            x = self.bn(x)
            x = self.relu(x)
            return x

    return ConvBnRelu(conv_op, bn_op, relu_op)


def conv_block(input_shape, config):
    conv_op, _ = conv(input_shape, config)

    class Conv(keras.Model):
        def __init__(self, conv_op):
            super().__init__()
            self.conv = conv_op

        def call(self, inputs):
            return self.conv(inputs)

    return Conv(conv_op)


def conv_bn_relu_maxpool(input_shape, config):
    
    conv_op, out_shape = conv(input_shape, config)
    bn_op, out_shape = batch_norm(out_shape, config)
    relu_op, out_shape = relu(out_shape, config)
    maxpool_op, _ = maxpool(out_shape, config)
    
    class ConvBnReluMaxpool(keras.Model):
        def __init__(self, conv_op, bn_op, relu_op, maxpool_op):
            super().__init__()
            self.conv = conv_op
            self.bn = bn_op
            self.relu = relu_op
            self.maxpool = maxpool_op

        def call(self, inputs):
            x = self.conv(inputs)
            x = self.bn(x)
            x = self.relu(x)
            return x

    return ConvBnReluMaxpool(conv_op, bn_op, relu_op, maxpool_op)


def conv_bn_hswish(input_shape, config):
    conv_op, out_shape = conv(input_shape, config)
    bn_op, out_shape = batch_norm(out_shape, config)
    hswish_op, _ = hswish(out_shape, config)
    
    class ConvBnHswish(keras.Model):
        def __init__(self, conv_op, bn_op, hswish_op):
            super().__init__()
            self.conv = conv_op
            self.bn = bn_op
            self.hswish = hswish_op

        def call(self, inputs):
            x = self.conv(inputs)
            x = self.bn(x)
            x = self.hswish(x)
            return x

    return ConvBnHswish(conv_op, bn_op, hswish_op)


def conv_bn_relu6(input_shape, config):
    conv_op, out_shape = conv(input_shape, config)
    bn_op, out_shape = batch_norm(out_shape, config)
    relu6_op, _ = relu6(out_shape, config)
    
    class ConvBnRelu6(keras.Model):
        def __init__(self, conv_op, bn_op, relu6_op):
            super().__init__()
            self.conv = conv_op
            self.bn = bn_op
            self.relu6 = relu6_op

        def call(self, inputs):
            x = self.conv(inputs)
            x = self.bn(x)
            x = self.relu6(x)
            return x

    return ConvBnRelu6(conv_op, bn_op, relu6_op)


def dwconv_bn_relu(input_shape, config):
    dwconv_op, out_shape = dwconv(input_shape, config)
    bn_op, out_shape = batch_norm(out_shape, config)
    relu_op, _ = relu(out_shape, config)
    
    class DwConvBnRelu(keras.Model):
        def __init__(self, dwconv_op, bn_op, relu_op):
            super().__init__()
            self.dwconv = dwconv_op
            self.bn = bn_op
            self.relu = relu_op

        def call(self, inputs):
            x = self.dwconv(inputs)
            x = self.bn(x)
            x = self.relu(x)
            return x

    return DwConvBnRelu(dwconv_op, bn_op, relu_op)


def dwconv_block(input_shape, config):
    dwconv_op, _ = dwconv(input_shape, config)

    class DwConv(keras.Model):
        def __init__(self, dwconv_op):
            super().__init__()
            self.dwconv = dwconv_op

        def call(self, inputs):
            return self.dwconv(inputs)

    return DwConv(dwconv_op)


def dwconv_bn_hswish(input_shape, config):
    dwconv_op, out_shape = dwconv(input_shape, config)
    bn_op, out_shape = batch_norm(out_shape, config)
    hswish_op, _ = hswish(out_shape, config)
    
    class DwConvBnHswish(keras.Model):
        def __init__(self, dwconv_op, bn_op, hswish_op):
            super().__init__()
            self.dwconv = dwconv_op
            self.bn = bn_op
            self.hswish = hswish_op

        def call(self, inputs):
            x = self.dwconv(inputs)
            x = self.bn(x)
            x = self.hswish(x)
            return x

    return DwConvBnHswish(dwconv_op, bn_op, hswish_op)


def hswish_block(input_shape, config):
    hswish_op, _ = hswish(input_shape, config)

    class Hswish(keras.Model):
        def __init__(self, hswish_op):
            super().__init__()
            self.hswish = hswish_op

        def call(self, inputs):
            return self.hswish(inputs)

    return Hswish(hswish_op)


def dwconv_bn_relu6(input_shape, config):
    dwconv_op, out_shape = dwconv(input_shape, config)
    bn_op, out_shape = batch_norm(out_shape, config)
    relu6_op, _ = relu6(out_shape, config)
    
    class DwConvBnRelu6(keras.Model):
        def __init__(self, dwconv_op, bn_op, relu6_op):
            super().__init__()
            self.dwconv = dwconv_op
            self.bn = bn_op
            self.relu6 = relu6_op

        def call(self, inputs):
            x = self.dwconv(inputs)
            x = self.bn(x)
            x = self.relu6(x)
            return x

    return DwConvBnRelu6(dwconv_op, bn_op, relu6_op)


def fc_block(input_shape, config):
    fc_op, _ = fc(input_shape, config)
    weight = tf.random.normal(shape=[config["CIN"], config["COUT"]])

    class FC(keras.Model):
        def __init__(self, fc_op, weight):
            super().__init__()
            self.fc = fc_op
            self.weight = weight

        def call(self, inputs):
            return self.fc(inputs, weight)

    return FC(fc_op, weight)


def maxpool_block(input_shape, config):
    maxpool_op, _ = maxpool(input_shape, config)

    class MaxPool(keras.Model):
        def __init__(self, maxpool_op):
            super().__init__()
            self.maxpool = maxpool_op

        def call(self, inputs):
            return self.maxpool(inputs)

    return MaxPool(maxpool_op)


def avgpool_block(input_shape, config):
    avgpool_op, _ = avgpool(input_shape, config)

    class AvgPool(keras.Model):
        def __init__(self, avgpool_op):
            super().__init__()
            self.avgpool = avgpool_op

        def call(self, inputs):
            return self.avgpool(inputs)

    return AvgPool(avgpool_op)


def bn_relu(input_shape, config):
    bn_op, out_shape = batch_norm(input_shape, config)
    relu_op, _ = relu(out_shape, config)
    
    class BnRelu(keras.Model):
        def __init__(self, bn_op, relu_op):
            super().__init__()
            self.bn = bn_op
            self.relu = relu_op

        def call(self, inputs):
            x = self.bn(inputs)
            x = self.relu(x)
            return x

    return BnRelu(bn_op, relu_op)


def bn_block(input_shape, config):
    bn_op, _ = batch_norm(input_shape, config)

    class BN(keras.Model):
        def __init__(self, bn_op):
            super().__init__()
            self.bn = bn_op

        def call(self, inputs):
            return self.bn(inputs)

    return BN(bn_op)


def relu_block(input_shape, config):
    relu_op, _ = relu(input_shape, config)

    class ReLu(keras.Model):
        def __init__(self, relu_op):
            super().__init__()
            self.relu = relu_op

        def call(self, inputs):
            return self.relu(inputs)

    return ReLu(relu_op)


def concat_block(input_shape, config):
    concat_op, _ = concat(input_shape, config)

    class Concat(keras.Model):
        def __init__(self, concat_op):
            super().__init__()
            self.concat = concat_op

        def call(self, inputs):
            return self.concat([inputs, inputs])

    return Concat(concat_op)


def add_block(input_shape, config):
    add_op, _ = add(input_shape, config)

    class Add(keras.Model):
        def __init__(self, add_op):
            super().__init__()
            self.add = add_op

        def call(self, inputs):
            return self.add([inputs, inputs])

    return Add(add_op)


def add_relu(input_shape, config):
    add_op, out_shape = add(input_shape, config)
    relu_op, _ = relu(out_shape, config)
    
    class AddRelu(keras.Model):
        def __init__(self, add_op, relu_op):
            super().__init__()
            self.add = add_op
            self.relu = relu_op

        def call(self, inputs):
            x = self.add([inputs, inputs])
            x = self.relu(x)
            return x

    return AddRelu(add_op, relu_op)


def global_avgpool_block(input_shape, config):
    global_avgpool_op, _ = global_avgpool(input_shape, config)

    class GlobalAvgPool(keras.Model):
        def __init__(self, global_avgpool_op):
            super().__init__()
            self.global_avgpool = global_avgpool_op

        def call(self, inputs):
            return self.global_avgpool(inputs)

    return GlobalAvgPool(global_avgpool_op)


def split_block(input_shape, config):
    split_op, _ = split(input_shape, config)

    class Split(keras.Model):
        def __init__(self, split_op):
            super().__init__()
            self.split = split_op

        def call(self, inputs):
            return self.split(inputs)

    return Split(split_op)
 
 
def channel_shuffle(input_shape, config):

    class ChannelShuffle(keras.Model):
        def __init__(self):
            super().__init__()

        def call(self, inputs):
            _, h, w, c = inputs.get_shape().as_list()
            x = tf.reshape(inputs, [-1, h, w, 2, c // 2])
            x = tf.transpose(x, (0, 1, 2, 4, 3))
            x = tf.reshape(x, [-1, h, w, c])
            return x

    return ChannelShuffle()


def se_block(input_shape, config):
    mid_channels = input_shape[-1] // 4
    hswish_op, _ = hswish(input_shape, config)

    class SE(keras.Model):
        def __init__(self, mid_channels, hswish_op):
            super().__init__()
            self.conv1 = keras.layers.Conv2D(
                filters=mid_channels,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="same",
            )
            self.conv2 = keras.layers.Conv2D(
                filters=input_shape[-1],
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="same",
            )
            self.hswish = hswish_op

        def call(self, inputs):
            x = tf.nn.avg_pool(
                inputs,
                ksize=[1] + input_shape[0:2] + [1],
                strides=[1, 1, 1, 1],
                padding="VALID",
            )
            x = self.conv1(x)
            x = tf.nn.relu(x)
            x = self.conv2(x)
            x = self.hswish(x)
            return x * inputs
    return SE(mid_channels, hswish_op)


def grouped_conv(input_shape, config = None):
    cout = input_shape[2] if "COUT" not in config else config["COUT"]
    num_groups = config['NUM_GROUPS']

    class GroupedConv(keras.Model):
        def __init__(self, cout, num_groups):
            super().__init__()
            self.cout = cout
            self.num_groups = num_groups

        def call(self, inputs):
            x = [keras.layers.Conv2D(
                    filters=self.cout // self.num_groups,
                    kernel_size=config['KERNEL_SIZE'],
                    strides=config['STRIDES'],
                    padding="same",
                )(x) for x in tf.split(inputs, self.num_groups, axis=3)
            ]
            return tf.concat(x, axis=3)    

    return GroupedConv(cout, num_groups)


def mix_conv(input_shape, config = None):
    cout = input_shape[2] if "COUT" not in config else config["COUT"]
    num_groups = config['NUM_GROUPS']

    class MixConv(keras.Model):
        def __init__(self, cout, num_groups):
            super().__init__()
            self.cout = cout
            self.num_groups = num_groups

        def call(self, inputs):
            x = [keras.layers.Conv2D(
                    filters=self.cout // self.num_groups,
                    kernel_size=i * 2 + 3,
                    strides=config['STRIDES'],
                    padding="same",
                )(x) for i, x in zip(range(self.num_groups), tf.split(inputs, self.num_groups, axis=3))
            ]
            return tf.concat(x, axis=3)    

    return MixConv(cout, num_groups)
