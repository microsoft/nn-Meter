# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf

from nn_meter.builder.utils.utils import get_tensor_by_shapes 
from .operators import *


def conv_bn_relu(input_shape, config):
    conv_op, out_shape = conv(input_shape, config)
    bn_op, out_shape = batch_norm(out_shape, config)
    relu_op, _ = relu(out_shape, config)
    
    class ConvBnRelu(tf.keras.Model):
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

    class Conv(tf.keras.Model):
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
    
    class ConvBnReluMaxpool(tf.keras.Model):
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
    
    class ConvBnHswish(tf.keras.Model):
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
    
    class ConvBnRelu6(tf.keras.Model):
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
    
    class DwConvBnRelu(tf.keras.Model):
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

    class DwConv(tf.keras.Model):
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
    
    class DwConvBnHswish(tf.keras.Model):
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

    class Hswish(tf.keras.Model):
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
    
    class DwConvBnRelu6(tf.keras.Model):
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

    class FC(tf.keras.Model):
        def __init__(self, fc_op, weight):
            super().__init__()
            self.fc = fc_op
            self.weight = weight

        def call(self, inputs):
            return self.fc(inputs, weight)

    return FC(fc_op, weight)


def maxpool_block(input_shape, config):
    maxpool_op, _ = maxpool(input_shape, config)

    class MaxPool(tf.keras.Model):
        def __init__(self, maxpool_op):
            super().__init__()
            self.maxpool = maxpool_op

        def call(self, inputs):
            return self.maxpool(inputs)

    return MaxPool(maxpool_op)


def avgpool_block(input_shape, config):
    avgpool_op, _ = avgpool(input_shape, config)

    class AvgPool(tf.keras.Model):
        def __init__(self, avgpool_op):
            super().__init__()
            self.avgpool = avgpool_op

        def call(self, inputs):
            return self.avgpool(inputs)

    return AvgPool(avgpool_op)


def bn_relu(input_shape, config):
    bn_op, out_shape = batch_norm(input_shape, config)
    relu_op, _ = relu(out_shape, config)
    
    class BnRelu(tf.keras.Model):
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

    class BN(tf.keras.Model):
        def __init__(self, bn_op):
            super().__init__()
            self.bn = bn_op

        def call(self, inputs):
            return self.bn(inputs)

    return BN(bn_op)


def relu_block(input_shape, config):
    relu_op, _ = relu(input_shape, config)

    class ReLu(tf.keras.Model):
        def __init__(self, relu_op):
            super().__init__()
            self.relu = relu_op

        def call(self, inputs):
            return self.relu(inputs)

    return ReLu(relu_op)


def concat_block(input_shape, config):
    concat_op, _ = concat(input_shape, config)

    class Concat(tf.keras.Model):
        def __init__(self, concat_op):
            super().__init__()
            self.concat = concat_op

        def call(self, inputs):
            return self.concat([inputs, inputs])

    return Concat(concat_op)


def add_block(input_shape, config):
    add_op, _ = add(input_shape, config)

    class Add(tf.keras.Model):
        def __init__(self, add_op):
            super().__init__()
            self.add = add_op

        def call(self, inputs):
            return self.add([inputs, inputs])

    return Add(add_op)


def add_relu(input_shape, config):
    add_op, out_shape = add(input_shape, config)
    relu_op, _ = relu(out_shape, config)
    
    class AddRelu(tf.keras.Model):
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

    class GlobalAvgPool(tf.keras.Model):
        def __init__(self, global_avgpool_op):
            super().__init__()
            self.global_avgpool = global_avgpool_op

        def call(self, inputs):
            return self.global_avgpool(inputs)

    return GlobalAvgPool(global_avgpool_op)


def split_block(input_shape, config):
    split_op, _ = split(input_shape, config)

    class Split(tf.keras.Model):
        def __init__(self, split_op):
            super().__init__()
            self.split = split_op

        def call(self, inputs):
            return self.split(inputs)

    return Split(split_op)
 
 
def channel_shuffle(input_shape, config):

    class ChannelShuffle(tf.keras.Model):
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
    se_op, _ = se(input_shape, config)

    class SE(tf.keras.Model):
        def __init__(self, se_op):
            super().__init__()
            self.se = se_op

        def call(self, inputs):
            return self.se(inputs)

    return SE(se_op)


def concat_pad(input_shape,  config):
    pass


def grouped_conv(input_shape, config = None):
    # cin = features.get_shape().as_list()[-1]
    # cout = num_outputs
    # #assert cin % num_groups == 0 and cout % num_groups == 0

    # with tf.compat.v1.variable_scope(opname+".grouped_conv"):
    #     groups = [
    #         tf.keras.layers.Conv2D(
    #             filters=num_outputs // num_groups,
    #             kernel_size=[kernel_size, kernel_size],
    #             strides=[stride, stride],
    #             padding="same",
    #             name="{}/conv".format(i)
    #         )(x) for i, x in zip(range(num_groups), tf.split(features, num_groups, axis=3))
    #     ]
    #     net = tf.concat(groups, axis=3, name="concat")

    # return net
    pass


def mix_conv(input_shape, config = None):
    # features, num_groups: int, stride: int
    # cin = features.get_shape().as_list()[-1]
    # assert cin % num_groups == 0

    # with tf.compat.v1.variable_scope("mix_conv"):
    #     groups = []
    #     for x, i in zip(tf.split(features, num_groups, axis=3), range(num_groups)):
    #         with tf.variable_scope("{}".format(i)):
    #             kernel_size = i * 2 + 3
    #             groups.append(depthwise_conv2d(x,kernel_size,stride))

    #     return tf.concat(groups, axis=3)
    pass