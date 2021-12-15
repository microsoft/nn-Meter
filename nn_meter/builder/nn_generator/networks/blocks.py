# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf 
from .operators import *


def conv_bn_relu(input_shape, config):
    conv_op, out_shape = conv(input_shape, config)
    bn_op, _ = batch_norm(out_shape, config)
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


def conv(input_shape, config):
    conv_op, _ = conv(input_shape, config)

    class Conv(tf.keras.Model):
        def __init__(self, conv_op):
            super().__init__()
            self.conv = conv_op

        def call(self, inputs):
            return self.conv(inputs)

    return Conv(conv_op)


def conv_bn_relu_maxpool(input, kernelsize, cin, cout, stride, mks, mstride, istraining = False, pad = False):
    x1 = input
    x = conv(x1, cout, kernelsize, stride=stride)
    x = batch_norm(x)
    x = relu(x)
    x = max_pooling(x, mks, mstride, opname='max_pool')
    return x


def conv_bn_hswish(input, kernelsize, cin, cout, stride, istraining = False, pad = False):
    x1 = input
    x = conv(x1, cout, kernelsize, stride=stride)
    x = batch_norm(x)
    x = hswish(x, config)
    return x


def conv_bn_relu6(input, kernelsize, cin, cout, stride, istraining = False):
    x = conv(input, cout, kernelsize, stride=stride)
    x = batch_norm(x)
    x = relu6(x, config)
    return x


def dwconv_bn_relu(input,kernelsize,cin,cout,stride,name='',istraining=False,pad=False):
    x1 = input
    x = dwconv(x1, kernelsize, stride=stride)
    x = batch_norm(x)
    x = relu(x, )    
    return x


def dwconv(input, kernelsize, cin, cout, stride, istraining = False, pad = False):
    x1 = input
    x = dwconv(x1, kernelsize, stride=stride)    
    return x


def dwconv_bn_hswish(input, kernelsize, cin, cout, stride, istraining = False, pad = False):
    x1 = input
    x = dwconv(x1, kernelsize, stride=stride)
    x = batch_norm(x, opname='dwconvbnrelu.1')
    x = hswish(x)
    return x


def hswish(input, istraining = False, pad = False):
    x = hswish(input)
    return x


def dwconv_bn_relu6(input, kernelsize, cin, cout, stride, istraining = False):
    x = dwconv(input, kernelsize, stride=stride, opname='dwconvbnrelu6.1')
    x = batch_norm(x, opname='dwconvbnrelu6.1')
    x = relu(x)
    return x


def fc(input, cout, pad = False):
    x1 = input
    x = fc(x1, cout, opname = 'fc')
    return x


def maxpool(input, kernelsize, stride, pad = False):
    x1 = input
    return max_pooling(x1, kernelsize, stride, opname='max_pool')


def avgpool(input, kernelsize, stride, name = ''):
    return avg_pooling(input, kernelsize, stride, opname='avg_pool')


def bn_relu(input, name = ''):
    x = batch_norm(input, opname='bnrelu.1')
    x = relu(x)
    return  x


def bn(input, name = ''):
    x = batch_norm(input, opname='bnrelu.1')
    return x


def relu(input,name = ""):
    x = relu(input)
    return x


def concats(inputs, name = ''):
    x = concats(input)
    return x


def add_relu(inputs, istraining = False, pad = False):
    x = tf.add(inputs[0], inputs[1])
    x = relu(x)   
    return x


def add(inputs, istraining = False, pad = False):
    x = tf.add(inputs[0], inputs[1]) 
    return x


def global_avgpool(input, name = ""):
    x = global_avgpooling(input, name=name)
    return x


def split(input, num_or_size_splits = 2,  name = ""):
     out1, out2 = tf.split(input, num_or_size_splits=2, axis=3)
     return out1, out2
 
 
def channel_shuffle(input, groups = 2, name = ""):
    x = channel_shuffle(input, groups, name)
    return x


def se(input, cin, name = ""):
    x = se(input,cin//4)
    return x
