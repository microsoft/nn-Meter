# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
import tensorflow as tf
from tensorflow import keras


''' 
This file contains the keras implementation of operators, return (the function of the operator (tf.keras.Model), the output shape of the operator)
'''

#---------------------- convolution layer ----------------------#

def conv(input_shape, config = None):
    cout = input_shape[2] if "cout" not in config else config["cout"]
    output_shape = [shape for shape in input_shape[:2]] + [cout]
    return keras.layers.Conv2D(
            cout,
            kernel_size=config['kernel_size'],
            padding=config['padding'],
            strides=config['strides']
        ), output_shape


def dwconv(input_shape, config = None):
    return keras.layers.DepthwiseConv2D(
            kernel_size=config['kernel_size'],
            padding=config['padding'],
            strides=config['strides']
        ), input_shape


def convtrans(input_shape, config = None):
    cout = input_shape[2] if "cout" not in config else config["cout"] 
    class Conv2dTranspose(keras.layers.Layer):
        def __init__(self, cout):
            super().__init__()
            self.filters = tf.Variable(tf.ones([config['kernel_size'], config['kernel_size'], cout, cout])) 
        def call(self, inputs):
            return tf.nn.conv2d_transpose(
                inputs,
                filters=self.filters,
                output_shape=[1] + input_shape,
                strides=[1, 1]
            )
    return Conv2dTranspose(cout), input_shape


def grouped_conv(input_shape, config = None):
    pass


def mix_conv(input_shape, config = None):
    # features, num_groups: int, stride: int
    pass


#------------------ normalization and pooling ------------------#

def batch_norm(input_shape, config = None):
    return keras.layers.BatchNormalization(), input_shape


def pooling(input_shape, config = None):
    output_shape = [int(input_shape[0] / 2), int(input_shape[1] / 2), input_shape[2]]
    return keras.layers.AveragePooling2D(padding=config['padding']), output_shape


def global_avgpooling(input_shape, config = None):
    pass


def max_pooling(input_shape, config = None):
    # features, kernelsize, stride, padding = 'SAME', opname = ''
    pass


def avg_pooling(input_shape, config = None):
    # features, kernelsize, stride, padding = 'SAME', opname = ''
    pass

#------------------------ other modules ------------------------#

def fc(input_shape, config = None):
    pass


def se(input_shape, config = None):
    class SE(keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.conv1 = tf.keras.layers.Conv2D(
                filters=input_shape[-1],
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="same",
            )
            self.conv2 = tf.keras.layers.Conv2D(
                filters=input_shape[-1],
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="same",
            )
        def call(self, inputs):
            x = tf.nn.avg_pool(
                inputs,
                ksize=[1] + input_shape[0:2] + [1],
                strides=[1, 1, 1, 1],
                padding='VALID',
            )
            x = self.conv1(x)
            x = tf.nn.relu(x)
            x = self.conv2(x)
            return x * inputs
    return SE(), input_shape


def dense(input_shape, config = None):
    return keras.layers.Dense(input_shape[0]), input_shape


def channel_shuffle():
    pass

#-------------------- activation function --------------------#

def relu(input_shape, config = None):
    return keras.layers.ReLU(), input_shape


def relu6(input_shape, config = None):
    def func(inputs):
        return tf.nn.relu6(inputs)
    return func, input_shape


def sigmoid(input_shape, config = None):
    def func(inputs):
        return tf.nn.sigmoid(inputs)
    return func, input_shape


def hswish(input_shape, config = None):
    def func(inputs):
        return tf.nn.relu6(tf.math.add(inputs, 3)) * 0.16667
    return func, input_shape

#---------------------- basic operation ----------------------#

def reshape(input_shape, config = None):
    if len(input_shape) == 3:
        output_shape = [input_shape[2], input_shape[0], input_shape[1]]
        def func(inputs):
            return tf.reshape(inputs, [1] + output_shape)
    else:
        output_shape = [1, 2, int(input_shape[0] / 2)]
        def func(inputs):
            return tf.reshape(inputs, [1] + output_shape)
    return func, output_shape


def add(input_shape, config = None):
    return keras.layers.Add(), input_shape


def concat(input_shape, config = None):
    if len(input_shape) == 3:
        output_shape = [input_shape[0], input_shape[1], input_shape[2] * 2]
    else:
        output_shape = [input_shape[0] * 2]
    return keras.layers.Concatenate(), output_shape


def flatten(input_shape, config = None):
    return keras.layers.Flatten(), input_shape
