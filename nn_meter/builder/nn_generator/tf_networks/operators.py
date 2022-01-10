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
    cout = input_shape[2] if "COUT" not in config else config["COUT"]
    output_h = (input_shape[0] - 1) // config["STRIDES"] + 1
    output_w = (input_shape[1] - 1) // config["STRIDES"] + 1
    output_shape = [output_h, output_w, cout]

    return keras.layers.Conv2D(
            cout,
            kernel_size=config["KERNEL_SIZE"],
            strides=config["STRIDES"],
            padding="same"
        ), output_shape


def dwconv(input_shape, config = None):
    output_h = (input_shape[0] - 1) // config["STRIDES"] + 1
    output_w = (input_shape[1] - 1) // config["STRIDES"] + 1
    output_shape = [output_h, output_w, input_shape[2]]
    return keras.layers.DepthwiseConv2D(
            kernel_size=config["KERNEL_SIZE"],
            strides=config["STRIDES"],
            padding="same"
        ), output_shape


def convtrans(input_shape, config = None):
    cout = input_shape[2] if "COUT" not in config else config["COUT"]
    output_shape = [input_shape[0] * config["STRIDES"], input_shape[1] * config["STRIDES"], cout]
    return keras.layers.Conv2DTranspose(
            cout,
            kernel_size=config["KERNEL_SIZE"],
            strides=config["STRIDES"],
            padding="same"
        ), output_shape

#------------------ normalization and pooling ------------------#

def bn(input_shape, config = None):
    return keras.layers.BatchNormalization(), input_shape


def globalavgpool(input_shape, config = None):
    return keras.layers.GlobalAveragePooling2D(), input_shape[2]


def maxpool(input_shape, config = None):
    output_h = (input_shape[0] - 1) // config["POOL_STRIDES"] + 1
    output_w = (input_shape[1] - 1) // config["POOL_STRIDES"] + 1

    return keras.layers.MaxPool2D(
        pool_size=config["KERNEL_SIZE"],
        strides=config["POOL_STRIDES"],
        padding="same"
        ), [output_h, output_w, input_shape[2]]


def avgpool(input_shape, config = None):
    output_h = (input_shape[0] - 1) // config["POOL_STRIDES"] + 1
    output_w = (input_shape[1] - 1) // config["POOL_STRIDES"] + 1

    return keras.layers.AveragePooling2D(
        pool_size=config["KERNEL_SIZE"],
        strides=config["POOL_STRIDES"],
        padding="same"
        ), [output_h, output_w, input_shape[2]]

#------------------------ other modules ------------------------#

def se(input_shape, config = None):
    class SE(keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.conv1 = keras.layers.Conv2D(
                filters=input_shape[-1] // 4,
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
            x = tf.nn.relu6(tf.math.add(x, 3)) * 0.16667
            return x * inputs
    return SE(), input_shape


def fc(input_shape, config = None):
    cout = input_shape[-1] if "COUT" not in config else config["COUT"]
    output_shape = input_shape[:-1] + [cout]
    return keras.layers.Dense(cout), output_shape

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
    if len(input_shape) == 2 and type(input_shape[0]) == list:
        output_shape = input_shape[0]
    else:
        output_shape = input_shape
    return keras.layers.Add(), output_shape


def concat(input_shape, config = None):
    if len(input_shape) > 1 and type(input_shape[0]) == list: # e.g. [[28, 28, 3], [28, 28, 5]] -> [28, 28, 8]
        output_shape = input_shape[0][:-1] + [sum([i[-1] for i in input_shape])]
    elif len(input_shape) == 3: # e.g. [28, 28, 4] -> [28, 28, 8]
        output_shape = input_shape[0:-1] + [input_shape[-1] * 2]
    else: # e.g. [1024] -> [2048]
        output_shape = [input_shape[0] * 2]
    return keras.layers.Concatenate(), output_shape


def flatten(input_shape, config = None):
    output_shape = [int(np.prod(input_shape))]
    return keras.layers.Flatten(), output_shape


def split(input_shape, config = None):
    output_shape = [input_shape[0], input_shape[1], input_shape[2] // 2]
    def func(inputs):
        return tf.split(inputs, num_or_size_splits=2, axis=3)
    return func, output_shape
