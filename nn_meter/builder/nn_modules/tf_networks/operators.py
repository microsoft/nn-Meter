# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
import tensorflow as tf
from tensorflow import keras
from ..interface import BaseOperator

''' 
This file contains the keras implementation of operators
'''

#---------------------- convolution layer ----------------------#

class Conv(BaseOperator):
    def get_model(self):
        cout = self.input_shape[2] if "COUT" not in self.config else self.config["COUT"]
        return keras.layers.Conv2D(
            cout,
            kernel_size=self.config["KERNEL_SIZE"],
            strides=self.config["STRIDES"],
            padding="same"
        )

    def get_output_shape(self):
        cout = self.input_shape[2] if "COUT" not in self.config else self.config["COUT"]
        output_h = (self.input_shape[0] - 1) // self.config["STRIDES"] + 1
        output_w = (self.input_shape[1] - 1) // self.config["STRIDES"] + 1
        return [output_h, output_w, cout]


class DwConv(BaseOperator):
    def get_model(self):
        return keras.layers.DepthwiseConv2D(
                kernel_size=self.config["KERNEL_SIZE"],
                strides=self.config["STRIDES"],
                padding="same"
            )

    def get_output_shape(self):
        output_h = (self.input_shape[0] - 1) // self.config["STRIDES"] + 1
        output_w = (self.input_shape[1] - 1) // self.config["STRIDES"] + 1
        return [output_h, output_w, self.input_shape[2]]


class ConvTrans(BaseOperator):
    def get_model(self):
        cout = self.input_shape[2] if "COUT" not in self.config else self.config["COUT"]
        return keras.layers.Conv2DTranspose(
                cout,
                kernel_size=self.config["KERNEL_SIZE"],
                strides=self.config["STRIDES"],
                padding="same"
            )

    def get_output_shape(self):
        cout = self.input_shape[2] if "COUT" not in self.config else self.config["COUT"]
        return [self.input_shape[0] * self.config["STRIDES"], self.input_shape[1] * self.config["STRIDES"], cout]

#------------------ normalization and pooling ------------------#

class BN(BaseOperator):
    def get_model(self):
        return keras.layers.BatchNormalization()


class GlobalAvgpool(BaseOperator):
    def get_model(self):
        return keras.layers.GlobalAveragePooling2D()

    def get_output_shape(self):
        return [self.input_shape[2]]


class MaxPool(BaseOperator):
    def get_model(self):
        if "POOL_STRIDES" not in self.config:
            self.config["POOL_STRIDES"] = self.config["STRIDES"]
        return keras.layers.MaxPool2D(
            pool_size=self.config["KERNEL_SIZE"],
            strides=self.config["POOL_STRIDES"],
            padding="same"
            )

    def get_output_shape(self):
        if "POOL_STRIDES" not in self.config:
            self.config["POOL_STRIDES"] = self.config["STRIDES"]
        output_h = (self.input_shape[0] - 1) // self.config["POOL_STRIDES"] + 1
        output_w = (self.input_shape[1] - 1) // self.config["POOL_STRIDES"] + 1
        return [output_h, output_w, self.input_shape[2]]


class AvgPool(BaseOperator):
    def get_model(self):
        if "POOL_STRIDES" not in self.config:
            self.config["POOL_STRIDES"] = self.config["STRIDES"]
        return keras.layers.AveragePooling2D(
            pool_size=self.config["KERNEL_SIZE"],
            strides=self.config["POOL_STRIDES"],
            padding="same"
            )

    def get_output_shape(self):
        if "POOL_STRIDES" not in self.config:
            self.config["POOL_STRIDES"] = self.config["STRIDES"]
        output_h = (self.input_shape[0] - 1) // self.config["POOL_STRIDES"] + 1
        output_w = (self.input_shape[1] - 1) // self.config["POOL_STRIDES"] + 1
        return [output_h, output_w, self.input_shape[2]]

#------------------------ other modules ------------------------#

class SE(BaseOperator):
    def get_model(self):
        class SE(keras.layers.Layer):
            def __init__(self, num_channels, se_ratio=0.25):
                super().__init__()
                self.pool = keras.layers.GlobalAveragePooling2D(keepdims=True)
                self.squeeze = keras.layers.Conv2D(filters=int(num_channels * se_ratio), kernel_size=1, padding='same')
                self.relu = keras.layers.ReLU()
                self.excite = keras.layers.Conv2D(filters=num_channels, kernel_size=1, padding='same')
                self.hswish = Hswish().get_model()

            def call(self, x):
                x0 = x
                x = self.pool(x)
                x = self.squeeze(x)
                x = self.relu(x)
                x = self.excite(x)
                x = self.hswish(x)
                return x * x0
        return SE(self.input_shape[-1])


class FC(BaseOperator):
    def get_model(self):
        cout = self.input_shape[-1] if "COUT" not in self.config else self.config["COUT"]
        return keras.layers.Dense(cout)

    def get_output_shape(self):
        cout = self.input_shape[-1] if "COUT" not in self.config else self.config["COUT"]
        return self.input_shape[:-1] + [cout]

#-------------------- activation function --------------------#

class Relu(BaseOperator):
    def get_model(self):
        return keras.layers.ReLU()


class Relu6(BaseOperator):
    def get_model(self):
        def func(inputs):
            return tf.nn.relu6(inputs)
        return func


class Sigmoid(BaseOperator):
    def get_model(self):
        def func(inputs):
            return tf.nn.sigmoid(inputs)
        return func


class Hswish(BaseOperator):
    def get_model(self):
        def func(inputs):
            relu6 = tf.keras.layers.ReLU(6)
            return inputs * relu6(inputs + 3.) * (1. / 6.)
        return func

#---------------------- basic operation ----------------------#

class Reshape(BaseOperator):
    def get_model(self):
        if len(self.input_shape) == 3:
            self.output_shape = [self.input_shape[2], self.input_shape[0], self.input_shape[1]]
            def func(inputs):
                return tf.reshape(inputs, [1] + self.output_shape)
        else:
            self.output_shape = [1, 2, int(self.input_shape[0] / 2)]
            def func(inputs):
                return tf.reshape(inputs, [1] + self.output_shape)
        return func

    def get_output_shape(self):
        return self.output_shape


class Add(BaseOperator):
    def get_model(self):
        return keras.layers.Add()

    def get_output_shape(self):
        if len(self.input_shape) == 2 and type(self.input_shape[0]) == list:
            output_shape = self.input_shape[0]
        else:
            output_shape = self.input_shape
        return output_shape

    def get_is_two_inputs(self):
        return True


class Concat(BaseOperator):
    def get_model(self):
        return keras.layers.Concatenate()

    def get_output_shape(self):
        if len(self.input_shape) > 1 and type(self.input_shape[0]) == list: # e.g. [[28, 28, 3], [28, 28, 5]] -> [28, 28, 8]
            output_shape = self.input_shape[0][:-1] + [sum([i[-1] for i in self.input_shape])]
        elif len(self.input_shape) == 3: # e.g. [28, 28, 4] -> [28, 28, 8]
            output_shape = self.input_shape[0:-1] + [self.input_shape[-1] * 2]
        else: # e.g. [1024] -> [2048]
            output_shape = [self.input_shape[0] * 2]
        return output_shape

    def get_is_two_inputs(self):
        return True


class Flatten(BaseOperator):
    def get_model(self):
        return keras.layers.Flatten()

    def get_output_shape(self):
        return [int(np.prod(self.input_shape))]


class Split(BaseOperator):
    def get_model(self):
        def func(inputs):
            return tf.split(inputs, num_or_size_splits=2, axis=3)
        return func

    def get_output_shape(self):
        return [self.input_shape[0], self.input_shape[1], self.input_shape[2] // 2]
