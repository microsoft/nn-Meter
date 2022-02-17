# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
import tensorflow as tf
from tensorflow import keras


class BaseOperator:
    def __init__(self, input_shape, config=None):
        self.input_shape = input_shape
        self.config = config

    def get_model(self):
        pass

    def get_output_shape(self):
        return self.input_shape

    def get_is_two_inputs(self):
        return False
    
    def test_operator():
        ''' for users to test the model when registration. Do not need to override by users.
        '''
        pass


''' 
This file contains the keras implementation of operators, return (the function of the operator (tf.keras.Model), the output shape of the operator)
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
            def __init__(self, input_shape):
                super().__init__()
                self.in_shape = input_shape
                self.conv1 = keras.layers.Conv2D(
                    filters=self.in_shape[-1] // 4,
                    kernel_size=[1, 1],
                    strides=[1, 1],
                    padding="same",
                )
                self.conv2 = keras.layers.Conv2D(
                    filters=self.in_shape[-1],
                    kernel_size=[1, 1],
                    strides=[1, 1],
                    padding="same",
                )

            def call(self, inputs):
                x = tf.nn.avg_pool(
                    inputs,
                    ksize=[1] + self.in_shape[0:2] + [1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                )
                x = self.conv1(x)
                x = tf.nn.relu(x)
                x = self.conv2(x)
                x = tf.nn.relu6(tf.math.add(x, 3)) * 0.16667
                return x * inputs
        return SE(self.input_shape)


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
            return tf.nn.relu6(tf.math.add(inputs, 3)) * 0.16667
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
