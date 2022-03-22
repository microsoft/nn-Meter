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

class BatchNorm(BaseOperator):
    def get_model(self):
        return keras.layers.BatchNormalization()


class LayerNorm(BaseOperator):
    def get_model(self):
        return keras.layers.LayerNormalization()


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
        class Layer(keras.layers.Layer):
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
        return Layer(self.input_shape)


class FC(BaseOperator):
    def get_model(self):
        cout = self.input_shape[-1] if "COUT" not in self.config else self.config["COUT"]
        return keras.layers.Dense(cout)

    def get_output_shape(self):
        cout = self.input_shape[-1] if "COUT" not in self.config else self.config["COUT"]
        return self.input_shape[:-1] + [cout]


class MultiHeadPositionalEmbedding(BaseOperator):
    def get_model(self):
        class Layer(keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def build(self, input_shape, **kwargs):
                print(input_shape)
                _, num_heads, qq_blocks, kk_blocks = input_shape
                self.bb = self.add_weight(shape=(kk_blocks, num_heads), initializer="zeros", trainable=True)
                strides = int(tf.math.ceil(tf.math.sqrt(float(kk_blocks / qq_blocks))))
                q_blocks_h = q_blocks_w = int(tf.math.sqrt(float(qq_blocks)))
                k_blocks_h = k_blocks_w = int(tf.math.sqrt(float(kk_blocks)))

                x1, y1 = tf.meshgrid(range(q_blocks_h), range(q_blocks_w))
                x2, y2 = tf.meshgrid(range(k_blocks_h), range(k_blocks_w))
                aa = tf.concat([tf.reshape(x1, (-1, 1)), tf.reshape(y1, (-1, 1))], axis=-1)
                bb = tf.concat([tf.reshape(x2, (-1, 1)), tf.reshape(y2, (-1, 1))], axis=-1)
                # print(f">>>> {aa.shape = }, {bb.shape = }") # aa.shape = (16, 2), bb.shape = (49, 2)
                cc = [tf.math.abs(bb - ii * strides) for ii in aa]
                self.bb_pos = tf.stack([ii[:, 0] + ii[:, 1] * k_blocks_h for ii in cc])
                # print(f">>>> {self.bb_pos.shape = }")    # self.bb_pos.shape = (16, 49)

                super().build(input_shape)

            def call(self, inputs, **kwargs):
                pos_bias = tf.gather(self.bb, self.bb_pos)
                pos_bias = tf.transpose(pos_bias, [2, 0, 1])
                return inputs + pos_bias

            def load_resized_pos_emb(self, source_layer):
                if isinstance(source_layer, dict):
                    source_bb = source_layer["positional_embedding:0"]  # weights
                else:
                    source_bb = source_layer.bb  # layer
                hh = ww = int(tf.math.sqrt(float(source_bb.shape[0])))
                ss = tf.reshape(source_bb, (hh, ww, source_bb.shape[-1]))  # [hh, ww, num_heads]
                target_hh = target_ww = int(tf.math.sqrt(float(self.bb.shape[0])))
                tt = tf.image.resize(ss, [target_hh, target_ww])  # [target_hh, target_ww, num_heads]
                tt = tf.reshape(tt, (self.bb.shape))  # [target_hh * target_ww, num_heads]
                self.bb.assign(tt)

        return Layer()


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


class Softmax(BaseOperator):
    def get_model(self):
        return keras.layers.Softmax()


class Hswish(BaseOperator):
    def get_model(self):
        def func(inputs):
            return tf.nn.relu6(tf.math.add(inputs, 3)) * 0.16667
        return func


class Gelu(BaseOperator):
    def get_model(self):
        def func(inputs):
            return tf.nn.gelu(inputs)
        return func

#---------------------- basic operation ----------------------#

class Reshape(BaseOperator):
    def get_model(self):
        if "SHAPE_TO" not in self.config:
            if len(self.input_shape) == 3:
                self.output_shape = [self.input_shape[2], self.input_shape[0], self.input_shape[1]]
                def func(inputs):
                    return tf.reshape(inputs, [1] + self.output_shape)
            else:
                self.output_shape = [1, 2, int(self.input_shape[0] / 2)]
                def func(inputs):
                    return tf.reshape(inputs, [1] + self.output_shape)
        else:
            self.output_shape = self.config["SHAPE_TO"]
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
        if "SPLIT_DIM" not in self.config:
            self.output_shape = [self.input_shape[0], self.input_shape[1], self.input_shape[2] // 2]
            def func(inputs):
                return tf.split(inputs, num_or_size_splits=2, axis=3)
        else:
            self.output_shape = [[self.input_shape[0], self.input_shape[1], dim] for dim in self.config["SPLIT_DIM"]]
            def func(inputs):
                return tf.split(inputs, self.config["SPLIT_DIM"], axis=-1)
        return func

    def get_output_shape(self):
        return self.output_shape


class Dropout(BaseOperator):
    def get_model(self):
        return keras.layers.Dropout(rate=0.2)


class Matmul(BaseOperator):
    def get_model(self):
        def func(inputs):
            return tf.matmul(inputs[0], inputs[1], transpose_b=self.config["TRANSPOSE"])
        return func

    def get_output_shape(self):
        return super().get_output_shape() #TODO


class Transpose(BaseOperator):
    def get_model(self):
        def func(inputs):
            return tf.transpose(inputs, perm=self.config["PERM"])
        return func

    def get_output_shape(self):
        return super().get_output_shape() #TODO
