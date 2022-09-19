# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from tensorflow import keras
from .interface import BaseTestCase


class SingleOpModel(keras.Model):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def call(self, inputs):
        return self.op(inputs)


class TwoOpModel(keras.Model):
    def __init__(self, op1, op2, op1_is_two_inputs, op2_is_two_inputs):
        super().__init__()
        self.op1 = op1
        self.op2 = op2
        self.op1_is_two_inputs = op1_is_two_inputs
        self.op2_is_two_inputs = op2_is_two_inputs

    def call(self, inputs):
        if self.op1_is_two_inputs:
            x = self.op1([inputs[0], inputs[1]])
        else:
            if self.op2_is_two_inputs:
                x = self.op1(inputs[0])
            else:
                x = self.op1(inputs)
        if self.op2_is_two_inputs:
            x = self.op2([x, inputs[-1]])
        else:
            x = self.op2(x)
        return x


class MultipleOutNodes(BaseTestCase):
    name = 'MON'
    cases = {
        'case1': ['relu_relu', 'relu_dwconv', 'dwconv'],
        'case2': ['dwconv_relu_relu', 'relu_dwconv'],
        'case3': ['dwconv_relu', 'dwconv', 'relu_relu']
    }
    true_case = 'case1'
    deps = {
        'BF_dwconv_relu': True,
    }
    implement = 'tensorflow'

    def _model_block(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)
        branch_1 = keras.layers.ReLU(negative_slope=0)(x)
        branch_1 = keras.layers.ReLU(negative_slope=0)(branch_1)
        branch_2 = keras.layers.ReLU(negative_slope=2)(x)
        branch_2 = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(branch_2)

        return keras.models.Model(input_layer, [branch_1, branch_2]), [self.input_shape]

    def _model_relu_relu(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.ReLU()(input_layer)
        x = keras.layers.ReLU()(x)

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_dwconv_relu_relu(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)
        x = keras.layers.ReLU()(x)
        x = keras.layers.ReLU()(x)

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_relu_dwconv(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.ReLU()(input_layer)
        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(x)

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_dwconv_relu(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)
        x = keras.layers.ReLU()(x)

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_dwconv(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)

        return keras.models.Model(input_layer, x), [self.input_shape]

