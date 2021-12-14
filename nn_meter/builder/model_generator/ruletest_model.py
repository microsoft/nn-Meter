# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf

from . import operators
from .utils import get_op_output_shape, get_op_is_two_inputs
from nn_meter.builder.utils import get_inputs_by_shapes


class SingleOpModel(tf.keras.Model):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def call(self, inputs):
        return self.op(inputs)


class TwoOpModel(tf.keras.Model):
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


def get_operator_by_name(name, input_shape, config = None):
    operator = getattr(operators, name)(input_shape, config)
    output_shape = get_op_output_shape(name, input_shape)
    op_is_two_inputs = get_op_is_two_inputs(name)
    return operator, output_shape, op_is_two_inputs


def get_ruletest_model(op1, op2, input_shape, config):
    layer1, op1_output_shape, op1_is_two_inputs = get_operator_by_name(op1, input_shape, config)
    layer2, _, op2_is_two_inputs = get_operator_by_name(op2, op1_output_shape, config)

    op1_model = SingleOpModel(layer1)
    op1_shapes = [input_shape] * (1 + op1_is_two_inputs)
    op1_model(get_inputs_by_shapes(op1_shapes))

    op2_model = SingleOpModel(layer2)
    op2_shapes = [op1_output_shape] * (1 + op2_is_two_inputs)
    op2_model(get_inputs_by_shapes(op2_shapes))

    block_model = TwoOpModel(layer1, layer2, op1_is_two_inputs, op2_is_two_inputs)
    block_shapes = [input_shape] * (1 + op1_is_two_inputs) + [op1_output_shape] * op2_is_two_inputs
    block_model(get_inputs_by_shapes(block_shapes))

    return op1_model, op2_model, block_model, op1_shapes, op2_shapes, block_shapes
