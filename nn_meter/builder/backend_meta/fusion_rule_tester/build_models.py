# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf

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
