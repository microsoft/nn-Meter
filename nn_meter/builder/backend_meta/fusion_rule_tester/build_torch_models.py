# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from torch import nn
from .interface import BaseTestCase


class SingleOpModel(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, inputs):
        return self.op(inputs)


class TwoOpModel(nn.Module):
    def __init__(self, op1, op2, op1_is_two_inputs, op2_is_two_inputs):
        super().__init__()
        self.op1 = op1
        self.op2 = op2
        self.op1_is_two_inputs = op1_is_two_inputs
        self.op2_is_two_inputs = op2_is_two_inputs

    def forward(self, inputs):
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
    implement = 'torch'

    def _model_block(self):
        raise NotImplementedError

    def _model_relu_relu(self):
        raise NotImplementedError

    def _model_dwconv_relu_relu(self):
        raise NotImplementedError

    def _model_relu_dwconv(self):
        raise NotImplementedError

    def _model_dwconv_relu(self):
        raise NotImplementedError

    def _model_dwconv(self):
        raise NotImplementedError

