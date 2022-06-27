# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import numpy as np
import torch.nn as nn
from .utils import get_padding
from ..interface import BaseOperator

'''
This file contains the torch implementation of operators
'''

#---------------------- convolution layer ----------------------#

class Conv(BaseOperator):
    def get_model(self):
        cin = self.input_shape[0]
        cout = cin if "COUT" not in self.config else self.config["COUT"]
        padding = get_padding(self.config["KERNEL_SIZE"], self.config["STRIDES"], self.input_shape[1])
        return nn.Conv2d(cin, cout, kernel_size=self.config["KERNEL_SIZE"], stride=self.config["STRIDES"], padding=padding)

    def get_output_shape(self):
        cout = self.input_shape[0] if "COUT" not in self.config else self.config["COUT"]
        output_h = (self.input_shape[1] - 1) // self.config["STRIDES"] + 1
        output_w = (self.input_shape[2] - 1) // self.config["STRIDES"] + 1
        return [cout, output_h, output_w]


class DwConv(BaseOperator):
    def get_model(self):
        cin = self.input_shape[0]
        padding = get_padding(self.config["KERNEL_SIZE"], self.config["STRIDES"], self.input_shape[1])
        return nn.Conv2d(cin, cin, kernel_size=self.config["KERNEL_SIZE"], stride=self.config["STRIDES"], padding=padding, groups=cin)

    def get_output_shape(self):
        cin = self.input_shape[0]
        output_h = (self.input_shape[1] - 1) // self.config["STRIDES"] + 1
        output_w = (self.input_shape[2] - 1) // self.config["STRIDES"] + 1
        return [cin, output_h, output_w]


class ConvTrans(BaseOperator):
    def get_model(self):
        cin = self.input_shape[0]
        cout = cin if "COUT" not in self.config else self.config["COUT"]
        padding = get_padding(self.config["KERNEL_SIZE"], self.config["STRIDES"], self.input_shape[1])
        output_padding = self.config["STRIDES"] + 2 * padding - self.config["KERNEL_SIZE"]
        return nn.ConvTranspose2d(cin, cout, kernel_size=self.config["KERNEL_SIZE"], stride=self.config["STRIDES"], padding=padding, output_padding=output_padding)

    def get_output_shape(self):
        cout = self.input_shape[0] if "COUT" not in self.config else self.config["COUT"]
        return [cout, self.input_shape[1] * self.config["STRIDES"], self.input_shape[2] * self.config["STRIDES"]]

#------------------ normalization and pooling ------------------#

class BN(BaseOperator):
    def get_model(self):
        cin = self.input_shape[0]
        return nn.BatchNorm2d(cin)


class GlobalAvgpool(BaseOperator):
    def get_model(self):
        return nn.AdaptiveAvgPool2d([1, 1])
    
    def get_output_shape(self):
        cin = self.input_shape[0]
        return [cin, 1, 1]


class MaxPool(BaseOperator):
    def get_model(self):
        if "POOL_STRIDES" not in self.config:
            self.config["POOL_STRIDES"] = self.config["STRIDES"]
        padding = get_padding(self.config["KERNEL_SIZE"], self.config["POOL_STRIDES"], self.input_shape[1])
        return nn.MaxPool2d(self.config["KERNEL_SIZE"], self.config["POOL_STRIDES"], padding=padding)

    def get_output_shape(self):
        cin = self.input_shape[0]
        if "POOL_STRIDES" not in self.config:
            self.config["POOL_STRIDES"] = self.config["STRIDES"]
        output_h = (self.input_shape[1] - 1) // self.config["POOL_STRIDES"] + 1
        output_w = (self.input_shape[2] - 1) // self.config["POOL_STRIDES"] + 1
        return [cin, output_h, output_w]


class AvgPool(BaseOperator):
    def get_model(self):
        if "POOL_STRIDES" not in self.config:
            self.config["POOL_STRIDES"] = self.config["STRIDES"]
        padding = get_padding(self.config["KERNEL_SIZE"], self.config["POOL_STRIDES"], self.input_shape[1])
        return nn.AvgPool2d(self.config["KERNEL_SIZE"], self.config["POOL_STRIDES"], padding=padding)

    def get_output_shape(self):
        cin = self.input_shape[0]
        if "POOL_STRIDES" not in self.config:
            self.config["POOL_STRIDES"] = self.config["STRIDES"]
        output_h = (self.input_shape[1] - 1) // self.config["POOL_STRIDES"] + 1
        output_w = (self.input_shape[2] - 1) // self.config["POOL_STRIDES"] + 1
        return [cin, output_h, output_w]

#------------------------ other modules ------------------------#

class SE(BaseOperator):
    def get_model(self):
        class SE(nn.Module):
            def __init__(self, num_channels, se_ratio=0.25):
                super().__init__()
                mid_channels = int(num_channels * se_ratio)
                self.squeeze = nn.Conv2d(num_channels, mid_channels, kernel_size=1, padding=0)
                self.relu = nn.ReLU()
                self.excite = nn.Conv2d(mid_channels, num_channels, kernel_size=1, padding=0)
                self.hswish = nn.Hardswish()

            def _scale(self, x):
                x = x.mean(3, keepdim=True).mean(2, keepdim=True)
                x = self.squeeze(x)
                x = self.relu(x)
                x = self.excite(x)
                x = self.hswish(x)
                return x

            def forward(self, x):
                scale = self._scale(x)
                return scale * x
        return SE(self.input_shape[0])


class FC(BaseOperator):
    def get_model(self):
        cin = self.input_shape[0]
        cout = self.input_shape[0] if "COUT" not in self.config else self.config["COUT"]
        return nn.Linear(cin, cout)

    def get_output_shape(self):
        cout = self.input_shape[0] if "COUT" not in self.config else self.config["COUT"]
        return [cout] + self.input_shape[1:]

#-------------------- activation function --------------------#

class Relu(BaseOperator):
    def get_model(self):
        return nn.ReLU()


class Relu6(BaseOperator):
    def get_model(self):
        return nn.ReLU()


class Sigmoid(BaseOperator):
    def get_model(self):
        return nn.Sigmoid()


class Hswish(BaseOperator):
    def get_model(self):
        return nn.Hardswish()

#---------------------- basic operation ----------------------#

class Reshape(BaseOperator):
    def get_model(self):
        if len(self.input_shape) == 3:
            self.output_shape = [self.input_shape[1], self.input_shape[2], self.input_shape[0]]
            def func(inputs):
                return torch.reshape(inputs, [1] + self.output_shape)
        else:
            self.output_shape = [1, 2, int(self.input_shape[0] / 2)]
            def func(inputs):
                return torch.reshape(inputs, [1] + self.output_shape)
        return func

    def get_output_shape(self):
        return self.output_shape


class Add(BaseOperator):
    def get_model(self):
        def func(inputs):
            return torch.add(inputs[0], inputs[1])
        return func

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
        def func(inputs):
            return torch.cat(tuple(inputs), dim=1)
        return func

    def get_output_shape(self):
        if len(self.input_shape) > 1 and type(self.input_shape[0]) == list: # e.g. [[3, 28, 28], [5, 28, 28]] -> [8, 28, 28]
            output_shape = [sum([i[0] for i in self.input_shape])] + self.input_shape[0][1:]
        elif len(self.input_shape) == 3: # e.g. [4, 28, 28] -> [8, 28, 28]
            output_shape = [self.input_shape[0] * 2] + self.input_shape[1:]
        else: # e.g. [1024] -> [2048]
            output_shape = [self.input_shape[0] * 2]
        return output_shape

    def get_is_two_inputs(self):
        return True


class Flatten(BaseOperator):
    def get_model(self):
        return nn.Flatten()

    def get_output_shape(self):
        return [int(np.prod(self.input_shape))]


class Split(BaseOperator):
    def get_model(self):
        cin = self.input_shape[0]
        def func(inputs):
            return torch.split(inputs, [cin // 2, cin - cin // 2], dim=1)
        return func

    def get_output_shape(self):
        return [self.input_shape[0] // 2, self.input_shape[1], self.input_shape[2]]
