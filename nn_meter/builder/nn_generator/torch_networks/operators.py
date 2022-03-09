# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
from ..interface import BaseOperator

''' 
This file contains the torch implementation of operators
'''

#---------------------- convolution layer ----------------------#

class Conv(BaseOperator):
    def get_model(self):
        cin = self.input_shape[0]
        cout = cin if "COUT" not in self.config else self.config["COUT"]
        return nn.Conv2d(cin, cout, kernel_size=self.config["KERNEL_SIZE"], stride=self.config["STRIDES"], padding=1)

    def get_output_shape(self):
        cout = self.input_shape[0] if "COUT" not in self.config else self.config["COUT"]
        output_h = (self.input_shape[1] - 1) // self.config["STRIDES"] + 1
        output_w = (self.input_shape[2] - 1) // self.config["STRIDES"] + 1
        return [cout, output_h, output_w]


class DwConv(BaseOperator):
    def get_model(self):
        cin = self.input_shape[0]
        return nn.Conv2d(cin, cin, kernel_size=self.config["KERNEL_SIZE"], stride=self.config["STRIDES"], padding=1, groups=cin)

    def get_output_shape(self):
        cin = self.input_shape[0]
        output_h = (self.input_shape[1] - 1) // self.config["STRIDES"] + 1
        output_w = (self.input_shape[2] - 1) // self.config["STRIDES"] + 1
        return [cin, output_h, output_w]


class ConvTrans(BaseOperator):
    def get_model(self):
        cin = self.input_shape[0]
        cout = cin if "COUT" not in self.config else self.config["COUT"]
        return nn.ConvTranspose2d(cin, cout, kernel_size=self.config["KERNEL_SIZE"], stride=self.config["STRIDES"], padding=1)

    def get_output_shape(self):
        cout = self.input_shape[2] if "COUT" not in self.config else self.config["COUT"]
        return [self.input_shape[0] * self.config["STRIDES"], self.input_shape[1] * self.config["STRIDES"], cout]

#------------------ normalization and pooling ------------------#

class BN(BaseOperator):
    def get_model(self):
        cin = self.input_shape[0]
        return nn.BatchNorm2d(cin)


class GlobalAvgpool(BaseOperator):
    pass


class MaxPool(BaseOperator):
    def get_model(self):
        if "POOL_STRIDES" not in self.config:
            self.config["POOL_STRIDES"] = self.config["STRIDES"]
        return nn.MaxPool2d(self.config["KERNEL_SIZE"], self.config["POOL_STRIDES"], padding=1)

    def get_output_shape(self):
        if "POOL_STRIDES" not in self.config:
            self.config["POOL_STRIDES"] = self.config["STRIDES"]
        output_h = (self.input_shape[1] - 1) // self.config["POOL_STRIDES"] + 1
        output_w = (self.input_shape[2] - 1) // self.config["POOL_STRIDES"] + 1
        return [output_h, output_w, self.input_shape[0]]


class AvgPool(BaseOperator):
    def get_model(self):
        if "POOL_STRIDES" not in self.config:
            self.config["POOL_STRIDES"] = self.config["STRIDES"]
        return nn.AvgPool2d(self.config["KERNEL_SIZE"], self.config["POOL_STRIDES"], padding=1)

    def get_output_shape(self):
        if "POOL_STRIDES" not in self.config:
            self.config["POOL_STRIDES"] = self.config["STRIDES"]
        output_h = (self.input_shape[1] - 1) // self.config["POOL_STRIDES"] + 1
        output_w = (self.input_shape[2] - 1) // self.config["POOL_STRIDES"] + 1
        return [output_h, output_w, self.input_shape[0]]

#------------------------ other modules ------------------------#

class SE(BaseOperator):
    pass


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
    pass


class Add(BaseOperator):
    pass


class Concat(BaseOperator):
    pass


class Flatten(BaseOperator):
    pass


class Split(BaseOperator):
    def get_model(self):
        cin = self.input_shape[0]
        def func(inputs):
            return torch.split(inputs, [cin // 2, cin - cin // 2], dim=1)
        return func

    def get_output_shape(self):
        return [self.input_shape[0] // 2, self.input_shape[1], self.input_shape[2]]
