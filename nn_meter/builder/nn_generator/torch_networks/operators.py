# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
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


class BN(BaseOperator):
    def get_model(self):
        cin = self.input_shape[0]
        return nn.BatchNorm2d(cin)


class Relu(BaseOperator):
    def get_model(self):
        return nn.ReLU()

