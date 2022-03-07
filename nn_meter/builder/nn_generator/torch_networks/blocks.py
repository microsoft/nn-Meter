# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import torch
import torch.nn as nn
from .operators import *
from .utils import get_inputs_by_shapes
from ..interface import BaseBlock
logging = logging.getLogger("nn-Meter")


class TorchBlock(BaseBlock):
    def test_block(self):
        pass
    
    def save_model(self, save_path):
        model = self.get_model()
        torch.onnx.export(
            model,
            get_inputs_by_shapes(self.input_tensor_shape),
            save_path,
            input_names=['input'],
            output_names=['output'],
            verbose=False,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
        )


class ConvBnRelu(TorchBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        class Model(nn.Module):
            def __init__(self, conv_op, bn_op, relu_op):
                super().__init__()
                self.conv = conv_op
                self.bn = bn_op
                self.relu = relu_op

            def call(self, inputs):
                x = self.conv(inputs)
                x = self.bn(x)
                x = self.relu(x)
                return x

        return Model(self.conv_op, self.bn_op, self.relu_op)


class DwConvBnRelu(TorchBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        class Model(nn.Module):
            def __init__(self, dwconv_op, bn_op, relu_op):
                super().__init__()
                self.dwconv = dwconv_op
                self.bn = bn_op
                self.relu = relu_op

            def call(self, inputs):
                x = self.dwconv(inputs)
                x = self.bn(x)
                x = self.relu(x)
                return x

        return Model(self.dwconv_op, self.bn_op, self.relu_op)
