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
    def __init__(self, config, batch_size = 1):
        self.config = config
        self.input_shape = [config["CIN"], config["HW"], config["HW"]]
        self.input_tensor_shape = [self.input_shape]
        self.batch_size = batch_size

    def test_block(self):
        input_data = get_inputs_by_shapes(self.input_tensor_shape, self.batch_size)
        model = self.get_model()
        model.eval()
        model(input_data)
        logging.keyinfo("Testing block is success!")

    def save_model(self, save_path):
        model = self.get_model()
        model.eval()
        torch.onnx.export(
            model,
            get_inputs_by_shapes(self.input_tensor_shape, self.batch_size),
            save_path,
            input_names=['input'],
            output_names=['output'],
            verbose=False,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
        )

    def build_model(self, ops):
        ''' convert a list of operators to torch model.
        '''
        class Model(nn.Module):
            def __init__(self, ops):
                super().__init__()
                self.layers = nn.Sequential(*ops)

            def forward(self, inputs):
                x = self.layers(inputs)
                return x

        return Model(ops)

    def get_model(self):
        raise NotImplementedError


class ConvBnRelu(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        return self.build_model([self.conv_op, self.bn_op, self.relu_op])


class ConvBnRelu6(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        relu6_op = Relu6(out_shape, config)
        self.relu6_op = relu6_op.get_model()

    def get_model(self):
        return self.build_model([self.conv_op, self.bn_op, self.relu6_op])


class ConvBn(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op = bn_op.get_model()

    def get_model(self):
        return self.build_model([self.conv_op, self.bn_op])


class ConvRelu(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        return self.build_model([self.conv_op, self.relu_op])


class ConvRelu6(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()
        
        relu6_op = Relu6(out_shape, config)
        self.relu6_op = relu6_op.get_model()

    def get_model(self):
        return self.build_model([self.conv_op, self.relu6_op])


class ConvHswish(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()
        
        hswish_op = Hswish(out_shape, config)
        self.hswish_op = hswish_op.get_model()

    def get_model(self):
        return self.build_model([self.conv_op, self.hswish_op])


class ConvBlock(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        conv_op = Conv(self.input_shape, config)
        self.conv_op = conv_op.get_model()

    def get_model(self):
        return self.build_model([self.conv_op])


class ConvBnHswish(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        hswish_op = Hswish(out_shape, config)
        self.hswish_op = hswish_op.get_model()

    def get_model(self):
        return self.build_model([self.conv_op, self.bn_op, self.hswish_op])


class ConvBnReluMaxPool(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op, out_shape = relu_op.get_model(), relu_op.get_output_shape()
        
        maxpool_op = MaxPool(out_shape, config)
        self.maxpool_op = maxpool_op.get_model()

    def get_model(self):
        return self.build_model([self.conv_op, self.bn_op, self.relu_op, self.maxpool_op])


class DwConvBn(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op = bn_op.get_model()

    def get_model(self):
        return self.build_model([self.dwconv_op, self.bn_op])


class DwConvRelu(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        return self.build_model([self.dwconv_op, self.relu_op])


class DwConvRelu6(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()
        
        relu6_op = Relu6(out_shape, config)
        self.relu6_op = relu6_op.get_model()

    def get_model(self):
        return self.build_model([self.dwconv_op, self.relu6_op])


class DwConvBnRelu(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        return self.build_model([self.dwconv_op, self.bn_op, self.relu_op])


class DwConvBnRelu6(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        relu6_op = Relu6(out_shape, config)
        self.relu6_op = relu6_op.get_model()

    def get_model(self):
        return self.build_model([self.dwconv_op, self.bn_op, self.relu6_op])


class DwConvBlock(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op = dwconv_op.get_model()

    def get_model(self):
        return self.build_model([self.dwconv_op])


class ConvBnHswish(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        hswish_op = Hswish(out_shape, config)
        self.hswish_op = hswish_op.get_model()

    def get_model(self):
        return self.build_model([self.dwconv_op, self.bn_op, self.hswish_op])


class MaxPoolBlock(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        maxpool_op = MaxPool(self.input_shape, config)
        self.maxpool_op = maxpool_op.get_model()

    def get_model(self):
        return self.build_model([self.maxpool_op])


class AvgPoolBlock(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        avgpool_op = AvgPool(self.input_shape, config)
        self.avgpool_op = avgpool_op.get_model()

    def get_model(self):
        return self.build_model([self.avgpool_op])


class FCBlock(TorchBlock):
    def __init__(self, config, batch_size = 1):
        self.config = config
        self.input_shape = [config["CIN"]]
        self.input_tensor_shape = [self.input_shape]
        self.batch_size = batch_size

        fc_op = FC(self.input_shape, config)
        self.fc_op = fc_op.get_model()

    def get_model(self):
        return self.build_model([self.fc_op])


class ConcatBlock(TorchBlock):
    def __init__(self, config, batch_size = 1):
        self.config = config
        self.input_shape = [[cin, config["HW"], config["HW"]]
                       for cin in [config['CIN1'], config['CIN2'], config['CIN3'], config['CIN4']]
                       if cin != 0]
        self.input_tensor_shape = self.input_shape
        self.batch_size = batch_size
        
        concat_op = Concat(self.input_shape, config)
        self.concat_op = concat_op.get_model()

    def get_model(self):
        class Model(nn.Module):
            def __init__(self, concat_op):
                super().__init__()
                self.concat = concat_op

            def forward(self, inputs):
                return self.concat(inputs)

        return Model(self.concat_op)


class SplitBlock(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        split_op = Split(self.input_shape, config)
        self.split_op = split_op.get_model()

    def get_model(self):
        class Model(nn.Module):
            def __init__(self, split_op):
                super().__init__()
                self.split = split_op

            def forward(self, inputs):
                return self.split(inputs)

        return Model(self.split_op)


class ChannelShuffle(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

    def get_model(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, inputs):
                _, c, h, w = list(inputs.shape)
                x = torch.reshape(inputs, [-1, c // 2, 2, h, w])
                x = torch.transpose(x, 2, 1)
                x = torch.reshape(x, [-1, c, h, w])
                return x

        return Model()


class SEBlock(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        se_op = SE(self.input_shape, config)
        self.se_op = se_op.get_model()

    def get_model(self):
        return self.build_model([self.se_op])


class GlobalAvgPoolBlock(TorchBlock):
    def __init__(self, config, batch_size = 1):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]
        self.batch_size = batch_size

        globalavgpool_op = GlobalAvgpool(self.input_shape, config)
        self.globalavgpool_op = globalavgpool_op.get_model()

    def get_model(self):
        return self.build_model([self.globalavgpool_op])


class BnRelu(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        bn_op = BN(self.input_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        return self.build_model([self.bn_op, self.relu_op])


class BnBlock(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        bn_op = BN(self.input_shape, config)
        self.bn_op = bn_op.get_model()

    def get_model(self):
        return self.build_model([self.bn_op])


class HswishBlock(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        hswish_op = Hswish(self.input_shape, config)
        self.hswish_op = hswish_op.get_model()

    def get_model(self):
        return self.build_model([self.hswish_op])


class ReluBlock(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        relu_op = Relu(self.input_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        return self.build_model([self.relu_op])


class AddRelu(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        add_op = Add(self.input_shape, config)
        self.add_op, out_shape = add_op.get_model(), add_op.get_output_shape()

        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        class Model(nn.Module):
            def __init__(self, add_op, relu_op):
                super().__init__()
                self.add = add_op
                self.relu = relu_op

            def forward(self, inputs):
                x = self.add([inputs, inputs])
                x = self.relu(x)
                return x

        return Model(self.add_op, self.relu_op)


class AddBlock(TorchBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)
        
        add_op = Add(self.input_shape, config)
        self.add_op = add_op.get_model()

    def get_model(self):
        class Model(nn.Module):
            def __init__(self, add_op):
                super().__init__()
                self.add = add_op

            def forward(self, inputs):
                return self.add([inputs, inputs])

        return Model(self.add_op)
