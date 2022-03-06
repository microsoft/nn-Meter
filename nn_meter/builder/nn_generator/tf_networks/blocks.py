# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import tensorflow as tf
import tensorflow.keras as keras
from .operators import *
from ..interface import BaseBlock
from nn_meter.builder.utils import get_inputs_by_shapes
logging = logging.getLogger("nn-Meter")


class ConvBnRelu(BaseBlock):
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
        class Model(keras.Model):
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


class ConvBnRelu6(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        relu6_op = Relu6(out_shape, config)
        self.relu6_op = relu6_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, conv_op, bn_op, relu6_op):
                super().__init__()
                self.conv = conv_op
                self.bn = bn_op
                self.relu6 = relu6_op

            def call(self, inputs):
                x = self.conv(inputs)
                x = self.bn(x)
                x = self.relu6(x)
                return x

        return Model(self.conv_op, self.bn_op, self.relu6_op)


class ConvBn(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op = bn_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, conv_op, bn_op):
                super().__init__()
                self.conv = conv_op
                self.bn = bn_op

            def call(self, inputs):
                x = self.conv(inputs)
                x = self.bn(x)
                return x

        return Model(self.conv_op, self.bn_op)


class ConvRelu(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, conv_op, relu_op):
                super().__init__()
                self.conv = conv_op
                self.relu = relu_op

            def call(self, inputs):
                x = self.conv(inputs)
                x = self.relu(x)
                return x

        return Model(self.conv_op, self.relu_op)


class ConvRelu6(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()
        
        relu6_op = Relu6(out_shape, config)
        self.relu6_op = relu6_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, conv_op, relu6_op):
                super().__init__()
                self.conv = conv_op
                self.relu6 = relu6_op

            def call(self, inputs):
                x = self.conv(inputs)
                x = self.relu6(x)
                return x

        return Model(self.conv_op, self.relu6_op)


class ConvHswish(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()
        
        hswish_op = Hswish(out_shape, config)
        self.hswish_op = hswish_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, conv_op, hswish_op):
                super().__init__()
                self.conv = conv_op
                self.hswish = hswish_op

            def call(self, inputs):
                x = self.conv(inputs)
                x = self.hswish(x)
                return x

        return Model(self.conv_op, self.hswish_op)


class ConvBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        conv_op = Conv(self.input_shape, config)
        self.conv_op = conv_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, conv_op):
                super().__init__()
                self.conv = conv_op

            def call(self, inputs):
                return self.conv(inputs)

        return Model(self.conv_op)


class ConvBnHswish(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        hswish_op = Hswish(out_shape, config)
        self.hswish_op = hswish_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, conv_op, bn_op, hswish_op):
                super().__init__()
                self.conv = conv_op
                self.bn = bn_op
                self.hswish = hswish_op

            def call(self, inputs):
                x = self.conv(inputs)
                x = self.bn(x)
                x = self.hswish(x)
                return x

        return Model(self.conv_op, self.bn_op, self.hswish_op)


class ConvBnReluMaxPool(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op, out_shape = relu_op.get_model(), relu_op.get_output_shape()
        
        maxpool_op = MaxPool(out_shape, config)
        self.maxpool_op = maxpool_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, conv_op, bn_op, relu_op, maxpool_op):
                super().__init__()
                self.conv = conv_op
                self.bn = bn_op
                self.relu = relu_op
                self.maxpool = maxpool_op

            def call(self, inputs):
                x = self.conv(inputs)
                x = self.bn(x)
                x = self.relu(x)
                x = self.maxpool(x)
                return x

        return Model(self.conv_op, self.bn_op, self.relu_op, self.maxpool_op)


class DwConvBn(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op = bn_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, dwconv_op, bn_op):
                super().__init__()
                self.dwconv = dwconv_op
                self.bn = bn_op      

            def call(self, inputs):
                x = self.dwconv(inputs)
                x = self.bn(x)
                return x

        return Model(self.dwconv_op, self.bn_op)


class DwConvRelu(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, dwconv_op, relu_op):
                super().__init__()
                self.dwconv = dwconv_op
                self.relu = relu_op

            def call(self, inputs):
                x = self.dwconv(inputs)
                x = self.relu(x)
                return x

        return Model(self.dwconv_op, self.relu_op)


class DwConvRelu6(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()
        
        relu6_op = Relu6(out_shape, config)
        self.relu6_op = relu6_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, dwconv_op, relu6_op):
                super().__init__()
                self.dwconv = dwconv_op
                self.relu6 = relu6_op

            def call(self, inputs):
                x = self.dwconv(inputs)
                x = self.relu6(x)
                return x

        return Model(self.dwconv_op, self.relu6_op)


class DwConvBnRelu(BaseBlock):
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
        class Model(keras.Model):
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


class DwConvBnRelu6(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        relu6_op = Relu6(out_shape, config)
        self.relu6_op = relu6_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, dwconv_op, bn_op, relu6_op):
                super().__init__()
                self.dwconv = dwconv_op
                self.bn = bn_op
                self.relu6 = relu6_op

            def call(self, inputs):
                x = self.dwconv(inputs)
                x = self.bn(x)
                x = self.relu6(x)
                return x

        return Model(self.dwconv_op, self.bn_op, self.relu6_op)


class DwConvBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op = dwconv_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, dwconv_op):
                super().__init__()
                self.dwconv = dwconv_op

            def call(self, inputs):
                return self.dwconv(inputs)

        return Model(self.dwconv_op)


class ConvBnHswish(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        hswish_op = Hswish(out_shape, config)
        self.hswish_op = hswish_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, dwconv_op, bn_op, hswish_op):
                super().__init__()
                self.dwconv = dwconv_op
                self.bn = bn_op
                self.hswish = hswish_op

            def call(self, inputs):
                x = self.dwconv(inputs)
                x = self.bn(x)
                x = self.hswish(x)
                return x

        return Model(self.dwconv_op, self.bn_op, self.hswish_op)


class MaxPoolBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        maxpool_op = MaxPool(self.input_shape, config)
        self.maxpool_op = maxpool_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, maxpool_op):
                super().__init__()
                self.maxpool = maxpool_op

            def call(self, inputs):
                return self.maxpool(inputs)

        return Model(self.maxpool_op)


class AvgPoolBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        avgpool_op = AvgPool(self.input_shape, config)
        self.avgpool_op = avgpool_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, avgpool_op):
                super().__init__()
                self.avgpool = avgpool_op

            def call(self, inputs):
                return self.avgpool(inputs)

        return Model(self.avgpool_op)


class FCBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        fc_op = FC(self.input_shape, config)
        self.fc_op = fc_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, fc_op):
                super().__init__()
                self.fc = fc_op

            def call(self, inputs):
                return self.fc(inputs)

        return Model(self.fc_op)


class ConcatBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [[config["HW"], config["HW"], cin]
                       for cin in [config['CIN1'], config['CIN2'], config['CIN3'], config['CIN4']]
                       if cin != 0]
        self.input_tensor_shape = self.input_shape
        
        concat_op = Concat(self.input_shape, config)
        self.concat_op = concat_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, concat_op):
                super().__init__()
                self.concat = concat_op

            def call(self, inputs):
                return self.concat(inputs)

        return Model(self.concat_op)


class SplitBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        split_op = Split(self.input_shape, config)
        self.split_op = split_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, split_op):
                super().__init__()
                self.split = split_op

            def call(self, inputs):
                return self.split(inputs)

        return Model(self.split_op)


class ChannelShuffle(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

    def get_model(self):
        class Model(keras.Model):
            def __init__(self):
                super().__init__()

            def call(self, inputs):
                _, h, w, c = inputs.get_shape().as_list()
                x = tf.reshape(inputs, [-1, h, w, 2, c // 2])
                x = tf.transpose(x, (0, 1, 2, 4, 3))
                x = tf.reshape(x, [-1, h, w, c])
                return x

        return Model()


class SEBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        se_op = SE(self.input_shape, config)
        self.se_op = se_op.get_model()

    def get_model(self):
        class Model(tf.keras.Model):
            def __init__(self, se_op):
                super().__init__()
                self.se = se_op

            def call(self, inputs):
                return self.se(inputs)

        return Model(self.se_op)


class GlobalAvgPoolBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        globalavgpool_op = GlobalAvgpool(self.input_shape, config)
        self.globalavgpool_op = globalavgpool_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, globalavgpool_op):
                super().__init__()
                self.globalavgpool = globalavgpool_op

            def call(self, inputs):
                return self.globalavgpool(inputs)

        return Model(self.globalavgpool_op)


class BnRelu(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        bn_op = BN(self.input_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, bn_op, relu_op):
                super().__init__()
                self.bn = bn_op
                self.relu = relu_op

            def call(self, inputs):
                x = self.bn(inputs)
                x = self.relu(x)
                return x

        return Model(self.bn_op, self.relu_op)


class BnBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        bn_op = BN(self.input_shape, config)
        self.bn_op = bn_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, bn_op):
                super().__init__()
                self.bn = bn_op

            def call(self, inputs):
                return self.bn(inputs)

        return Model(self.bn_op)


class HswishBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        hswish_op = Hswish(self.input_shape, config)
        self.hswish_op = hswish_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, hswish_op):
                super().__init__()
                self.hswish = hswish_op

            def call(self, inputs):
                return self.hswish(inputs)

        return Model(self.hswish_op)


class ReluBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        relu_op = Relu(self.input_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, relu_op):
                super().__init__()
                self.relu = relu_op

            def call(self, inputs):
                return self.relu(inputs)

        return Model(self.relu_op)


class AddRelu(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

        add_op = Add(self.input_shape, config)
        self.add_op, out_shape = add_op.get_model(), add_op.get_output_shape()

        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, add_op, relu_op):
                super().__init__()
                self.add = add_op
                self.relu = relu_op

            def call(self, inputs):
                x = self.add([inputs, inputs])
                x = self.relu(x)
                return x

        return Model(self.add_op, self.relu_op)


class AddBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]
        
        add_op = Add(self.input_shape, config)
        self.add_op = add_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, add_op):
                super().__init__()
                self.add = add_op

            def call(self, inputs):
                return self.add([inputs, inputs])

        return Model(self.add_op)


class GroupedConvBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]
        self.cout = self.input_shape[2] if "COUT" not in config else config["COUT"]
        self.num_groups = config['NUM_GROUPS']

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, cout, num_groups, kernel_size, strides):
                super().__init__()
                self.cout = cout
                self.num_groups = num_groups
                self.kernel_size = kernel_size
                self.strides = strides

            def call(self, inputs):
                x = [keras.layers.Conv2D(
                        filters=self.cout // self.num_groups,
                        kernel_size=self.kernel_size,
                        strides=self.strides,
                        padding="same",
                    )(x) for x in tf.split(inputs, self.num_groups, axis=3)
                ]
                return tf.concat(x, axis=3)    

        return Model(self.cout, self.num_groups, self.config['KERNEL_SIZE'], self.config['STRIDES'])


class MixedConvBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]
        self.cout = self.input_shape[2] if "COUT" not in config else config["COUT"]
        self.num_groups = config['NUM_GROUPS']

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, cout, num_groups, strides):
                super().__init__()
                self.cout = cout
                self.num_groups = num_groups
                self.strides = strides

            def call(self, inputs):
                x = [keras.layers.Conv2D(
                        filters=self.cout // self.num_groups,
                        kernel_size=i * 2 + 3,
                        strides=self.strides,
                        padding="same",
                    )(x) for i, x in zip(range(self.num_groups), tf.split(inputs, self.num_groups, axis=3))
                ]
                return tf.concat(x, axis=3)    

        return Model(self.cout, self.num_groups, self.config['STRIDES'])
