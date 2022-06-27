# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import tensorflow as tf
import tensorflow.keras as keras
from .operators import *
from ..interface import BaseBlock
from .utils import get_inputs_by_shapes
logging = logging.getLogger("nn-Meter")


class TFBlock(BaseBlock):
    def __init__(self, config, batch_size = 1):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]
        self.batch_size = batch_size

    def test_block(self):
        import os, shutil
        from typing import List
        model_path = "./temp_model"
        model = self.get_model()
        model_output = model(get_inputs_by_shapes(self.input_tensor_shape))
        
        # check model save and reload
        keras.models.save_model(model, model_path)
        restore_model = keras.models.load_model(model_path)
        if isinstance(model_output, List):
            output_shape = [mod.shape for mod in model_output]
            restore_output_shape = [mod.shape for mod in restore_model(get_inputs_by_shapes(self.input_tensor_shape))]
        else:
            output_shape = model_output.shape
            restore_output_shape = restore_model(get_inputs_by_shapes(self.input_tensor_shape)).shape
        assert output_shape == restore_output_shape
        shutil.rmtree(model_path)

        # check model convert to tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(restore_model)
        tflite_model = converter.convert()
        open(model_path + '.tflite', 'wb').write(tflite_model)
        os.remove(model_path + '.tflite')
        logging.keyinfo("Testing block is success!")

    def save_model(self, save_path):
        model = self.get_model()
        keras.models.save_model(model, save_path)

    def build_model(self, ops):
        ''' convert a list of operators to keras model.
        '''
        class Model(keras.Model):
            def __init__(self, ops):
                super().__init__()
                self.ops = keras.Sequential(ops)

            def call(self, inputs):
                x = self.ops(inputs)
                return x

        model = Model(ops)
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model

    def get_model(self):
        raise NotImplementedError


class ConvBnRelu(TFBlock):
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


class ConvBnRelu6(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

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

        model = Model(self.conv_op, self.bn_op, self.relu6_op)
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model


class ConvBn(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op = bn_op.get_model()

    def get_model(self):
        return self.build_model([self.conv_op, self.bn_op])


class ConvRelu(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        conv_op = Conv(self.input_shape, config)
        self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        return self.build_model([self.conv_op, self.relu_op])


class ConvRelu6(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

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

        model = Model(self.conv_op, self.relu6_op)
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model


class ConvHswish(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

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

        model = Model(self.conv_op, self.hswish_op)
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model


class ConvBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        conv_op = Conv(self.input_shape, config)
        self.conv_op = conv_op.get_model()

    def get_model(self):
        return self.build_model([self.conv_op])


class ConvBnHswish(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

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

        model = Model(self.conv_op, self.bn_op, self.hswish_op)
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model


class ConvBnReluMaxPool(TFBlock):
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


class DwConvBn(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

        bn_op = BN(out_shape, config)
        self.bn_op = bn_op.get_model()

    def get_model(self):
        return self.build_model([self.dwconv_op, self.bn_op])


class DwConvRelu(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        return self.build_model([self.dwconv_op, self.relu_op])


class DwConvRelu6(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

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

        model = Model(self.dwconv_op, self.relu6_op)
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model


class DwConvBnRelu(TFBlock):
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


class DwConvBnRelu6(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

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

        model = Model(self.dwconv_op, self.bn_op, self.relu6_op)
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model


class DwConvBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        dwconv_op = DwConv(self.input_shape, config)
        self.dwconv_op = dwconv_op.get_model()

    def get_model(self):
        return self.build_model([self.dwconv_op])


class ConvBnHswish(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

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

        model = Model(self.dwconv_op, self.bn_op, self.hswish_op)
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model


class MaxPoolBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        maxpool_op = MaxPool(self.input_shape, config)
        self.maxpool_op = maxpool_op.get_model()

    def get_model(self):
        return self.build_model([self.maxpool_op])


class AvgPoolBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        avgpool_op = AvgPool(self.input_shape, config)
        self.avgpool_op = avgpool_op.get_model()

    def get_model(self):
        return self.build_model([self.avgpool_op])


class FCBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        self.config = config
        self.input_shape = [config["CIN"]]
        self.input_tensor_shape = [self.input_shape]
        self.batch_size = batch_size

        fc_op = FC(self.input_shape, config)
        self.fc_op = fc_op.get_model()

    def get_model(self):
        return self.build_model([self.fc_op])


class ConcatBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        self.config = config
        self.input_shape = [[config["HW"], config["HW"], cin]
                       for cin in [config['CIN1'], config['CIN2'], config['CIN3'], config['CIN4']]
                       if cin != 0]
        self.input_tensor_shape = self.input_shape
        self.batch_size = batch_size
        
        concat_op = Concat(self.input_shape, config)
        self.concat_op = concat_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, concat_op):
                super().__init__()
                self.concat = concat_op

            def call(self, inputs):
                return self.concat(inputs)

        model = Model(self.concat_op)
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model


class SplitBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        split_op = Split(self.input_shape, config)
        self.split_op = split_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, split_op):
                super().__init__()
                self.split = split_op

            def call(self, inputs):
                return self.split(inputs)

        model = Model(self.split_op)
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model


class ChannelShuffle(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

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

        model = Model()
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model


class SEBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        se_op = SE(self.input_shape, config)
        self.se_op = se_op.get_model()

    def get_model(self):
        return self.build_model([self.se_op])


class GlobalAvgPoolBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        globalavgpool_op = GlobalAvgpool(self.input_shape, config)
        self.globalavgpool_op = globalavgpool_op.get_model()

    def get_model(self):
        return self.build_model([self.globalavgpool_op])


class BnRelu(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        bn_op = BN(self.input_shape, config)
        self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
        relu_op = Relu(out_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        return self.build_model([self.bn_op, self.relu_op])


class BnBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        bn_op = BN(self.input_shape, config)
        self.bn_op = bn_op.get_model()

    def get_model(self):
        return self.build_model([self.bn_op])


class HswishBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        hswish_op = Hswish(self.input_shape, config)
        self.hswish_op = hswish_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, hswish_op):
                super().__init__()
                self.hswish = hswish_op

            def call(self, inputs):
                return self.hswish(inputs)

        model = Model(self.hswish_op)
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model


class ReluBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

        relu_op = Relu(self.input_shape, config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        return self.build_model([self.relu_op])


class AddRelu(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)

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

        model = Model(self.add_op, self.relu_op)
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model


class AddBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)
        
        add_op = Add(self.input_shape, config)
        self.add_op = add_op.get_model()

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, add_op):
                super().__init__()
                self.add = add_op

            def call(self, inputs):
                return self.add([inputs, inputs])

        model = Model(self.add_op)
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model


class GroupedConvBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)
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

        model = Model(self.cout, self.num_groups, self.config['KERNEL_SIZE'], self.config['STRIDES'])
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model


class MixedConvBlock(TFBlock):
    def __init__(self, config, batch_size = 1):
        super().__init__(config, batch_size)
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

        model = Model(self.cout, self.num_groups, self.config['STRIDES'])
        model(get_inputs_by_shapes(self.input_tensor_shape, self.batch_size))
        return model
