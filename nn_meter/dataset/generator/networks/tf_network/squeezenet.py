# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf
from .ops import *
from ..utils import *


def fire_block(input, squeeze_cin, cout, kernel_size, name = '', log = False):
    (h1, w1) = input.shape.as_list()[1:3]
    in_features = int(input.get_shape()[3])

    x = conv2d(input, squeeze_cin, 1, stride=1, opname=name + '.1')
    x1 = activation(x, activation='relu', opname=name + '.1')
    x = conv2d(x1, cout, 1, stride=1, opname=name + '.2')
    x2 = activation(x, activation='relu', opname=name + '.2')
    x = conv2d(x1, cout, kernel_size, stride=1, opname=name + '.3')
    x3 = activation(x, activation='relu', opname=name + '.3')
    x = tf.concat([x2, x3], axis=3)

    logs = {}
    if log:
        logs[name + '.1'] = add_to_log('conv-relu', in_features, squeeze_cin, 1, 1, h1, w1)
        logs[name + '.2'] = add_to_log('conv-relu', squeeze_cin, cout, 1, 1, h1, w1)
        logs[name + '.3'] = add_to_log('conv-relu', squeeze_cin, cout, kernel_size, 1, h1, w1)
        logs[name + '.4'] = add_ele_to_log('concat', [x2.shape.as_list()[1:4], x3.shape.as_list()[1:4]])
    return x, logs


class SqueezeNet(object):
    def __init__(self, input, cfg, version = None, sample = False):
        ''' change channel number, kernel size
        '''
        self.input = input
        self.num_classes = cfg['n_classes']

        # fixed block channels and kernel size
        self.bmcs = [16, 16, 32, 32, 48, 48, 64, 64]
        self.bcs = [64, 64, 128, 128, 192, 192, 256, 256]
        self.bks = [3] * 8

        # sampling block channels and kernel size
        self.cs = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bcs))
        self.ks = get_sampling_ks(cfg['sample_space']['kernelsize'], len(self.bks))
       
        if sample == True:
            self.ncs = [int(self.bcs[index] * self.cs[index]) for index in range(len(self.bcs))]
            self.nmcs = [int(self.bmcs[index] * self.cs[index]) for index in range(len(self.bmcs))]
            self.nks = self.ks
        else:
            self.ncs = self.bcs
            self.nmcs = self.bmcs
            self.nks = self.bks

        self.config = {}
        self.sconfig = '_'.join([str(x) for x in self.nks]) + '-' + '_'.join([str(x) for x in self.ncs])

        # build SqueezeNet model
        self.out = self.build()

    def add_to_log(self, op, cin, cout, ks, stride, layername, inputh, inputw):
        self.config[layername] = {
            'op': op,
            'cin': cin,
            'cout': cout,
            'ks': ks,
            'stride': stride,
            'inputh': inputh,
            'inputw': inputw
        }

    def build(self):
        ''' build SqueezeNet model according to model config
        '''
        x = conv2d(self.input, 96, 7, opname='conv1', stride=2, padding='SAME')
        x = activation(x, 'relu', opname='conv1.relu')
        self.add_to_log('conv-relu', 3, 96, 7, 2, 'layer1', self.input.shape.as_list()[1], self.input.shape.as_list()[2])

        r = [3, 4, 1]

        layer_count = 2
        index = 0
        for layers in r:
            (h, w, cin) = x.shape.as_list()[1:4]
            x = max_pooling(x, 3, 2, opname = 'conv1')
            self.add_to_log('max-pool', cin, cin, 3, 2, 'layer' + str(layer_count), h, w)

            layer_count += 1
            for _ in range(layers):
                x, log = fire_block(x, self.nmcs[index], self.ncs[index], self.nks[index], name='layer' + str(layer_count), log=True)
                self.config.update(log)
                layer_count += 1
                index += 1

        (h, w, lastcin) = x.shape.as_list()[1:4]
        x = conv2d(x, 512, 1, opname='conv' + str(layer_count), stride=1, padding='SAME')
        x = activation(x, 'relu', opname='conv' + str(layer_count) + '.relu')
        self.add_to_log('conv-relu', lastcin, 512, 1, 1, 'layer' + str(layer_count), h, w)

        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = flatten(x)
        self.add_to_log('global-pool', 512, 512, None, None, 'layer' + str(layer_count + 1), 1, 1)

        x = fc_layer(x, self.num_classes, opname='fc3')
        self.add_to_log('fc', 512, self.num_classes, None, None, 'layer' + str(layer_count + 2), None, None)

        return x