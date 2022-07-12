# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf
from .ops import *
from ..utils import *

def dense_layer(input, growth_rate, kernel_size, name = '', is_training = False, log = False):
    ''' build Dense block
    '''
    (h1, w1, cin) = input.shape.as_list()[1:4]
    x = batch_norm(input, is_training, opname=name + '.1')
    x = activation(x, activation='relu', opname=name + '.1')

    x = conv2d(x, 4 * growth_rate, 1, stride=1, opname=name + '.2')
    x = batch_norm(x, is_training, opname=name + '.2')
    x = activation(x, activation='relu', opname=name + '.2')

    x1 = conv2d(x, growth_rate, kernel_size, stride=1, opname=name + '.3')
    x2 = tf.concat([input, x1], axis=3)

    logs = {}
    if log:
        logs[name + '.1'] = add_to_log('bn-relu', cin, cin, 1, 1, h1, w1)
        logs[name + '.2'] = add_to_log('conv-bn-relu', cin, 4 * growth_rate, 1, 1, h1, w1)
        logs[name + '.3'] = add_to_log('conv', 4 * growth_rate, growth_rate, kernel_size, 1, h1, w1)
        logs[name + '.4'] = add_ele_to_log('concat', [input.shape.as_list()[1:4], x1.shape.as_list()[1:4]])
    return x2, logs


class DenseNet(object):
    def __init__(self, input, cfg, version = 18, sample = False, enable_out = False):
        ''' change channel number, kernel size
        '''
        self.input = input
        self.num_classes = cfg['n_classes']
        self.enable_out = enable_out
        self.dense_num = [6, 12, 24, 16]

        # fixed block channels and kernel size
        self.bcs = [32] * sum(self.dense_num)
        self.bks = [3] * sum(self.dense_num)

        # sampling block channels and kernel size
        self.cs = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bcs))
        self.ks = get_sampling_ks(cfg['sample_space']['kernelsize'], len(self.bks))

        if sample == True:
            self.ncs = [int(self.bcs[index] * self.cs[index]) for index in range(len(self.bcs))]
            self.nks = self.ks
        else:
            self.ncs = self.bcs
            self.nks = self.bks

        self.config = {}
        self.sconfig = '_'.join([str(x) for x in self.nks]) + '-' + '_'.join([str(x) for x in self.ncs])

        # build DenseNet model
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
        ''' build DenseNet model according to model config
        '''
        x = conv2d(self.input, 64, 7, opname='conv1', stride=2, padding='SAME')
        x = batch_norm(x, opname='conv1.bn')
        x = activation(x, 'relu', opname='conv1.relu')
        self.add_to_log('conv-bn-relu', 3, 64, 7, 2, 'layer1', self.input.shape.as_list()[1], self.input.shape.as_list()[2])

        (h, w) = x.shape.as_list()[1:3]
        x = max_pooling(x, 3, 2, opname = 'maxpool1')
        self.add_to_log('max-pool', 64, 64, 3, 2, 'layer2', h, w)  # bug: input size error

        layer_count = 3
        index = 0
        for layers in self.dense_num:
            for _ in range(layers):
                x, log = dense_layer(x, self.ncs[index], self.nks[index], name='layer' + str(layer_count), log=True)
                self.config.update(log)
                index += 1
                layer_count += 1

            (h, w, cin) = x.shape.as_list()[1:4]
            x = batch_norm(x, opname='conv' + str(layer_count) + '.bn')
            x = activation(x, 'relu', opname='conv' + str(layer_count) + '.relu')
            self.add_to_log('bn-relu', cin, cin, None, None, 'layer' + str(layer_count), h, w)
            layer_count += 1

            x = conv2d(x, cin // 2, 1, opname='conv' + str(layer_count), stride=1)
            self.add_to_log('conv', cin, cin // 2, 1, 1, 'layer' + str(layer_count), h, w)
            layer_count += 1

            (h1, w1, cin1) = x.shape.as_list()[1:4]
            x = avg_pooling(x, 2, 2, opname='conv' + str(layer_count) + 'avgpool')          
            self.add_to_log('avg-pool', cin1, cin1, 2, 2, 'layer' + str(layer_count), h1, w1) 
            layer_count += 1

        (h, w, cin) = x.shape.as_list()[1:4]
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = flatten(x)
        self.add_to_log('global-pool', cin, cin, None, None, 'layer' + str(layer_count), 1, 1)

        x = fc_layer(x, self.num_classes, opname='fc' + str(layer_count + 1))
        self.add_to_log('fc', cin, self.num_classes, None, None, 'layer' + str(layer_count + 1), None, None)

        return x
