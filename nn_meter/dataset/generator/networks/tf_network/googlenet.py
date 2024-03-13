# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf
from .ops import *
from ..utils import *

def inception_block(input, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, name = '', istraining = False, is_slice = False, log = False):
    (h1, w1) = input.shape.as_list()[1:3]
    in_features = int(input.get_shape()[3])

    x = conv2d(input, ch1x1, 1, stride=1, opname=name + '.1')
    x = batch_norm(x, istraining, opname=name + '.1')
    x1 = activation(x, activation='relu', opname=name + '.1')

    x = conv2d(input, ch3x3red, 1, stride=1, opname=name + '.2.1')
    x = batch_norm(x, istraining, opname=name + '.2.1')
    x = activation(x, activation='relu', opname=name + '.2.1')
    x = conv2d(x, ch3x3, 3, stride=1, opname=name + '.2.2')
    x = batch_norm(x, istraining, opname=name + '.2.2')
    x2 = activation(x, activation='relu', opname=name + '.2.2')

    x = conv2d(input, ch5x5red, 1, stride=1, opname=name + '.3.1')
    x = batch_norm(x, istraining, opname=name + '.3.1')
    x = activation(x, activation='relu', opname=name + '.3.1')
    x = conv2d(x, ch5x5, 5, stride=1, opname=name + '.3.2')
    x = batch_norm(x, istraining, opname=name + '.3.2')
    x3 = activation(x, activation='relu', opname=name + '.3.2')

    x = max_pooling(input, 3, 1, opname=name + '.4')
    x = conv2d(x, pool_proj, 1, stride=1, opname=name + '.4')
    x = batch_norm(x, istraining, opname=name + '.4')
    x4 = activation(x, activation='relu', opname=name + '.4')

    out = tf.concat([x1, x2, x3, x4], axis=3)

    logs = {}
    if log:
        logs[name + '.1'] = add_to_log('conv-bn-relu', in_features, ch1x1, 1, 1, h1, w1)
        logs[name + '.2.1'] = add_to_log('conv-bn-relu', in_features, ch3x3red, 1, 1, h1, w1)
        logs[name + '.2.2'] = add_to_log('conv-bn-relu', ch3x3red, ch3x3, 3, 1, h1, w1)
        logs[name + '.3.1'] = add_to_log('conv-bn-relu', in_features, ch5x5red, 1, 1, h1, w1)
        logs[name + '.3.2'] = add_to_log('conv-bn-relu', ch5x5red, ch5x5, 5, 1, h1, w1)
        logs[name + '.4.1'] = add_to_log('max-pool', in_features, in_features, 3, 2, h1, w1)
        logs[name + '.4.2'] = add_to_log('conv-bn-relu', in_features, pool_proj, 1, 1, h1, w1)
        logs[name + '.5'] = add_ele_to_log('concat', [x1.shape.as_list()[1:4], x2.shape.as_list()[1:4], x3.shape.as_list()[1:4], x4.shape.as_list()[1:4]])

    return out, logs


class GoogleNet(object):
    def __init__(self, input, cfg, version = None, sample = False):
        ''' change channel number, kernel size
        '''
        self.input = input
        self.num_classes = cfg['n_classes']

        # fixed block channels and kernel size
        self.bc1s = [64, 128, 192, 160, 128, 112, 256, 256, 384]
        self.bc2ms = [96, 128, 96, 112, 128, 144, 160, 160, 192]
        self.bc2s = [128, 192, 208, 224, 256, 288, 320, 320, 384]
        self.bc3ms = [16, 32, 16, 24, 24, 32, 32, 32, 48]
        self.bc3s = [32, 96, 48, 64, 64, 64, 128, 128, 128]
        self.bc4s = [32, 64, 64, 64, 64, 64, 128, 128, 128]

        # sampling block channels and kernel size
        self.cs1 = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bc1s))
        self.cs2 = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bc1s))
        self.cs3 = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bc1s))
        self.cs4 = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bc1s))
        self.cs5 = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bc1s))
        self.cs6 = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bc1s))

        if sample == True:
            self.nc1s = [int(self.bc1s[index] * self.cs1[index]) for index in range(len(self.bc1s))]
            self.nc2ms = [int(self.bc2ms[index] * self.cs2[index]) for index in range(len(self.bc2ms))]
            self.nc2s = [int(self.bc2s[index] * self.cs3[index]) for index in range(len(self.bc2s))]
            self.nc3ms = [int(self.bc3ms[index] * self.cs4[index]) for index in range(len(self.bc3ms))]
            self.nc3s = [int(self.bc3s[index] * self.cs5[index]) for index in range(len(self.bc3s))]
            self.nc4s = [int(self.bc4s[index] * self.cs6[index]) for index in range(len(self.bc4s))]
        else:
            self.nc1s = self.bc1s
            self.nc2ms = self.bc2ms
            self.nc2s = self.bc2s
            self.nc3ms = self.bc3ms
            self.nc3s = self.bc3s
            self.nc4s = self.bc4s

        self.config = {}
        self.sconfig = '_'.join([str(x) for x in self.nc1s]) + '-' + '_'.join([str(x) for x in self.nc2ms]) + '-' + '_'.join([str(x) for x in self.nc2s]) + '-' + '_'.join([str(x) for x in self.nc3ms]) + '-' + '_'.join([str(x) for x in self.nc3s]) + '-' + '_'.join([str(x) for x in self.nc4s])

        # build GoogleNet model
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
        ''' build GoogleNet model according to model config
        '''
        x = conv2d(self.input, 64, 7, opname='conv1', stride=2, padding='SAME')
        x = batch_norm(x, opname='conv1.bn')
        x = activation(x, 'relu', opname='conv1.relu')     
        self.add_to_log('conv-bn-relu', 3, 64, 7, 2, 'layer1', self.input.shape.as_list()[1], self.input.shape.as_list()[2])

        (h, w) = x.shape.as_list()[1:3]
        x = max_pooling(x, 3, 2, opname='maxpool1')
        self.add_to_log('max-pool', 64, 64, 3, 2, 'layer2', h, w)  ## bug: input size error

        (h, w) = x.shape.as_list()[1:3]
        x = conv2d(x, 64, 1, opname='conv3', stride=1, padding='SAME')
        x = batch_norm(x, opname='conv3.bn')
        x = activation(x, 'relu', opname='conv3.relu')     
        self.add_to_log('conv-bn-relu', 64, 64, 1, 1, 'layer3', h, w)

        (h, w) = x.shape.as_list()[1:3]
        x = conv2d(x, 192, 3, opname='conv4', stride=1, padding='SAME')
        x = batch_norm(x, opname='conv4.bn')
        x = activation(x, 'relu', opname='conv4.relu')     
        self.add_to_log('conv-bn-relu', 64, 192, 3, 1, 'layer4', h, w)

        r = [2, 5, 2]
        layer_count, index = 5, 0
        curr_channel = 0
        for layers in r:
            (h, w, cin) = x.shape.as_list()[1:4]          
            x = max_pooling(x, 3, 2, opname='maxpool' + str(layer_count))
            self.add_to_log('max-pool', cin, cin, 3, 2, 'layer' + str(layer_count), h, w)  ## bug: input size error
            layer_count += 1

            for _ in range(layers):
                x, log = inception_block(
                    x, self.nc1s[index],
                    self.nc2ms[index], self.nc2s[index],
                    self.nc3ms[index], self.nc3s[index],
                    self.nc4s[index], name='layer' + str(layer_count), log=True
                )
                self.config.update(log)
                layer_count += 1

                (h, w, curr_channel) = x.shape.as_list()[1:4]
                index += 1

        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = flatten(x)
        self.add_to_log('global-pool', curr_channel, curr_channel, None, None, 'layer' + str(layer_count + 1), 1, 1)

        x = fc_layer(x, self.num_classes, opname='fc3')
        self.add_to_log('fc', curr_channel, self.num_classes, None, None, 'layer' + str(layer_count + 2), None, None)

        return x
