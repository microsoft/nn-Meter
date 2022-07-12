# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf
from .ops import *
from ..utils import *


class MnasNet(object):
    def __init__(self, input, cfg, version = 'large', sample = False, enable_out = False):
        ''' change channel number, kernel size
        '''
        self.input = input
        self.num_classes = cfg['n_classes']
        self.enable_out = enable_out
        self.lastc = 1280
        self.s = [2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1]
        self.se = [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]

        # fixed block channels and kernel size
        self.bks = [3] * 2 + [5] * 3 + [3] * 4 + [3] * 2 + [5] * 3 + [3] * 1
        self.bcs = [24] * 2 + [40] * 2 + [80] * 4 + [112] * 2 + [160] * 3 + [320] * 1
        self.bes = [6] * 2 + [3] * 3 + [6] * 4 + [6] * 2 + [6] * 3 + [6] * 1

        # sampling block channels and kernel size
        self.cs = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bcs))
        self.ks = get_sampling_ks(cfg['sample_space']['kernelsize'], len(self.bks))
        self.es = get_sampling_ks(cfg['sample_space']['es'], len(self.bes))

        if sample == True:
            self.ncs = [int(self.bcs[index] * self.cs[index]) for index in range(len(self.bcs))]
            self.nks = self.ks
            self.nes = self.es
        else:
            self.ncs = self.bcs
            self.nks = self.bks
            self.nes = self.bes

        self.config = {}
        self.sconfig = '_'.join([str(x) for x in self.nks]) + '-' + '_'.join([str(x) for x in self.ncs]) + '-' + '_'.join([str(x) for x in self.nes])

        # build MnasNet model
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
        ''' build MnasNet model according to model config
        '''
        x = conv2d(self.input, 32, 3, opname='conv1', stride=2, padding='SAME')
        x = batch_norm(x, opname='conv1.bn')
        x = activation(x, 'relu', opname='conv1.relu')
        self.add_to_log('conv-bn-relu', 3, 32, 3, 2, 'layer1', self.input.shape.as_list()[1], self.input.shape.as_list()[2])

        (h, w) = x.shape.as_list()[1:3]
        x = depthwise_conv2d(x, 3, stride=1, opname='dwconv2')
        x = batch_norm(x, opname='dwconv2.bn')
        x = activation(x, activation='relu', opname='dwconv2.relu')
        self.add_to_log('dwconv-bn-relu', 32, 32, 3, 1, 'layer2', h, w)

        x = conv2d(x, 16, 1, stride=1, opname='conv3')
        x = batch_norm(x, opname='conv3.bn')
        self.add_to_log('conv-bn', 32, 16, 1, 1, 'layer3', h, w)

        layer_count = 0
        curr_channel = 16
        curr_out = 16
        for _ in range(len(self.ncs)):
            myk = self.nks[layer_count]
            myexp = self.nes[layer_count]
            myout = self.ncs[layer_count]
            s = self.s[layer_count]
            out = self.bcs[layer_count]
            if s == 1 and out == curr_out and self.enable_out == False:
                myout = curr_channel

            x, log = inverted_block(x, myk, myout, s, myexp, self.se[layer_count], ac='relu', name='layer' + str(layer_count + 4), log=True)
            self.config.update(log)

            curr_channel = myout
            curr_out = out
            layer_count += 1

        (h, w, lastc) = x.shape.as_list()[1:4]
        x = conv2d(x, self.lastc, 1, opname='conv' + str(layer_count + 4), stride=1)
        x = batch_norm(x, opname='conv' + str(layer_count + 4) + '.bn')
        x = activation(x, 'relu', opname='conv' + str(layer_count + 4) + '.relu')
        self.add_to_log('conv-bn-relu', lastc, self.lastc, 1, 1, 'layer' + str(layer_count + 4), h, w)

        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = flatten(x)
        self.add_to_log('global-pool', self.lastc, self.lastc, None, None, 'layer' + str(layer_count + 5), 1, 1)

        x = fc_layer(x, self.num_classes, opname='fc' + str(layer_count + 6))
        self.add_to_log('fc', self.lastc, self.num_classes, None, None, 'layer' + str(layer_count + 6), None, None)

        return x
