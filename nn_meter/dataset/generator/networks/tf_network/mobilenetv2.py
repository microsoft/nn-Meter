# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf
from .ops import *
from ..utils import  *

class MobileNetV2(object):
    def __init__(self, input, cfg, version = None, sample = False, enable_out = False):
        ''' change channel number, kernel size
        '''
        self.input = input
        self.num_classes = cfg['n_classes']
        self.enable_out = enable_out

        # fixed block channels and kernel size
        self.bcs = [32, 16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320, 1280]
        self.bks = [3] * 18
        self.bes = [1] + [6] * 16

        # sampling block channels and kernel size
        self.cs = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bcs))
        self.ks = get_sampling_ks(cfg['sample_space']['kernelsize'], len(self.bks))
        self.es = get_sampling_ks(cfg['sample_space']['es'], len(self.bes))

        if sample == True:
            if len(self.cs) < 19:
                self.cs.append([1 for _ in range(len(self.cs))])
            if len(self.ks) < 18:
                self.ks.append([3 for _ in range(len(self.ks))])
            self.ncs = [int(self.bcs[index] * self.cs[index]) for index in range(len(self.bcs))]
            self.nks = self.ks
            self.nes = self.es
        else:
            self.ncs = self.bcs
            self.nks = self.bks
            self.nes = self.bes

        self.config = {}
        self.sconfig = '_'.join([str(x) for x in self.nks]) + '-' + '_'.join([str(x) for x in self.ncs]) + '-' + '_'.join([str(x) for x in self.nes])

        # build MobileNetV2 model
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
        ''' build MobileNetV2 model according to model config
        '''
        x = conv2d(self.input, self.ncs[0], self.nks[0], opname='conv1', stride=2, padding='SAME')
        x = batch_norm(x, opname='conv1.bn')
        x = activation(x, 'relu6', opname='conv1.relu6')
        self.add_to_log('conv-bn-relu', 3, self.ncs[0], self.nks[0], 2, 'layer1', self.input.shape.as_list()[1], self.input.shape.as_list()[2])

        r = [1, 2, 3, 4, 3, 3, 1]
        s = [1, 2, 2, 2, 1, 2, 1]
        layer_count = 2
        curr_channel = self.ncs[0]

        for index in range(len(r)):
            stride = s[index]
            layers = r[index]
            for j in range(layers):
                sr = stride if j == 0 else 1
                use_re_connect = sr == 1 and self.ncs[layer_count - 2] == self.ncs[layer_count - 1]
                if self.enable_out == False:
                    use_re_connect = sr == 1 and self.bcs[layer_count - 2] == self.bcs[layer_count - 1]
                c_current = curr_channel if self.enable_out == False and use_re_connect else self.ncs[layer_count - 1]

                (h, w) = x.shape.as_list()[1:3]
                x, log = inverted_block(x, self.nks[layer_count - 2], c_current, sr, self.nes[layer_count - 2], name='layer' + str(layer_count), log=True)
                self.config.update(log)
                curr_channel = c_current
                layer_count += 1

        (h, w) = x.shape.as_list()[1:3]
        x = conv2d(x, self.ncs[layer_count - 1], 1, opname='conv' + str(layer_count) + '.1', stride=1)
        x = batch_norm(x, opname='conv' + str(layer_count) + '.bn')
        x = activation(x, 'relu', opname='conv' + str(layer_count) + '.relu')
        self.add_to_log('conv-bn-relu', self.ncs[layer_count - 2], self.ncs[layer_count - 1], 1, 1, 'layer' + str(layer_count), h, w)

        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = flatten(x)
        self.add_to_log('global-pool', self.ncs[layer_count - 1], self.ncs[layer_count - 1], None, None, 'layer' + str(layer_count + 1), 1, 1)

        x = fc_layer(x, self.num_classes, opname='fc' + str(layer_count + 2))
        self.add_to_log('fc', self.ncs[layer_count - 1], self.num_classes, None, None, 'layer' + str(layer_count + 2), None, None)

        return x
