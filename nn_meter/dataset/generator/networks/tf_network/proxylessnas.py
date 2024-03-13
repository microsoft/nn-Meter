# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf
from .ops import *
from ..utils import  *

class ProxylessNAS(object):
    def __init__(self, input, cfg, version = None, sample = False, enable_out = False):
        ''' change channel number, kernel size
        '''
        self.input = input
        self.num_classes = cfg['n_classes']
        self.enable_out = enable_out

        # fixed block channels and kernel size
        self.bcs = [16] * 1 + [32] * 2 + [40] * 4 + [80] * 4 + [96] * 4 + [192] * 4 + [320] * 1
        self.bks = [3, 5, 3, 7, 3, 5, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7]
        self.bes = [1, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3, 6, 3, 3, 3, 6, 6, 3, 3, 6]

        # sampling block channels and kernel size
        self.cs = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bcs))
        self.ks = get_sampling_ks(cfg['sample_space']['kernelsize'], len(self.bks))
        self.es = get_sampling_ks(cfg['sample_space']['es'], len(self.bes))

        if sample == True:
            if len(self.cs) < 19:
                self.cs.append([1 for j in range(len(self.cs))])
            if len(self.ks) < 18:
                self.ks.append([3 for j in range(len(self.ks))] )
            self.ncs = [int(self.bcs[index] * self.cs[index]) for index in range(len(self.bcs))]
            self.nks = self.ks
            self.nes = self.es
        else:
            self.ncs = self.bcs
            self.nks = self.bks
            self.nes = self.bes

        self.config = {}
        self.sconfig = '_'.join([str(x) for x in self.nks]) + '-' + '_'.join([str(x) for x in self.ncs]) + '-' + '_'.join([str(x) for x in self.nes])

        # build ProxylessNas model
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
        ''' build ProxylessNas model according to model config
        '''
        x = conv2d(self.input, 32, 3, opname='conv1', stride=2, padding='SAME')
        x = batch_norm(x, opname='conv1.bn')
        x = activation(x, 'relu6', opname='conv1.relu6')

        self.add_to_log('conv-bn-relu6', 3, 32, 3, 2, 'layer1', self.input.shape.as_list()[1], self.input.shape.as_list()[2])
        r = [1, 2, 4, 4, 4, 4, 1]
        s = [1, 2, 2, 2, 1, 2, 1]
        layer_count = 0
        curr_channel = 32

        for index in range(len(r)):
            stride = s[index]
            layers = r[index]
            for j in range(layers):
                sr = stride if j == 0 else 1
                use_re_connect = sr == 1 and self.ncs[layer_count - 1] == self.ncs[layer_count]
                if self.enable_out == False:
                    use_re_connect = sr == 1 and self.bcs[layer_count - 1] == self.bcs[layer_count]
                if self.enable_out == False and use_re_connect:
                    c_current = curr_channel
                else:
                    c_current = self.ncs[layer_count]

                (h, w) = x.shape.as_list()[1:3]
                x, log = inverted_block(x, self.nks[layer_count], c_current, sr, self.nes[layer_count], name='layer' + str(layer_count + 2), log=True)
                self.config.update(log)

                curr_channel = c_current
                layer_count += 1

        (h, w) = x.shape.as_list()[1:3]
        x = conv2d(x, 1280, 1, opname='conv' + str(layer_count + 2), stride=1)
        x = batch_norm(x, opname='conv' + str(layer_count + 2) + '.bn')
        x = activation(x, 'relu', opname='conv' + str(layer_count + 2) + '.relu')
        self.add_to_log('conv-bn-relu', curr_channel, 1280, 1, 1, 'layer' + str(layer_count + 2), h, w)

        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = flatten(x)
        self.add_to_log('global-pool', 1280, 1280, None, None, 'layer' + str(layer_count + 3), 1, 1)

        x = fc_layer(x, self.num_classes, opname='fc' + str(layer_count + 4))
        self.add_to_log('fc', 1280, self.num_classes, None, None, 'layer' + str(layer_count + 4), None, None)

        return x
