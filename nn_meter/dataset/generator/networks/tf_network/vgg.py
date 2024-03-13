# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf
from .ops import *
from ..utils import *

cfgs = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(object):
    def __init__(self, input, cfg, version = 11, sample = False):
        ''' change channel number, kernel size
        '''
        self.input = input
        self.num_classes = cfg['n_classes']
        self.clayers = cfgs[version]
        da = cfgs[version]
        da = list(filter(('M').__ne__, da))

        # fixed block channels and kernel size
        self.bcs = da
        self.bcs.extend([4096, 4096])
        self.bks = [3] * len(da)

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

        # build VGG model
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
        ''' build VGG model according to model config
        '''
        layer_count, layer_num = 1, 0
        curr_channel = 3
        x = self.input
        for v in self.clayers:
            (h, w) = x.shape.as_list()[1:3]
            if v == 'M':
                x = max_pooling(x, 2, 2, opname='max-pool' + str(layer_count), padding='VALID')
                self.add_to_log('max-pool', curr_channel, curr_channel, 2, 2, 'layer' + str(layer_count), h, w)
            else:
                x = conv2d(x, self.ncs[layer_num], self.nks[layer_num], opname='conv' + str(layer_count), stride=1, padding='SAME')
                x = batch_norm(x, opname='conv' + str(layer_count) + '.bn')
                x = activation(x, 'relu', opname='conv' + str(layer_count) + '.relu')
                self.add_to_log('conv-bn-relu', curr_channel, self.ncs[layer_num], self.nks[layer_num], 1, 'layer' + str(layer_count), h, w)
                curr_channel = self.ncs[layer_num]
                layer_num += 1
            layer_count += 1

        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = flatten(x)
        self.add_to_log('global-pool', self.ncs[layer_num - 1], self.ncs[layer_num - 1], None, None, 'layer' + str(layer_count + 1), 1, 1)
       
        x = fc_layer(x, self.ncs[layer_num], opname='fc1')       
        x = activation(x, 'relu', opname='fc1.relu')
        self.add_to_log('fc-relu', self.ncs[layer_num], self.ncs[layer_num], None, None, 'layer' + str(layer_count + 2), None, None)

        x = fc_layer(x, self.ncs[layer_num + 1], opname='fc2')
        x = activation(x, 'relu', opname = 'fc2.relu')
        self.add_to_log('fc-relu', self.ncs[layer_num], self.ncs[layer_num + 1], None, None, 'layer' + str(layer_count + 3), None, None)

        x = fc_layer(x, self.num_classes, opname='fc3')
        self.add_to_log('fc', self.ncs[layer_num + 1], self.num_classes, None, None, 'layer' + str(layer_count + 4), None, None)

        return x
