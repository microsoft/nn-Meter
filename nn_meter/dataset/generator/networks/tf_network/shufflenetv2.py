# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf
from .ops import *
from ..utils import  *

def shufflev2_unit(input, kernelsize, oup, stride, mid_cscale = 1, name = '', istraining = False, is_slice = False, log = False):
    (h1, w1, c1) = input.shape.as_list()[1:]
    oup_inc = oup // 2
    base_mid_c = oup_inc
    mid_c = int(base_mid_c * mid_cscale)

    if stride == 1:
        out1, out2 = tf.split(input, num_or_size_splits=2, axis=3)

        (h2, w2) = out1.shape.as_list()[1:3]
        x = conv2d(out1, mid_c, 1, stride=1, opname=name + '.1_2')
        x = batch_norm(x, istraining, opname=name + '.1_2')
        x = activation(x, activation='relu', opname=name + '.1_2')

        (h3, w3) = x.shape.as_list()[1:3]
        x = depthwise_conv2d(x, kernelsize, stride, opname=name + '.2_2')
        x = batch_norm(x, istraining, opname=name + '.2_2')

        (h4, w4) = x.shape.as_list()[1:3]
        x = conv2d(x, oup_inc, 1, stride=1, opname=name + '.3_2')
        x = batch_norm(x, istraining, opname=name + '.3_2')
        x1 = activation(x, activation='relu', opname=name + '.3_2')
        out = tf.concat([x1, out2], axis=3)

        logs = {}
        if log:
            logs[name + '.1'] = add_to_log('split', c1, oup_inc, None, None, h1, w1)
            logs[name + '.2.1'] = add_to_log('conv-bn-relu', oup_inc, mid_c, 1, 1, h2, w2)
            logs[name + '.2.2'] = add_to_log('dwconv-bn', mid_c, mid_c, kernelsize, stride, h3, w3)
            logs[name + '.2.3'] = add_to_log('conv-bn-relu', mid_c, oup_inc, 1, 1, h4, w4)
            logs[name + '.3'] = add_ele_to_log('concat', [x1.shape.as_list()[1:4], out2.shape.as_list()[1:4]])
            logs[name + '.4'] = add_ele_to_log('channel_shuffle', [out.shape.as_list()[1:4]])
        return channel_shuffle(out, 2, is_slice), logs

    else:
        (h1, w1) = input.shape.as_list()[1:3]
        x = depthwise_conv2d(input, kernelsize, stride=stride, opname=name + '.1_1')
        x = batch_norm(x, istraining, opname=name + '.1_1')

        (h2, w2) = x.shape.as_list()[1:3]
        x = conv2d(x, oup_inc, 1, stride=1, opname=name + '.2_1')
        x = batch_norm(x, istraining, opname=name + '.2_1')
        x1 = activation(x, activation='relu', opname=name + '.2_1')

        x = conv2d(input, mid_c, 1, stride=1, opname=name + '.1_2')
        x = batch_norm(x, istraining, opname=name + '.1_2')
        x = activation(x, activation='relu', opname=name + '.1_2')

        (h3, w3) = x.shape.as_list()[1:3]
        x = depthwise_conv2d(x, kernelsize, stride, opname=name + '.2_2')
        x = batch_norm(x, istraining, opname=name + '.2_2')

        (h4, w4) = x.shape.as_list()[1:3]
        x = conv2d(x, oup_inc, 1, stride=1, opname=name + '.3_2')
        x = batch_norm(x, istraining, opname=name + '.3_2')
        x2 = activation(x, activation='relu', opname=name + '.3_2')
        out = tf.concat([x1, x2], axis=3)

        logs = {}
        if log:
            logs[name + '.1.1'] = add_to_log('dwconv-bn-relu', c1, c1, kernelsize, stride, h1, w1)
            logs[name + '.1.2'] = add_to_log('conv-bn-relu', c1, oup_inc, 1, 1, h2, w2)
            logs[name + '.2.1'] = add_to_log('conv-bn-relu', c1, mid_c, 1, 1, h1, w1)
            logs[name + '.2.2'] = add_to_log('dwconv-bn-relu', mid_c, mid_c, kernelsize, stride, h3, w3)
            logs[name + '.2.3'] = add_to_log('conv-bn-relu', mid_c, oup_inc, 1, 1, h4, w4)
            logs[name + '.3'] = add_ele_to_log('concat', [x1.shape.as_list()[1:4], x2.shape.as_list()[1:4]])
            logs[name + '.4'] = add_ele_to_log('channel_shuffle', [out.shape.as_list()[1:4]])

        return channel_shuffle(out, 2, is_slice), logs


class ShuffleNetV2:
    def __init__(self, input, cfg, version = None, sample = False, enable_out = False):
        ''' change channel number, kernel size
        '''
        self.input = input
        self.num_classes = cfg['n_classes']
        self.enable_out = enable_out

        self.bcs = [24] + [116] * 4 + [232] * 8 + [464] * 4 + [1024]
        self.bks = [3] + [3] * (4 + 8 + 4) + [1]
        self.bes = [1] + [6] * 16

        self.cs = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bcs))
        self.ks = get_sampling_ks(cfg['sample_space']['kernelsize'], len(self.bks))
        self.mcs = get_sampling_channels(cfg['sample_space']['mid_channel']['start'], cfg['sample_space']['mid_channel']['end'], cfg['sample_space']['mid_channel']['step'], len(self.bcs))

        if sample == True:
            if len(self.cs) < 18:
                self.cs.append([1 for _ in range(len(self.cs))])
            if len(self.ks) < 18:
                self.ks.append([3 for _ in range(len(self.ks))] )
            self.ncs = [int(self.bcs[index] * self.cs[index]) for index in range(len(self.bcs))]
            self.nks = self.ks
        else:
            self.ncs = self.bcs
            self.mcs = [1] * 17
            self.nks = self.bks
            self.nes = self.bes

        self.config = {}
        self.sconfig = '_'.join([str(x) for x in self.nks]) + '-' + '_'.join([str(x) for x in self.ncs]) + '-' + '_'.join([str(x) for x in self.mcs])
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
        ''' build ShuffleNetV2 model according to model config
        '''
        x = conv2d(self.input, self.ncs[0], self.nks[0], opname='conv1', stride=2, padding='SAME')
        x = batch_norm(x, opname='conv1.bn')
        x = activation(x, 'relu', opname='conv1.relu')
        self.add_to_log('conv-bn-relu', 3, self.ncs[0], self.nks[0], 2, 'layer1', self.input.shape.as_list()[1], self.input.shape.as_list()[2])

        (h, w) = x.shape.as_list()[1:3]
        x = max_pooling(x, 3, 2, opname='conv1')
        self.add_to_log('max-pool', self.ncs[0], self.ncs[0], 3, 2, 'layer2', h, w)

        r = [4, 8, 4]
        s = [2, 2, 2]
        layer_count = 2
        curr_channel = self.ncs[0]

        for index in range(len(r)):
            stride = s[index]
            layers = r[index]
            # ep = r[index]
            for j in range(layers):
                sr = stride if j == 0 else 1

                if self.enable_out == False and sr == 1:
                    c_current = curr_channel
                else:
                    c_current = self.ncs[layer_count - 1]

                (h, w) = x.shape.as_list()[1:3]
                x, log = shufflev2_unit(x, self.nks[layer_count - 2], c_current, sr,
                                        self.mcs[layer_count - 2], name='layer' + str(layer_count), log=True)
                self.config.update(log)
                curr_channel = c_current
                layer_count += 1

        (h, w) = x.shape.as_list()[1:3]
        x = conv2d(x, self.ncs[layer_count - 1], 1, opname='conv' + str(layer_count) + '.1', stride=1)
        x = batch_norm(x, opname='conv' + str(layer_count) + '.bn')
        x = activation(x, 'relu', opname='conv' + str(layer_count) + '.relu')
        self.add_to_log('conv-bn-relu', self.ncs[layer_count - 2], self.ncs[layer_count - 1], 1, 1, 'layer' + str(layer_count) + '.1', h, w)

        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = flatten(x)
        self.add_to_log('global-pool', self.ncs[layer_count - 1], self.ncs[layer_count - 1], None, None, 'layer' + str(layer_count + 1), 1, 1)

        x = fc_layer(x, self.num_classes, opname='fc' + str(layer_count + 1))
        self.add_to_log('fc', self.ncs[layer_count - 1], self.num_classes, None, None, 'layer' + str(layer_count + 2), None, None)

        return x
