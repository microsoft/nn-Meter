# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf 
from .ops import  *
from ..utils import * 

class AlexNet(object):
    def __init__(self, x, cfg, version = None, sample = False):  ## change channel number, kernel size
        self.input = x
        self.num_classes = cfg['n_classes']

        self.bcs = [64, 192, 384, 256, 256, 4096, 4096]
        self.bks = [11, 5, 3, 3, 3]
        #print(cfg['sample_space'])
        self.cs = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bcs))
        self.ks = get_sampling_ks(cfg['sample_space']['kernelsize'], len(self.bks))
        #print(self.cs)
        #print(self.ks)
        self.config = {}
        if sample == True:

            if len(self.cs) < 7:
                i = len(self.cs)
                self.cs.append([1 for j in range(i)])
            if len(self.ks)<5:
                i = len(self.ks)
                self.ks.append([3 for j in range(i)] )
            self.ncs = [int(self.bcs[index] * self.cs[index]) for index in range(len(self.bcs))]
            self.nks = self.ks
        else:
            self.ncs = self.bcs 
            self.nks = self.bks 
        self.sconfig = '_'.join([str(x) for x in self.nks]) + '-' + '_'.join([str(x) for x in self.ncs])
        self.out = self.build()

    def add_to_log(self, op, cin, cout, ks, stride, layername, inputh, inputw):
        self.config[layername] = {}
        self.config[layername]['op'] = op
        self.config[layername]['cin'] = cin
        self.config[layername]['cout'] = cout
        self.config[layername]['ks'] = ks
        self.config[layername]['stride'] = stride 
        self.config[layername]['inputh'] = inputh
        self.config[layername]['inputw'] = inputw 

    def build(self):
        x = conv2d(self.input, self.ncs[0], self.nks[0], opname = 'conv1', stride = 4, padding = 'VALID') #def conv2d(_input, out_features, kernel_size, opname = '', stride = 1, padding = 'SAME', param_initializer = None):
        x = activation(x, 'relu', opname = 'conv1.relu')
        self.add_to_log('conv-relu', 3, self.ncs[0], self.nks[0], 4, 'layer1', self.input.shape.as_list()[1], self.input.shape.as_list()[2])

        (h, w) = x.shape.as_list()[1:3]
        x = max_pooling(x, 3, 2, opname = 'conv1')
        self.add_to_log('max-pool', self.ncs[0], self.ncs[0], 3, 2, 'layer2', h, w)  ## bug: input size error

        x = conv2d(x, self.ncs[1], self.nks[1], opname = 'conv2', padding = 'SAME')
        x = activation(x, 'relu', opname = 'conv2.relu')
        self.add_to_log('conv-relu', self.ncs[0], self.ncs[1], self.nks[1], 1, 'layer3', x.shape.as_list()[1], x.shape.as_list()[2])

        (h, w) = x.shape.as_list()[1:3]
        x = max_pooling(x, 3, 2, padding = 'VALID', opname = 'conv2')
        self.add_to_log('max-pool', self.ncs[1], self.ncs[1], 3, 2, 'layer4', h, w)

        x = conv2d(x, self.ncs[2], self.nks[2], opname = 'conv3', padding = 'SAME')
        x = activation(x, 'relu', opname = 'conv3.relu')
        self.add_to_log('conv-relu', self.ncs[1], self.ncs[2], self.nks[2], 1, 'layer5', x.shape.as_list()[1], x.shape.as_list()[2])

        x = conv2d(x, self.ncs[3], self.nks[3], opname = 'conv4', padding = 'SAME')
        x = activation(x, 'relu', opname = 'conv4.relu')
        self.add_to_log('conv-relu', self.ncs[2], self.ncs[3], self.nks[3], 1, 'layer6', x.shape.as_list()[1], x.shape.as_list()[2])

        x = conv2d(x, self.ncs[4], self.nks[4], opname = 'conv5', padding = 'SAME')
        x = activation(x, 'relu', opname = 'conv5.relu')
        self.add_to_log('conv-relu', self.ncs[3], self.ncs[4], self.nks[4], 1, 'layer7', x.shape.as_list()[1], x.shape.as_list()[2])

        (h, w) = x.shape.as_list()[1:3]

        x = max_pooling(x, 3, 2, opname = 'conv5', padding = 'VALID')
        self.add_to_log('max-pool', self.ncs[4], self.ncs[4], 3, 2, 'layer8', h, w)

        x = tf.reduce_mean(x, axis = [1, 2], keep_dims = True)
        x = flatten(x)
        self.add_to_log('global-pool', self.ncs[4], self.ncs[4], None, None, 'layer9', 1, 1)

        x = fc_layer(x, self.ncs[5], opname = 'fc1')        
        x = activation(x, 'relu', opname = 'fc1.relu')
        self.add_to_log('fc-relu', self.ncs[4], self.ncs[5], None, None, 'layer10', None, None)
        x = fc_layer(x, self.ncs[6], opname = 'fc2')

        x = activation(x, 'relu', opname = 'fc2.relu')
        self.add_to_log('fc-relu', self.ncs[5], self.ncs[6], None, None, 'layer11', None, None)
        x = fc_layer(x, self.num_classes, opname = 'fc3')
        self.add_to_log('fc', self.ncs[5], self.num_classes, None, None, 'layer12', None, None)

        return x
