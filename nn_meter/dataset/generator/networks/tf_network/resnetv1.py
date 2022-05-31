# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf 
from .ops import  *
from ..utils import  *


def res_basic_block(input, kernelsize, oup, stride, name = '', istraining = False, log = False):
    logs = {}
    (h1, w1) = input.shape.as_list()[1:3]
    in_features = int(input.get_shape()[3])

    x = conv2d(input, oup, kernelsize, stride = stride, opname = name + '.1')
    x = batch_norm(x, istraining, opname = name + '.1')
    x = activation(x, activation = 'relu', opname = name + '.1')

    (h2, w2) = x.shape.as_list()[1:3]
    x = conv2d(x, oup, kernelsize, stride = 1, opname = name + '.2')
    x2 = batch_norm(x, istraining, opname = name + '.2')

    if stride != 1 or oup != in_features:
        x = conv2d(input, oup, 1, stride = stride, opname = name + '.0')
        x1 = batch_norm(x, istraining, opname = name + '.0')
    else:
        x1 = input

    x = x2 + x1
    x = activation(x, activation = 'relu', opname = name + '.4')

    if log:
        logs[name + '.1'] = add_to_log('conv-bn-relu', in_features, oup, kernelsize, stride, h1, w1)
        logs[name + '.2'] = add_to_log('conv-bn-relu', oup, oup, kernelsize, 1, h2, w2)
        if stride != 1 or oup != in_features:
             logs[name + '.0'] = add_to_log('conv-bn', in_features, oup, 1, stride, h1, w1)
        logs[name + '.4'] = add_ele_to_log('add', [x1.shape.as_list()[1:4], x2.shape.as_list()[1:4]])
        logs[name + '.5'] = add_to_log('relu', oup, oup, None, None, h2, w2)
    return x, logs


def res_bottleneck(input, kernelsize, midp, stride, exp = 4, name = '', istraining = False, log = False):
    logs = {}
    (h1, w1) = input.shape.as_list()[1:3]
    in_features = int(input.get_shape()[3])

    x = conv2d(input, midp, 1, stride = 1, opname = name + '.1')
    x = batch_norm(x, istraining, opname = name + '.1')
    x = activation(x, activation = 'relu', opname = name + '.1')

    x = conv2d(x, midp, kernelsize, stride = stride, opname = name + '.2')
    x = batch_norm(x, istraining, opname = name + '.2')
    x = activation(x, activation = 'relu', opname = name + '.2')

    (h2, w2) = x.shape.as_list()[1:3]
    x = conv2d(x, midp * exp, 1, stride = 1, opname = name + '.3')
    x2 = batch_norm(x, istraining, opname = name + '.3')

    if stride != 1 or midp * exp != in_features:
        x = conv2d(input, midp * exp, 1, stride = stride, opname = name + '.0')
        x1 = batch_norm(x, istraining, opname = name + '.0')
    else:
        x1 = input 

    x = x2 + x1
    x = activation(x, activation = 'relu', opname = name + '.4')

    logs = {}
    if log:
        logs[name + '.1'] = add_to_log('conv-bn-relu', in_features, midp, 1, 1, h1, w1)
        logs[name + '.2'] = add_to_log('conv-bn-relu', midp, midp, kernelsize, stride, h1, w1)
        logs[name + '.3'] = add_to_log('conv-bn', midp, midp * exp, 1, 1, h2, w2)
        if stride != 1 or midp * exp != in_features:
             logs[name + '.0'] = add_to_log('conv-bn', in_features, midp * exp, 1, stride, h1, w1)
        logs[name + '.4'] = add_ele_to_log('add', [x1.shape.as_list()[1:4], x2.shape.as_list()[1:4]])
        logs[name + '.5'] = add_to_log('relu', midp * exp, midp * exp, None, None, h2, w2)

    return x, logs


class ResNetV1(object):
    def __init__(self, x, cfg, version = 18, sample = False, enable_out = False):  ## change channel number, kernel size
        self.input = x
        self.num_classes = cfg['n_classes']
        self.enable_out = enable_out
        self.bneck18 = [
            # kernelsize   cout  stride
            [3, 64, 1, None], 
            [3, 64, 1, None], 
            [3, 128, 2, None], 
            [3, 128, 1, None], 
            [3, 256, 2, None], 
            [3, 256, 1, None], 
            [3, 512, 2, None], 
            [3, 512, 1, None]      
        ]
        self.bneck34 = [
             # kernelsize   cout  stride
            [3, 64, 1, None], 
            [3, 64, 1, None], 
            [3, 64, 1, None], 
            [3, 128, 2, None], 
            [3, 128, 1, None], 
            [3, 128, 1, None], 
            [3, 128, 1, None], 
            [3, 256, 2, None], 
            [3, 256, 1, None], 
            [3, 256, 1, None], 
            [3, 256, 1, None], 
            [3, 256, 1, None], 
            [3, 256, 1, None], 
            [3, 512, 2, None], 
            [3, 512, 1, None], 
            [3, 512, 1, None]     
             # kernelsize   cout  stride
        ]
        self.bneck50 = [
            [3, 256, 1, 4], 
            [3, 256, 1, 4], 
            [3, 256, 1, 4], 

            [3, 512, 2, 4], 
            [3, 512, 1, 4], 
            [3, 512, 1, 4], 
            [3, 512, 1, 4], 

            [3, 1024, 2, 4], 
            [3, 1024, 1, 4], 
            [3, 1024, 1, 4], 
            [3, 1024, 1, 4], 
            [3, 1024, 1, 4], 
            [3, 1024, 1, 4], 

            [3, 2048, 2, 4], 
            [3, 2048, 1, 4], 
            [3, 2048, 1, 4]
        ]
        self.bes = []

        if version == 18:
            self.bneck = self.bneck18           
        elif version == 34:
            self.bneck = self.bneck34 
        elif version == 50:
            self.bneck = self.bneck50
            self.bes = [int(x) for x in np.array(self.bneck).T[3]]
        self.version = version
        self.bks = [int(x) for x in np.array(self.bneck).T[0]]
        self.bcs = [int(x) for x in np.array(self.bneck).T[1]]
        
        self.cs = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bcs))
        self.ks = get_sampling_ks(cfg['sample_space']['kernelsize'], len(self.bks))
        self.es = get_sampling_ks(cfg['sample_space']['es'], len(self.bes))

        self.config = {}
        if sample == True:
            self.ncs = [int(self.bcs[index] * self.cs[index]) for index in range(len(self.bcs))]
            self.nks = self.ks
            if len(self.bes)>0:
                self.nes = [int(self.bes[index] * self.es[index]) for index in range(len(self.bes))]
            else:
                self.nes = []
        else:
            self.ncs = self.bcs 
            self.nks = self.bks 
            self.nes = self.bes
        self.sconfig = '_'.join([str(x) for x in self.nks]) + '-' + '_'.join([str(x) for x in self.ncs]) + '-' + '_'.join([str(x) for x in self.nes])
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

    def addblock_to_log(self, op, cin, cout, ks, stride, layername, inputh, inputw, es):
        self.config[layername] = {}
        self.config[layername]['op'] = op
        self.config[layername]['cin'] = cin
        self.config[layername]['cout'] = cout
        self.config[layername]['ks'] = ks
        self.config[layername]['stride'] = stride 
        self.config[layername]['inputh'] = inputh
        self.config[layername]['inputw'] = inputw 
        self.config[layername]['es_channel'] = es

    def build(self):
        x = conv2d(self.input, 64, 7, opname = 'conv1', stride = 2, padding = 'SAME') #def conv2d(_input, out_features, kernel_size, opname = '', stride = 1, padding = 'SAME', param_initializer = None):
        x = batch_norm(x, opname = 'conv1.bn')
        x = activation(x, 'relu', opname = 'conv1.relu')
       # print(x.shape)
        self.add_to_log('conv-bn-relu', 3, 64, 7, 2, 'layer1', self.input.shape.as_list()[1], self.input.shape.as_list()[2])

        (h, w) = x.shape.as_list()[1:3]

        x = max_pooling(x, 3, 2, opname = 'conv1')
        self.add_to_log('max-pool', 64, 64, 3, 2, 'layer2', h, w)  ## bug: input size error
        #print(x.shape)
        layercount = 0
        lastchannel = 64
        lastout = 64
        for idx, (k, out, s, exp) in enumerate(self.bneck):
            myk = self.nks[layercount]
            myout = self.ncs[layercount]
            if s == 1 and out  == lastout and self.enable_out == False:
                myout = lastchannel
           # print(myk, myexp, myout)
           # print(myexp, myexp//4)
            if self.version in [18, 34]:
                x, log = res_basic_block(x, myk, myout, s, name = 'layer' + str(layercount + 3), log = True)
                lastchannel = myout 
            else:
                myexp = self.nes[layercount]
                x, log = res_bottleneck(x, myk, myout//myexp, s, myexp, name = 'layer' + str(layercount + 3), log = True)
                lastchannel = myout//myexp  * myexp

            #x, log = inverted_block_v3(x, myk, myout, s, myexp, NL, SE, name = 'layer' + str(layercount + 2), log = True)
            self.config.update(log)
            #print(layercount + 3, x.shape)

            lastout = out 
            layercount  += 1

        x = tf.reduce_mean(x, axis = [1, 2], keep_dims = True)
        x = flatten(x)
        self.add_to_log('global-pool', lastchannel, lastchannel, None, None, 'layer' + str(layercount + 4), 1, 1)
        #print(x.shape)

        x = fc_layer(x, self.num_classes, opname = 'fc' + str(layercount + 5))
        self.add_to_log('fc', lastchannel, self.num_classes, None, None, 'layer' + str(layercount + 5), None, None)
        #print(x.shape)

        return x