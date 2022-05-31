# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf 
from .ops import  *
from ..utils import  *

def inverted_block_v3(input, kernelsize, oup, stride, exp_ch, ac, se = False, name = '', istraining = False, log = False):
    (h1, w1) = input.shape.as_list()[1:3]
    in_features = int(input.get_shape()[3])

    x = conv2d(input, exp_ch, 1, stride = 1, opname = name + '.1')
    x = batch_norm(x, istraining, opname = name + '.1')
    x = activation(x, activation = ac, opname = name + '.1')

    (h2, w2) = x.shape.as_list()[1:3]
    x = depthwise_conv2d(x, kernelsize, stride = stride, opname = name + '.2')
    x = batch_norm(x, istraining, opname = name + '.2')
    if se:
        x = SE(x, exp_ch // 4)
    x = activation(x, activation = ac, opname = name + '.2')

    (h3, w3) = x.shape.as_list()[1:3]
    x = conv2d(x, oup, 1, stride = 1, opname = name + '.3')
    x1 = batch_norm(x, istraining, opname = name + '.3')

    logs = {}
    if log:
        logs[name + '.1'] = add_to_log('conv-bn-' + ac, in_features, exp_ch, 1, 1, h1, w1)
        logs[name + '.2'] = add_to_log('dwconv-bn-' + ac, exp_ch, exp_ch, kernelsize, stride, h2, w2)
        if se:
             logs[name + '.3'] = add_to_log('se', exp_ch, 4, None, None, h3, w3)
             logs[name + '.4'] = add_to_log('conv-bn', exp_ch, oup, 1, 1, h3, w3)
        else:
            logs[name + '.3'] = add_to_log('conv-bn', exp_ch, oup, 1, 1, h3, w3)

    if stride == 1 and in_features == oup:
        x2 = input
        x = tf.add(x1, x2)
        if log:
            logs[name + '.5'] = add_to_log('add', in_features, oup, None, None, h3, w3)
        return x, logs 
    else:
        return x1, logs


class MobileNetV3(object):
    def __init__(self, x, cfg, version = 'large', sample = False, enable_out = False):  ## change channel number,  kernel size
        self.input = x
        self.num_classes = cfg['n_classes']
        self.enable_out = enable_out
        self.bneck_settings_large  =  [
            # k   exp   out   SE      NL         s
            [ 3,   16,    16,    False,   "relu",     1 ], 
            [ 3,   64,    24,    False,   "relu",     2 ], 
            [ 3,   72,    24,    False,   "relu",     1 ], 
            [ 5,   72,    40,    True,    "relu",     2 ], 
            [ 5,   120,   40,    True,    "relu",     1 ], 
            [ 5,   120,   40,    True,    "relu",     1 ], 
            [ 3,   240,   80,    False,   "hswish",   2 ], 
            [ 3,   200,   80,    False,   "hswish",   1 ], 
            [ 3,   184,   80,    False,   "hswish",   1 ], 
            [ 3,   184,   80,    False,   "hswish",   1 ], 
            [ 3,   480,   112,   True,    "hswish",   1 ], 
            [ 3,   672,   112,   True,    "hswish",   1 ], 
            [ 5,   672,   160,   True,    "hswish",   2 ], 
            [ 5,   960,   160,   True,    "hswish",   1 ], 
            [ 5,   960,   160,   True,    "hswish",   1 ], 
        ]
        self.bneck_settings_small = [
            # k   exp   out  SE      NL         s
            [ 3,   16,    16,   True,    "relu",     2 ], 
            [ 3,   72,    24,   False,   "relu",     2 ], 
            [ 3,   88,    24,   False,   "relu",     1 ], 
            [ 5,   96,    40,   True,    "hswish",   2 ], 
            [ 5,   240,   40,   True,    "hswish",   1 ], 
            [ 5,   240,   40,   True,    "hswish",   1 ], 
            [ 5,   120,   48,   True,    "hswish",   1 ], 
            [ 5,   144,   48,   True,    "hswish",   1 ], 
            [ 5,   288,   96,   True,    "hswish",   2 ], 
            [ 5,   576,   96,   True,    "hswish",   1 ], 
            [ 5,   576,   96,   True,    "hswish",   1 ], 
        ]

        if version == 'large':
            self.bneck = self.bneck_settings_large
            self.lastc = 1280
        else:
            self.bneck = self.bneck_settings_small
            self.lastc = 1024
        self.bks = [int(x) for x in np.array(self.bneck).T[0]]
        self.bes = [int(x) for x in np.array(self.bneck).T[1]]
        self.bcs = [int(x) for x in np.array(self.bneck).T[2]]
        
        self.cs = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bcs))
        self.ks = get_sampling_ks(cfg['sample_space']['kernelsize'], len(self.bks))
        self.es = get_sampling_channels(cfg['sample_space']['es']['start'], cfg['sample_space']['es']['end'], cfg['sample_space']['es']['step'], len(self.bes))

        self.config = {}
        if sample == True:
            self.ncs = [int(self.bcs[index] * self.cs[index]) for index in range(len(self.bcs))]
            self.nks = self.ks
            self.nes = [int(self.bes[index] * self.es[index]) for index in range(len(self.bes))]
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
        x = conv2d(self.input, 16, 3, opname = 'conv1', stride = 2, padding = 'SAME') #def conv2d(_input, out_features, kernel_size, opname = '', stride = 1, padding = 'SAME', param_initializer = None):
        x = batch_norm(x, opname = 'conv1.bn')
        x = activation(x, 'hswish', opname = 'conv1.hswish')
        self.add_to_log('conv-bn-hswish', 3, 16, 3, 2, 'layer1', self.input.shape.as_list()[1], self.input.shape.as_list()[2])

        r = [1, 2, 3, 4, 3, 3, 1]
        e = [1, 6, 6, 6, 6, 6, 6]
        s = [1, 2, 2, 2, 1, 2, 1]
        layercount = 0
        lastchannel = 16
        lastout = 16
        for idx,  (k,  exp,  out,  SE,  NL,  s) in enumerate(self.bneck):
            myk = self.nks[layercount]
            myexp = self.nes[layercount]
            myout = self.ncs[layercount]
            if s == 1 and out  == lastout and self.enable_out == False:  ##
                myout = lastchannel
            x, log = inverted_block_v3(x, myk, myout, s, myexp, NL, SE, name = 'layer' + str(layercount + 2), log = True)
            self.config.update(log)

            lastchannel = myout 
            lastout = out 

            layercount  += 1

        (h, w, lastc) = x.shape.as_list()[1:4]

        x = conv2d(x, self.lastc, 1, opname = 'conv' + str(layercount + 2) + '.1', stride = 1)
        x = batch_norm(x, opname = 'conv' + str(layercount) + '.1')
        x = activation(x, 'hswish', opname = 'conv' + str(layercount + 2) + '.1')
        self.add_to_log('conv-bn-hswish', lastc, self.lastc, 1, 1, 'layer' + str(layercount + 2) + '.1', h, w)

        x = tf.reduce_mean(x,  axis = [1,  2], keep_dims = True)
        x = flatten(x)
        self.add_to_log('global-pool', self.lastc, self.lastc, None, None, 'layer' + str(layercount + 3), 1, 1)

        x = fc_layer(x, self.num_classes, opname = 'fc' + str(layercount + 4))
        self.add_to_log('fc', self.lastc, self.num_classes, None, None, 'layer' + str(layercount + 4), None, None)

        return x