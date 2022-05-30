import tensorflow as tf 
from .ops import  *
from .utils import  *

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


        #print(self.cs)
        #print(self.ks)
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
        #print(x.shape)

        self.add_to_log('conv-bn-hswish', 3, 16, 3, 2, 'layer1', self.input.shape.as_list()[1], self.input.shape.as_list()[2])
        #print(x.shape)
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
        #sys.exit()
        #print(layercount)
        #print(layercount + 2)
    
        x = conv2d(x, self.lastc, 1, opname = 'conv' + str(layercount + 2) + '.1', stride = 1)
        x = batch_norm(x, opname = 'conv' + str(layercount) + '.1')
        x = activation(x, 'hswish', opname = 'conv' + str(layercount + 2) + '.1')
       # print(layercount, x.shape)

        self.add_to_log('conv-bn-hswish', lastc, self.lastc, 1, 1, 'layer' + str(layercount + 2) + '.1', h, w)

        x  =  tf.reduce_mean(x,  axis = [1,  2], keep_dims = True)
        x = flatten(x)
        self.add_to_log('global-pool', self.lastc, self.lastc, None, None, 'layer' + str(layercount + 3), 1, 1)
        #print(x.shape)

        x = fc_layer(x, self.num_classes, opname = 'fc' + str(layercount + 4))
        self.add_to_log('fc', self.lastc, self.num_classes, None, None, 'layer' + str(layercount + 4), None, None)
        #print(x.shape)

        return x