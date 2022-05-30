import tensorflow as tf 
from .ops import  *
from .utils import * 

class SqueezeNet(object):
    def __init__(self, x, cfg, version = None, sample = False):  ## change channel number, kernel size
        self.input = x
        self.num_classes = cfg['n_classes']
     
        self.bmcs = [16, 16, 32, 32, 48, 48, 64, 64]
        self.bcs = [64, 64, 128, 128, 192, 192, 256, 256]

        self.bks = [3] * 8
        #print(cfg['sample_space'])
        self.cs = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bcs))
        self.ks = get_sampling_ks(cfg['sample_space']['kernelsize'], len(self.bks))
        #print(self.cs)
        #print(self.ks)
        self.config = {}
        if sample == True:
        
            self.ncs = [int(self.bcs[index] * self.cs[index]) for index in range(len(self.bcs))]
            self.nmcs = [int(self.bmcs[index] * self.cs[index]) for index in range(len(self.bmcs))]
            self.nks = self.ks
        else:
            self.ncs = self.bcs 
            self.nmcs = self.bmcs
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
        x = conv2d(self.input, 96, 7, opname = 'conv1', stride = 2, padding = 'SAME') #def conv2d(_input, out_features, kernel_size, opname = '', stride = 1, padding = 'SAME', param_initializer = None):
        x = activation(x, 'relu', opname = 'conv1.relu')
        #print(x.shape)
      
        self.add_to_log('conv-relu', 3, 96, 7, 2, 'layer1', self.input.shape.as_list()[1], self.input.shape.as_list()[2])

        r = [3, 4, 1]
        layercount = 2
        index = 0
        lastc = 0
        for layers in r:
            (h, w, cin) = x.shape.as_list()[1:4]           
           
            x = max_pooling(x, 3, 2, opname = 'conv1')
            #print(layercount, x.shape)
            self.add_to_log('max-pool', cin, cin, 3, 2, 'layer' + str(layercount), h, w)  ## bug: input size error
            layercount  += 1
            for lay in range(layers):
                x, log = fire_block(x, self.nmcs[index], self.ncs[index], self.nks[index], name = 'layer' + str(layercount), log = True)
                self.config.update(log)
                #print(layercount, x.shape)
                layercount  += 1
                
                lastc = self.bcs[index] * 2
                index  += 1

        (h, w, lastcin) = x.shape.as_list()[1:4]
        x = conv2d(x, 512, 1, opname = 'conv' + str(layercount), stride = 1, padding = 'SAME') #def conv2d(_input, out_features, kernel_size, opname = '', stride = 1, padding = 'SAME', param_initializer = None):
        x = activation(x, 'relu', opname = 'conv' + str(layercount) + '.relu')
        print(layercount, x.shape)
      
        self.add_to_log('conv-relu', lastcin, 512, 1, 1, 'layer' + str(layercount), h, w)

        x = tf.reduce_mean(x, axis = [1, 2], keep_dims = True)
        x = flatten(x)
        self.add_to_log('global-pool', 512, 512, None, None, 'layer' + str(layercount + 1), 1, 1)
        
        x = fc_layer(x, self.num_classes, opname = 'fc3')
        self.add_to_log('fc', 512, self.num_classes, None, None, 'layer' + str(layercount + 2), None, None)
       
        return x