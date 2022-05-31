import tensorflow as tf 
from .ops import  *
from ..utils import * 
import numpy as np
cfgs = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], 
}
class VGG(object):
    def __init__(self, x, cfg, version = 11, sample = False):  ## change channel number, kernel size
        self.input = x
        self.num_classes = cfg['n_classes']
        da = cfgs[version]
        da = list(filter(('M').__ne__, da))
        self.sconfig = ""
        
        self.bcs = da
        #print(self.bcs, da)
        self.clayers = cfgs[version]
        self.bcs.append(4096)
        self.bcs.append(4096)
        
        self.bks = [3] * len(da)
        #print(cfg['sample_space'])
        self.cs = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bcs))
        self.ks = get_sampling_ks(cfg['sample_space']['kernelsize'], len(self.bks))
        #print(self.cs)
        #print(self.ks)
        self.config = {}
        if sample == True:
            self.ncs = [int(self.bcs[index] * self.cs[index]) for index in range(len(self.bcs))]
            self.nks = self.ks
        else:
            self.ncs = self.bcs 
            self.nks = self.bks 
        #print('nks', self.nks)
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
        layercount = 1
        x = self.input
        layernum = 0
        lastc = 3
        for v in self.clayers:
            if v == 'M':
                (h, w) = x.shape.as_list()[1:3]
                x = max_pooling(x, 2, 2, opname = 'max-pool' + str(layercount), padding = 'VALID')
                self.add_to_log('max-pool', lastc, lastc, 2, 2, 'layer' + str(layercount), h, w)  ## update vgg.json maxpool
            else:
                (h, w) = x.shape.as_list()[1:3]
                x = conv2d(x, self.ncs[layernum], self.nks[layernum], opname = 'conv' + str(layercount), stride = 1, padding = 'SAME') #def conv2d(_input, out_features, kernel_size, opname = '', stride = 1, padding = 'SAME', param_initializer = None):
                x = batch_norm(x, opname = 'conv' + str(layercount) + '.bn')
                x = activation(x, 'relu', opname = 'conv' + str(layercount) + '.relu')
                self.add_to_log('conv-bn-relu', lastc, self.ncs[layernum], self.nks[layernum], 1, 'layer' + str(layercount), h, w)
                #print(x.shape)
                lastc = self.ncs[layernum]
                layernum  += 1
            layercount  += 1
                
        
        #print(x.shape)
        x = tf.reduce_mean(x, axis = [1, 2], keep_dims = True)
        x = flatten(x)
        self.add_to_log('global-pool', self.ncs[layernum-1], self.ncs[layernum-1], None, None, 'layer' + str(layercount + 1), 1, 1)
        #print(x.shape, layernum, len(self.bcs))
        
        x = fc_layer(x, self.ncs[layernum], opname = 'fc1')        
        x = activation(x, 'relu', opname = 'fc1.relu')
        self.add_to_log('fc-relu', self.ncs[layernum], self.ncs[layernum], None, None, 'layer' + str(layercount + 2), None, None)
       # print(x.shape)

        x = fc_layer(x, self.ncs[layernum + 1], opname = 'fc2')
        x = activation(x, 'relu', opname = 'fc2.relu')
        self.add_to_log('fc-relu', self.ncs[layernum], self.ncs[layernum + 1], None, None, 'layer' + str(layercount + 3), None, None)
        #print(x.shape)

        x = fc_layer(x, self.num_classes, opname = 'fc3')
        self.add_to_log('fc', self.ncs[layernum + 1], self.num_classes, None, None, 'layer' + str(layercount + 4), None, None)
        #print(x.shape)




       
        return x