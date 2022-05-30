import tensorflow as tf
from .ops import  *
from .utils import  *
class MobileNetV1(object):
    def __init__(self, x, cfg, version = None, sample = False):  ## change channel number, kernel size
        self.input = x
        self.num_classes = cfg['n_classes']
        
        self.bcs = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        self.bks = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        #print(cfg['sample_space'])
        self.cs = get_sampling_channels(cfg['sample_space']['channel']['start'], cfg['sample_space']['channel']['end'], cfg['sample_space']['channel']['step'], len(self.bcs))
        self.ks = get_sampling_ks(cfg['sample_space']['kernelsize'], len(self.bks))
        #print(self.cs)
        #print(self.ks)
        self.config = {}
        if sample == True:

            if len(self.cs)<13:
                i = len(self.cs)
                self.cs.append([1 for j in range(i)])
            if len(self.ks)<13:
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
        x = conv2d(self.input, self.ncs[0], self.nks[0], opname = 'conv1', stride = 2, padding = 'SAME') #def conv2d(_input, out_features, kernel_size, opname = '', stride = 1, padding = 'SAME', param_initializer = None):
        x = batch_norm(x, opname = 'conv1.bn')
        x = activation(x, 'relu', opname = 'conv1.relu')
        #print(x.shape)

        self.add_to_log('conv-bn-relu', 3, self.ncs[0], self.nks[0], 2, 'layer1', self.input.shape.as_list()[1], self.input.shape.as_list()[2])

        r = [1, 2, 2, 6, 2]
        s = [1, 2, 2, 2, 2]
        layercount = 2
        for index in range(len(r)):
            stride = s[index]
            layers = r[index]
            for j in range(layers):
                if j == 0:
                    sr = stride
                else:
                    sr = 1

                (h, w) = x.shape.as_list()[1:3]

                x = depthwise_conv2d(x, self.nks[layercount-2], sr, opname = 'dwconv' + str(layercount) + '.1')
                x = batch_norm(x, opname = 'dwconv' + str(layercount) + '.1.bn')
                x = activation(x, 'relu', opname = 'dwconv' + str(layercount) + '.1.relu')
                #print(layercount, x.shape)
                self.add_to_log('dwconv-bn-relu', self.ncs[layercount-2], self.ncs[layercount-2], self.nks[layercount-2], sr, 'layer' + str(layercount) + '.1', h, w)

                (h, w) = x.shape.as_list()[1:3]
                x = conv2d(x, self.ncs[layercount-1], 1, opname = 'conv' + str(layercount) + '.2', stride = 1)
                x = batch_norm(x, opname = 'conv' + str(layercount) + '.2.bn')
                x = activation(x, 'relu', opname = 'conv' + str(layercount) + '.2.relu')
                self.add_to_log('conv-bn-relu', self.ncs[layercount-2], self.ncs[layercount-1], 1, 1, 'layer' + str(layercount) + '.2', h, w)
               # print(layercount, x.shape)
                layercount  += 1

        x = tf.reduce_mean(x, axis = [1, 2], keep_dims = True)
        x = flatten(x)
        self.add_to_log('global-pool', self.ncs[layercount-2], self.ncs[layercount-2], None, None, 'layer' + str(layercount + 1), 1, 1)
        print(x.shape)

        x = fc_layer(x, self.num_classes, opname = 'fc3')
        self.add_to_log('fc', self.ncs[layercount-2], self.num_classes, None, None, 'layer' + str(layercount + 2), None, None)
        #print(x.shape)

        return x