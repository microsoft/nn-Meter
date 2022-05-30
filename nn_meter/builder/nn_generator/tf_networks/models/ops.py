
import tensorflow as tf

import numpy as np
def fc_layer(_input, out_units, opname = '', use_bias = False, param_initializer = None):
    features_total = int(_input.get_shape()[-1])
    if not param_initializer:
        param_initializer = {}
    with tf.compat.v1.variable_scope(opname + '.fc'):
        init_key = '%s/weight' % tf.get_variable_scope().name
        initializer = param_initializer.get(init_key, tf.contrib.layers.xavier_initializer())
        weight = tf.compat.v1.get_variable(name = 'weight', shape = [features_total, out_units], initializer = initializer)
        output = tf.matmul(_input, weight)
        if use_bias:
            init_key = '%s/bias' % tf.get_variable_scope().name
            initializer = param_initializer.get(init_key, tf.constant_initializer([0.0] * out_units))
            bias = tf.get_variable(name = 'bias', shape = [out_units], initializer = initializer)
            output = output  +  bias
    return output
def activation(_input, activation = 'relu6', opname = ''):
    with tf.compat.v1.variable_scope(opname + '.' + activation):
        if activation  ==  'relu6':
            #print(opname + '.' + activation)
            return tf.nn.relu6(_input)
        elif activation == 'relu':
            return tf.nn.relu(_input)
        elif activation == 'sigmoid':
            return tf.nn.sigmoid(_input)
        elif activation == 'hswish':
            return _input * (tf.nn.relu6(_input +  3) * 0.16667)
          
        else:
            raise ValueError('Do not support %s' % activation)
def Hswish(input, opname = ''):
    with tf.compat.v1.variable_scope(opname + '.hwise'):
        return tf.math.multiply(input, tf.nn.relu6(input + 3.)/6.)
def Sigmoid(x, opname = '') :
    with tf.compat.v1.variable_scope(opname + '.sigmoid'):
        return tf.nn.sigmoid(x)
def conv2d(_input, out_features, kernel_size, opname = '', stride = 1, padding = 'SAME', param_initializer = None):
    in_features = int(_input.get_shape()[3])

    if not param_initializer:
        param_initializer = {}
    output = _input
    with tf.compat.v1.variable_scope(opname + '.conv'):
       # print(opname + '.conv')
        init_key = '%s/weight' % tf.compat.v1.get_variable_scope().name
        initializer = param_initializer.get(init_key, tf.contrib.layers.variance_scaling_initializer())
        weight = tf.compat.v1.get_variable(name = 'weight', shape = [kernel_size, kernel_size, in_features, out_features], initializer = initializer)
        output = tf.nn.conv2d(output, weight, [1, stride, stride, 1], padding = padding, data_format = 'NHWC')
    return output
def depthwise_conv2d(_input, kernel_size, stride = 1, opname = '', padding = 'SAME', param_initializer = None):
    in_features = int(_input.get_shape()[3])
    if not param_initializer:
        param_initializer = {}
    output = _input
    with tf.compat.v1.variable_scope(opname + '.depconv'):
        #print(opname + '.depconv')
        init_key = '%s/weight' % tf.get_variable_scope().name
        initializer = param_initializer.get(init_key, tf.contrib.layers.variance_scaling_initializer())
        weight = tf.get_variable(name = 'weight', shape = [kernel_size, kernel_size, in_features, 1], initializer = initializer)
        output = tf.nn.depthwise_conv2d(output, weight, [1, stride, stride, 1], padding = padding, data_format = 'NHWC')
    return output
def batch_norm(_input, is_training = False, opname = '', epsilon = 1e-3, decay = 0.9):
    with tf.compat.v1.variable_scope(opname + ".batchnorm"):
        #print(opname + '.batchnorm')
        scope = tf.compat.v1.get_variable_scope().name
        #bn_init = {'beta': param_initializer['%s/bias' % scope], 'gamma': param_initializer['%s/weight' % scope], 'moving_mean': param_initializer['%s/running_mean' % scope], 'moving_variance': param_initializer['%s/running_var' % scope], }
        output = tf.contrib.layers.batch_norm(_input, scale = True, is_training = is_training, updates_collections = None, epsilon = epsilon, decay = decay, data_format = 'NHWC', )
    return output

def channel_shuffle(input, groups, need_slice = False):
    #with tf.variable_scope(opname + '.shuffle_unit'):

    if need_slice == False:
        x = input
        n, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape = tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
        x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
        x = tf.reshape(x, shape = tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
        return x
    else:  ##only support batch = 1
        x = input
        n, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape = tf.convert_to_tensor([h, w, c]))
        x = tf.reshape(x, shape = tf.convert_to_tensor([h, w, groups, c // groups]))

        x = tf.transpose(x, [0, 1, 3, 2])
        x = tf.reshape(x, shape = tf.convert_to_tensor([h, w, c]))
        x = tf.reshape(x, shape = tf.convert_to_tensor([n, h, w, c]))
        return x
def global_avgpool(x, name = ''):
    with tf.compat.v1.variable_scope(name + ".globalpool"):
        x = tf.reduce_mean(x, axis = [1, 2], keep_dims = True)
        x = flatten(x)
        return x

def grouped_conv(features, num_groups: int, stride: int, kernel_size: int, num_outputs: int, opname = ''):
    cin = features.get_shape().as_list()[-1]
    cout = num_outputs
    #assert cin % num_groups  ==  0 and cout % num_groups  ==  0

    with tf.compat.v1.variable_scope(opname + ".grouped_conv"):
        groups = [
            tf.keras.layers.Conv2D(
                filters = num_outputs // num_groups, 
                kernel_size = [kernel_size, kernel_size], 
                strides = [stride, stride], 
                padding = "same", 
                name = "{}/conv".format(i)
            )(x) for i, x in zip(range(num_groups), tf.split(features, num_groups, axis = 3))
        ]
        net = tf.concat(groups, axis = 3, name = "concat")

    return net



def flatten(_input):
    input_shape = _input.shape.as_list()
    if len(input_shape) !=  2:
        return tf.reshape(_input, [-1, np.prod(input_shape[1:])])
    else:
        return _input

def global_pooling(features):
    return tf.nn.avg_pool(
        features, 
        ksize = [1]  +  features.get_shape().as_list()[1: 3]  +  [1], 
        strides = [1, 1, 1, 1], 
        padding = 'VALID'
    )
def max_pooling(features, kernelsize, stride, padding = 'SAME', opname = ''):
    with tf.compat.v1.variable_scope(opname + ".maxpool"):
        return tf.nn.max_pool(features, 
                              ksize = [1, kernelsize, kernelsize, 1], 
                              strides = [1, stride, stride, 1], 
                              padding = padding, 
                              name = "maxpool")
def avg_pooling(features, kernelsize, stride, padding = 'SAME', opname = ''):
    with tf.compat.v1.variable_scope(opname + ".avgpool"):
        return tf.nn.avg_pool(features, 
                              ksize = [1, kernelsize, kernelsize, 1], 
                              strides = [1, stride, stride, 1], 
                              padding = padding, 
                              name = "avgpool")
def SE(features, mid_channels: int):
    """SE layer
    https://github.com/tensorflow/models/blob/89dd9a4e2548e8a5214bd4e564428d01c206a7db/research/slim/nets/mobilenet/conv_blocks.py#L408
    """
    def gating_fn(features):
        return tf.nn.relu6(tf.math.add(features, 3)) * 0.16667

    with tf.compat.v1.variable_scope("SE"):
        net = global_pooling(features)
        net = tf.keras.layers.Conv2D(
            filters = mid_channels, 
            kernel_size = [1, 1], 
            strides = [1, 1], 
            padding = "same", 
        )(net)

        net = tf.nn.relu(net)

        net = tf.keras.layers.Conv2D(
            filters = features.get_shape().as_list()[-1], 
            kernel_size = [1, 1], 
            strides = [1, 1], 
            padding = "same", 
        )(net)

    return gating_fn(net) * features



def mix_conv(features, num_groups: int, stride: int):
    cin = features.get_shape().as_list()[-1]
    assert cin % num_groups  ==  0

    with tf.compat.v1.variable_scope("mix_conv"):
        groups = []
        for x, i in zip(tf.split(features, num_groups, axis = 3), range(num_groups)):
            with tf.variable_scope("{}".format(i)):
                kernel_size = i * 2  +  3
                groups.append(depthwise_conv2d(x, kernel_size, stride))

        return tf.concat(groups, axis = 3)


import tensorflow as tf 

def add_to_log(op, cin, cout, ks, stride, inputh, inputw):
        config = {}
       
        config['op'] = op
        config['cin'] = cin
        config['cout'] = cout
        config['ks'] = ks
        config['stride'] = stride 
        config['inputh'] = inputh
        config['inputw'] = inputw 
        return config
def add_ele_to_log(op, tensorshapes):
        config = {}
       
        config['op'] = op
        config['input_tensors'] = tensorshapes
        return config

def inverted_block(input, kernelsize, oup, stride, expansion = 1, se = False, ac = 'relu6', name = '', istraining = False, log = False):

    (h1, w1) = input.shape.as_list()[1:3]
    in_features = int(input.get_shape()[3])
    feature_dim = round(in_features * expansion)
   
    x = conv2d(input, feature_dim, 1, stride = 1, opname = name + '.1')
    x = batch_norm(x, istraining, opname = name + '.1')
    x = activation(x, activation = ac, opname = name + '.1')

    (h2, w2) = x.shape.as_list()[1:3]

    x = depthwise_conv2d(x, kernelsize, stride = stride, opname = name + '.2')
    x = batch_norm(x, istraining, opname = name + '.2')

    if se:
        x = SE(x, feature_dim//4)

    x = activation(x, activation = ac, opname = name + '.2')

    (h3, w3) = x.shape.as_list()[1:3]
    x = conv2d(x, oup, 1, stride = 1, opname = name + '.3')
    x1 = batch_norm(x, istraining, opname = name + '.3')


    logs = {}
    if log:
        logs[name + '.1'] = add_to_log('conv-bn-' + ac, in_features, feature_dim, 1, 1, h1, w1)
        logs[name + '.2'] = add_to_log('dwconv-bn-' + ac, feature_dim, feature_dim, kernelsize, stride, h2, w2)
        if se:
             logs[name + '.3'] = add_to_log('se', feature_dim, 4, None, None, h3, w3)
             
       

        logs[name + '.4'] = add_to_log('conv-bn', feature_dim, oup, 1, 1, h3, w3)
       

    if stride == 1 and in_features == oup:

        x2 = input
        x = tf.add(x1, x2)
        if log:
            logs[name + '.5'] = add_ele_to_log('add', [x1.shape.as_list()[1:4], x2.shape.as_list()[1:4]])
          
        return x, logs 
    else:
        return x1, logs

    
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
        x = SE(x, exp_ch//4)
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

   
def dense_layer(input, growthrate, kernelsize, name = '', istraining = False, is_slice = False, log = False):
    (h1, w1, cin) = input.shape.as_list()[1:4]
    x = batch_norm(input, istraining, opname = name + '.1')
    x = activation(x, activation = 'relu', opname = name + '.1')

    x = conv2d(x, 4 * growthrate, 1, stride = 1, opname = name + '.2')
    x = batch_norm(x, istraining, opname = name + '.2')
    x = activation(x, activation = 'relu', opname = name + '.2')
    x1 = conv2d(x, growthrate, kernelsize, stride = 1, opname = name + '.3')
    x2 = tf.concat([input, x1], axis = 3)
    logs = {}
    if log:
        logs[name + '.1'] = add_to_log('bn-relu', cin, cin, 1, 1, h1, w1)
        logs[name + '.2'] = add_to_log('conv-bn-relu', cin, 4 * growthrate, 1, 1, h1, w1)
        logs[name + '.3'] = add_to_log('conv', 4 * growthrate, growthrate, kernelsize, 1, h1, w1)
        logs[name + '.4'] = add_ele_to_log('concat', [input.shape.as_list()[1:4], x1.shape.as_list()[1:4]])
    return x2, logs 




def fire_block(input, squeezecin, cout, kernelsize, name = '', istraining = False, is_slice = False, log = False):
    (h1, w1) = input.shape.as_list()[1:3]
    in_features = int(input.get_shape()[3])


    x = conv2d(input, squeezecin, 1, stride = 1, opname = name + '.1')
    x1 = activation(x, activation = 'relu', opname = name + '.1')

    x = conv2d(x1, cout, 1, stride = 1, opname = name + '.2')
    x2 = activation(x, activation = 'relu', opname = name + '.2')

    x = conv2d(x1, cout, kernelsize, stride = 1, opname = name + '.3')
    x3 = activation(x, activation = 'relu', opname = name + '.3')
    x = tf.concat([x2, x3], axis = 3)

    logs = {}
    if log:
        logs[name + '.1'] = add_to_log('conv-relu', in_features, squeezecin, 1, 1, h1, w1)
        logs[name + '.2'] = add_to_log('conv-relu', squeezecin, cout, 1, 1, h1, w1)
        logs[name + '.3'] = add_to_log('conv-relu', squeezecin, cout, kernelsize, 1, h1, w1)
        logs[name + '.4'] = add_ele_to_log('concat', [x2.shape.as_list()[1:4], x3.shape.as_list()[1:4]])
    return x, logs





def inception_block(input, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, name = '', istraining = False, is_slice = False, log = False):
    (h1, w1) = input.shape.as_list()[1:3]
    in_features = int(input.get_shape()[3])
   # print(ch1x1, ch3x3red, ch3x3, ch5x5red, pool_proj)

    x = conv2d(input, ch1x1, 1, stride = 1, opname = name + '.1')
    x = batch_norm(x, istraining, opname = name + '.1')
    x1 = activation(x, activation = 'relu', opname = name + '.1')
    #print(x1.shape)

    x = conv2d(input, ch3x3red, 1, stride = 1, opname = name + '.2.1')
    x = batch_norm(x, istraining, opname = name + '.2.1')
    x = activation(x, activation = 'relu', opname = name + '.2.1')
    x = conv2d(x, ch3x3, 3, stride = 1, opname = name + '.2.2')
    x = batch_norm(x, istraining, opname = name + '.2.2')
    x2 = activation(x, activation = 'relu', opname = name + '.2.2')
    #print(x2.shape)


    x = conv2d(input, ch5x5red, 1, stride = 1, opname = name + '.3.1')
    x = batch_norm(x, istraining, opname = name + '.3.1')
    x = activation(x, activation = 'relu', opname = name + '.3.1')
    x = conv2d(x, ch5x5, 5, stride = 1, opname = name + '.3.2')
    x = batch_norm(x, istraining, opname = name + '.3.2')
    x3 = activation(x, activation = 'relu', opname = name + '.3.2')
    #print(x3.shape)

    x = max_pooling(input, 3, 1, opname = name + '.4')
        #self.add_to_log('max-pool', self.ncs[0], self.ncs[0], 3, 2, 'layer2', h, w)  ## bug: input size error
    x = conv2d(x, pool_proj, 1, stride = 1, opname = name + '.4')
    x = batch_norm(x, istraining, opname = name + '.4')
    x4 = activation(x, activation = 'relu', opname = name + '.4')
    #print(x4.shape)

    out = tf.concat([x1, x2, x3, x4], axis = 3)
    #print(out.shape)
    #print(x1.shape, x2.shape, x3.shape, x4.shape)
    logs = {}
    if log:

        logs[name + '.1'] = add_to_log('conv-bn-relu', in_features, ch1x1, 1, 1, h1, w1)
        logs[name + '.2.1'] = add_to_log('conv-bn-relu', in_features, ch3x3red, 1, 1, h1, w1)
        logs[name + '.2.2'] = add_to_log('conv-bn-relu', ch3x3red, ch3x3, 3, 1, h1, w1)
        logs[name + '.3.1'] = add_to_log('conv-bn-relu', in_features, ch5x5red, 1, 1, h1, w1)
        logs[name + '.3.2'] = add_to_log('conv-bn-relu', ch5x5red, ch5x5, 5, 1, h1, w1)
        logs[name + '.4.1'] = add_to_log('max-pool', in_features, in_features, 3, 2, h1, w1)  ## bug: input size error
        logs[name + '.4.2'] = add_to_log('conv-bn-relu', in_features, pool_proj, 1, 1, h1, w1)
        logs[name + '.5'] = add_ele_to_log('concat', [x1.shape.as_list()[1:4], x2.shape.as_list()[1:4], x3.shape.as_list()[1:4], x4.shape.as_list()[1:4]])



    return out, logs


def shufflev2_unit(input, kernelsize, inp, oup, stride, mid_cscale = 1, name = '', istraining = False, is_slice = False, log = False):
    (h1, w1, c1) = input.shape.as_list()[1:4]
    oup_inc = oup//2
    base_mid_c = oup_inc 
    mid_c = int(base_mid_c * mid_cscale)
    logs = {}
    
    if stride == 1: 
        out1, out2 = tf.split(input, num_or_size_splits = 2, axis = 3)

        (h2, w2) = out1.shape.as_list()[1:3]
        x = conv2d(out1, mid_c, 1, stride = 1, opname = name + '.1_2')
        x = batch_norm(x, istraining, opname = name + '.1_2')
        x = activation(x, activation = 'relu', opname = name + '.1_2')

        (h3, w3) = x.shape.as_list()[1:3]
        x = depthwise_conv2d(x, kernelsize, stride, opname = name + '.2_2')
        x = batch_norm(x, istraining, opname = name + '.2_2')

        (h4, w4) = x.shape.as_list()[1:3]
        x = conv2d(x, oup_inc, 1, stride = 1, opname = name + '.3_2')
        x = batch_norm(x, istraining, opname = name + '.3_2')
        x1 = activation(x, activation = 'relu', opname = name + '.3_2')
        out = tf.concat([x1, out2], axis = 3)

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
         x = depthwise_conv2d(input, kernelsize, stride = stride, opname = name + '.1_1')
         x = batch_norm(x, istraining, opname = name + '.1_1')


         (h2, w2) = x.shape.as_list()[1:3]
         x = conv2d(x, oup_inc, 1, stride = 1, opname = name + '.2_1')
         x = batch_norm(x, istraining, opname = name + '.2_1')
         x1 = activation(x, activation = 'relu', opname = name + '.2_1')



         x = conv2d(input, mid_c, 1, stride = 1, opname = name + '.1_2')
         x = batch_norm(x, istraining, opname = name + '.1_2')
         x = activation(x, activation = 'relu', opname = name + '.1_2')


         (h3, w3) = x.shape.as_list()[1:3]
         x = depthwise_conv2d(x, kernelsize, stride, opname = name + '.2_2')
         x = batch_norm(x, istraining, opname = name + '.2_2')

         (h4, w4) = x.shape.as_list()[1:3]
         x = conv2d(x, oup_inc, 1, stride = 1, opname = name + '.3_2')
         x = batch_norm(x, istraining, opname = name + '.3_2')
         x2 = activation(x, activation = 'relu', opname = name + '.3_2')
         out = tf.concat([x1, x2], axis = 3)

         
         if log:
            logs[name + '.1.1'] = add_to_log('dwconv-bn-relu', c1, c1, kernelsize, stride, h1, w1)
            logs[name + '.1.2'] = add_to_log('conv-bn-relu', c1, oup_inc, 1, 1, h2, w2)

            logs[name + '.2.1'] = add_to_log('conv-bn-relu', c1, mid_c, 1, 1, h1, w1)
            logs[name + '.2.2'] = add_to_log('dwconv-bn-relu', mid_c, mid_c, kernelsize, stride, h3, w3)
            logs[name + '.2.3'] = add_to_log('conv-bn-relu', mid_c, oup_inc, 1, 1, h4, w4)
            logs[name + '.3'] = add_ele_to_log('concat', [x1.shape.as_list()[1:4], x2.shape.as_list()[1:4]])
            logs[name + '.4'] = add_ele_to_log('channel_shuffle', [out.shape.as_list()[1:4]])
         
         return channel_shuffle(out, 2, is_slice), logs