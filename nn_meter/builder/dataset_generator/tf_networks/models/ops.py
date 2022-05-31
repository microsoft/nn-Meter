import numpy as np
import tensorflow as tf
from tensorflow import keras
from ..utils import *

def fc_layer(_input, out_channels, opname = '', use_bias = False, param_initializer = None):
    return keras.layers.Dense(out_channels, use_bias=use_bias)(_input)


def Hswish(input, opname = ''):
    relu6 = tf.keras.layers.ReLU(6)
    return input * relu6(input + 3.) * (1. / 6.)


def Sigmoid(x, opname = '') :
    return tf.nn.sigmoid(x)


def activation(_input, activation = 'relu6', opname = ''):
    if activation  ==  'relu6':
        return tf.nn.relu6(_input)
    elif activation == 'relu':
        return tf.nn.relu(_input)
    elif activation == 'sigmoid':
        return Sigmoid(_input)
    elif activation == 'hswish':
        return Hswish(_input)
    else:
        raise ValueError('Do not support %s' % activation)


def conv2d(_input, out_channels, kernel_size, stride = 1, padding = 'SAME', opname = '', param_initializer = None):
    return keras.layers.Conv2D(out_channels, kernel_size=kernel_size, strides=stride, padding=padding)(_input)


def depthwise_conv2d(_input, kernel_size, stride = 1, opname = '', padding = 'SAME', param_initializer = None):
    return keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding)(_input)


def batch_norm(_input, is_training = False, opname = '', epsilon = 1e-3, decay = 0.9):
    output = keras.layers.BatchNormalization()(_input)
    return output

def convbnrelu(_input, out_channels, kernel_size, stride = 1, relu = False, padding = 'SAME', opname = ''):
    x = conv2d(_input, out_channels, kernel_size, stride)
    x = batch_norm(x)
    if(relu): x = tf.nn.relu(x)
    return x


def avgpool(_input, kernel_size, stride = 1, padding = 'VALID', opname = ''):
    return tf.nn.avg_pool2d(_input, kernel_size, stride, padding = padding)


def reluconvbn(_input, out_channels, kernel_size, stride = 1, padding = 'SAME', opname = ''):
    x = tf.nn.relu(_input)
    x = conv2d(x, out_channels, kernel_size, stride)
    x = batch_norm(x)
    return x


def pooling(_input, out_channels, mode, stride = 1, padding = 'SAME', opname = ''):
    tensor_shape = _input.get_shape().as_list()
    if tensor_shape[-1] !=  out_channels:
        _input = reluconvbn(_input, out_channels, 1, 1, 1, opname = opname+".pooling")
    if(mode == 'avg'):
        x = tf.nn.avg_pool2d(_input, 3, stride, padding = padding)
    else:
        if(mode == 'max'):
            x = tf.nn.max_pool2d(_input, 3, stride, padding = padding)
        else:
            raise NotImplementedError
    return x


def zero(_input, out_channels, stride, opname = ''):
    tensor_shape = _input.get_shape().as_list()
    if tensor_shape[-1] == out_channels:
        if stride == 1:
            return tf.math.multiply(_input, .0)
        else:
            raise NotImplementedError
    else:
        tensor_shape[-1] = out_channels
        return tf.zeros(shape = tensor_shape, dtype = "float32") 


def skip(_input, out_channels, identity, stride, opname = ''):
    tensor_shape = _input.get_shape().as_list()
    if stride == 1 and tensor_shape[-1] == out_channels:
        return _input
    else:
        if stride == 1:
            x = conv2d(_input, out_channels, 1, stride)
            x = batch_norm(x)
            return x
        else:
            raise NotImplementedError


def channel_shuffle(input, groups, need_slice = False):
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
    x = tf.reduce_mean(x, axis = [1, 2], keep_dims = True)
    x = flatten(x)
    return x


def grouped_conv(features, num_groups, stride, kernel_size, num_outputs, opname = ''):
    cin = features.get_shape().as_list()[-1]
    cout = num_outputs
    assert cin % num_groups  ==  0 and cout % num_groups  ==  0

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
        ksize = [1] + features.get_shape().as_list()[1:3] + [1], 
        strides = [1, 1, 1, 1], 
        padding = 'VALID'
    )


def max_pooling(features, kernelsize, stride, padding = 'SAME', opname = ''):
    return tf.nn.max_pool(features,
                        ksize = [1, kernelsize, kernelsize, 1], 
                        strides = [1, stride, stride, 1], 
                        padding = padding, 
                        name = "maxpool")


def avg_pooling(features, kernelsize, stride, padding = 'SAME', opname = ''):
    return tf.nn.avg_pool(features, 
                            ksize = [1, kernelsize, kernelsize, 1], 
                            strides = [1, stride, stride, 1], 
                            padding = padding, 
                            name = "avgpool")


def SE(features, mid_channels: int):
    """SE layer
    https://github.com/tensorflow/models/blob/89dd9a4e2548e8a5214bd4e564428d01c206a7db/research/slim/nets/mobilenet/conv_blocks.py#L408
    """
    x = global_pooling(features)
    x = keras.layers.Conv2D(filters = mid_channels, kernel_size = [1, 1], strides = [1, 1], padding = "same")(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(
            filters = features.get_shape().as_list()[-1], 
            kernel_size = [1, 1], strides = [1, 1], padding = "same", 
        )(x)
    x = tf.nn.relu6(tf.math.add(x, 3)) * 0.16667
    return x * features


def mix_conv(features, num_groups: int, stride: int):
    cin = features.get_shape().as_list()[-1]
    assert cin % num_groups  ==  0

    groups = []
    for x, i in zip(tf.split(features, num_groups, axis = 3), range(num_groups)):
        with tf.variable_scope("{}".format(i)):
            kernel_size = i * 2  +  3
            groups.append(depthwise_conv2d(x, kernel_size, stride))
    return tf.concat(groups, axis = 3)


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
