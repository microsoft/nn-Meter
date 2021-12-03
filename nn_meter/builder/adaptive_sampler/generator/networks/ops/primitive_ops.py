import tensorflow as tf 
from .operators import *


def convbnrelu(input,  kernelsize, cin, cout, stride, name = '', istraining = False, pad = False):
    x = conv2d(input, cout, kernelsize, stride=stride, opname='convbnrelu.1')
    x = batch_norm(x, opname='convbnrelu.1')
    x = activation(x, activation='relu',opname='convbnrelu.1')
    return x


def conv(input, kernelsize, cin, cout, stride,name = '',  istraining = False, pad = False):
    x1 = input  
    x = conv2d(x1, cout, kernelsize, stride=stride, opname='convbnrelu.1')
    return x


def convbnrelumaxpool(input, kernelsize, cin, cout, stride, mks, mstride, name = '', istraining = False, pad = False):
    x1 = input
    x = conv2d(x1, cout, kernelsize, stride=stride, opname='convbnrelu.1')
    x = batch_norm(x, opname='convbnrelu.1')
    x = activation(x, activation='relu', opname='convbnrelu.1')
    x = max_pooling(x, mks, mstride, opname='max_pool')
    return x


def convbnhswish(input, kernelsize, cin, cout, stride, name = '', istraining = False, pad = False):
    x1 = input
    x = conv2d(x1, cout, kernelsize, stride=stride, opname='convbnrelu.1')
    x = batch_norm(x, opname='convbnrelu.1')
    x = activation(x, activation='hswish', opname='convbnrelu.1')
    return x


def convbnrelu6(input, kernelsize, cin, cout, stride, name = '', istraining = False):
    x = conv2d(input, cout, kernelsize, stride=stride, opname='convbnrelu6.1')
    x = batch_norm(x, opname='convbnrelu6.1')
    x = activation(x, activation='relu6', opname='convbnrelu6.1')
    return x


def dwconvbnrelu(input,kernelsize,cin,cout,stride,name='',istraining=False,pad=False):
    x1 = input
    x = depthwise_conv2d(x1, kernelsize, stride=stride, opname='dwconvbnrelu.1')
    x = batch_norm(x, opname='dwconvbnrelu.1')
    x = activation(x, activation='relu', opname='dwconvbnrelu.1')    
    return x


def dwconv(input, kernelsize, cin, cout, stride, name = '', istraining = False, pad = False):
    x1 = input
    x = depthwise_conv2d(x1, kernelsize, stride=stride, opname='dwconvbnrelu.1')    
    return x


def dwconvbnhswish(input, kernelsize, cin, cout, stride, name = '', istraining = False, pad = False):
    x1 = input
    x = depthwise_conv2d(x1, kernelsize, stride=stride, opname='dwconvbnrelu.1')
    x = batch_norm(x, opname='dwconvbnrelu.1')
    x = activation(x, activation='hswish', opname='dwconvbnrelu.1')
    return x


def hswish(input, name = '', istraining = False, pad = False):
    x = activation(input,activation='hswish',opname='hswish')
    return x


def dwconvbnrelu6(input, kernelsize, cin, cout, stride, name = '', istraining = False):
    x = depthwise_conv2d(input, kernelsize, stride=stride, opname='dwconvbnrelu6.1')
    x = batch_norm(x, opname='dwconvbnrelu6.1')
    x = activation(x, activation='relu6', opname='dwconvbnrelu6.1')
    return x


def fc(input, cout, name = '', pad = False):
    x1 = input
    x = fc_layer(x1, cout, opname = 'fc')
    return x


def max_pool(input, kernelsize, stride, name = '', pad = False):
    x1 = input
    return max_pooling(x1, kernelsize, stride, opname='max_pool')


def avgpool(input, kernelsize, stride, name = ''):
    return avg_pooling(input, kernelsize, stride, opname='avg_pool')


def bnrelu(input, name = ''):
    x = batch_norm(input, opname='bnrelu.1')
    x = activation(x, activation='relu', opname='bnrelu.2')
    return  x


def bn(input, name = ''):
    x = batch_norm(input, opname='bnrelu.1')
    return x


def relu(input,name = ""):
    x = activation(input,activation='relu',opname='bnrelu.2')
    return x


def concats(inputs, name = ''):
    x = tf.concat(inputs, axis=3)
    return x


def addrelu(inputs, name = '', istraining = False, pad = False):
    x = tf.add(inputs[0], inputs[1])
    x = activation(x, activation='relu', opname='addrelu.relu')   
    return x


def add(inputs, name = '', istraining = False, pad = False):
    x = tf.add(inputs[0], inputs[1]) 
    return x


def global_avgpool(input, name = ""):
    x = global_avgpooling(input, name=name)
    return x


def split(input, num_or_size_splits = 2,  name = ""):
     out1, out2 = tf.split(input, num_or_size_splits=2, axis=3)
     return out1, out2
 
 
def channelshuffle(input, groups = 2, name = ""):
    x = channel_shuffle(input, groups, name)
    return x


def se(input, cin, name = ""):
    x = SE(input,cin//4)
    return x
