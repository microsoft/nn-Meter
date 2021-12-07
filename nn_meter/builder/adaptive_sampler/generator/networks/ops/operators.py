import numpy as np
import tensorflow as tf

def fc_layer(_input, out_units, opname = '',use_bias = False, param_initializer = None):
    features_total = int(_input.get_shape()[-1])
    if not param_initializer:
        param_initializer = {}
    with tf.compat.v1.variable_scope(opname + '.fc'):
        init_key = '%s/weight' % tf.get_variable_scope().name
        initializer = param_initializer.get(init_key, tf.contrib.layers.xavier_initializer())
        weight = tf.compat.v1.get_variable(name='weight', shape=[features_total, out_units], initializer=initializer)
        output = tf.matmul(_input, weight)
        if use_bias:
            init_key = '%s/bias' % tf.get_variable_scope().name
            initializer = param_initializer.get(init_key, tf.constant_initializer([0.0] * out_units))
            bias = tf.get_variable(name='bias', shape=[out_units], initializer=initializer)
            output = output + bias
    return output


def activation(_input, activation = 'relu6', opname = ''):
    with tf.compat.v1.variable_scope(opname + '.'+activation):
        if activation == 'relu6':
            #print(opname+'.'+activation)
            return tf.nn.relu6(_input)
        elif activation == 'relu':
            return tf.nn.relu(_input)
        elif activation == 'sigmoid':
            return tf.nn.sigmoid(_input)
        elif activation == 'hswish':
            return _input * (tf.nn.relu6(_input + 3) * 0.16667)
          
        else:
            raise ValueError('Do not support %s' % activation)


def Hswish(input, opname = ''):
    with tf.compat.v1.variable_scope(opname + '.hwise'):
        return tf.math.multiply(input, tf.nn.relu6(input + 3.) / 6.)


def Sigmoid(x, opname = '') :
    with tf.compat.v1.variable_scope(opname + '.sigmoid'):
        return tf.nn.sigmoid(x)


def conv2d(_input, out_features, kernel_size, opname = '', stride = 1,padding = 'SAME', param_initializer = None):
    in_features = int(_input.get_shape()[3])

    if not param_initializer:
        param_initializer = {}
    output = _input
    with tf.compat.v1.variable_scope(opname + '.conv'):
        init_key = '%s/weight' % tf.compat.v1.get_variable_scope().name
        initializer = param_initializer.get(init_key, tf.contrib.layers.variance_scaling_initializer())
        weight = tf.compat.v1.get_variable(name='weight', shape=[kernel_size,kernel_size,in_features,out_features], initializer=initializer)
        output = tf.nn.conv2d(output, weight, [1, stride, stride, 1], padding=padding, data_format='NHWC')
    return output


def depthwise_conv2d(_input, kernel_size, stride = 1, opname = '', padding = 'SAME', param_initializer = None):
    in_features = int(_input.get_shape()[3])
    if not param_initializer:
        param_initializer = {}
    output = _input
    with tf.compat.v1.variable_scope(opname + '.depconv'):
        #print(opname+'.depconv')
        init_key = '%s/weight' % tf.get_variable_scope().name
        initializer = param_initializer.get(init_key, tf.contrib.layers.variance_scaling_initializer())
        weight = tf.get_variable(name='weight', shape=[kernel_size, kernel_size, in_features, 1], initializer=initializer)
        output = tf.nn.depthwise_conv2d(output, weight, [1, stride, stride, 1], padding=padding, data_format='NHWC')
    return output


def batch_norm(_input, is_training = False, opname = '', epsilon = 1e-3, decay = 0.9):
    with tf.compat.v1.variable_scope(opname + ".batchnorm"):
        scope = tf.compat.v1.get_variable_scope().name
        output = tf.contrib.layers.batch_norm(_input, scale=True, is_training=is_training, updates_collections=None, 
                                              epsilon=epsilon, decay=decay, data_format='NHWC')
    return output


def channel_shuffle(input, groups, need_slice = False):
    if need_slice == False:
        x = input
        n, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
        x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
        return x
    else:  # only support batch = 1
        x = input
        n, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=tf.convert_to_tensor([h,w,c]))
        x = tf.reshape(x, shape=tf.convert_to_tensor([h, w, groups, c // groups]))

        x=tf.transpose(x, [0,1,3,2])
        x=tf.reshape(x, shape=tf.convert_to_tensor([h,w,c]))
        x=tf.reshape(x, shape=tf.convert_to_tensor([n,h,w,c]))
        return x


def global_avgpool(x, name=''):
    with tf.compat.v1.variable_scope(name + ".globalpool"):
        x = tf.reduce_mean(x, axis = [1, 2], keep_dims = True)
        x = flatten(x)
        return x


def global_pooling(features):
    return tf.nn.avg_pool(
        features,
        ksize=[1] + features.get_shape().as_list()[1: 3] + [1],
        strides=[1, 1, 1, 1],
        padding='VALID'
    )


def grouped_conv(features, num_groups: int, stride: int, kernel_size: int, num_outputs: int,opname=''):
    cin = features.get_shape().as_list()[-1]
    cout = num_outputs
    #assert cin % num_groups == 0 and cout % num_groups == 0

    with tf.compat.v1.variable_scope(opname + ".grouped_conv"):
        groups = [
            tf.keras.layers.Conv2D(
                filters=num_outputs // num_groups,
                kernel_size=[kernel_size, kernel_size],
                strides=[stride, stride],
                padding="same",
                name="{}/conv".format(i)
            )(x) for i, x in zip(range(num_groups), tf.split(features, num_groups, axis=3))
        ]
        net = tf.concat(groups, axis=3, name="concat")

    return net


def flatten(_input):
    input_shape = _input.shape.as_list()
    if len(input_shape) != 2:
        return tf.reshape(_input, [-1, np.prod(input_shape[1:])])
    else:
        return _input


def global_avgpooling(x, name=''):
    with tf.compat.v1.variable_scope(name + ".globalpool"):
        x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
        x = flatten(x)
        return x


def max_pooling(features, kernelsize, stride, padding = 'SAME', opname = ''):
    with tf.compat.v1.variable_scope(opname+".maxpool"):
        return tf.nn.max_pool(features,
                              ksize=[1, kernelsize, kernelsize, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding,
                              name="maxpool")


def avg_pooling(features, kernelsize, stride, padding = 'SAME', opname = ''):
    with tf.compat.v1.variable_scope(opname + ".avgpool"):
        return tf.nn.avg_pool(features,
                              ksize=[1, kernelsize, kernelsize, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding,
                              name="avgpool")


def SE(features, mid_channels: int):
    """ SE layer
    reference: https://github.com/tensorflow/models/blob/89dd9a4e2548e8a5214bd4e564428d01c206a7db/research/slim/nets/mobilenet/conv_blocks.py#L408
    """
    def gating_fn(features):
        return tf.nn.relu6(tf.math.add(features, 3)) * 0.16667

    with tf.compat.v1.variable_scope("SE"):
        net = global_pooling(features)
        net = tf.keras.layers.Conv2D(
            filters=mid_channels,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="same",
        )(net)

        net = tf.nn.relu(net)

        net = tf.keras.layers.Conv2D(
            filters=features.get_shape().as_list()[-1],
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="same",
        )(net)

    return gating_fn(net) * features


def mix_conv(features, num_groups: int, stride: int):
    cin = features.get_shape().as_list()[-1]
    assert cin % num_groups == 0

    with tf.compat.v1.variable_scope("mix_conv"):
        groups = []
        for x, i in zip(tf.split(features, num_groups, axis = 3), range(num_groups)):
            with tf.variable_scope("{}".format(i)):
                kernel_size = i * 2 + 3
                groups.append(depthwise_conv2d(x, kernel_size, stride))

        return tf.concat(groups, axis=3)
