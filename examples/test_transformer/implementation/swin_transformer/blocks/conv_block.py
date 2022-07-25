import tensorflow as tf
from keras_cv_attention_models.attention_layers import (
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
)
# from space_utils import ACT
ACT = 'hard_swish'

def dsconv(inputs, channel, act, strides, kernel_size, exp, use_se=False):
    inp_channel = inputs.shape[-1]
    
    nn = conv2d_no_bias(inputs, inp_channel * exp, 1, strides=1, padding="same")
    nn = batchnorm_with_activation(nn, activation=act)
    nn = depthwise_conv2d_no_bias(inputs=nn, kernel_size=kernel_size, strides=strides, padding="same")

    nn = batchnorm_with_activation(nn, activation=act)

    if use_se:
        se_d = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(nn)
        # # print(se_d.shape)
        # se_d = tf.expand_dims(se_d, axis=1)
        # se_d = tf.expand_dims(se_d, axis=1)
        # se_d = conv2d_no_bias(se_d, inp_channel * exp //4, 1, strides=1, use_bias=True)
        # se_d = tf.nn.relu(se_d)
        # se_d = conv2d_no_bias(se_d, inp_channel * exp, 1, strides=1, use_bias=True)
        # se_d = tf.nn.relu6(se_d + 3) / 6
        # nn = se_d * nn
        se_d = conv2d_no_bias(se_d, inp_channel * exp //4, 1, strides=1, use_bias=True)
        se_d = tf.nn.relu(se_d)
        se_d = conv2d_no_bias(se_d, inp_channel * exp, 1, strides=1, use_bias=True)
        se_d = tf.nn.relu6(se_d + 3) / 6
        nn = se_d * nn

    nn = conv2d_no_bias(nn, channel, 1, strides=1, padding="same")
    nn = batchnorm_with_activation(nn, activation=None)
    return nn

# use this function to obtain the first conv layer
def first_conv(inputs, channels, act=ACT):
    nn = conv2d_no_bias(inputs, channels, 3, strides=2, padding='same', use_bias=False)
    nn = batchnorm_with_activation(nn, activation=act)
    nn = depthwise_conv2d_no_bias(nn, kernel_size=3, strides=1, padding='same',use_bias=False)
    nn = batchnorm_with_activation(nn, activation=act)
    nn = conv2d_no_bias(nn, channels, 1, strides=1, padding='same', use_bias=False)
    return nn

# use this function to obtain a single layer in conv stage
def conv_layer(inputs, channel, expansion_ratio, kernel_size, stride, use_se, act=ACT):
    return dsconv(inputs, channel, strides=stride, kernel_size=kernel_size, exp=expansion_ratio, act=act, use_se=use_se)