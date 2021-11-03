import tensorflow as tf
from tensorflow import keras
from ...config_manager import config


def reshape(input_shape):
    if len(input_shape) == 3:
        output_shape = [input_shape[2], input_shape[0], input_shape[1]]
        def func(inputs):
            return tf.reshape(inputs, [1] + output_shape)
    else:
        output_shape = [1, 2, int(input_shape[0] / 2)]
        def func(inputs):
            return tf.reshape(inputs, [1] + output_shape)
    return func, output_shape, False


def dwconv(input_shape):
    return keras.layers.DepthwiseConv2D(config.get('kernel_size', 'ruletest'), padding='same'), input_shape, False


def relu(input_shape):
    return keras.layers.ReLU(), input_shape, False


def add(input_shape):
    return keras.layers.Add(), input_shape, True

def conv(input_shape):
    return keras.layers.Conv2D(input_shape[2], config.get('kernel_size', 'ruletest'), padding='same'), input_shape, False


def concat(input_shape):
    if len(input_shape) == 3:
        output_shape = [input_shape[0], input_shape[1], input_shape[2] * 2]
    else:
        output_shape = [input_shape[0] * 2]

    return keras.layers.Concatenate(), output_shape, True


def convtrans(input_shape):
    class Conv2dTranspose(keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.filters = tf.Variable(tf.ones([config.get('kernel_size', 'ruletest'), config.get('kernel_size', 'ruletest'), input_shape[2], input_shape[2]])) 
        def call(self, inputs):
            return tf.nn.conv2d_transpose(
                inputs,
                filters=self.filters,
                output_shape=[1] + input_shape,
                strides=[1, 1],
            )

    return Conv2dTranspose(), input_shape, False


def dense(input_shape):
    return keras.layers.Dense(input_shape[0]), input_shape, False


def pooling(input_shape):
    output_shape = [int(input_shape[0] / 2), int(input_shape[1] / 2), input_shape[2]]
    return keras.layers.AveragePooling2D(padding='same'), output_shape, False


def se(input_shape):
    class SE(keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.conv1 = tf.keras.layers.Conv2D(
                filters=input_shape[-1],
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="same",
            )
            self.conv2 = tf.keras.layers.Conv2D(
                filters=input_shape[-1],
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="same",
            )
        def call(self, inputs):
            x = tf.nn.avg_pool(
                inputs,
                ksize=[1] + input_shape[0:2] + [1],
                strides=[1, 1, 1, 1],
                padding='VALID',
            )
            x = self.conv1(x)
            x = tf.nn.relu(x)
            x = self.conv2(x)
            return x * inputs

    return SE(), input_shape, False


def hswish(input_shape):
    def func(inputs):
        return tf.nn.relu6(tf.math.add(inputs, 3)) * 0.16667
    return func, input_shape, False

