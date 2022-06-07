# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tensorflow as tf
from tensorflow import keras

def get_tensor_by_shapes(shapes, batch_size = 1):
    if len(shapes) == 1:
        return tf.random.normal(shape = [batch_size] + shapes[0])
    else:
        return [tf.random.normal(shape = [batch_size] + shape) for shape in shapes]


def get_inputs_by_shapes(shapes, batch_size = 1):
    if len(shapes) == 1:
        return keras.Input(shape=shapes[0], batch_size=batch_size)
    else:
        return [keras.Input(shape=shape, batch_size=batch_size) for shape in shapes]
