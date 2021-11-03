import tensorflow as tf
from tensorflow import keras


def get_inputs_by_shapes(shapes):
    if len(shapes) == 1:
        return keras.Input(shape=shapes[0])
    else:
        return [keras.Input(shape=shape) for shape in shapes]

def get_tensor_by_shapes(shapes):
    if len(shapes) == 1:
        return tf.random.normal(shape=[1] + shapes[0])
    else:
        return [tf.random.normal(shape=[1] + shape) for shape in shapes]
