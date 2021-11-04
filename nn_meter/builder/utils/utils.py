# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import copy
import tensorflow as tf
from tensorflow import keras
from .latency import Latency


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


def dump_testcases(testcases):
    testcases_copy = copy.deepcopy(testcases)
    for item in testcases_copy.values():
        for model in item.values():
            if hasattr(model, 'latency'):
                model['latency'] = str(model['latency'])
    return testcases_copy


def read_testcases(testcases):
    testcases_copy = copy.deepcopy(testcases)
    for item in testcases_copy.values():
        for model in item.values():
            if hasattr(model, 'latency'):
                model['latency'] = Latency(model['latency'])
    return testcases_copy
