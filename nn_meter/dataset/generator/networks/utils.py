# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import time
import random
import tensorflow as tf
from typing import List, Union

def get_sampling_channels(c_start, c_end, c_ratio, c_layers):
    nc = []
    channel_scale = []
    while c_start <= c_end:
        channel_scale.append(c_start)
        c_start += c_ratio

    for _ in range(c_layers):
        index = random.choice(channel_scale)
        nc.append(index)
    return nc


def get_sampling_ks(kernelsizes, layers):
    return [random.choice(kernelsizes) for _ in range(layers)]


def get_sampling_es(es, layers):
    return [random.choice(es) for _ in range(layers)]


def add_to_log(op, cin, cout, ks, stride, inputh, inputw):
    config = {
        'op': op,
        'cin': cin,
        'cout': cout,
        'ks': ks,
        'stride': stride,
        'inputh': inputh,
        'inputw': inputw
    } 
    return config


def add_ele_to_log(op, tensorshapes):
    config = {
        'op': op,
        'input_tensors': tensorshapes
    }
    return config


def save_to_models(modelpath, inputs, outputs, blockname, label = ""):
    savepath = os.path.join(modelpath, "models")
    os.makedirs(savepath, exist_ok=True)

    if label == "":
        graphname = blockname + "_" + str(int(time.time() * 1000))
    else:
        graphname = blockname+"_"+label
    savepath = os.path.join(savepath, graphname)
    os.makedirs(savepath, exist_ok=True)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        inputs_name, outputs_name = to_saved_model(
                sess, inputs, outputs, savepath
            )
    return savepath, inputs_name, outputs_name



def to_saved_model(
    sess, 
    inputs: List[Union[tf.Tensor, tf.Operation]], 
    outputs: List[Union[tf.Tensor, tf.Operation]], 
    path: str
):
    from tensorflow.python.saved_model import signature_constants
    from tensorflow.python.saved_model import tag_constants

    # set input/output nodename
    inputs_dic = {
        "input_{}".format(idx): i if isinstance(i, tf.Tensor) else i.outputs[0]
        for idx, i in zip(range(len(inputs)), inputs)
    }
    outputs_dic = {
        "output_{}".format(idx): o if isinstance(o, tf.Tensor) else o.outputs[0]
        for idx, o in zip(range(len(outputs)), outputs)
    }

    saved_model = tf.compat.v1.saved_model

    builder = saved_model.builder.SavedModelBuilder(path)
    sigs = {}
    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        saved_model.signature_def_utils.predict_signature_def(
        inputs_dic, outputs_dic
    )
    builder.add_meta_graph_and_variables(
        sess, 
        [tag_constants.SERVING], 
        signature_def_map = sigs
    )
    builder.save()
    return list(inputs_dic.keys()), list(outputs_dic.keys())
