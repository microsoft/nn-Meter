# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import random


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


def get_sampling_ks(ks_list, layers):
    return [random.choice(ks_list) for _ in range(layers)]


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


def add_ele_to_log(op, tensor_shapes):
    config = {
        'op': op,
        'input_tensors': tensor_shapes
    }
    return config
