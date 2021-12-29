# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
from sklearn.metrics import mean_squared_error


def get_flop(input_channel, output_channel, k, H, W, stride):
    paras = output_channel * (k * k * input_channel + 1)
    flops = 2 * H / stride * W / stride * paras
    return flops, paras


def get_conv_mem(input_channel, output_channel, k, H, W, stride):
    paras = output_channel * (k * k * input_channel + 1)
    mem = paras + output_channel * H / stride * W / stride + input_channel * H * W
    return mem


def get_depthwise_flop(input_channel, output_channel, k, H, W, stride):
    paras = output_channel * (k * k + 1)
    flops = 2 * H / stride * W / stride * paras
    return flops, paras


def get_flops_params(blocktype, hw, cin, cout, kernelsize, stride):
    if "dwconv" in blocktype:
        return get_depthwise_flop(cin, cout, kernelsize, hw, hw, stride)
    elif "conv" in blocktype:
        return get_flop(cin, cout, kernelsize, hw, hw, stride)
    elif "fc" in blocktype:
        flop = (2 * cin + 1) * cout
        return flop, flop


def read_kernel_latency(filename):
    pass


def get_feature(op, inputh, cin, cout, ks, s):
    if s != None and "conv" in op:

        flops, params = get_flops_params(op, inputh, cin, cout, ks, s)
        features = [inputh, cin, cout, ks, s, flops / 2e6, params / 1e6]

    elif "fc" in op:
        flop = (2 * cin + 1) * cout
        features = [cin, cout, flop / 2e6, flop / 1e6]

    elif "pool" in op:
        features = [inputh, cin, cout, ks, s]

    elif "se" in op:
        features = [inputh, cin]

    elif op in ["hwish", "hswish"]:
        features = [inputh, cin]

    return features
