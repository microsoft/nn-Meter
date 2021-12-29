# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np


def get_flop(cin, cout, k, H, W, stride):
    paras = cout * (k * k * cin + 1)
    flops = 2 * H / stride * W / stride * paras
    return flops, paras


def get_conv_mem(cin, cout, k, H, W, stride):
    paras = cout * (k * k * cin + 1)
    mem = paras + cout * H / stride * W / stride + cin * H * W
    return mem


def get_depthwise_flop(cin, cout, k, H, W, stride):
    paras = cout * (k * k + 1)
    flops = 2 * H / stride * W / stride * paras
    return flops, paras


def get_flops_params(kernel_type, hw, cin, cout, kernelsize, stride):
    if "dwconv" in kernel_type:
        return get_depthwise_flop(cin, cout, kernelsize, hw, hw, stride)
    elif "conv" in kernel_type:
        return get_flop(cin, cout, kernelsize, hw, hw, stride)
    elif "fc" in kernel_type:
        flop = (2 * cin + 1) * cout
        return flop, flop


def get_features_by_config(kernel_type, config):
    
    # if s != None and "conv" in kernel_type:

    #     flops, params = get_flops_params(op, inputh, cin, cout, ks, s)
    #     features = [inputh, cin, cout, ks, s, flops / 2e6, params / 1e6]

    # elif "fc" in op:
    #     flop = (2 * cin + 1) * cout
    #     features = [cin, cout, flop / 2e6, flop / 1e6]

    # elif "pool" in op:
    #     features = [inputh, cin, cout, ks, s]

    # elif "se" in op:
    #     features = [inputh, cin]

    # elif op in ["hwish", "hswish"]:
    #     features = [inputh, cin]

    # return features
    pass
