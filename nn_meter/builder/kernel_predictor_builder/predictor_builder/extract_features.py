# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from ..utils import config_for_kernel

def get_conv_flop_params(hw, cin, cout, kernel_size, stride):
    params = cout * (kernel_size * kernel_size * cin + 1)
    flops = 2 * hw / stride * hw / stride * params
    return flops, params


def get_dwconv_flop_params(hw, cout, kernel_size, stride):
    params = cout * (kernel_size * kernel_size + 1)
    flops = 2 * hw / stride * hw / stride * params
    return flops, params


def get_fc_flop_params(cin, cout):
    params = (2 * cin + 1) * cout
    flops = params
    return flops, params


def get_flops_params(kernel_type, config):
    hw, cin, cout, kernel_size, stride = config["HW"], config["CIN"], config["COUT"], \
                                         config["KERNEL_SIZE"], config["STRIDE"]
    if "conv" in kernel_type:
        return get_conv_flop_params(hw, cin, cout, kernel_size, stride)
    elif "dwconv" in kernel_type:
        return get_dwconv_flop_params(hw, cout, kernel_size, stride)
    elif "fc" in kernel_type:
        return get_fc_flop_params(cin, cout)


def get_data_by_profiled_results(result):
    # read result
    
    # parse profiled results
    features, lats = [], []
    profiled_kernels = ...
    for data in profiled_kernels.values():
        configs = data["configs"]
        feature = get_features_by_config(configs)
        features.append(feature)
        latency = data["Latency"]["avg"]
        lats.append(latency)
    return features, lats


def get_features_by_config(kernel_type, config):
    # TODO: change concat implementation to clear here
    feature = [config[data] for data in config_for_kernel[kernel_type]]
    if "conv" in kernel_type or "dwconv" in kernel_type or "fc" in kernel_type:
        flop, param = get_flops_params(kernel_type, config) 
        feature.extend([flop, param])
    return feature
