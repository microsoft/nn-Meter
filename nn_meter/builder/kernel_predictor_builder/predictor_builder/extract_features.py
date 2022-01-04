# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
from ..utils import config_for_kernel
from nn_meter.builder.backend_meta.utils import read_profiled_results

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
                                         config["KERNEL_SIZE"], config["STRIDES"]
    if "conv" in kernel_type:
        return get_conv_flop_params(hw, cin, cout, kernel_size, stride)
    elif "dwconv" in kernel_type:
        return get_dwconv_flop_params(hw, cout, kernel_size, stride)
    elif "fc" in kernel_type:
        return get_fc_flop_params(cin, cout)


def get_data_by_profiled_results(kernel_type, cfgs_path, lats_path = None):
    ''' return (features, latency)
    kernel_type (str): type of kernel

    cfgs_path: path of config information dict, or dict of "origin_kernels.json", such as
    {
        "conv_bn_relu": {
            "id_0": {
                "model": "...",
                "shapes": [[14, 14, 98]],
                "config": {
                    "HW": 14,
                    "CIN": 98,
                    "COUT": 120,
                    "KERNEL_SIZE": 3,
                    "STRIDES": 1
                },
            }
        }
    }

    lats_path: pathe of profiled latency information dict, or dict of "profiled_results", such as
    {
        "conv_bn_relu": {
            "id_0": {
                "latency": "42.001 +- 1.0"
            }
        }
    }
    if lats_path == None, it means latency information are also included in cfgs_path.
    '''
    if lats_path == None:
        lats_path = cfgs_path
    if isinstance(cfgs_path, str):
        with open(cfgs_path, 'r') as fp:
            cfgs_dict = json.load(fp)[kernel_type]
    else:
        cfgs_dict = cfgs_path[kernel_type] if kernel_type in cfgs_path else cfgs_path
    if isinstance(lats_path, str):
        with open(lats_path, 'r') as fp:
            lats_path = read_profiled_results(json.load(fp))[kernel_type]
    else:
        lats_dict = lats_path[kernel_type] if kernel_type in lats_path else lats_path

    features, lats = [], []
    for id in cfgs_dict.keys():
        configs = cfgs_dict[id]["config"]
        feature = get_features_by_config(kernel_type, configs)
        features.append(feature)

        latency = lats_dict[id]["latency"]
        lats.append(latency)
    return (features, lats)


def get_features_by_config(kernel_type, config):
    feature = [config[data] for data in config_for_kernel[kernel_type]]
    if "conv" in kernel_type or "dwconv" in kernel_type or "fc" in kernel_type:
        flop, param = get_flops_params(kernel_type, config) 
        feature.extend([flop, param])
    return feature
