# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import yaml
import json
import logging
import importlib
from nn_meter.builder.backend_meta.utils import read_profiled_results
logging = logging.getLogger("nn-Meter")


feature_for_kernel = {
    # conv
    "conv-bn-relu":         ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-bn-relu6":        ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-bn":              ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-relu":            ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-relu6":           ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-hswish":          ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-block":           ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-bn-hswish":       ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    # dwconv ("COUT" will always be the same as "CIN")
    "dwconv-bn":            ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "dwconv-relu":          ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "dwconv-relu6":         ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "dwconv-bn-relu":       ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "dwconv-bn-relu6":      ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "dwconv-block":         ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "dwconv-bn-hswish":     ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    # pooling ("COUT" will always be the same as "CIN")
    "maxpool":              ["HW", "CIN", "COUT", "KERNEL_SIZE", "POOL_STRIDES"],
    "avgpool":              ["HW", "CIN", "COUT", "KERNEL_SIZE", "POOL_STRIDES"],
    # others
    "fc":                   ["CIN", "COUT"],
    "concat":               ["HW", "CIN1", "CIN2", "CIN3", "CIN4"],
    "split":                ["HW", "CIN"],
    "channelshuffle":       ["HW", "CIN"],
    "se":                   ["HW", "CIN"],
    "global-avgpool":       ["HW", "CIN"],
    "bnrelu":               ["HW", "CIN"],
    "bn":                   ["HW", "CIN"],
    "hswish":               ["HW", "CIN"],
    "relu":                 ["HW", "CIN"],
    # In "addrelu" block and "add" block, the second feature "CIN" will always be the same as
    # the third feature
    "addrelu":              ["HW", "CIN", "CIN"],
    "add":                  ["HW", "CIN", "CIN"], 
}

__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__registry_cfg_filename__ = 'registry.yaml'
__REG_KERNELS__ = {}
if os.path.isfile(os.path.join(__user_config_folder__, __registry_cfg_filename__)):
    with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'r') as fp:
        registry_modules = yaml.load(fp, yaml.FullLoader)
    if "kernels" in registry_modules:
        __REG_KERNELS__ = registry_modules["kernels"]


class BaseFeatureParser:
    def __init__(self, kernel_type):
        self.kernel_type = kernel_type
        self.needed_config = feature_for_kernel[kernel_type]

    def get_feature_by_config(self, config_dict):
        if "COUT" not in config_dict and "COUT" in self.needed_config:
            config_dict["COUT"] = config_dict["CIN"]
        feature = [config_dict[data] for data in self.needed_config]
        return feature

    def get_config_by_feature(self, feature):
        assert len(self.needed_config) == len(feature)
        config = {k: v for k, v in zip(self.needed_config, feature)}
        return config


class FlopsParamParser(BaseFeatureParser):
    def get_feature_by_config(self, config_dict):
        if "COUT" not in config_dict and "COUT" in self.needed_config:
            config_dict["COUT"] = config_dict["CIN"]
        feature = [config_dict[data] for data in self.needed_config]
        from .utils import get_flops_params
        flop, param = get_flops_params(self.kernel_type, config_dict)
        flop /= 2e6
        param /= 1e6
        feature.extend([flop, param])
        return feature

    def get_config_by_feature(self, feature):
        # remove flops and params num feature from feature vector
        feature = feature[:-2]
        assert len(self.needed_config) == len(feature)
        config = {k: v for k, v in zip(self.needed_config, feature)}
        return config


def get_feature_parser(kernel_type):
    if kernel_type in __REG_KERNELS__:
        kernel_info = __REG_KERNELS__[kernel_type]
        sys.path.append(kernel_info["package_location"])
        parser_name = kernel_info["parser_name"]
        parser_module = importlib.import_module(kernel_info["parser_module"])
        return getattr(parser_module, parser_name)(kernel_type)
    elif kernel_type in feature_for_kernel:
        if "conv" in kernel_type or "dwconv" in kernel_type or "fc" in kernel_type:
            return FlopsParamParser(kernel_type)
        else:
            return BaseFeatureParser(kernel_type)


def get_data_by_profiled_results(kernel_type, feature_parser, cfgs_path, labs_path = None,
                                 save_path = None, predict_label = "latency"):
    ''' return (features, latency)
    kernel_type (str): type of kernel
    
    feature_parser (subclass instance of BaseFeatureParser) the parser containing the feature parsing script

    cfgs_path (str or dict): path of config information dict, or dict of "origin_kernels.json", such as
        {
            "conv-bn-relu": {
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

    labs_path (str or dict): path of profiled label information dict, or dict of "profiled_results", such as
        {
            "conv-bn-relu": {
                "id_0": {
                    "latency": "42.001 +- 1.0"
                }
            }
        }
        if labs_path == None, it means latency (or other label) information are also included in cfgs_path.

    save_path (str): the path to save the feature and latency information, including file name. If save_path is None, the data will not be saved.

    predict_label (str): the predicting label to build kernel predictor
    '''
    if labs_path == None:
        if type(cfgs_path) == tuple:
            cfgs_path, labs_path = cfgs_path
        else:
            labs_path = cfgs_path
    if isinstance(cfgs_path, str):
        with open(cfgs_path, 'r') as fp:
            cfgs_dict = json.load(fp)[kernel_type]
    else:
        cfgs_dict = cfgs_path[kernel_type] if kernel_type in cfgs_path else cfgs_path
    if isinstance(labs_path, str):
        with open(labs_path, 'r') as fp:
            labs_dict = read_profiled_results(json.load(fp))[kernel_type]
    else:
        labs_dict = labs_path[kernel_type] if kernel_type in labs_path else labs_path

    paths, features, labs = [], [], []
    for id in labs_dict.keys():
        try:
            path = cfgs_dict[id]["model"]
            configs = cfgs_dict[id]["config"]
            feature = feature_parser.get_feature_by_config(configs)
            if predict_label == "latency":
                label = labs_dict[id]["latency"].avg
            else:
                label = labs_dict[id][predict_label]
            if label != 0.0:
                paths.append(os.path.basename(path))
                features.append(feature)
                labs.append(label)
        except:
            pass

    # save features and latency information to `save_path`
    if save_path:
       import pandas as pd
       cols = feature_parser.needed_config[:]
       if len(features[0]) - len(feature_parser.needed_config) > 0: # there are extra features beyond needed config
           cols += [f'feature_{i}' for i in range(len(features[0]) - len(feature_parser.needed_config))]
       data_df = pd.DataFrame(features, columns=cols)
       data_df = pd.concat([pd.DataFrame(paths, columns=["model_path"]), data_df], axis=1)
       data_df[predict_label] = labs
       data_df.to_csv(save_path, index=False)
       logging.info(f'Saved the feature table of all data for {kernel_type} in path {save_path}.')

    return (features, labs)
