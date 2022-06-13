# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import yaml
import logging
import importlib
from . import config_sampler
logging = logging.getLogger("nn-Meter")


__BUILTIN_KERNELS__ = {
    # builtin name: [kernel class name, kernel sampler class name]
    "conv-bn-relu": ["ConvBnRelu", "ConvSampler"],
    "conv-bn-relu6": ["ConvBnRelu6", "ConvSampler"],
    "conv-bn": ["ConvBn", "ConvSampler"],
    "conv-relu": ["ConvRelu", "ConvSampler"],
    "conv-relu6": ["ConvRelu6", "ConvSampler"],
    "conv-hswish": ["ConvHswish", "ConvSampler"],
    "conv-block": ["ConvBlock", "ConvSampler"],
    "conv-bn-hswish": ["ConvBnHswish", "ConvSampler"],
    # dwconv
    "dwconv-bn": ["DwConvBn", "DwConvSampler"],
    "dwconv-relu": ["DwConvRelu", "DwConvSampler"],
    "dwconv-relu6": ["DwConvRelu6", "DwConvSampler"],
    "dwconv-bn-relu": ["DwConvBnRelu", "DwConvSampler"],
    "dwconv-bn-relu6": ["DwConvBnRelu6", "DwConvSampler"],
    "dwconv-block": ["DwConvBlock", "DwConvSampler"],
    "dwconv-bn-hswish": ["ConvBnHswish", "DwConvSampler"],
    # others
    "maxpool": ["MaxPoolBlock", "PoolingSampler"],
    "avgpool": ["AvgPoolBlock", "PoolingSampler"],
    "fc": ["FCBlock", "FCSampler"],
    "concat": ["ConcatBlock", "ConcatSampler"],
    "split": ["SplitBlock", "CinEvenSampler"],
    "channelshuffle": ["ChannelShuffle", "CinEvenSampler"],
    "se": ["SEBlock", "CinEvenSampler"],
    "global-avgpool": ["GlobalAvgPoolBlock", "GlobalAvgPoolSampler"],
    "bnrelu": ["BnRelu", "HwCinSampler"],
    "bn": ["BnBlock", "HwCinSampler"],
    "hswish": ["HswishBlock", "HwCinSampler"],
    "relu": ["ReluBlock", "HwCinSampler"],
    "addrelu": ["AddRelu", "HwCinSampler"],
    "add": ["AddBlock", "HwCinSampler"],
}


__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__registry_cfg_filename__ = 'registry.yaml'
__REG_KERNELS__ = {}
if os.path.isfile(os.path.join(__user_config_folder__, __registry_cfg_filename__)):
    with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'r') as fp:
        registry_modules = yaml.load(fp, yaml.FullLoader)
    if "kernels" in registry_modules:
        __REG_KERNELS__ = registry_modules["kernels"]


def generate_model_for_kernel(kernel_type, config, save_path, implement='tensorflow', batch_size=1):
    """ get the nn model for predictor build.
    """
    # get kernel class information
    if kernel_type in __REG_KERNELS__:
        kernel_info = __REG_KERNELS__[kernel_type]
        assert kernel_info["implement"] == implement
        sys.path.append(kernel_info["package_location"])
        kernel_name = kernel_info["class_name"]
        kernel_module = importlib.import_module(kernel_info["class_module"])

    elif kernel_type in __BUILTIN_KERNELS__:
        kernel_name = __BUILTIN_KERNELS__[kernel_type][0]
        if implement == 'tensorflow':
            from nn_meter.builder.nn_modules.tf_networks import blocks
        elif implement == 'torch':
            from nn_meter.builder.nn_modules.torch_networks import blocks
        else:
            raise NotImplementedError('You must choose one implementation of kernel from "tensorflow" or "pytorch"')
        kernel_module = blocks

    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}. Please register the kernel first.")

    # get kernel class and create kernel instance by needed_config
    kernel_class = getattr(kernel_module, kernel_name)(config, batch_size)
    input_tensor_shape = kernel_class.input_tensor_shape
    model = kernel_class.get_model()

    # save model file to savepath
    kernel_class.save_model(save_path)
    logging.info(f"{kernel_type} model is generated and saved to {save_path}.")

    return model, input_tensor_shape, config


def get_sampler_for_kernel(kernel_type, sample_num, sampling_mode, configs = None):
    """ return the list of sampled data configurations in prior and finegrained sampling mode
    """
    # get kernel sampler class information
    if kernel_type in __REG_KERNELS__:
        kernel_info = __REG_KERNELS__[kernel_type]
        sys.path.append(kernel_info["package_location"])
        sampler_name = kernel_info["sampler_name"]
        sampler_module = importlib.import_module(kernel_info["sampler_module"])
    elif kernel_type in __BUILTIN_KERNELS__:
        sampler_name = __BUILTIN_KERNELS__[kernel_type][1]
        sampler_module = config_sampler

    # get kernel class and create kernel instance by needed_config
    sampler_class = getattr(sampler_module, sampler_name)()

    # initialize sampling, based on prior distribution
    if sampling_mode == 'prior':
        sampled_cfgs = sampler_class.prior_config_sampling(sample_num)
    # fine-grained sampling for data with large error points
    elif sampling_mode == 'finegrained':
        sampled_cfgs = sampler_class.finegrained_config_sampling(configs, sample_num)
    return sampled_cfgs


def list_kernels():
    return list(__BUILTIN_KERNELS__.keys()) + ["* " + item for item in list(__REG_KERNELS__.keys())]
