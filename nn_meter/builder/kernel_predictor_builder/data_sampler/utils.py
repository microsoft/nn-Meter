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
    "conv_bn_relu": ["ConvBnRelu", "ConvSampler"],
    "conv_bn_relu6": ["ConvBnRelu6", "ConvSampler"],
    "conv_bn": ["ConvBn", "ConvSampler"],
    "conv_relu": ["ConvRelu", "ConvSampler"],
    "conv_relu6": ["ConvRelu6", "ConvSampler"],
    "conv_hswish": ["ConvHswish", "ConvSampler"],
    "conv_block": ["ConvBlock", "ConvSampler"],
    "conv_bn_hswish": ["ConvBnHswish", "ConvSampler"],
    # dwconv
    "dwconv_bn": ["DwConvBn", "DwConvSampler"],
    "dwconv_relu": ["DwConvRelu", "DwConvSampler"],
    "dwconv_relu6": ["DwConvRelu6", "DwConvSampler"],
    "dwconv_bn_relu": ["DwConvBnRelu", "DwConvSampler"],
    "dwconv_bn_relu6": ["DwConvBnRelu6", "DwConvSampler"],
    "dwconv_block": ["DwConvBlock", "DwConvSampler"],
    "dwconv_bn_hswish": ["ConvBnHswish", "DwConvSampler"],
    # others
    "maxpool_block": ["MaxPoolBlock", "PoolingSampler"],
    "avgpool_block": ["AvgPoolBlock", "PoolingSampler"],
    "fc_block": ["FCBlock", "FCSampler"],
    "concat_block": ["ConcatBlock", "ConcatSampler"],
    "split_block": ["SplitBlock", "CinOddSampler"],
    "channel_shuffle": ["ChannelShuffle", "CinOddSampler"],
    "se_block": ["SEBlock", "CinOddSampler"],
    "globalavgpool_block": ["GlobalAvgPoolBlock", "GlobalAvgPoolSampler"],
    "bn_relu": ["BnRelu", "HwCinSampler"],
    "bn_block": ["BnBlock", "HwCinSampler"],
    "hswish_block": ["HswishBlock", "HwCinSampler"],
    "relu_block": ["ReluBlock", "HwCinSampler"],
    "add_relu": ["AddRelu", "HwCinSampler"],
    "add_block": ["AddBlock", "HwCinSampler"],
}


__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__registry_cfg_filename__ = 'registry.yaml'
__REG_KERNELS__ = {}
if os.path.isfile(os.path.join(__user_config_folder__, __registry_cfg_filename__)):
    with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'r') as fp:
        registry_modules = yaml.load(fp, yaml.FullLoader)
    if "kernels" in registry_modules:
        __REG_KERNELS__ = registry_modules["kernels"]


def generate_model_for_kernel(kernel_type, config, save_path, implement='tensorflow'):
    """ get the nn model for predictor build. returns: input_tensors, output_tensors, configuration_key, and graphname, they are for saving tensorflow v1.x models
    """
    if implement == 'tensorflow':
        from nn_meter.builder.nn_generator.tf_networks import blocks
    elif implement == 'torch':
        from nn_meter.builder.nn_generator.torch_networks import blocks
    else:
        raise NotImplementedError('You must choose one implementation of kernel from "tensorflow" or "pytorch"')

    # get kernel class information
    if kernel_type in __REG_KERNELS__:
        kernel_info = __REG_KERNELS__[kernel_type]
        sys.path.append(kernel_info["package_location"])
        kernel_name = kernel_info["class_name"]
        kernel_module = importlib.import_module(kernel_info["class_module"])
    elif kernel_type in __BUILTIN_KERNELS__:
        kernel_name = __BUILTIN_KERNELS__[kernel_type][0]
        kernel_module = blocks

    # get kernel class and create kernel instance by needed_config
    kernel_class = getattr(kernel_module, kernel_name)(config)
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
