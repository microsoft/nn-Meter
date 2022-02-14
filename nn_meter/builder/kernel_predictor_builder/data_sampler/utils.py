# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import yaml
import logging
import importlib
import numpy as np
from . import config_sampler
from nn_meter.builder.utils import get_inputs_by_shapes
from nn_meter.builder.nn_generator.tf_networks import blocks


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


def generate_model_for_kernel(kernel_type, config, savepath):
    """ get the nn model for predictor build. returns: input_tensors, output_tensors, configuration_key, and graphname, they are for saving tensorflow v1.x models
    """
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
    model(get_inputs_by_shapes(input_tensor_shape))

    # save model file to savepath
    from tensorflow import keras
    keras.models.save_model(model, savepath)
    logging.info(f"{kernel_type} model is generated and saved to {savepath}.")

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
        sampled_cfgs = sampler_class.prior_kernel_sampling(sample_num)
    # fine-grained sampling for data with large error points
    elif sampling_mode == 'finegrained':
        sampled_cfgs = sampler_class.finegrained_config_sampling(configs, sample_num)
    return sampled_cfgs


def list_kernels():
    return __BUILTIN_KERNELS__.keys() + ["* " + item for item in list(__REG_KERNELS__.keys())]


def inverse_transform_sampling(data, n_bins = 40, n_samples = 1000):
    ''' calculate inversed cdf, for sampling by possibility
    '''
    import scipy.interpolate as interpolate
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    data = inv_cdf(r)
    ndata = [int(x) for x in data]
    return ndata


def sample_based_on_distribution(data, count):
    ''' use data to calculate a inversed cdf, and sample `count` data from such distribution
    '''
    return inverse_transform_sampling(data, n_samples=count)


def data_validation(data, cdata):
    ''' convert sampled data to valid configuration, e.g.,: kernel size = 1, 3, 5, 7

    @params:
    data: the origin data value.
    cdata: valid configuration value.
    '''
    newlist = []
    for da in cdata:
        value = [abs(da - x) for x in data]
        newlist.append(value)

    newlist = list(np.asarray(newlist).T)    
    cda = [list(d).index(min(d)) for d in newlist]
    redata = [cdata[x] for x in cda]
    return redata
