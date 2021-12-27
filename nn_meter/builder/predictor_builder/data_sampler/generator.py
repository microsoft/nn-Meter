# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
import tensorflow as tf
from finegrained_sampler import *
from prior_distribution_sampler import *
from nn_meter.builder.utils import builder_config
from nn_meter.builder.nn_generator.predbuild_model import get_predbuild_model
        

config_for_blocks = {
      "conv_bn_relu":         ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
      "conv_bn_relu6":        ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
      "conv_bn":              ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
      "conv_relu":            ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
      "conv_relu6":           ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
      "conv_hswish":          ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
      "conv_block":           ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
      "conv_bn_hswish":       ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
      # "conv_bn_relu_maxpool": ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES", "POOL_STRIDES"],

      "dwconv_bn":            ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
      "dwconv_relu":          ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
      "dwconv_relu6":         ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
      "dwconv_bn_relu":       ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
      "dwconv_bn_relu6":      ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
      "dwconv_block":         ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
      "dwconv_bn_hswish":     ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],

      "maxpool_block":        ["HW", "CIN", "KERNEL_SIZE", "POOL_STRIDES"],
      "avgpool_block":        ["HW", "CIN", "KERNEL_SIZE", "POOL_STRIDES"],

      "fc_block":             ["CIN", "COUT"],

      "concat_block":         ["HW", "NS", "CINS"],
      "concat_pad":           ["HW", "NS", "CINS"],

      "split_block":          ["HW", "CIN"],
      "channel_shuffle_block":["HW", "CIN"],
      "se_block":             ["HW", "CIN"],

      "global_avgpool_block": ["HW", "CIN"],
      "bn_relu":              ["HW", "CIN"],
      "bn_block":             ["HW", "CIN"],
      "hswish_block":         ["HW", "CIN"],
      "relu_block":           ["HW", "CIN"],
      "add_relu":             ["HW", "CIN"],
      "add_block":            ["HW", "CIN"], 
}


def block_sampling_from_prior(block_type, count):
        """
        return the list of sampled data configurations in initial sampling phase

        @params

        block_type: identical kernel name 
        count: for the target kernel, we sample `count` data from prior distribution.
        
        """
        assert block_type in config_for_blocks.keys(), f"not supported block type: {block_type}. Supported type includes {config_for_blocks.keys()}."

        if "conv" in block_type:
            return sampling_conv(count)
        elif "dwconv" in block_type:
            return sampling_dwconv(count)
        elif block_type == 'maxpool_block': # we only sample hws and cins, and keep ks=3, strides=2 fixed
            return sampling_pooling(count, fix_ks=3, fix_stride=2)
        elif block_type == 'avgpool_block': # we only sample hws and cins, and keep ks=3, strides=1 fixed
            return sampling_pooling(count, fix_ks=3, fix_stride=1)
        elif block_type == 'fc_block': # half samples have fixed cout as 1000, other samples have random cout
            return sampling_fc(int(count * 0.5), fix_cout = 1000) + sampling_fc(int(count * 0.5), fix_cout = False)
        elif "concat" in block_type:
            return sampling_concats(count)
        elif block_type in ['split_block', 'channel_shuffle_block', 'se_block']:
            return sampling_hw_cin_odd(count)
        elif block_type == 'global_avgpool_block': 
            cfgs = sampling_hw_cin(count)
            for cfg in cfgs: cfg["HW"] = 7
            return cfgs
        else: # 'hswish_block', 'bn_relu', 'bn_block', 'relu_block'
            return sampling_hw_cin(count, resize = True)
            


def block_sampling_with_finegrained(block_type, count, cfgs):
        """
        return the list of sampled data configurations in finegrained sampling phase

        @params

        blocktype: identical kernel name 
        count: int 
        for each large-error-data-point, we sample `count` more data around it.
        cfgs: list
        each item in the list represent a large-error-data-point. each item is a dictionary, storing the configuration
        
        """
        assert block_type in config_for_blocks.keys(), f"not supported block type: {block_type}. Supported type includes {config_for_blocks.keys()}."

        if "conv" in block_type:
            return finegrained_sampling_conv(cfgs, count)
        elif "dwconv" in block_type:
            return finegrained_sampling_dwconv(cfgs, count)
        elif block_type in ['maxpool_block', 'avgpool_block']:
            return finegrained_sampling_pooling(count, fix_ks=3, fix_stride=1)
        if block_type == 'fc_block':
            return finegrained_sampling_fc(cfgs, count)
        if block_type in ['concat_block']:
            return finegrained_sampling_concat(cfgs, count)
        if block_type in ['split_block', 'se_block', 'channel_shuffle_block']:
            return finegrained_sampling_hw_cin_odd(cfgs, count)
        else:
            return finegrained_sampling_hw_cin(cfgs, count)


class Generator:
    def __init__(self, block_type, sample_num):
        self.block_type = block_type
        self.config = builder_config(block_type)
        self.sample_num = sample_num
        self.savepath = os.path.join(self.config.workspace_path, "adaptive")
        self.testcase = {}

    def generate_block_from_cfg(self, block_cfgs, block_type):
        """
        generate tensorflow models for sampled data

        @params

        blockcfgs: each item in the list represent a sampled data. each item is a dictionary, storing the configuration
        block_type: identical kernel name 
        """
        id = 0
        logging.info(f"building block for {block_type}...")
        for block_cfg in block_cfgs:
            tf.reset_default_graph()
            model, input_shape, config = get_predbuild_model(block_type, block_cfg, savepath=self.savepath)
            self.testcase[id] = {
                'model': self.savepath,
                'shape': input_shape,
                'config': config
            }
            id += 1

    def run(self, sample_stage = 'prior'):
        """ sample N configurations for target kernel, generate tensorflow model files (pb).
        
        @params
        
        sample_stage: path of the directory containing all experiment runs. choose from ['prior', 'finegrained']
        """
        # initialize sampling, based on prior distribution
        if sample_stage == 'prior':
            sampled_cfgs = block_sampling_from_prior(self.block_type, self.sample_num)
        # fine-grained sampling for data with large error points
        elif sample_stage == 'finegrained':
            sampled_cfgs = block_sampling_with_finegrained(self.block_type, self.sample_num, self.cfg['BLOCK']['DATA'])

        # for all sampled configurations, generate tensorflow model files 
        self.generate_block_from_cfg(sampled_cfgs, self.block_type)
