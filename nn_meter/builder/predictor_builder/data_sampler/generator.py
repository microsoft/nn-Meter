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
      "conv_block":           ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
      "conv_bn_relu_maxpool": ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES", "POOL_STRIDES"],
      "conv_bn_hswish":       ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
      "dwconv_bn_relu":       ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
      "dwconv_block":         ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
      "dwconv_bn_hswish":     ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
      "maxpool_block":        ["HW", "CIN", "KERNEL_SIZE", "POOL_STRIDES"],
      "avgpool_block":        ["HW", "CIN", "KERNEL_SIZE", "POOL_STRIDES"],
      "fc_block":             ["CIN", "COUT"],
      "hswish_block":         ["HW", "CIN"],
      "se_block":             ["HW", "CIN"],
      "global_avgpool_block": ["HW", "CIN"],
      "split_block":          ["HW", "CIN"],
      "channel_shuffle":      ["HW", "CIN"],
      "bn_relu":              ["HW", "CIN"],
      "concat_block":         ["HW", "CIN"],
      "concat_pad":           ["HW", "CIN"],
      "add_relu":             ["HW", "CIN"],
      "add_block":            ["HW", "CIN"],
      "bn_block":             ["HW", "CIN"],
      "relu_block":           ["HW", "CIN"]
}



def block_sampling_from_prior(block_type, count):
        """
        return the list of sampled data configurations in initial sampling phase

        @params

        block_type: identical kernel name 
        count: for the target kernel, we sample `count` data from prior distribution.
        
        """
        if block_type in['conv_bn_relu', 'conv_bn_relu6', 'conv_bn', 'conv_relu', 'conv_relu6', 'conv_bn_hswish', 'conv_hswish']:
            cins, couts, hws, ks, strides = sampling_conv(count)
            return (hws, ks, strides, cins, couts)
        if block_type in['dwconv_bn_relu', 'dwconv_bn_relu6', 'dwconv_bn', 'dwconv_relu', 'dwconv_relu6', 'se_block', 'dwconv_bn_hswish']:
            cs, hws, ks, strides = sampling_dwconv(count)
            return (hws, ks, strides, cs)
        if block_type in ['fc']:
            cins, couts = sampling_fc(int(count * 0.5), fix_cout = 1000)
            cins1, couts1 = sampling_fc(int(count * 0.5), fix_cout = False)
            cins.extend(cins1)
            couts.extend(couts1)
            return (cins, couts)
        if block_type in ['maxpool']:
            hws, cins, ks, strides = sampling_pooling(count)
            return (hws, ks, strides, cins)
        if block_type in ['avgpool']:
            hws, cins, ks, strides = sampling_pooling(count)
            ks = [3] * len(ks)
            strides = [1] * len(ks)
            return (hws, ks, strides, cins)
        if block_type == 'global_avgpool':
            cins, couts = sampling_fc(count)
            hws = [7] * count 
            return (hws, cins)
        if block_type in ['split_block', 'channel_shuffle_block', 'hswish_block', 'bnrelu_block', 'bn_block', 'relu_block']:
            cs, hws, ks, strides = sampling_dwconv(count, resize = True)
            ncs = []
            for c in cs:
                nc = c if c % 2 == 0 else c + 1
                ncs.append(nc)
            return (hws, ncs)
        if block_type == 'concat_block':
            hws, ns, cin1, cin2, cin3, cin4 = sampling_concats(count)
            return (hws, ns, cin1, cin2, cin3, cin4)
        if block_type == 'addrelu_block':
            hws, cs1, cs2 = sampling_addrelu(count)
            return (hws, cs1, cs2)


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
        if block_type in['conv_bn_relu', 'conv_bn_relu6', 'conv_bn', 'conv_relu', 'conv_relu6', 'conv_bn_hswish', 'conv_hswish']:
            return finegrained_sampling_conv(cfgs, count)
        if block_type in['dwconv_bn_relu', 'dwconv_bn_relu6', 'dwconv_bn', 'dwconv_relu', 'dwconv_relu6', 'dwconv_bn_hswish', 'maxpool', 'avgpool']:
            return finegrained_sampling_dwconv(cfgs, count)
        if block_type in ['fc']:
            return finegrained_sampling_fc(cfgs, count)
        if block_type in ['bnrelu', 'bn', 'relu', 'global_avgpool']:
            return finegrained_sampling_CIN(cfgs, count)
        if block_type in ['split', 'hswish', 'se', 'channel_shuffle']:
            return finegrained_sampling_CIN_odd(cfgs, count)
        if block_type in ['addrelu', 'add']:
            return finegrained_sampling_CIN(cfgs, count)
        if block_type in ['concat']:
            return finegrained_sampling_concat(cfgs, count)


class Generator:
    def __init__(self, block_type, sample_num):
        self.block_type = block_type
        self.config = builder_config(block_type)
        self.sample_num = sample_num
        self.savepath = os.path.join(self.config.workspace_path, "adaptive")
        self.testcase = {}

    def generate_block_from_cfg(self, blockcfgs, block_type):
        """
        generate tensorflow models for sampled data

        @params

        blockcfgs: each item in the list represent a sampled data. each item is a dictionary, storing the configuration
        block_type: identical kernel name 
        """
        id = 0
        for blockcfg in blockcfgs:
            logging.info(f"building block for {block_type} with config: {blockcfg}")
            tf.reset_default_graph()
            savemodelpath, model, config = get_predbuild_model(block_type, blockcfg, savepath=self.savepath)
            self.testcase[id] = {}
            id += 1

    def run(self, sample_stage = 'prior'):
        """ sample N configurations for target kernel, generate tensorflow model files (pb).
        
        @params
        
        sample_stage: path of the directory containing all experiment runs.
        """
        # initialize sampling, based on prior distribution
        if sample_stage == 'prior':
            samples = block_sampling_from_prior(self.block_type, self.sample_num)
        # fine-grained sampling for data with large error points
        else:
            samples = block_sampling_with_finegrained(self.block_type, self.sample_num, self.cfg['BLOCK']['DATA'])
       
        # for all sampled configurations, generate tensorflow model files 
        self.generate_block_from_cfg(samples, self.block_type)
