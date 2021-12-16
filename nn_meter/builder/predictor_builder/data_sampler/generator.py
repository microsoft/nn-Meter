# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
import tensorflow as tf
from finegrained_sampler import *
from prior_distribution_sampler import *
from nn_meter.builder.utils import builder_config
from nn_meter.builder.nn_generator.predbuild_model import get_predbuild_model


    

def block_sampling_with_finegrained(blocktype, count, cfgs):
        """
        return the list of sampled data configurations in finegrained sampling phase
        @params
        ----------
        blocktype: str
        identical kernel name 
        count: int 
        for each large-error-data-point, we sample `count` more data around it.
        cfgs: list
        each item in the list represent a large-error-data-point. each item is a dictionary, storing the configuration
        
        """
  
        if blocktype in['conv-bn-relu', 'conv-bn-relu6', 'conv-bn', 'conv-relu', 'conv-relu6', 'conv-bn-hswish', 'conv-hswish']:
            return finegrained_sampling_conv(cfgs, count)
        if blocktype in['dwconv-bn-relu', 'dwconv-bn-relu6', 'dwconv-bn', 'dwconv-relu', 'dwconv-relu6', 'dwconv-bn-hswish', 'maxpool', 'avgpool']:
            return finegrained_sampling_dwconv(cfgs, count)
        if blocktype in ['fc']:
            return finegrained_sampling_fc(cfgs, count)
        if blocktype in ['bnrelu', 'bn', 'relu', 'global-avgpool']:
            return finegrained_sampling_CIN(cfgs, count)
        if blocktype in ['split', 'hswish', 'se', 'channel-shuffle']:
            return finegrained_sampling_CIN_odd(cfgs, count)
        if blocktype in ['addrelu', 'add']:
            return finegrained_sampling_CIN(cfgs, count)
        if blocktype in ['concat']:
            return finegrained_sampling_concat(cfgs, count)
        

def block_sampling_from_prior(blocktype, count):
        """
        return the list of sampled data configurations in initial sampling phase
        @params
        ----------
        blocktype: str
        identical kernel name 
        count: int 
        for the target kernel, we sample `count` data from prior distribution.
        
        """
        if blocktype in['conv-bn-relu', 'conv-bn-relu6', 'conv-bn', 'conv-relu', 'conv-relu6', 'conv-bn-hswish', 'conv-hswish']:
            cins, couts, hws, ks, strides = sampling_conv(count)
            return (hws, ks, strides, cins, couts)
        if blocktype in['dwconv-bn-relu', 'dwconv-bn-relu6', 'dwconv-bn', 'dwconv-relu', 'dwconv-relu6', 'se', 'dwconv-bn-hswish']:
            cs, hws, ks, strides = sampling_dwconv(count)
            return (hws, ks, strides, cs)
        if blocktype in ['fc']:
            cins, couts = sampling_fc(int(count * 0.5), fix_cout = 1000)
            cins1, couts1 = sampling_fc(int(count * 0.5), fix_cout = False)
            cins.extend(cins1)
            couts.extend(couts1)
            return (cins, couts)
        if blocktype in ['maxpool']:
            hws, cins, ks, strides = sampling_pooling(count)
            return (hws, ks, strides, cins)
        if blocktype in ['avgpool']:
            hws, cins, ks, strides = sampling_pooling(count)
            ks = [3] * len(ks)
            strides = [1] * len(ks)
            return (hws, ks, strides, cins)
        if blocktype == 'global-avgpool':
            cins, couts = sampling_fc(count)
            hws = [7] * count 
            return (hws, cins)
        if blocktype in ['split', 'channel_shuffle', 'hswish', 'bnrelu', 'bn', 'relu']:
            cs, hws, ks, strides = sampling_dwconv(count, resize = True)
            ncs = []
            for c in cs:
                nc = c if c % 2 == 0 else c + 1
                ncs.append(nc)
            return (hws, ncs)
        if blocktype == 'concat':
            hws, ns, cin1, cin2, cin3, cin4 = sampling_concats(count)
            return (hws, ns, cin1, cin2, cin3, cin4)
        if blocktype == 'addrelu':
            hws, cs1, cs2 = sampling_addrelu(count)
            return (hws, cs1, cs2)


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
            savemodelpath, model, config, graphname = get_predbuild_model(block_type, blockcfg, saved=True)
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

