# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .build_block import build_block
from .networks.block_utils import save_to_models
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from data_sampler.block_sampler import block_sampling_from_prior, block_sampling_with_finegrained
import logging
from nn_meter.builder.utils import builder_config

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
            input_tensor, output_tensor, config, graphname = build_block(block_type, blockcfg)

            output_tensors = output_tensor if isinstance(output_tensor, list) else [output_tensor]
            input_tensors = input_tensor if isinstance(input_tensor, list) else [input_tensor]

            try:
                savemodelpath, tfpath, pbpath, inputnames, outputnames = save_to_models(self.args.savepath, input_tensors, output_tensors, graphname, config)
                self.testcase[id] = {}
                self.testcase[id]['saved_model'] = savemodelpath
                self.testcase[id]['tflite_path'] = tfpath
                self.testcase[id]['pb_path'] = pbpath
                self.testcase[id]['config'] = config
            except:
                pass
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
