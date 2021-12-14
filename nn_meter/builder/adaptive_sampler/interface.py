# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from argparse import ArgumentParser

from regression.build_regression_model import*
from ..model_generator.adasample_model import get_adasample_model

from silence_tensorflow import silence_tensorflow
silence_tensorflow()


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf

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
            input_tensor, output_tensor, config, graphname = get_adasample_model(block_type, blockcfg)

            output_tensors = output_tensor if isinstance(output_tensor, list) else [output_tensor]
            input_tensors = input_tensor if isinstance(input_tensor, list) else [input_tensor]

            try:
                savemodelpath, tfpath, pbpath, inputnames, outputnames = save_to_models(self.args.savepath, input_tensors, output_tensors, graphname, config)
                self.testcase[id] = {}
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



# sampling from prior distribution
def init_sampler():
    
    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument("--config", type=str,default="configs/conv.yaml")
   
    parser.add_argument("--rootdir", type=str, default="data")
    args=parser.parse_args()
    
  
    generator=generation(args)
    generator.setconfig_by_file(args.config)
    generator.run('prior')


# fine-grained sampling for data with large errors
def run_adaptive_sampler():
    
    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument("--kernel", type=str,default="conv-bn-relu")
    parser.add_argument("--rootdir", type=str, default="data")  ## where to save model files
    parser.add_argument("--sample_count", type=int, default=10)## for each large-error-data, we sample😊
    parser.add_argument("--iteration", type=int, default=10)
    args=parser.parse_args()
   
    acc10,cfgs=build_predictor('cpu','kernel_latency',args.kernel,large_error_threshold=0.2)## use current sampled data to build regression model, and locate data with large errors in testset
    print('cfgs',cfgs)
    ### for data with large-errors, we conduct fine-grained data sampling in the channel number dimensions
    generator=generation(args)
    generator.setconfig(args.kernel,args.sample_count,cfgs)
    generator.run('finegrained')
