# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .net_from_cfg import block_from_cfg
from .networks.block_utils import save_to_models
import os
import tensorflow as tf
from data_sampler.block_sampler import *


def get_output_folder(parent_dir, run_name):
    """
    Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    run_name: str
      Name of the run

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok = True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, run_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok = True)
    return parent_dir


class generation:
    def __init__(self, args):
        self.args = args
       
    def setconfig_by_file(self, config):
        """ read configuration files in config/xxx.yaml, we only need two parameters: block_type and sample_num
        """
        self.cfg = get_config(config)
        self.args.savepath = get_output_folder(self.args.rootdir, self.cfg['BLOCK']['BLOCK_TYPE'])
  
        print(self.args.savepath)

    def generate_block_from_cfg(self, blockcfgs, bottleneck_type):
        """
        generate tensorflow models for sampled data

        @params

        blockcfgs: list
        each item in the list represent a sampled data. each item is a dictionary, storing the configuration
        block_type: str
        identical kernel name 
        """
        graphs = {}
        graphs[bottleneck_type] = {}
        id = 0
        for blockcfg in blockcfgs:
                print(bottleneck_type, blockcfg)
                tf.reset_default_graph()
                input_tensor, output_tensor, config, graphname = block_from_cfg(bottleneck_type, blockcfg)

                output_tensors = []
                input_tensors = []
                if isinstance(output_tensor, list):
                    output_tensors = output_tensor
                else:
                    output_tensors = [output_tensor]
                
                if isinstance(input_tensor, list):
                    input_tensors = input_tensor
                else:
                    input_tensors = [input_tensor]

                try:
                    savemodelpath, tfpath, pbpath, inputnames, outputnames = save_to_models(self.args.savepath, input_tensors, output_tensors, graphname, config)
                    graphs[bottleneck_type][id] = {}
                    graphs[bottleneck_type][id]['saved_model'] = savemodelpath
                    graphs[bottleneck_type][id]['tflite_path'] = tfpath
                    graphs[bottleneck_type][id]['pb_path'] = pbpath
                    graphs[bottleneck_type][id]['config'] = config
                except:
                    pass
                id += 1
        return graphs

    def setconfig(self, block_type, sample_num, cfgs):
        """
        set parameters to self.cfg
        
        @params
        
        block_type: identical kernel name 
        sample_num: for each large-error-data, we conduct fine-grained sampling for sample_num more data
        cfgs: each item in the list represent a large-error-data-point. each item is a dictionary, storing the configuration
        """
        self.cfg = {}
        self.cfg['BLOCK'] = {}
        self.cfg['BLOCK']['BLOCK_TYPE'] = block_type
        self.cfg['BLOCK']['SAMPLE_NUM'] = sample_num
        self.cfg['BLOCK']['DATA'] = cfgs
        self.args.savepath =  get_output_folder(self.args.rootdir, block_type)
  
        print(self.args.savepath)

    def run(self, sample_stage = 'prior'):
        """
        sample N configurations for target kernel, generate tensorflow model files (pb and tflite).
        
        @params
        
        sample_stage: path of the directory containing all experiment runs.
        """
        bottleneck_type = self.cfg['BLOCK']['BLOCK_TYPE']

        # samples# a list of sampled configurations
        if sample_stage == 'prior': ## initial sampling, based on prior distribution
            samples = block_sampling_from_prior(self.cfg['BLOCK']['BLOCK_TYPE'], self.cfg['BLOCK']['SAMPLE_NUM'])
        else:  # fine-grained sampling for data with large error points
            samples = block_sampling_with_finegrained(self.cfg['BLOCK']['BLOCK_TYPE'], self.cfg['BLOCK']['SAMPLE_NUM'], self.cfg['BLOCK']['DATA'])
       
        # for all sampled configurations, generate tensorflow model files 
        self.generate_block_from_cfg(samples, bottleneck_type)
        


    

   

    



