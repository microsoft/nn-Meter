# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging

from .generator import config_for_kernel
from nn_meter.builder.utils import get_inputs_by_shapes
from nn_meter.builder.utils import builder_config as config
from nn_meter.builder.nn_generator import blocks
from nn_meter.builder.nn_generator.utils import save_model
from .finegrained_sampler import finegrained_kernel_sampling
from .prior_distribution_sampler import prior_kernel_sampling


def generate_model_for_kernel(kernel_type, config, savepath=None):
    """ get the nn model for predictor build. returns: input_tensors, output_tensors, configuration_key, and graphname, they are for saving tensorflow v1.x models
    """
    try:
        needed_config = {k: config[k] for k in config_for_kernel[kernel_type]}
        if "POOL_STRIDES" in config_for_kernel[kernel_type] and "POOL_STRIDES" not in config:
                needed_config["POOL_STRIDES"] = config["STRIDES"]
    except:
        raise NotImplementedError(f"The kernel_type={kernel_type} you called is not exist in our model zoo. Please implement the block and try again.")
    if kernel_type == "fc_block":
        input_shape = [1, config["CIN"]]
        input_tensor_shape = [input_shape]
    elif kernel_type == "concat_block":
        input_shape = [[config["HW"], config["HW"], config["CINS"][i]] for i in range(config["NS"])]
        input_tensor_shape = input_shape
    else:
        input_shape = [config["HW"], config["HW"], config["CIN"]]
        input_tensor_shape = [input_shape]
    model = getattr(blocks, kernel_type)(input_shape, needed_config)
    model(get_inputs_by_shapes(input_tensor_shape))
    if savepath:
        save_model(model, savepath)
        logging.info(f"{kernel_type} model is generated and saved to {savepath}.")
    else:
        logging.info(f"{kernel_type} model is generated.")
    return model, input_shape, needed_config


class Generator:
    def __init__(self, kernel_type, sample_num, mark = ""):
        self.kernel_type = kernel_type
        self.sample_num = sample_num
        self.ws_path = config.get('MODEL_DIR', 'predbuild')
        self.case_save_path = os.path.join(self.ws_path, 'testcases')
        self.info_save_path = os.path.join(self.ws_path, 'results')
        self.testcase = {}
        self.mark = mark

    def generate_config(self, sample_stage = 'prior', configs = None):
        # initialize sampling, based on prior distribution
        if sample_stage == 'prior':
            sampled_cfgs = prior_kernel_sampling(self.kernel_type, self.sample_num)
        # fine-grained sampling for data with large error points
        elif sample_stage == 'finegrained':
            sampled_cfgs = finegrained_kernel_sampling(self.kernel_type, configs, self.sample_num)
        for i in range(len(sampled_cfgs)):
            self.testcase["id_" + str(i)] = {}
            self.testcase["id_" + str(i)]['config'] = sampled_cfgs[i]

    def generate_kernel_by_cfg(self):
        """ generate tensorflow models for sampled data
        """
        kernel_type = self.kernel_type
        logging.info(f"building kernel for {kernel_type}...")
        for id, value in self.testcase.items():
            model_path = os.path.join(self.case_save_path, "_".join([kernel_type, self.mark, id]))
            kernel_cfg = value['config']
            _, input_shape, config = generate_model_for_kernel(kernel_type, kernel_cfg, savepath=model_path)
            self.testcase[id] = {
                'model': model_path,
                'shapes': [input_shape] if kernel_type != 'concat_block' else input_shape,
                'config': config
            }
        self.save_testcase_info()

    def save_testcase_info(self):
        """ save generated testcases to `self.info_save_path`
        """
        import json
        info_save_path = os.path.join(self.info_save_path, "origin_testcases.json")
        os.makedirs(os.path.dirname(info_save_path), exist_ok=True)
        with open(info_save_path, 'w') as fp:
            json.dump(self.testcases, fp, indent=4)
        
    def run(self, sample_stage = 'prior', configs = None):
        """ sample N configurations for target kernel, generate tensorflow keras model files.

        @params
        sample_stage: path of the directory containing all experiment runs. choose from ['prior', 'finegrained']
        configs: init configs for finegrained sampling
        """
        # sample configs
        self.generate_config(sample_stage, configs)
        
        # for all sampled configurations, save testcases info and generate tensorflow model files 
        self.generate_kernel_by_cfg()


def generate_config_sample(kernel_type, sample_num, mark = '', sample_stage = 'prior', configs = None):
    g = Generator(kernel_type, sample_num)
    g.run(sample_stage=sample_stage, configs=configs)
    logging.info(f'Generate {len(g.testcase)} testcases with testcases model \
                 saved in {g.case_save_path} and information saved in {g.info_save_path}.')
    return g.testcase
