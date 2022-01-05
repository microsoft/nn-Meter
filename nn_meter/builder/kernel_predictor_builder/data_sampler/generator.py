# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import random
import string
import logging

from ..utils import config_for_kernel
from .finegrained_sampler import finegrained_kernel_sampling
from .prior_distribution_sampler import prior_kernel_sampling
from nn_meter.builder.utils import get_inputs_by_shapes, merge_prev_info
from nn_meter.builder.utils import builder_config as config
from nn_meter.builder.nn_generator.tf_networks import blocks
from nn_meter.builder.nn_generator.utils import save_model


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
        input_shape = [[config["HW"], config["HW"], cin] 
                       for cin in [config['CIN1'], config['CIN2'], config['CIN3'], config['CIN4']]
                       if cin != 0]
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


class KernelGenerator:
    def __init__(self, kernel_type, sample_num, mark = ""):
        self.kernel_type = kernel_type
        self.sample_num = sample_num
        self.ws_path = config.get('MODEL_DIR', 'predbuild')
        self.case_save_path = os.path.join(self.ws_path, 'models')
        self.kernel_info = {kernel_type: {}}
        self.kernels = self.kernel_info[self.kernel_type]
        self.mark = mark

    def generate_config(self, sampling_mode = 'prior', configs = None):
        # initialize sampling, based on prior distribution
        if sampling_mode == 'prior':
            sampled_cfgs = prior_kernel_sampling(self.kernel_type, self.sample_num)
        # fine-grained sampling for data with large error points
        elif sampling_mode == 'finegrained':
            sampled_cfgs = finegrained_kernel_sampling(self.kernel_type, configs, self.sample_num)
        for i in range(len(sampled_cfgs)):
            random_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
            self.kernels[random_id] = {}
            self.kernels[random_id]['config'] = sampled_cfgs[i]

    def generate_kernel_by_cfg(self):
        """ generate tensorflow models for sampled data
        """
        kernel_type = self.kernel_type
        logging.info(f"building kernel for {kernel_type}...")
        for id, value in self.kernels.items():
            model_path = os.path.join(self.case_save_path, "_".join([kernel_type, self.mark, id]))
            kernel_cfg = value['config']
            _, input_shape, config = generate_model_for_kernel(kernel_type, kernel_cfg, savepath=model_path)
            self.kernels[id] = {
                'model': model_path,
                'shapes': [input_shape] if kernel_type != 'concat_block' else input_shape,
                'config': config
            }
        
    def run(self, sampling_mode = 'prior', configs = None):
        """ sample N configurations for target kernel, generate tensorflow keras model files.

        @params
        sampling_mode: path of the directory containing all experiment runs. choose from ['prior', 'finegrained']
        configs: init configs for finegrained sampling
        """
        # sample configs
        self.generate_config(sampling_mode, configs)
        
        # for all sampled configurations, save kernels info and generate tensorflow model files 
        self.generate_kernel_by_cfg()
        logging.info(f'Generate {len(self.kernels)} kernels with kernels model saved in {self.case_save_path}.')
        return self.kernel_info


def generate_config_sample(kernel_type, sample_num, mark = '', sampling_mode = 'prior', configs = None):
    """ Generate config sample and return sampled configs.

    @params
    kernel_type (str): type of kernel

    sample_num (int): the sampling number of configs

    mark (str, optional): the mark to run . Defaults to ''.

    sampling_mode (str, optional): the sampling mode for config generation, supporting mode includes 'prior' and 'finegrained'.
        Defaults to be 'prior'.

    configs (list, optional): is required when the sampling_mode=='finegrained'. The fingrained samples will based on the config 
        in `configs`. Defaults to None.

    """
    generator = KernelGenerator(kernel_type, sample_num, mark=mark)
    kernels_info = generator.run(sampling_mode=sampling_mode, configs=configs)

    # save information to json file in incrementally mode
    ws_mode_path = config.get('MODEL_DIR', "predbuild")
    info_save_path = os.path.join(ws_mode_path, "results", f"{kernel_type}.json")
    new_kernels_info = merge_prev_info(new_info=kernels_info, info_save_path=info_save_path)
    os.makedirs(os.path.dirname(info_save_path), exist_ok=True)
    with open(info_save_path, 'w') as fp:
        json.dump(new_kernels_info, fp, indent=4)
    logging.keyinfo(f"Save the kernel model information to {info_save_path}")

    return kernels_info
