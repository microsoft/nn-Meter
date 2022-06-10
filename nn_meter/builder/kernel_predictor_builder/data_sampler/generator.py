# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import random
import string
import logging
from nn_meter.builder import builder_config
from nn_meter.builder.utils import merge_info
from .utils import get_sampler_for_kernel, generate_model_for_kernel
logging = logging.getLogger("nn-Meter")


class KernelGenerator:
    def __init__(self, kernel_type, sample_num, mark = ""):
        self.kernel_type = kernel_type
        self.sample_num = sample_num
        self.workspace_path = builder_config.get('WORKSPACE', 'predbuild')
        self.case_save_path = os.path.join(self.workspace_path, 'kernels')
        self.kernel_info = {kernel_type: {}}
        self.kernels = self.kernel_info[self.kernel_type]
        self.implement = builder_config.get('IMPLEMENT', 'predbuild')
        self.batch_size = builder_config.get('BATCH_SIZE', 'predbuild')
        self.model_suffix = "" if self.implement == 'tensorflow' else ".onnx"
        self.mark = mark
        os.makedirs(self.case_save_path, exist_ok=True)

    def generate_config(self, sampling_mode = 'prior', configs = None):
        sampled_cfgs = get_sampler_for_kernel(self.kernel_type, self.sample_num, sampling_mode, configs)
        for i in range(len(sampled_cfgs)):
            random_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
            self.kernels[random_id] = {}
            self.kernels[random_id]['config'] = sampled_cfgs[i]

    def generate_kernel_by_cfg(self):
        """ generate tensorflow models for sampled data
        """
        kernel_type = self.kernel_type
        logging.info(f"building kernel for {kernel_type}...")
        count = 0
        error_save_path = os.path.join(self.workspace_path, 'results', 'generate_error.log')
        for id, value in self.kernels.items():
            model_path = os.path.join(self.case_save_path, ("_".join([kernel_type, self.mark, id]) + self.model_suffix))
            kernel_cfg = value['config']
            try:
                _, input_tensor_shape, config = generate_model_for_kernel(
                    kernel_type, kernel_cfg, save_path=model_path,
                    implement=self.implement, batch_size=self.batch_size
                )
                self.kernels[id] = {
                    'model': model_path,
                    'shapes': input_tensor_shape,
                    'config': config
                }
                count += 1
            except Exception as e:
                open(os.path.join(self.workspace_path, "results", "generate_error.log"), 'a').write(f"{id}: {e}\n")

        # save information to json file in incrementally mode
        info_save_path = os.path.join(self.workspace_path, "results", f"{kernel_type}_{self.mark}.json")
        new_kernels_info = merge_info(new_info=self.kernel_info, info_save_path=info_save_path)
        os.makedirs(os.path.dirname(info_save_path), exist_ok=True)
        with open(info_save_path, 'w') as fp:
            json.dump(new_kernels_info, fp, indent=4)
        logging.keyinfo(f"Generate {len(self.kernels)} kernels and save info to {info_save_path} " \
                        f"Failed information are saved in {error_save_path} (if any).")

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

    mark (str, optional): the mark for the running results. Defaults to ''.

    sampling_mode (str, optional): the sampling mode for config generation, supporting mode includes 'prior' and 'finegrained'.
        Defaults to be 'prior'.

    configs (list, optional): is required when the sampling_mode=='finegrained'. The fingrained samples will based on the config 
        in `configs`. Defaults to None.

    """
    generator = KernelGenerator(kernel_type, sample_num, mark=mark)
    kernels_info = generator.run(sampling_mode=sampling_mode, configs=configs)

    return kernels_info
