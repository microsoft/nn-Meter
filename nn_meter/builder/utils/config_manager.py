# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import yaml
import pkg_resources
import logging
from shutil import copyfile

__backend_tflite_cfg_filename__ = 'backend_tflite_config.yaml'
__backend_openvino_cfg_filename__ = 'backend_openvino_config.yaml'
__ruletest_cfg_filename__ = 'ruletest_config.yaml'


def copy_to_workspace(platform_type, workspace_path):
    """copy the default config file to user's workspace
    """
    os.makedirs(os.path.join(workspace_path, 'configs'), exist_ok=True)
    # backend config
    if platform_type == 'tflite':
        config_name = __backend_tflite_cfg_filename__
    elif platform_type == 'openvino':
        config_name = __backend_openvino_cfg_filename__
    copyfile(
        pkg_resources.resource_filename(".".join(__name__.split('.')[:-3]), 'configs/builder/' + config_name), 
        os.path.join(os.path.join(workspace_path, 'configs'), 'backend_config.yaml'))
    # rule test config
    copyfile(
        pkg_resources.resource_filename(".".join(__name__.split('.')[:-3]), f'configs/builder/' + __ruletest_cfg_filename__), 
        os.path.join(os.path.join(workspace_path, 'configs'), 'ruletest_config.yaml'))


def load_config_file(platform_type, workspace_path):
    """load config file from workspace_path;
    if the file not located in workspace_path, copy it from distribution
    """
    backend_filepath = os.path.join(workspace_path, "configs", 'backend_config.yaml')
    ruletest_filepath = os.path.join(workspace_path, "configs", 'ruletest_config.yaml')
    try:
        with open(backend_filepath) as fp:
            backend = yaml.load(fp, yaml.FullLoader)
        with open(ruletest_filepath) as fp:
            ruletest = yaml.load(fp, yaml.FullLoader)
        return backend, ruletest
            
    except FileNotFoundError:
        logging.info(f"config file in {workspace_path} not found, created")
        copy_to_workspace(platform_type, workspace_path)
        return load_config_file(platform_type, workspace_path)


class ConfigData:
    def __init__(self):
        self.workspace_path = None
        self._global_settings = None

    def set(self, name, value, module=''):
        self._global_settings[module][name] = value
    
    def set_module(self, value, module=''):
        self._global_settings[module] = value
        
    def get(self, name, module=''):
        return self._global_settings[module].get(name)

    def get_settings(self):
        return self._global_settings


class ConfigManager(ConfigData):
    def init(self, platform_type, workspace_path):
        self.workspace_path = workspace_path
        self._load_from_config_file(platform_type, workspace_path)
    
    def _load_from_config_file(self, platform_type, workspace_path):
        backend, ruletest = load_config_file(platform_type, workspace_path)
        self.set_module(backend, 'backend')
        self.set_module(ruletest, 'ruletest')
        self.set(self, 'model_dir', os.path.join(self.workspace_path, "testcases"), 'ruletest')


builder_config = ConfigManager()
