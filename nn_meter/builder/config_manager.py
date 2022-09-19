# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import yaml
import pkg_resources
from shutil import copyfile

__backend_tflite_cfg_filename__ = 'backend_tflite_config.yaml'
__backend_openvino_cfg_filename__ = 'backend_openvino_config.yaml'
__ruletest_cfg_filename__ = 'ruletest_config.yaml'
__predbuild_cfg_filename__ = 'predictorbuild_config.yaml'


def copy_to_workspace(backend_type, workspace_path, backendConfigFile = None):
    """copy the default config file to user's workspace
    """
    os.makedirs(os.path.join(workspace_path, 'configs'), exist_ok=True)

    # backend config
    if backend_type == 'customized' and backendConfigFile:
        copyfile(backendConfigFile, os.path.join(workspace_path, 'configs', 'backend_config.yaml'))
    else:
        if backend_type == 'tflite':
            config_name = __backend_tflite_cfg_filename__
        elif backend_type == 'openvino':
            config_name = __backend_openvino_cfg_filename__
        copyfile(
            pkg_resources.resource_filename(".".join(__name__.split('.')[:-2]), 'configs/builder/backends/' + config_name), 
            os.path.join(workspace_path, 'configs', 'backend_config.yaml'))
    # rule test config
    copyfile(
        pkg_resources.resource_filename(".".join(__name__.split('.')[:-2]), f'configs/builder/fusion_rule_tester/' + __ruletest_cfg_filename__), 
        os.path.join(os.path.join(workspace_path, 'configs'), 'ruletest_config.yaml'))
    # predictor builder config
    copyfile(
        pkg_resources.resource_filename(".".join(__name__.split('.')[:-2]), f'configs/builder/predictor_builder/' + __predbuild_cfg_filename__), 
        os.path.join(os.path.join(workspace_path, 'configs'), 'predictorbuild_config.yaml'))


def load_config_file(workspace_path):
    """load config file from workspace_path;
    if the file not located in workspace_path, copy it from distribution
    """
    backend_filepath = os.path.join(workspace_path, "configs", 'backend_config.yaml')
    ruletest_filepath = os.path.join(workspace_path, "configs", 'ruletest_config.yaml')
    predictorbuild_filepath = os.path.join(workspace_path, "configs", 'predictorbuild_config.yaml')
    try:
        try:
            with open(backend_filepath) as fp:
                backend = yaml.load(fp, yaml.FullLoader)
        except: # In customized backend, sometimes there is no backend config file, skip loading config for this case
            backend = None
        with open(ruletest_filepath) as fp:
            ruletest = yaml.load(fp, yaml.FullLoader)
        with open(predictorbuild_filepath) as fp:
            predictorbuild = yaml.load(fp, yaml.FullLoader)
        return backend, ruletest, predictorbuild
    except:
        raise FileNotFoundError(f"config file in {workspace_path} not found, please create the workspace first." \
                                "Docs: https://github.com/microsoft/nn-Meter/blob/main/docs/builder/overview.md#-create-workspace-")


class ConfigData:
    def __init__(self):
        self.workspace_path = ''
        self._global_settings = {}

    def set(self, name, value, module=''):
        self._global_settings[module][name] = value
    
    def set_module(self, value, module=''):
        self._global_settings[module] = value
        
    def get(self, name, module=''):
        try:
            return self._global_settings[module].get(name)
        except:
            raise ValueError(f"Could not find {module} or {module}.{name} in builder config. \
                Please run `builder_config.init('path/to/workspace')` first.")

    def get_module(self, module=''):
        try:
            return self._global_settings[module]
        except:
            raise ValueError(f"Module [{module}] from config file not found, please create the workspace first. " \
                "Docs: https://github.com/microsoft/nn-Meter/blob/main/docs/builder/overview.md#-create-workspace-")

    def get_settings(self):
        return self._global_settings


class ConfigManager(ConfigData):
    def init(self, workspace_path):
        self.workspace_path = workspace_path
        self._load_from_config_file(workspace_path)
    
    def _load_from_config_file(self, workspace_path):
        backend, ruletest, predbuild = load_config_file(workspace_path)
        self.set_module(backend, 'backend')
        self.set_module(ruletest, 'ruletest')
        self.set_module(predbuild, 'predbuild')
        self.set('WORKSPACE', os.path.join(self.workspace_path, "fusion_rule_test"), 'ruletest')
        self.set('WORKSPACE', os.path.join(self.workspace_path, "predictor_build"), 'predbuild')

builder_config = ConfigManager()
