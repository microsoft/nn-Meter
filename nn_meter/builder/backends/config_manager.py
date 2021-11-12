# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import yaml
import pkg_resources
from shutil import copyfile

def copy_to_workspace(platform_type, workspace_path):
    """copy the default config file to user's workspace
    """
    os.makedirs(os.path.join(workspace_path, 'configs'), exist_ok=True)
    if platform_type == 'tflite':
        config_name = 'backend_tflite_config.yaml'
    elif platform_type == 'openvino':
        config_name = 'backend_openvino_config.yaml'
    copyfile(
        pkg_resources.resource_filename(".".join(__name__.split('.')[:-3]), 'configs/builder/' + config_name), 
        os.path.join(os.path.join(workspace_path, 'configs'), 'backend_config.yaml'))


def copy_cusconfig_to_workspace(workspace_path, config_path):
    """copy the config file of the customized backend to user's workspace
    """
    os.makedirs(os.path.join(workspace_path, 'configs'), exist_ok=True)
    copyfile(
        config_path, 
        os.path.join(os.path.join(workspace_path, 'configs'), 'backend_config.yaml'))


def load_backend_config(platform_type, workspace_path):
    filepath = os.path.join(workspace_path, "configs", 'backend_config.yaml')
    try:
        with open(filepath) as fp:
            return os.path.join(yaml.load(fp, yaml.FullLoader)['data_folder'])
    except FileNotFoundError:
        copy_to_workspace(platform_type, workspace_path)
        return load_backend_config(platform_type, workspace_path)
