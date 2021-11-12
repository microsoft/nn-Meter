# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import pkg_resources
from shutil import copyfile

class ConfigData:
    def __init__(self, global_setting=None):
        self._global_settings = global_setting if global_setting else {}

    def set(self, name, value, module=''):
        self._global_settings[module][name] = value
        
    def get(self, name, module=''):
        return self._global_settings[module].get(name)

    def get_settings(self):
        return self._global_settings


class ConfigManager(ConfigData):

    def init(self,workspace_path ):
        if os.path.isdir(workspace_path):
            config_file = os.path.join(workspace_path, 'configs', 'ruletester_config.yaml')
        elif os.path.isfile(workspace_path):
            config_file = workspace_path
        self._load_from_config_file(config_file)
    
    def _load_from_config_file(self, config_file):
        with open(config_file, 'r') as fp:
            setting = json.load(fp)
        self._global_settings = setting

config = ConfigManager()


def copy_to_workspace(workspace_path):
    """copy the default config file to user's workspace
    """
    os.makedirs(os.path.join(workspace_path, 'configs'), exist_ok=True)
    copyfile(
        pkg_resources.resource_filename(".".join(__name__.split('.')[:-3]), f'configs/builder/ruletester_config.yaml'), 
        os.path.join(os.path.join(workspace_path, 'configs'), 'ruletester_config.yaml'))
