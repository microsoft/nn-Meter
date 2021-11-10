# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json


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
    def __init__(self, config_path=None, save_path='./data/ruletest_config.json'):
        super().__init__()
        if config_path: # load config from file
            self._load_from_config_file(config_path)
        else: # set default config
            self._set_default_config()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as fp:
                json.dump(self._global_settings, fp, indent=4)

    def _set_default_config(self):
        self._global_settings['ruletest'] = {
            'default_input_shape': [28, 28, 16],
            'd1_input_shape': [428],
            'filters': 256,
            'kernel_size': 3,
            'enabled': ['BF', 'MON', 'RT'],
            'params': {
                'BF': {
                    'eps': 0.5,
                }
            },
            'model_dir': '',
            'detail': False,
        }
    
    def _load_from_config_file(self, config_path):
        with open(config_path, 'r') as fp:
            setting = json.load(fp)
        self._global_settings = setting


config = ConfigManager()
