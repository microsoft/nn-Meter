class ConfigData:
    def __init__(self, init_defaults=True):
        # default settings
        self._global_settings = {}
        if init_defaults:
            self._init_config()

    def _init_config(self):
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

        # init config from files
        # ...

    def set(self, name, value, module=''):
        self._global_settings[module][name] = value
        
    def get(self, name, module=''):
        return self._global_settings[module].get(name)

    def get_settings(self):
        return self._global_settings

    def load_from_config_file(self, config_file, module=''):
        raise NotImplementedError()

class ConfigManager:
    def __init__(self):
        # self.data = ConfigData
        # # load from file
        # # default config
        pass
    
    def _init_config(self):
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
