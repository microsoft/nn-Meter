import yaml
import os
import logging
import pkg_resources
from shutil import copyfile


__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__default_user_data_folder__ = os.path.expanduser('~/.nn_meter/data')

__predictors_cfg_filename__ = 'predictors.yaml'


def create_user_configs():
    """create user configs from distributed configs
    """
    os.makedirs(__user_config_folder__, exist_ok=True)
    # TODO/backlog: to handle config merging when upgrading
    for f in pkg_resources.resource_listdir(__name__, 'configs'):
        copyfile(pkg_resources.resource_filename(__name__, f'configs/{f}'), os.path.join(__user_config_folder__, f))
    # make default setting yaml file
    with open(os.path.join(__user_config_folder__, 'settings.yaml'), 'w') as fp:
        yaml.dump({'data_folder': __default_user_data_folder__}, fp)


def get_user_data_folder():
    """get user data folder in settings.yaml
    """
    filepath = os.path.join(__user_config_folder__, 'settings.yaml')
    try:
        with open(filepath) as fp:
            return os.path.join(yaml.load(fp, yaml.FullLoader)['data_folder'])
    except FileNotFoundError:
        logging.info(f"setting file {filepath} not found, created")
        create_user_configs()
        return get_user_data_folder()


def change_user_data_folder(new_folder):
    """change user data folder in settings.yaml
    """
    os.makedirs(new_folder, exist_ok=True)
    with open(os.path.join(__user_config_folder__, 'settings.yaml')) as fp:
        setting = yaml.load(fp, yaml.FullLoader)
    with open(os.path.join(__user_config_folder__, 'settings.yaml'), 'w') as fp:
        setting['data_folder'] = new_folder
        yaml.dump(setting, fp)


def load_config_file(fname: str, loader=None):
    """load config file from __user_config_folder__;
    if the file not located in __user_config_folder__, copy it from distribution
    """
    filepath = os.path.join(__user_config_folder__, fname)
    try:
        with open(filepath) as fp:
            if loader is None:
                return yaml.load(fp, yaml.FullLoader)
            else:
                return loader(fp)
    except FileNotFoundError:
        logging.info(f"config file {filepath} not found, created")
        create_user_configs()
        return load_config_file(fname)