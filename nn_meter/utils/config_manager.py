# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import yaml
import logging
import pkg_resources
from shutil import copyfile
logging = logging.getLogger("nn-Meter")


__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__default_user_data_folder__ = os.path.expanduser('~/.nn_meter/data')

__predictors_cfg_filename__ = 'predictors.yaml'


def create_user_configs():
    """create user configs from distributed configs
    """
    os.makedirs(__user_config_folder__, exist_ok=True)
    # TODO/backlog: to handle config merging when upgrading
    for f in pkg_resources.resource_listdir(".".join(__name__.split('.')[:-2]), 'configs'):
        if f.endswith(".yaml"):
            copyfile(
                pkg_resources.resource_filename(".".join(__name__.split('.')[:-2]), f'configs/{f}'),
                os.path.join(__user_config_folder__, f))
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
