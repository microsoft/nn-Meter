# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import yaml
import logging


def import_module(meta_data):
    pass


def register_module(module_type, meta_file):
    meta_data = yaml.load(meta_file)
    if module_type not in []:
        raise KeyError
    try:
        import_module(meta_data)
    except:
        raise NotImplementedError() # the meta data cannot be imported as a module
    
    if os.path.isfile('registry.yaml'):
        with open('r') as fp:
            prev_info = yaml.load(...)
    else:
        prev_info = {}
    with open('w') as fp:
        if module_type in prev_info:
            prev_info[module_type].update(meta_data)
        else:
            prev_info[module_type] = {meta_data}
    logging.info(f"Successfully register {meta_data['className']}")


def register_module_cli():
    pass


def unregister_module_cli():
    pass
