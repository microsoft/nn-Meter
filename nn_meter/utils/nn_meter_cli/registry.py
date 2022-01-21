# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import yaml
import importlib
import logging

__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__registry_cfg_filename__ = 'registry.yaml'

    
def import_module(meta_data):
    sys.path.append(meta_data["packageLocation"])
    module_path = meta_data["classModule"]
    class_name = meta_data["className"]
    module = importlib.import_module(module_path)   
    backend_cls = getattr(module, class_name)
    return backend_cls


def register_module(module_type, meta_file):
    with open(meta_file, "r") as fp:
        meta_data = yaml.load(fp, yaml.FullLoader)
    builtin_name = meta_data.pop("builtinName")
    import_module(meta_data)
    # TODO: check necessary feature
    
    if module_type == "predictors":
        pass
    # for backend, check if there exits the default config file:
    if module_type == "backends":
        if not os.path.isfile(meta_data["defaultConfigFile"]):
            raise ValueError(f"The default config file {meta_data['defaultConfigFile']} does not exist")

    prev_info = {}
    if os.path.isfile(os.path.join(__user_config_folder__, __registry_cfg_filename__)):
        with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'r') as fp:
            prev_info = yaml.load(fp, yaml.FullLoader)
    if module_type in prev_info:
        prev_info[module_type][builtin_name] = meta_data
    else:
        prev_info[module_type] = {builtin_name: meta_data}

    with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'w') as fp:
        yaml.dump(prev_info, fp)
    logging.keyinfo(f"Successfully register {module_type[:-1]}: {builtin_name}")


def register_module_cli(args):
    if args.predictor:
        register_module("predictors", args.predictor)
    elif args.backend:
        register_module("backends", args.backend)
    elif args.operator:
        register_module("operators", args.operator)
    elif args.testcase:
        register_module("testcases", args.testcase)
    elif args.kernel:
        register_module("kernels", args.kernel)
    else:
        logging.keyinfo('please run "nn-meter register --help" to see guidance.')


def unregister_module(module_type, module_name):
    success = False
    if os.path.isfile(os.path.join(__user_config_folder__, __registry_cfg_filename__)):
        with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'r') as fp:
            prev_info = yaml.load(fp, yaml.FullLoader)
        if module_name in prev_info[module_type]:
            prev_info[module_type].pop(module_name)
            with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'w') as fp:
                yaml.dump(prev_info, fp)
            success = True
            logging.keyinfo(f"Successfully unregister {module_name}.")
    if not success:
        logging.keyinfo(f"Unregister failed: there is no module named {module_name}.")


def unregister_module_cli(args):
    if args.predictor:
        unregister_module("predictors", args.predictor)
    elif args.backend:
        unregister_module("backends", args.backend)
    elif args.operator:
        unregister_module("operators", args.operator)
    elif args.testcase:
        unregister_module("testcases", args.testcase)
    elif args.kernel:
        unregister_module("kernels", args.kernel)
    else:
        logging.keyinfo('please run "nn-meter unregister --help" to see guidance.')
