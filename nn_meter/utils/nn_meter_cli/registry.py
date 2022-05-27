# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import yaml
import logging
import importlib


__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__registry_cfg_filename__ = 'registry.yaml'
__predictors_cfg_filename__ = 'predictors.yaml'

    
def import_module(package_location, module_path, class_name):
    sys.path.append(package_location)
    module = importlib.import_module(module_path)   
    backend_cls = getattr(module, class_name)
    return backend_cls


def register_module(module_type, meta_file):
    with open(meta_file, "r") as fp:
        meta_data = yaml.load(fp, yaml.FullLoader)

    # load module information to '~/.nn_meter/config/registry.yaml'
    builtin_name = meta_data.pop("builtin_name")
    import_module(meta_data["package_location"], meta_data["class_module"], meta_data["class_name"])

    # check necessary feature and run test script
    # for backend, check if there exits the default config file:
    if module_type == "backends":
        if meta_data["defaultConfigFile"] != None and not os.path.isfile(meta_data["defaultConfigFile"]):
            raise ValueError(f"The default config file {meta_data['defaultConfigFile']} does not exist")
    elif module_type == "kernels":
        import_module(meta_data["package_location"], meta_data["sampler_module"], meta_data["sampler_name"])
        import_module(meta_data["package_location"], meta_data["parser_module"], meta_data["parser_name"])
    if module_type == "operators" or module_type == "testcases":
        implement = meta_data.pop("implement")
        meta_data = {implement: meta_data}

    prev_info = {}
    if os.path.isfile(os.path.join(__user_config_folder__, __registry_cfg_filename__)):
        with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'r') as fp:
            prev_info = yaml.load(fp, yaml.FullLoader)

    if module_type in prev_info:
        if (module_type == "operators" or module_type == "testcases") and builtin_name in prev_info[module_type]:
            prev_info[module_type][builtin_name].update(meta_data)
        else:
            prev_info[module_type][builtin_name] = meta_data
    else:
        prev_info[module_type] = {builtin_name: meta_data}

    with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'w') as fp:
        yaml.dump(prev_info, fp)
    logging.keyinfo(f"Successfully register {module_type[:-1]}: {builtin_name}")


def register_predictor(meta_file):
    with open(meta_file, "r") as fp:
        meta_data = yaml.load(fp, yaml.FullLoader)
    
    # for predictors registration, load predictor information to '~/.nn_meter/config/predictors.yaml'
    builtin_name = meta_data["name"]
    with open(os.path.join(__user_config_folder__, __predictors_cfg_filename__), 'r') as fp:
        prev_info = yaml.load(fp, yaml.FullLoader)
    prev_info.append(meta_data)
    
    with open(os.path.join(__user_config_folder__, __predictors_cfg_filename__), 'w') as fp:
        yaml.dump(prev_info, fp)
    logging.keyinfo(f"Successfully register predictor: {builtin_name}")


def register_module_cli(args):
    if args.predictor:
        register_predictor(args.predictor)
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


def unregister_module_with_implement(module_type, module_name, module_implement):
    success = False
    if os.path.isfile(os.path.join(__user_config_folder__, __registry_cfg_filename__)):
        with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'r') as fp:
            prev_info = yaml.load(fp, yaml.FullLoader)
        try:
            prev_info[module_type][module_name].pop(module_implement)
            with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'w') as fp:
                yaml.dump(prev_info, fp)
            success = True
            logging.keyinfo(f"Successfully unregister {module_name} ({module_implement}).")
        except:
            pass
    if not success:
        logging.keyinfo(f"Unregister failed: there is no module named {module_name} ({module_implement}).")


def unregister_predictor(predictor_name, predictor_version):
    success = False
    with open(os.path.join(__user_config_folder__, __predictors_cfg_filename__), 'r') as fp:
        prev_info = yaml.load(fp, yaml.FullLoader)
    for i, p in enumerate(prev_info):
        if p['name'] == predictor_name and (predictor_version is None or p['version'] == predictor_version):
            prev_info.pop(i)
            success = True
    if success:
        with open(os.path.join(__user_config_folder__, __predictors_cfg_filename__), 'w') as fp:
            yaml.dump(prev_info, fp)
        logging.keyinfo(f"Successfully unregister {predictor_name}.")
    else:
        logging.keyinfo(f"Unregister failed: there is no predictor named {predictor_name}.")


def unregister_module_cli(args):
    if args.predictor:
        unregister_predictor(args.predictor, args.predictor_version)
    elif args.backend:
        unregister_module("backends", args.backend)
    elif args.operator:
        unregister_module_with_implement("operators", args.operator[0], args.operator[1])
    elif args.testcase:
        unregister_module_with_implement("testcases", args.testcase[0], args.testcase[1])
    elif args.kernel:
        unregister_module("kernels", args.kernel)
    else:
        logging.keyinfo('please run "nn-meter unregister --help" to see guidance.')
