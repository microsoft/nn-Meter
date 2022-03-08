# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import yaml
import pickle
import logging
from glob import glob
from nn_meter.utils import download_from_url, create_user_configs
logging = logging.getLogger("nn-Meter")


__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')


def loading_to_local(pred_info, dir):
    """ loading builtin predictors to local

    @params:

    pred_info: a dictionary containing predictor information
    dir: the local directory to store the kernel predictors and fusion rules
    """
    os.makedirs(dir, exist_ok=True)
    hardware = pred_info['name']
    ppath = os.path.join(dir, hardware)

    isdownloaded = check_predictors(ppath, pred_info["kernel_predictors"])
    if not isdownloaded:
        logging.keyinfo(f'Download from {pred_info["download"]} ...')
        download_from_url(pred_info["download"], dir)

    # load predictors
    predictors = {}
    ps = glob(os.path.join(ppath, "**.pkl"))
    for p in ps:
        pname =  os.path.basename(p).replace(".pkl", "")
        with open(p, "rb") as f:
            logging.info("load predictor %s" % p)
            model = pickle.load(f)
            predictors[pname] = model
    fusionrule = os.path.join(ppath, "fusion_rules.json")
    # logging.info(fusionrule)
    if not os.path.isfile(fusionrule):
        raise ValueError(
            "check your fusion rule path, file " + fusionrule + " does not exist！"
        )
    return predictors, fusionrule


def loading_customized_predictor(pred_info):
    """ loading customized predictor

    @params:
    pred_info: a dictionary containing predictor information
    """
    hardware = pred_info['name']
    ppath = pred_info['package_location']

    isexist = check_predictors(ppath, pred_info["kernel_predictors"])
    if not isexist:
        raise FileExistsError(f"The predictor {hardware} in {ppath} does not exist.")

    # load predictors
    predictors = {}
    ps = glob(os.path.join(ppath, "**.pkl"))
    for p in ps:
        pname =  os.path.basename(p).replace(".pkl", "")
        with open(p, "rb") as f:
            logging.info("load predictor %s" % p)
            model = pickle.load(f)
            predictors[pname] = model
    fusionrule = os.path.join(ppath, "fusion_rules.json")
    # logging.info(fusionrule)
    if not os.path.isfile(fusionrule):
        raise ValueError(
            "check your fusion rule path, file " + fusionrule + " does not exist！"
        )
    return predictors, fusionrule


def check_predictors(ppath, kernel_predictors):
    """
    @params:

    model: a pytorch/onnx/tensorflow model object or a str containing path to the model file
    """
    logging.info("checking local kernel predictors at " + ppath)
    if os.path.isdir(ppath):
        filenames = glob(os.path.join(ppath, "**.pkl"))
        # check if all the pkl files are included
        for kp in kernel_predictors:
            fullpath = os.path.join(ppath, kp + ".pkl")
            if fullpath not in filenames:
                return False
        return True
    else:
        return False


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
