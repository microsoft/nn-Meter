# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import pickle
import os
from glob import glob
from zipfile import ZipFile
from tqdm import tqdm
import requests
import logging
from nn_meter.utils.utils import download_from_url


def loading_to_local(pred_info, dir="data/predictorzoo"):
    """
    @params:

    configs: the default predictor.yaml that describes the supported hardware+backend
    hardware: the targeting hardware_inferenceframework name
    dir: the local directory to store the kernel predictors and fusion rules

    """
    os.makedirs(dir, exist_ok=True)
    hardware = pred_info['name']
    ppath = os.path.join(dir, hardware)

    isdownloaded = check_predictors(ppath, pred_info["kernel_predictors"])
    if not isdownloaded:
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
