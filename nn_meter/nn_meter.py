# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from nn_meter.prediction.predictors.predict_by_kernel import nn_predict
from nn_meter.kerneldetection import KernelDetector
from nn_meter.ir_converters import model_to_graph, model_file_to_graph
from nn_meter.prediction.load_predictors import loading_to_local

import yaml
import os
import argparse
import pkg_resources
from shutil import copyfile
from packaging import version


__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__user_data_folder__ = os.path.expanduser('~/.nn_meter/data')

__predictors_cfg_filename__ = 'predictors.yaml'


def create_user_configs():
    """create user configs from distributed configs
    """
    os.makedirs(__user_config_folder__, exist_ok=True)
    # TODO: to handle config merging when upgrading
    for f in pkg_resources.resource_listdir(__name__, 'configs'):
        copyfile(pkg_resources.resource_filename(__name__, f'configs/{f}'), os.path.join(__user_config_folder__, f))


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
        print(f"config file {filepath} not found, created")
        create_user_configs()
        return load_config_file(fname)


def list_latency_predictors():
    """ return the list of latency predictors specified in ~/.nn_meter/predictors
    """
    return load_config_file(__predictors_cfg_filename__)


def load_predictor_config(predictor_name: str, predictor_version: str):
    """load predictor config according to name and version
    """
    config = load_config_file(__predictors_cfg_filename__)
    preds_info = [p for p in config if p['name'] == predictor_name and (predictor_version is None or p['version'] == predictor_version)]
    n_preds = len(preds_info)
    if n_preds == 1:
        return preds_info[0]
    elif n_preds > 1:
        # find the latest version of the predictor
        latest_version, latest_version_idx = version.parse(preds_info[0]['version']), 0
        for i in range(1, n_preds):
            if version.parse(preds_info[i]['version']) > latest_version:
                latest_version = version.parse(preds_info[i]['version'])
                latest_version_idx = i
        print(f'WARNING: There are multiple version for {predictor}, use the latest one ({str(latest_version)})')
        return preds_info[latest_version_idx]
    else:
        raise NotImplementedError('No predictor that meet the required version, please try again.')


def load_latency_predictor(predictor_name: str, predictor_version: str):
    """load predictor model according to name and version
    """
    pred_info = load_predictor_config(predictor_name, predictor_version)
    kernel_predictors, fusionrule = loading_to_local(pred_info, __user_data_folder__)
    return nnMeter(kernel_predictors, fusionrule)


class nnMeter:
    def __init__(self, predictors, fusionrule):
        self.kernel_predictors = predictors
        self.fusionrule = fusionrule
        self.kd = KernelDetector(self.fusionrule)

    def predict(
        self, model, model_type=None, input_shape=(1, 3, 224, 224), modelname="test"
    ):
        """
        @params:

        model: a pytorch/onnx/tensorflow model object or a str containing path to the model file
        """
        if isinstance(model, str):
            graph = model_file_to_graph(model, model_type)
        else:
            graph = model_to_graph(model, model_type, input_shape=input_shape)
       # print(graph)
        self.kd.load_graph(graph)

        py = nn_predict(self.kernel_predictors, self.kd.kernels)
        return py


def nn_meter_cli():
    parser = argparse.ArgumentParser('nn-meter')
    parser.add_argument(
        '--list-predictors',
        help='list all supported predictors',
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--input_model",
        type=str,
        required=True,
        help="Path to input model. ONNX, FrozenPB or JSON",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        required=True,
        help="name of target predictor (hardware)"
    )
    parser.add_argument(
        '--predictor-version',
        help="the version of the latency predictor (If not specified, use the lateast version)",
        default=None
    )
    args = parser.parse_args()

    if args.list_predictors:
        preds = load_config_file(__predictors_cfg_filename__)
        print("Supported latency predictors:")
        for p in preds:
            print(f"{p['name']}: version={p['version']}")
        return

    predictor = load_latency_predictor(args.predictor, args.predictor_version)
    latency = predictor.predict(args.input_model)
    print('predict latency', latency)
