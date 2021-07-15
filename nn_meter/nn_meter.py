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


__user_config_folder__ = os.path.expanduser('~/.nn_meter')


def create_user_configs():
    """create user configs from distributed configs
    """
    os.makedirs(__user_config_folder__, exist_ok=True)
    # TODO: to handle config merging when upgrading
    for f in pkg_resources.resource_listdir(__name__, 'configs'):
        copyfile(pkg_resources.resource_filename(__name__, f'configs/{f}'), os.path.join(__user_config_folder__, f))


def load_latency_predictors(config, hardware):
    kernel_predictors, fusionrule = loading_to_local(config, hardware)
    nnmeter = nnMeter(kernel_predictors, fusionrule)
    return nnmeter



def list_latency_predictors():
    """ return the list of latency predictors specified in ~/.nn_meter/predictors
    """
    fn_pred = os.path.join(__user_config_folder__, 'predictors.yaml')
    try:
        with open(fn_pred) as fp:
            return yaml.load(fp, yaml.FullLoader)
    except FileNotFoundError:
        print(f"config file {fn_pred} not found, created")
        create_user_configs()
        return list_latency_predictors()


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
    parser.add_argument('--list-predictors', help='list all supported predictors', action='store_true', default=False)
    parser.add_argument('--predictor', help='name of predictor')
    parser.add_argument('--predictor-version', help='version of specified predictor', default=None)
    args = parser.parse_args()

    if args.list_predictors:
        preds = list_latency_predictors()
        print("Supported latency predictors:")
        for p in preds:
            print(f"{p['name']}: version={p['version']}")
        return

    predictor = None
