# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from glob import glob
from nn_meter.prediction.predictors.predict_by_kernel import nn_predict
from nn_meter.kerneldetection import KernelDetector
from nn_meter.ir_converters import model_file_to_graph
from nn_meter.prediction.load_predictors import loading_to_local

import yaml
import os
import sys
import argparse
import pkg_resources
from shutil import copyfile
from packaging import version
import logging

__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__user_data_folder__ = os.path.expanduser('~/.nn_meter/data')

__predictors_cfg_filename__ = 'predictors.yaml'


def create_user_configs():
    """create user configs from distributed configs
    """
    os.makedirs(__user_config_folder__, exist_ok=True)
    # TODO/backlog: to handle config merging when upgrading
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
        logging.info(f"config file {filepath} not found, created")
        create_user_configs()
        return load_config_file(fname)


def list_latency_predictors():
    """ return the list of latency predictors specified in ~/.nn_meter/predictors
    """
    return load_config_file(__predictors_cfg_filename__)


def load_predictor_config(predictor_name: str, predictor_version: float = None):
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
        print(f'WARNING: There are multiple version for {predictor_name}, use the latest one ({str(latest_version)})')
        return preds_info[latest_version_idx]
    else:
        raise NotImplementedError('No predictor that meets the required name and version, please try again.')


def load_latency_predictor(predictor_name: str, predictor_version: float = None):
    """load predictor model according to name and version
    """
    pred_info = load_predictor_config(predictor_name, predictor_version)
    kernel_predictors, fusionrule = loading_to_local(pred_info, __user_data_folder__)
    return nnMeter(kernel_predictors, fusionrule)


def apply_latency_predictor(args):
    # specify model type
    if args.tensorflow:
        input_model, model_type, model_suffix = args.tensorflow, "pb", ".pb"
    elif args.onnx:
        input_model, model_type, model_suffix = args.onnx, "onnx", ".onnx"
    elif args.nn_meter_ir:
        input_model, model_type, model_suffix = args.nn_meter_ir, "nnmeter", ".json"
    elif args.nni_ir:
        input_model, model_type, model_suffix = args.nni_ir, "nni", ".json"
    elif args.torch: # find torch model name from torch model zoo
        input_model, model_type = args.torch, "torch" # TODO: fix here

    # load predictor
    predictor = load_latency_predictor(args.predictor, args.predictor_version)

    # specify model for prediction
    input_model_list = []
    if os.path.isfile(input_model):
        input_model_list = [input_model]
    elif os.path.isdir(input_model):
        input_model_list = glob(os.path.join(input_model, "**" + model_suffix))
        input_model_list.sort()
        logging.info(f'Found {len(input_model_list)} model in {input_model}. Start prediction ...')
    else:
        logging.error(f'Cannot find any model satisfying the arguments.')

    # predict latency
    result = {}
    for model in input_model_list:
        latency = predictor.predict(model, model_type)
        result[os.path.basename(model)] = latency
        logging.result(f'[RESULT] predict latency for {os.path.basename(model)}: {latency}')
    
    return result


def get_nnmeter_ir(args):
    if args.tensorflow:
        graph = model_file_to_graph(args.tensorflow, 'pb')
    elif args.onnx:
        graph = model_file_to_graph(args.onnx, 'pb')
    else:
        raise ValueError(f"Unsupported model.")
    return graph


class nnMeter:
    def __init__(self, predictors, fusionrule):
        self.kernel_predictors = predictors
        self.fusionrule = fusionrule
        self.kd = KernelDetector(self.fusionrule)

    def predict(
        self, model, model_type, input_shape=(1, 3, 224, 224)
    ):
        """
        @params:

        model: a pytorch/onnx/tensorflow model object or a str containing path to the model file
        """
        graph = model_file_to_graph(model, model_type, input_shape)
        
        # logging.info(graph)
        self.kd.load_graph(graph)

        py = nn_predict(self.kernel_predictors, self.kd.kernels)
        return py


def nn_meter_cli():
    parser = argparse.ArgumentParser('nn-meter')

    # Usage 1: list predictors
    parser.add_argument(
        '--list-predictors',
        help='list all supported predictors',
        action='store_true',
        default=False
    )

    # Usage 2: latency predictors
    parser.add_argument(
        "--predictor",
        type=str,
        help="name of target predictor (hardware)"
    )
    parser.add_argument(
        '--predictor-version',
        type=float,
        help="the version of the latency predictor (If not specified, use the lateast version)",
        default=None
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tensorflow",
        type=str,
        help="Path to input Tensorflow model (*.pb)"
    )
    group.add_argument(
        "--onnx",
        type=str,
        help="Path to input ONNX model (*.onnx)"
    )
    group.add_argument(
        "--nni-ir",
        type=str,
        help="Path to input NNI IR model (*.json)"
    )
    group.add_argument(
        "--nn-meter-ir",
        type=str,
        help="Path to input nn-Meter IR model (*.json)"
    )
    group.add_argument(
        "--torch",      # Jiahang: --torch only can support the model object. The argument specifies 
        type=str,       # the name of the model, and we will look for the model in torch model zoo.
        nargs='+',
        help="Name of the input torch model from the torch model zoo"
    )

    # Usage 3: get nn-meter-ir model from tensorflow pbfile or onnx file
    # Usags: nn-meter getir --tensorflow <pb-file>
    subprasers = parser.add_subparsers(dest='getir')
    getir = subprasers.add_parser(
        'getir',
        help='specify a model type to convert to nn-meter ir graph'
    )
    getir.add_argument(
        "--tensorflow",
        type = str,
        help="Path to input Tensorflow model (*.pb)"
    )
    getir.add_argument(
        "--onnx",
        type=str,
        help="Path to input ONNX model (*.onnx)"
    )
    getir.add_argument(
        "--output",
        type=str,
        help="Path to save the output nn-meter ir graph for tensorflow and onnx (*.json)"
    )

    # Other utils
    parser.add_argument(
        "-v", "--verbose", 
        help="increase output verbosity",
        action="store_true"
    )

    # parse args
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    else:
        logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.KEYINFO)
    
    # Usage 1
    if args.list_predictors:
        preds = list_latency_predictors()
        logging.keyinfo("Supported latency predictors:")
        for p in preds:
            logging.result(f"[Predictor] {p['name']}: version={p['version']}")
        return

    # Usage 2
    if not args.getir:
        _ = apply_latency_predictor(args)

    # Usage 3
    if args.getir:
        import json
        from nn_meter.utils.graph_tool import NumpyEncoder
        graph = get_nnmeter_ir(args)
        filename = args.output
        with open(filename, "w+") as fp:
            json.dump(graph,
                fp,
                indent=4,
                skipkeys=True,
                sort_keys=True,
                cls=NumpyEncoder,
            )
