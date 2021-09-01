# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from glob import glob
import os
import sys
import argparse
import logging
from nn_meter.nn_meter import *

__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__user_data_folder__ = os.path.expanduser('~/.nn_meter/data')

__predictors_cfg_filename__ = 'predictors.yaml'


def list_latency_predictors_cli():
    preds = list_latency_predictors()
    logging.keyinfo("Supported latency predictors:")
    for p in preds:
        logging.result(f"[Predictor] {p['name']}: version={p['version']}")
    return


def apply_latency_predictor_cli(args):
    """apply latency predictor to predict model latency according to the command line interface arguments
    """
    if not args.predictor:
        logging.keyinfo('You must specify a predictor. Use "nn-meter --list-predictors" to see all supporting predictors.')
        return

    # specify model type
    if args.tensorflow:
        input_model, model_type, model_suffix = args.tensorflow, "pb", ".pb"
    elif args.onnx:
        input_model, model_type, model_suffix = args.onnx, "onnx", ".onnx"
    elif args.nn_meter_ir:
        input_model, model_type, model_suffix = args.nn_meter_ir, "nnmeter-ir", ".json"
    elif args.torchvision: # torch model name from torchvision model zoo
        input_model_list, model_type = args.torchvision, "torch" 

    # load predictor
    predictor = load_latency_predictor(args.predictor, args.predictor_version)

    # specify model for prediction
    if not args.torchvision: # input of tensorflow, onnx, nnmeter-ir and nni-ir is file name, while input of torchvision is string list
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
        latency = predictor.predict(model, model_type) # in unit of ms
        result[os.path.basename(model)] = latency
        logging.result(f'[RESULT] predict latency for {os.path.basename(model)}: {latency} ms')
    
    return result

def get_nnmeter_ir_cli(args):
    """convert pb file or onnx file to nn-Meter IR graph according to the command line interface arguments
    """
    import json
    from nn_meter.utils.graph_tool import NumpyEncoder
    if args.tensorflow:
        graph = model_file_to_graph(args.tensorflow, 'pb')
        filename = args.output if args.output else args.tensorflow.replace(".pb", "_pb_ir.json") 
    elif args.onnx:
        graph = model_file_to_graph(args.onnx, 'onnx')
        filename = args.output if args.output else args.onnx.replace(".onnx", "_onnx_ir.json") 
    else:
        raise ValueError(f"Unsupported model.")
    
    if not str.endswith(filename, '.json'): filename += '.json'
    with open(filename, "w+") as fp:
        json.dump(graph,
            fp,
            indent=4,
            skipkeys=True,
            sort_keys=True,
            cls=NumpyEncoder,
        )
    
    logging.result(f'The nn-meter ir graph has been saved. Saved path: {os.path.abspath(filename)}')


def nn_meter_info(args):
    if args.list_predictors:
        list_latency_predictors_cli()
    else:
        logging.keyinfo('please run "nn-meter {positional argument} --help" to see nn-meter guidance')


def nn_meter_cli():
    parser = argparse.ArgumentParser('nn-meter', description='please run "nn-meter {positional argument} --help" to see nn-meter guidance')
    parser.set_defaults(func=nn_meter_info)

    # optional arguments
    parser.add_argument(
        "-v", "--verbose", 
        help="increase output verbosity",
        action="store_true"
    )
    parser.add_argument(
        '--list-predictors',
        help='list all supported predictors',
        action='store_true',
        default=False
    )

    # create subparsers for args with sub values
    subparsers = parser.add_subparsers()

    # Usage 1: latency predictors
    lat_pred = subparsers.add_parser('lat_pred', help='apply latency predictor for testing model')
    lat_pred.add_argument(
        "--predictor",
        type=str,
        help="name of target predictor (hardware)"
    )
    lat_pred.add_argument(
        "--predictor-version",
        type=float,
        help="the version of the latency predictor (if not specified, use the lateast version)",
        default=None
    )
    group = lat_pred.add_mutually_exclusive_group()
    group.add_argument(
        "--tensorflow",
        type=str,
        help="path to input Tensorflow model (*.pb file or floder)"
    )
    group.add_argument(
        "--onnx",
        type=str,
        help="path to input ONNX model (*.onnx file or floder)"
    )
    group.add_argument(
        "--nn-meter-ir",
        type=str,
        help="path to input nn-Meter IR model (*.json file or floder)"
    )
    group.add_argument(
        "--torchvision",        # --torchvision only can support the model object. The argument specifies 
        type=str,               # the name of the model, and we will look for the model in torchvision model zoo.
        nargs='+',
        help="name of the input torch model from the torchvision model zoo"
    )
    lat_pred.set_defaults(func=apply_latency_predictor_cli)

    # Usage 2: get nn-meter-ir model from tensorflow pbfile or onnx file
    # Usage: nn-meter get_ir --tensorflow <pb-file>
    get_ir = subparsers.add_parser(
        'get_ir', 
        help='specify a model type to convert to nn-meter ir graph'
    )
    group2 = get_ir.add_mutually_exclusive_group()
    group2.add_argument(
        "--tensorflow",
        type = str,
        help="path to input Tensorflow model (*.pb)"
    )
    group2.add_argument(
        "--onnx",
        type=str,
        help="path to input ONNX model (*.onnx)"
    )
    get_ir.add_argument(
        "-o", "--output",
        type=str,
        help="path to save the output nn-meter ir graph for tensorflow and onnx (*.json), default to be /path/to/input/file/<input_file_name>_ir.json"
    )
    get_ir.set_defaults(func=get_nnmeter_ir_cli)

    # parse args
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(stream=sys.stdout, format="(nn-Meter) %(message)s", level=logging.INFO)
    else:
        logging.basicConfig(stream=sys.stdout, format="(nn-Meter) %(message)s", level=logging.KEYINFO)
    args.func(args)


if __name__ == '__main__':
    nn_meter_cli()
