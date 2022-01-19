# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
import logging
import argparse
from .registry import register_module_cli, unregister_module_cli
from .predictor import list_latency_predictors_cli, apply_latency_predictor_cli, get_nnmeter_ir_cli
from .builder import list_backends_cli, list_kernels_cli, list_operators_cli, list_special_testcases_cli, \
    test_backend_connection_cli, create_workspace_cli

def nn_meter_info(args):
    if args.list_predictors:
        list_latency_predictors_cli()
    if args.list_backends:
        list_backends_cli()
    if args.list_kernels:
        list_kernels_cli()
    if args.list_operators:
        list_operators_cli()
    if args.list_testcases:
        list_special_testcases_cli()
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
    parser.add_argument(
        '--list-backends',
        help='list all supported backends',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--list-kernels',
        help='list all supported kernels when building kernel predictors',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--list-operators',
        help='list all supported operators when building fusion rule test cases',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--list-testcases',
        help='list all supported special test cases when building fusion rule test cases',
        action='store_true',
        default=False
    )

    # create subparsers for args with sub values
    subparsers = parser.add_subparsers()

    # Usage 1: latency predictors
    # Usage: nn-meter predict --predictor <hardware> --tensorflow <pb-file>
    lat_pred = subparsers.add_parser('predict', aliases=['lat_pred'], help='apply latency predictor for testing model')
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
    
    # Usage 3: nn-Meter backend:    
    # test connection to backend 
    # Usage: nn-meter connect --backend <backend-name> --workspace <path/to/workspace>
    test_connection = subparsers.add_parser(
        'connect', 
        help='connect to backend'
    )
    test_connection.add_argument(
        "--workspace",
        type=str,
        help="path to the workspace with configuration completed"
    )
    test_connection.add_argument(
        "--backend",
        type=str,
        help="the name of the testing backend"
    )
    test_connection.set_defaults(func=test_backend_connection_cli)
    
    # register and unregister backend 
    # Usage: nn-meter build register  <path/to/workspace>
    # TODO
    

    # Usage 4: create workspace folder for nn-Meter builder 
    # Usage: nn-meter create --tflite-workspace <path/to/workspace>
    create_workspace = subparsers.add_parser(
        'create', 
        help='create a workspace folder for nn-Meter builder'
    )
    platform = create_workspace.add_mutually_exclusive_group()
    platform.add_argument(
        "--tflite-workspace",
        type=str,
        help="path to place a tflite workspace for rule testing"
    )
    platform.add_argument(
        "--openvino-workspace",
        type=str,
        help="path to place a openvino workspace for rule testing"
    )
    platform.add_argument(
        "--customized-workspace",
        type=str,
        nargs='+',
        help="create a customized backend workspace for rule testing. The first word indicates the name of the ' \
            'customized backend, the second word indicates the path to place the customized backend workspace, ' \
            'and the third word indicates the path to the customized config .yaml file. The customized backend ' \
            'should be register first (refer to `nn-meter backend reg --h` for more help)."
    )
    create_workspace.set_defaults(func=create_workspace_cli)

    
    # Usage 5: change data floder
    # Usage: nn-meter set change --data <path/to/new-folder>
    #TODO

    # parse args
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(stream=sys.stdout, format="(nn-Meter) %(message)s", level=logging.INFO)
    else:
        logging.basicConfig(stream=sys.stdout, format="(nn-Meter) %(message)s", level=logging.KEYINFO)
    args.func(args)


if __name__ == '__main__':
    nn_meter_cli()
