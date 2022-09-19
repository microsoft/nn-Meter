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
    elif args.list_backends:
        list_backends_cli()
    elif args.list_kernels:
        list_kernels_cli()
    elif args.list_operators:
        list_operators_cli()
    elif args.list_testcases:
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

    # Usage 1: predict latency for models
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
    model_type = lat_pred.add_mutually_exclusive_group()
    model_type.add_argument(
        "--tensorflow",
        type=str,
        help="path to input Tensorflow model (*.pb file or floder)"
    )
    model_type.add_argument(
        "--onnx",
        type=str,
        help="path to input ONNX model (*.onnx file or floder)"
    )
    model_type.add_argument(
        "--nn-meter-ir",
        type=str,
        help="path to input nn-Meter IR model (*.json file or floder)"
    )
    model_type.add_argument(
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
    model_type = get_ir.add_mutually_exclusive_group()
    model_type.add_argument(
        "--tensorflow",
        type = str,
        help="path to input Tensorflow model (*.pb)"
    )
    model_type.add_argument(
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

    # Usage 3: create workspace folder for nn-Meter builder 
    # Usage: nn-meter create --tflite-workspace <path/to/workspace>
    create_workspace = subparsers.add_parser(
        'create', 
        help='create a workspace folder for nn-Meter builder'
    )
    platform = create_workspace.add_mutually_exclusive_group()
    platform.add_argument(
        "--tflite-workspace",
        type=str,
        help="path to place a tflite workspace for nn-Meter builder"
    )
    platform.add_argument(
        "--openvino-workspace",
        type=str,
        help="path to place a openvino workspace for nn-Meter builder"
    )
    platform.add_argument(
        "--customized-workspace",
        type=str,
        help="path to place a customized workspace for nn-Meter builder. A customized backend should be register first (refer to `nn-meter register --h` for more help)."
    )
    create_workspace.add_argument(
        "--backend",
        type=str,
        help="the backend name for registered backend"
    )
    create_workspace.set_defaults(func=create_workspace_cli)

    # Usage 4: test the connection to backend 
    # Usage: nn-meter connect --backend <backend-name> --workspace <path/to/workspace>
    test_connection = subparsers.add_parser(
        'connect', 
        help='connect to backend'
    )
    test_connection.add_argument(
        "--backend",
        type=str,
        help="the name of the testing backend"
    )
    test_connection.add_argument(
        "--workspace",
        type=str,
        help="path to the workspace with configuration completed"
    )
    test_connection.set_defaults(func=test_backend_connection_cli)
    
    # Usage 5: register customized module 
    # Usage: nn-meter register --backend <path/to/meta/file>
    register = subparsers.add_parser(
        'register', 
        help='register customized module to nn-Meter, supporting type: predictor, backend, operator, testcase, operator'
    )
    module_type = register.add_mutually_exclusive_group()
    module_type.add_argument(
        "--predictor",
        type=str,
        help="path to the meta file to register a customized predictor"
    )
    module_type.add_argument(
        "--backend",
        type=str,
        help="path to the meta file to register a customized backend"
    )
    module_type.add_argument(
        "--operator",
        type=str,
        help="path to the meta file to register a customized operator"
    )
    module_type.add_argument(
        "--testcase",
        type=str,
        help="path to the meta file to register a customized testcase"
    )
    module_type.add_argument(
        "--kernel",
        type=str,
        help="path to the meta file to register a customized kernel"
    )
    register.set_defaults(func=register_module_cli)
    
    # Usage 6: unregister customized module 
    # Usage: nn-meter unregister --backend <path/to/meta/file>
    unregister = subparsers.add_parser(
        'unregister', 
        help='unregister customized module from nn-Meter, supporting type: predictor, backend, operator, testcase, operator'
    )
    module_type = unregister.add_mutually_exclusive_group()
    module_type.add_argument(
        "--predictor",
        type=str,
        help="name of the registered predictor"
    )
    module_type.add_argument(
        "--backend",
        type=str,
        help="name of the registered backend"
    )
    module_type.add_argument(
        "--operator",
        type=str,
        nargs='+',
        help="name of the registered operator"
    )
    module_type.add_argument(
        "--testcase",
        type=str,
        nargs='+',
        help="name of the registered testcase"
    )
    module_type.add_argument(
        "--kernel",
        type=str,
        help="name of the registered kernel"
    )
    unregister.add_argument(
        "--predictor-version",
        type=float,
        help="the version of the predictor (if not specified, unregister all version)",
        default=None
    )
    unregister.set_defaults(func=unregister_module_cli)

    # Usage 7: change data folder
    # Usage: nn-meter set_data --data <path/to/new-folder>
    # TODO

    # parse args
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(stream=sys.stdout, format="(nn-Meter) %(message)s", level=logging.INFO)
    else:
        logging.basicConfig(stream=sys.stdout, format="(nn-Meter) %(message)s", level=logging.KEYINFO)
    args.func(args)


if __name__ == '__main__':
    nn_meter_cli()
