# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
from glob import glob
from nn_meter import list_latency_predictors, load_latency_predictor, model_file_to_graph


def list_latency_predictors_cli():
    preds = list_latency_predictors()
    logging.keyinfo("Supported latency predictors:")
    for p in preds:
        logging.result(f"[Predictor] {p['name']}: version={p['version']}")
    return


def apply_latency_predictor_cli(args):
    """apply latency predictor to predict model latency according to the command line interface arguments
    """

    # specify model type
    if args.tensorflow:
        input_model, model_type, model_suffix = args.tensorflow, "pb", ".pb"
    elif args.onnx:
        input_model, model_type, model_suffix = args.onnx, "onnx", ".onnx"
    elif args.nn_meter_ir:
        input_model, model_type, model_suffix = args.nn_meter_ir, "nnmeter-ir", ".json"
    elif args.torchvision: # torch model name from torchvision model zoo
        input_model_list, model_type = args.torchvision, "torch" 
    else:
        logging.keyinfo('please run "nn-meter predict --help" to see guidance.')
        return
    
    if not args.predictor:
        logging.keyinfo('You must specify a predictor. Use "nn-meter --list-predictors" to see all supporting predictors.')
        return

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
    from nn_meter.utils.utils import NumpyEncoder
    if args.tensorflow:
        graph = model_file_to_graph(args.tensorflow, 'pb')
        filename = args.output if args.output else args.tensorflow.replace(".pb", "_pb_ir.json") 
    elif args.onnx:
        graph = model_file_to_graph(args.onnx, 'onnx')
        filename = args.output if args.output else args.onnx.replace(".onnx", "_onnx_ir.json") 
    else:
        logging.keyinfo('please run "nn-meter get_ir --help" to see guidance.')
        return
    
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
