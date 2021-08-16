# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from nn_meter.utils.utils import try_import_torchvision_models
from nn_meter.nn_meter import apply_latency_predictor, get_nnmeter_ir
from nn_meter import list_latency_predictors
import argparse
import os
import sys
import logging
from glob import glob
from functools import partial, partialmethod

logging.KEYINFO = 22
logging.addLevelName(logging.KEYINFO, 'KEYINFO')
logging.Logger.keyinfo = partialmethod(logging.Logger.log, logging.KEYINFO)
logging.keyinfo = partial(logging.log, logging.KEYINFO)

logging.RESULT = 25
logging.addLevelName(logging.RESULT, 'RESULT')
logging.Logger.result = partialmethod(logging.Logger.log, logging.RESULT)
logging.result = partial(logging.log, logging.RESULT)


def test_ir_graphs(predictor, ppath="data/testmodels"):
    # will remove this to examples once we have the pip package
    from glob import glob
    from nn_meter import download_from_url

    url = "https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/ir_graphs.zip"
    download_from_url(url, ppath)
    models = glob(os.path.join(ppath, "**.json"))
    print(models)
    for model in models:
        latency = predictor.predict(model) # in unit of ms
        logging.info(os.path.basename(model), latency)


def test_pb_models(predictor, ppath="data/testmodels"):
    # will remove this to examples once we have the pip package
    from glob import glob
    from nn_meter import download_from_url

    url = "https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/pb_models.zip"
    download_from_url(url, ppath)
    models = glob(os.path.join(ppath, "**.pb"))
    for model in models:
        latency = predictor.predict(model) # in unit of ms
        logging.info(os.path.basename(model), latency)


def test_onnx_models(predictor, ppath="data/testmodels"):
    # will remove this to examples once we have the pip package
    from glob import glob
    from nn_meter import download_from_url

    url = "https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/onnx_models.zip"
    download_from_url(url, ppath)
    models = glob(os.path.join(ppath, "**.onnx"))
    for model in models:
        latency = predictor.predict(model) # in unit of ms
        logging.info(os.path.basename(model), latency)


def test_pytorch_models(args, predictor):
    # will remove this to examples once we have the pip package
    models = try_import_torchvision_models()

    resnet18 = models.resnet18()
    alexnet = models.alexnet()
    vgg16 = models.vgg16()
    squeezenet = models.squeezenet1_0()
    densenet161 = models.densenet161()
    inception_v3 = models.inception_v3()
    googlenet = models.googlenet()
    shufflenet_v2 = models.shufflenet_v2_x1_0()
    mobilenet_v2 = models.mobilenet_v2()  # noqa: F841
    resnext50_32x4d = models.resnext50_32x4d()
    wide_resnet50_2 = models.wide_resnet50_2()
    mnasnet = models.mnasnet1_0()
    models = []
    models.append(alexnet)
    models.append(resnet18)
    models.append(vgg16)
    models.append(squeezenet)
    models.append(densenet161)
    models.append(inception_v3)
    models.append(googlenet)
    models.append(shufflenet_v2)
    models.append(resnext50_32x4d)
    models.append(wide_resnet50_2)
    models.append(mnasnet)
    logging.info("start to test")
    for model in models:
        latency = predictor.predict(
            model, model_type="torch", input_shape=(1, 3, 224, 224)
        ) # the resulting latency is in unit of ms
        logging.info(model.__class__.__name__, latency)


if __name__ == "__main__":
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
        "--predictor-version",
        type=float,
        default=None,
        help="the version of the latency predictor (If not specified, use the lateast version)",
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
        "--nn-meter-ir",
        type=str,
        help="Path to input nn-Meter IR model (*.json)"
    )
    group.add_argument(
        "--torchvision",        # --torchvision only can support the model object. The argument specifies the name 
        type=str,               # of the model, and we will look for the model in torchvision model zoo.
        nargs='+',
        help="Name of the input torch model from the torchvision model zoo"
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
        "-o", "--output",
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

    # Usage 2
    if not args.getir:
        _ = apply_latency_predictor(args)

    # Usage 3
    if args.getir:
        get_nnmeter_ir(args)

    
