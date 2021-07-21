# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from nn_meter.utils.utils import try_import_torchvision_models
<<<<<<< HEAD
from nn_meter import load_predictor_config, load_latency_predictors
=======
from nn_meter import load_latency_predictor
import yaml
>>>>>>> 52bfa50decd5440281db743fedc65a552eeae754
import argparse
import os
import logging


def test_ir_graphs(args, predictor):
    # will remove this to examples once we have the pip package
    from glob import glob
    from nn_meter import download_from_url

    url = "https://github.com/Lynazhang/nnmeter/releases/download/0.1/ir_graphs.zip"
    download_from_url(url, "data/testmodels")
    models = glob("data/testmodels/**.json")
    for model in models:
        latency = predictor.predict(model)
        logging.debug(os.path.basename(model), latency)


def test_pb_models(args, predictor):
    # will remove this to examples once we have the pip package
    from glob import glob
    from nn_meter import download_from_url

    url = "https://github.com/Lynazhang/nnmeter/releases/download/0.1/pb_models.zip"
    download_from_url(url, "data/testmodels")
    models = glob("data/testmodels/**.pb")
    for model in models:
        latency = predictor.predict(model)
        logging.debug(os.path.basename(model), latency)


def test_onnx_models(args, predictor):
    # will remove this to examples once we have the pip package
    from glob import glob
    from nn_meter import download_from_url

    url = "https://github.com/Lynazhang/nnmeter/releases/download/0.1/onnx_models.zip"
    download_from_url(url, "data/testmodels")
    models = glob("data/testmodels/**.onnx")
    for model in models:
        latency = predictor.predict(model)
        logging.debug(os.path.basename(model), latency)


def test_pytorch_models(args, predictor):
    # will remove this to examples once we have the pip package
    models = try_import_torchvision_models()

    resnet18 = models.resnet18()
    alexnet = models.alexnet()
    vgg16 = models.vgg16()
    squeezenet = models.squeezenet1_0()
    densenet = models.densenet161()
    inception = models.inception_v3()
    googlenet = models.googlenet()
    shufflenet = models.shufflenet_v2_x1_0()
    mobilenet_v2 = models.mobilenet_v2()  # noqa: F841
    resnext50_32x4d = models.resnext50_32x4d()
    wide_resnet50_2 = models.wide_resnet50_2()
    mnasnet = models.mnasnet1_0()
    models = []
    models.append(alexnet)
    models.append(resnet18)
    models.append(vgg16)
    models.append(squeezenet)
    models.append(densenet)
    models.append(inception)
    models.append(googlenet)
    models.append(shufflenet)
    models.append(resnext50_32x4d)
    models.append(wide_resnet50_2)
    models.append(mnasnet)
    logging.debug("start to test")
    for model in models:
        latency = predictor.predict(
            model, model_type="torch", input_shape=(1, 3, 224, 224)
        )
        logging.debug(model.__class__.__name__, latency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("predict model latency on device")
    parser.add_argument(
<<<<<<< HEAD
        "--predictor", 
        type=str, 
        required=True, 
        help="name of target predictor (hardware)",
=======
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
>>>>>>> 52bfa50decd5440281db743fedc65a552eeae754
    )
    parser.add_argument(
        "--predictor-version",
        type=str,
        default=None,
        help="the version of the latency predictor (If not specified, use the lateast version)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="nn_meter/configs/predictors.yaml",
        help="config file to store current supported edge platform",
    )
    group = parser.add_mutually_exclusive_group() # Jiahang: can't handle model_type == "torch" now.
    group.add_argument(
        "--tensorflow",
        type=str,
        # required=True,
        help="Path to input Tensorflow model (*.pb)"
    )
    group.add_argument(
        "--onnx",
        type=str,
        # required=True,
        help="Path to input ONNX model (*.onnx)"
    )
    group.add_argument(
        "--nn-meter-ir",
        type=str,
        # required=True,
        help="Path to input nn-Meter IR model (*.json)"
    )
    group.add_argument(
        "--nni-ir",
        type=str,
        # required=True,
        help="Path to input NNI IR model (*.json)"
    )
    parser.add_argument(
        "-v", "--verbose", 
        help="increase output verbosity",
        action="store_true"
    )
    args = parser.parse_args()

<<<<<<< HEAD
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    pred_info = load_predictor_config(args.config, args.predictor, args.predictor_version)
    predictor = load_latency_predictors(pred_info)
    if args.tensorflow:
        input_model, model_type = args.tensorflow, "pb"
    elif args.onnx:
        input_model, model_type = args.onnx, "onnx"
    elif args.nn_meter_ir:
        input_model, model_type = args.nn_meter_ir, "json"
    elif args.nni_ir:
        input_model, model_type = args.nni_ir, "json"
    
    latency = predictor.predict(input_model, model_type)
    logging.info('predict latency: %f' % latency)
=======
    predictor = load_latency_predictor(args.predictor, args.predictor_version)
    latency = predictor.predict(args.input_model)
    print('predict latency', latency)
>>>>>>> 52bfa50decd5440281db743fedc65a552eeae754
