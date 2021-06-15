# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from nn_meter import load_latency_predictors
import yaml
import argparse


def test_pytorch_models(args, predictor):
    import torchvision.models as models

    resnet18 = models.resnet18()
    alexnet = models.alexnet()
    vgg16 = models.vgg16()
    squeezenet = models.squeezenet1_0()
    densenet = models.densenet161()
    inception = models.inception_v3()
    googlenet = models.googlenet()
    shufflenet = models.shufflenet_v2_x1_0()
    mobilenet_v2 = models.mobilenet_v2()
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
    print("start to test")
    for model in models:
        latency = predictor.predict(
            model, model_type="torch", input_shape=(1, 3, 224, 224)
        )
        print(model.__class__.__name__, latency)


def test_onnx_models(args, predictor):
    from glob import glob

    models = glob("data/testmodels/**.onnx")
    for model in models:
        latency = predictor.predict(model)
        print(model.split("/")[-1], latency)


def test_pb_models(args, predictor):
    from glob import glob

    models = glob("data/testmodels/**.pb")
    for model in models:
        latency = predictor.predict(model)
        print(model.split("/")[-1], latency)
       # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser("predict model latency on device")
    parser.add_argument( "--input_model", type=str, required=True, help="Path to input model. ONNX, FrozenPB or JSON")
    parser.add_argument( "--hardware", type=str, default='cortexA76cpu_tflite21', help="target hardware")
    parser.add_argument( "--config", type=str, default="configs/devices.yaml", help="config file to store current supported edge platform")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)['predictors']
        if args.hardware in config:
            print(config)
       
            predictor = load_latency_predictors(config,args.hardware)
            latency=predictor.predict(args.input_model)
            #test_onnx_models(args,predictor)
            #test_pb_models(args,predictor)
            #test_pytorch_models(args, predictor)
            print('predict latency',latency)
        else:
            raise NotImplementedError

