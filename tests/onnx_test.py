import re
import os
import time
import logging
import subprocess
from integration_test import *
import argparse


__torchvision_model_zoo__ = {
        'resnet18': 'models.resnet18()',
        'alexnet': 'models.alexnet()',
        'vgg16': 'models.vgg16()',
        'squeezenet': 'models.squeezenet1_0()',
        'densenet161': 'models.densenet161()',
        'inception_v3': 'models.inception_v3()',
        'googlenet': 'models.googlenet()',
        'shufflenet_v2': 'models.shufflenet_v2_x1_0()',
        'mobilenet_v2': 'models.mobilenet_v2()',
        'resnext50_32x4d': 'models.resnext50_32x4d()',
        'wide_resnet50_2': 'models.wide_resnet50_2()',
        'mnasnet': 'models.mnasnet1_0()',
    }

    
# integration test to predict model latency
def integration_test_onnx_based_torch(model_type, model_list, output_name = "tests/onnx_test.txt"):
    """
    download the kernel predictors from the url
    @params:

    model_type: torch
    model_list:  the torchvision model waiting for latency prediction
    output_name: a summary file to save the testing results
    """
    # if the output is not created, create it and add a title
    if not os.path.isfile(output_name):
        with open(output_name,"w") as f:
            f.write('model_name, model_type, predictor, predictor_version, latency\n')
    
    # start testing
    for pred_name, pred_version in get_predictors():
        for model in model_list:
            try:
                since = time.time()
                print(" ".join(['nn-meter', 'lat_pred', '--torchvision', model, '--predictor', pred_name, '--predictor-version', pred_version]))
                result = subprocess.run(
                    ['nn-meter', 'lat_pred', '--torchvision', model, '--predictor', pred_name, '--predictor-version', pred_version],
                    stdout=subprocess.PIPE)
                print(result.stderr, result.stdout)

                runtime = time.time() - since
                print("Run time: ", runtime)
                latency_list = parse_latency_info(result.stdout.decode('utf-8'))
                for model, latency in latency_list:
                    item = f'{model}, {model_type}, {pred_name}, {pred_version}, {round(float(latency), 4)}\n'
                    with open(output_name, "a") as f:
                        f.write(item)
            except NotImplementedError:
                logging.error(f"Meets ERROR when checking --torchvision {model} --predictor {pred_name} --predictor-version {pred_version}")


if __name__ == "__main__":

    integration_test_onnx_based_torch(
        model_type='torch',
        model_list=[
            'resnet18', 'alexnet', 'list', 'squeezenet', 'densenet161', 'inception_v3', 'googlenet', 
            'shufflenet_v2', 'mobilenet_v2', 'resnext50_32x4d', 'wide_resnet50_2', 'mnasnet']
    )
    