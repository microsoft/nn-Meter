import re
import os
import time
from glob import glob
from tqdm import tqdm
import logging
import subprocess
from nn_meter import download_from_url


# check package status
def check_package_status():
    try:
        output1 = subprocess.check_output(['nn-meter', '-h'])
    except NotImplementedError:
        logging.error("Meets ERROR when checking 'nn-meter -h'")

# check predictors list
def get_predictors():
    try:
        predictors_list = subprocess.check_output(['nn-meter', '--list-predictors'])
    except NotImplementedError:
        logging.error("Meets ERROR when checking 'nn-meter --list-predictors'")

    predictors_list = predictors_list.decode('utf-8')
    pattern = re.compile(r'(?<=\[Predictor\] ).+(?=\n)')
    predictors_info = pattern.findall(predictors_list)
    predictors = list(map(lambda x: re.sub('\s*', '', x).split(':version='), predictors_info))
    return predictors


def parse_latency_info(info):
    # (nn-Meter) [RESULT] predict latency for shufflenetv2_0.onnx: 5.423898780782251 ms
    pattern = re.compile(r'(?<=\[RESULT\] predict latency for ).*(?= ms\n)') 
    latency_info = pattern.findall(info)
    latency_list = list(map(lambda x: re.sub('\s*', '', x).split(':'), latency_info))
    return latency_list
    

# integration test to predict model latency
def integration_test_torch(model_type, model_list, output_name = "tests/test_result_torch.txt"):
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
        try:
            since = time.time()
            # print(f'nn-meter --{model_type} {model} --predictor {pred_name} --predictor-version {pred_version}')
            result = subprocess.check_output(['nn-meter', f'--torchvision'] + model_list + ['--predictor', f'{pred_name}', '--predictor-version', f'{pred_version}'])
            runtime = time.time() - since
        except NotImplementedError:
            logging.error("Meets ERROR when checking --torchvision {model_string} --predictor {pred_name} --predictor-version {pred_version}")

        latency_list = parse_latency_info(result.decode('utf-8'))
        for model, latency in latency_list:
            item = f'{model}, {model_type}, {pred_name}, {pred_version}, {round(float(latency), 4)}\n'
            # print(item)
            with open(output_name, "a") as f:
                f.write(item)

if __name__ == "__main__":
    check_package_status()

    # check torch model
    integration_test_torch(
        model_type='torch',
        model_list=[
            'resnet18', 'alexnet', 'vgg16', 'squeezenet', 'densenet', 'inception', 'googlenet', 
            'shufflenet', 'mobilenet_v2', 'resnext50_32x4d', 'wide_resnet50_2', 'mnasnet']
    )

    