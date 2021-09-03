# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
import os
import time
from glob import glob
import logging
import subprocess
from nn_meter import download_from_url


__model_suffix__ = {
    "tensorflow": ".pb",
    "onnx": ".onnx"
}


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


def get_models(model_type, ppath = "data/testmodels/pb"):
    models = glob(os.path.join(ppath, "**" + __model_suffix__[model_type]))
    models.sort() # sort the models list by alphabetical order
    return models


def parse_latency_info(info):
    # (nn-Meter) [RESULT] predict latency for shufflenetv2_0.onnx: 5.423898780782251 ms
    pattern = re.compile(r'(?<=\[RESULT\] predict latency for ).*(?= ms\n)')
    latency_info = pattern.findall(info)
    latency_list = list(map(lambda x: re.sub('\s*', '', x).split(':'), latency_info))
    return latency_list
    

# integration test to predict model latency
def integration_test(model_type, url, ppath, output_name = "tests/test_result.txt"):
    """
    download the kernel predictors from the url
    @params:

    model_type: tensorflow, onnx, 
    url: github release url address for testing model file
    ppath:  the targeting dir to save the download model file
    output_name: a summary file to save the testing results
    """
    if not os.path.isdir("../data/testmodels"):
        os.mkdir("../data")
        os.mkdir("../data/testmodels")

    # download data and unzip
    if not os.path.isdir(ppath):
        os.mkdir(ppath)
        logging.keyinfo(f'Download from {url} ...')
        download_from_url(url, ppath)

    # if the output_name is not created, create it and add a title
    if not os.path.isfile(output_name):
        with open(output_name,"w") as f:
            f.write('model_name, model_type, predictor, predictor_version, latency\n')
    
    # start testing
    for pred_name, pred_version in get_predictors():
        try:
            since = time.time()
            # print(f'nn-meter --{model_type} {ppath} --predictor {pred_name} --predictor-version {pred_version}')
            result = subprocess.check_output(['nn-meter', 'lat_pred', f'--{model_type}', f'{ppath}', '--predictor', f'{pred_name}', '--predictor-version', f'{pred_version}'])
            runtime = time.time() - since
        except NotImplementedError:
            logging.error(f"Meets ERROR when checking --{model_type} {ppath} --predictor {pred_name} --predictor-version {pred_version}")

        latency_list = parse_latency_info(result.decode('utf-8'))
        for model, latency in latency_list:
            item = f'{model}, {model_type}, {pred_name}, {pred_version}, {round(float(latency), 4)}\n'
            # print(item)
            with open(output_name, "a") as f:
                f.write(item)       
    

def check_getir_module(model_type, ppath):
    for model in get_models(model_type, ppath):
        try:
            _ = subprocess.check_output(['nn-meter', 'get_ir', f'--{model_type}', model])
            _ = subprocess.check_output(['nn-meter', 'get_ir', f'--{model_type}', model, '--output', f'temp.json'])
            if os.path.exists('temp.json'):
                os.remove('temp.json')
            break # test just one file to avoid time cosuming
        except NotImplementedError:
            logging.error("Meets ERROR when checking get_ir --{model_type} {ppath}'")


if __name__ == "__main__":
    check_package_status()
    output_name = "tests/test_result.txt"

    # check tensorflow model
    integration_test(
        model_type='tensorflow',
        url="https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/pb_models.zip",
        ppath="../data/testmodels/pb",
        output_name=output_name
    )

    # check onnx model
    integration_test(
        model_type='onnx',
        url="https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/onnx_models.zip",
        ppath="../data/testmodels/onnx",
        output_name=output_name
    )

    # check nnmeter-ir graph model
    integration_test(
        model_type='nn-meter-ir',
        url="https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/ir_graphs.zip",
        ppath="../data/testmodels/ir",
        output_name=output_name
    )

    # check getir
    check_getir_module(
        model_type='tensorflow',
        ppath = "../data/testmodels/pb"
    )

    check_getir_module(
        model_type='onnx',
        ppath = "../data/testmodels/onnx"
    )
