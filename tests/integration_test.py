import re
import os
import time
from glob import glob
from tqdm import tqdm
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
    predictors = list(map(lambda x: x.split(': version='), predictors_info))
    return predictors


def get_models(model_type, ppath = "data/testmodels/pb"):
    models = glob(os.path.join(ppath, "**" + __model_suffix__[model_type]))
    models.sort()
    return models


def parse_latency_info(info):
    pattern = re.compile(r'(?<=\[RESULT\] predict latency: )[0-9.]+(?=\s+?$)')
    latency = pattern.findall(info)[0]
    return latency
    
# integration test to predict model latency
def integration_test(model_type, url, ppath, outcsv_name = "tests/test_result.txt"):
    """
    download the kernel predictors from the url
    @params:

    model_type: tensorflow, onnx, 
    url: github release url address for testing model file
    ppath:  the targeting dir to save the download model file
    outcsv_name: a summary file to save the testing results
    """
    if not os.path.isdir("../data/testmodels"):
        os.mkdir("../data")
        os.mkdir("../data/testmodels")

    # download data and unzip
    if not os.path.isdir(ppath):
        os.mkdir(ppath)
        download_from_url(url, ppath)

    # if the outcsv is not created, create it and add a title
    if not os.path.isfile(outcsv_name):
        with open(outcsv_name,"w") as f:
            f.write('model_name, model_type, predictor, predictor_version, latency\n')

    # start testing
    for model in get_models(model_type, ppath):
        for pred_name, pred_version in get_predictors():
            try:
                since = time.time()
                # print(f'nn-meter --{model_type} {model} --predictor {pred_name} --predictor-version {pred_version}')
                result = subprocess.check_output(['nn-meter', f'--{model_type}', f'{model}', '--predictor', f'{pred_name}', '--predictor-version', f'{pred_version}'])
                runtime = time.time() - since
            except NotImplementedError:
                logging.error("Meets ERROR when checking --{model_type} {model} --predictor {pred_name} --predictor-version {pred_version}")

            latency = parse_latency_info(result.decode('utf-8'))
            item = f'{os.path.basename(model)}, {model_type}, {pred_name}, {pred_version}, {latency}\n'
            # print(item)
            with open(outcsv_name, "a") as f:
                f.write(item)
    

if __name__ == "__main__":
    check_package_status()

    # check tensorflow model
    integration_test(
        model_type='tensorflow',
        url = "https://github.com/Lynazhang/nnmeter/releases/download/0.1/pb_models.zip",
        ppath = "../data/testmodels/pb",
    )

    # check onnx model
    integration_test(
        model_type='onnx',
        url = "https://github.com/Lynazhang/nnmeter/releases/download/0.1/onnx_models.zip",
        ppath = "../data/testmodels/onnx",
    )


