# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import logging
from .utils import builder_config as config
from nn_meter.builder.backends import connect_backend


def profile_models(backend, testcases, mode = 'ruletest', metrics = ["latency"]):
    """ run testcases with given backend and return latency of testcase models
    @params:

    backend: applied backend with its config, should be a subclass of BaseBackend
    testcases: the Dict of testcases or the path of the testcase json file
    mode: the mode for running testcases, including ['ruletest', 'predbuild']
    metrics: required metrics to report. We only support latency for metric by now.

    """
    if isinstance(testcases, str):
        with open(testcases, 'r') as fp:
            testcases = json.load(fp)

    ws_mode_path = config.get('MODEL_DIR', mode)
    model_save_path = os.path.join(ws_mode_path, 'testcases')
    os.makedirs(model_save_path, exist_ok=True)
    for _, testcase in testcases.items():
        for _, model in testcase.items():
            model_path = model['model']
            profiled_res = backend.profile_model_file(model_path, model_save_path, model['shapes'])
            for metric in metrics:
                model[metric] = profiled_res[metric]

    info_save_path = os.path.join(ws_mode_path, "results", "profiled_testcases.json")
    os.makedirs(os.path.dirname(info_save_path), exist_ok=True)
    from .backend_meta.utils import dump_testcases_with_latency
    with open(info_save_path, 'w') as fp:
        json.dump(dump_testcases_with_latency(testcases), fp, indent=4)
    logging.keyinfo(f"Save the profiled testcases information to {info_save_path}")
    return testcases


def sample_and_profile_kernel_data(kernel_type, sample_num, backend, sampling_mode = 'prior', configs = None, mark = ''):
    ''' sample kernel configs and profile kernel model based on configs
    '''
    from nn_meter.builder.kernel_predictor_builder import generate_config_sample

    # generate test cases
    testcase = generate_config_sample(kernel_type, sample_num, mark=mark, 
                                      sampling_mode=sampling_mode, configs=configs)
    
    # connect to backend, run test cases and get latency
    backend = connect_backend(backend=backend)
    profiled_testcases = profile_models(backend, testcase, mode='predbuild')
    return profiled_testcases


def build_predictor_for_kernel(kernel_type, hardware = None, init_sample_num = 1000, finegrained_sample_num = 10, iteration = 5, error_threshold = 0.2):
    """ 
    Build latency predictor for given kernel. This method contains three main steps:
    1. sample kernel configs and profile kernel model based on configs;
    2. initialize latency predictor of kernel based on the profiled data;
    3. adopt adaptive sampler with iteratively doing step 1 for finegrained sampling to improve predictor performance

    @params
    
    kernel_type (str): the type of kernel
    
    hardware (str): the hardware of data
    
    init_sample_num (int, optional): the data size for predictor initialization. Defaults to 1000.
    
    finegrained_sample_num (int, optional): the data size for adaptive sampling. For each data with error higher than 
        error_threshold, #finegrained_sample_num data will be generated based the the large error data. Defaults to 10.

    iteration (int, optional): the iteration for adaptive sampler. Defaults to 5.

    error_threshold (float, optional): the threshold of large error. Defaults to 0.2.
    """
    from nn_meter.builder.kernel_predictor_builder import build_predictor_by_data
    
    # init predictor builder with prior data sampler
    data = sample_and_profile_kernel_data(init_sample_num)
    # use current sampled data to build regression model, and locate data with large errors in testset
    predictor, acc10, error_configs = build_predictor_by_data(kernel_type=kernel_type, hardware=hardware, data=data, error_threshold=0.2)
    logging.info(f'Iteration 0: acc10 {acc10}, error_configs number: {len(error_configs)}')
    
    for i in range(1, iteration):
        new_data = sample_and_profile_kernel_data(finegrained_sample_num, configs=error_configs)
        data.extend(new_data)
        # TODO: save_data
        predictor, acc10, error_configs = build_predictor_by_data(kernel_type=kernel_type, hardware=hardware, data=data, error_threshold=error_threshold)
        logging.info(f'Iteration {i}: acc10 {acc10}, error_configs number: {len(error_configs)}')

    return predictor, data
