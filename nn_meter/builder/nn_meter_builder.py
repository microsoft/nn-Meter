# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import logging
from .utils import builder_config as config
from nn_meter.builder.backends import connect_backend


def profile_models(backend, models, mode = 'ruletest', metrics = ["latency"], details = False):
    """ run models with given backend and return latency of testcase models
    @params:

    backend: applied backend with its config, should be a subclass of BaseBackend

    models: the Dict of models or the path of the json file about models information 

    mode: the mode for running models, including ['ruletest', 'predbuild']

    metrics: required metrics to report. We only support latency for metric by now.

    details: if False, only metrics result will be dumped to the profiled results. Otherwise 
        models information will be dumpled, too.

    """
    if isinstance(models, str):
        with open(models, 'r') as fp:
            models = json.load(fp)

    ws_mode_path = config.get('MODEL_DIR', mode)
    model_save_path = os.path.join(ws_mode_path, 'models')
    os.makedirs(model_save_path, exist_ok=True)
    for _, modules in models.items():
        for _, model in modules.items():
            model_path = model['model']
            profiled_res = backend.profile_model_file(model_path, model_save_path, model['shapes'])
            for metric in metrics:
                model[metric] = profiled_res[metric]

    # save information to json file
    info_save_path = os.path.join(ws_mode_path, "results", "profiled_results.json")
    os.makedirs(os.path.dirname(info_save_path), exist_ok=True)
    from .backend_meta.utils import dump_profiled_results
    with open(info_save_path, 'w') as fp:
        json.dump(dump_profiled_results(models, details=details), fp, indent=4)
    logging.keyinfo(f"Save the profiled models information to {info_save_path}")
    return models

def sample_and_profile_kernel_data(kernel_type, sample_num, backend, sampling_mode = 'prior', configs = None, mark = ''):
    ''' sample kernel configs and profile kernel model based on configs
    '''
    from nn_meter.builder.kernel_predictor_builder import generate_config_sample

    # generate test cases
    kernels = generate_config_sample(kernel_type, sample_num, mark=mark, 
                                     sampling_mode=sampling_mode, configs=configs)
    
    # connect to backend, run test cases and get latency
    backend = connect_backend(backend=backend)
    profiled_results = profile_models(backend, kernels, mode='predbuild')
    return profiled_results


def build_predictor_for_kernel(kernel_type, backend = None, init_sample_num = 1000, finegrained_sample_num = 10, iteration = 5, error_threshold = 0.1):
    """ 
    Build latency predictor for given kernel. This method contains three main steps:
    1. sample kernel configs and profile kernel model based on configs;
    2. initialize latency predictor of kernel based on the profiled data;
    3. adopt adaptive sampler with iteratively doing step 1 for finegrained sampling to improve predictor performance

    @params
    
    kernel_type (str): the type of kernel
    
    backend (str): the backend to profile models
    
    init_sample_num (int, optional): the data size for predictor initialization. Defaults to 1000.
    
    finegrained_sample_num (int, optional): the data size for adaptive sampling. For each data with error higher than 
        error_threshold, #finegrained_sample_num data will be generated based the the large error data. Defaults to 10.

    iteration (int, optional): the iteration for adaptive sampler. Defaults to 5.

    error_threshold (float, optional): the threshold of large error. Defaults to 0.2.
    """
    from nn_meter.builder.kernel_predictor_builder import build_predictor_by_data, get_data_by_profiled_results
    
    # init predictor builder with prior data sampler
    kernel_data = sample_and_profile_kernel_data(kernel_type, init_sample_num, backend, sampling_mode='prior', mark='prior')
    data = get_data_by_profiled_results(kernel_data)

    # use current sampled data to build regression model, and locate data with large errors in testset
    predictor, acc10, error_configs = build_predictor_by_data(kernel_type, data, backend, error_threshold=error_threshold)
    logging.info(f'Iteration 0: acc10 {acc10}, error_configs number: {len(error_configs)}')
    
    for i in range(1, iteration):
        new_data = sample_and_profile_kernel_data(kernel_type, finegrained_sample_num, backend, 
                                                  sampling_mode = 'finegrained', configs=error_configs, mark='finegrained_iter{i}')
        # TODO: data.extend(new_data)
        predictor, acc10, error_configs = build_predictor_by_data(kernel_type, new_data, backend, error_threshold=error_threshold)
        logging.info(f'Iteration {i}: acc10 {acc10}, error_configs number: {len(error_configs)}')

    return predictor, data
