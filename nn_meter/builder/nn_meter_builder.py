# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import logging
from .utils import builder_config as config
from nn_meter.builder.backends import connect_backend
from nn_meter.builder.predictor_builder import generate_config_sample, build_predictor


def run_testcases(backend, testcases, mode = 'ruletest', metrics = ["latency"]):
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
    from .utils.utils import dump_testcases_with_latency
    with open(info_save_path, 'w') as fp:
        json.dump(dump_testcases_with_latency(testcases), fp, indent=4)
    logging.keyinfo(f"Save the profiled testcases information to {info_save_path}")
    return testcases


def get_sampled_data(block_type, sample_num, backend, sample_stage='prior', configs=None, mark = ''):
    ''' init sample of testcases
    '''
    # generate test cases
    testcase = generate_config_sample(block_type, sample_num, mark=mark, 
                                      sample_stage=sample_stage, configs=configs)
    
    # connect to backend, run test cases and get latency
    backend = connect_backend(backend=backend)
    profiled_testcases = run_testcases(backend, testcase, mode='predbuild')
    return profiled_testcases


def build_adaptived_predictor(block_type, init_sample_num = 1000, finegrained_sample_num = 10, iteration = 5, error_threshold = 0.2):
    """[summary]

    Args:
        block_type ([type]): [description]
        init_sample_num (int, optional): [description]. Defaults to 1000.
        finegrained_sample_num (int, optional): [description]. Defaults to 10.
        iteration (int, optional): [description]. Defaults to 5.
        error_threshold (float, optional): [description]. Defaults to 0.2.
    """
    # init predictor builder with prior data sampler
    data = get_sampled_data(init_sample_num)
    # use current sampled data to build regression model, and locate data with large errors in testset
    acc10, cfgs = build_predictor(block_type=block_type, hardware='cpu', data=data, error_threshold=0.2)
    
    for i in range(1, iteration):
        new_data = get_sampled_data(finegrained_sample_num, configs=cfgs)
        data.extend(new_data)
        acc10, cfgs = build_predictor(block_type=block_type, hardware='cpu', data=data, error_threshold=error_threshold)
        print('cfgs', cfgs)
        logging.info(f'iteration {i}: acc10 {acc10}')
