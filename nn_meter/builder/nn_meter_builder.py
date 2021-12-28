# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import logging
from .utils import builder_config as config
from .utils.utils import dump_testcases_with_latency


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

    case_save_path = os.path.join(ws_mode_path, "results", "profiled_testcases.json")
    os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
    with open(case_save_path, 'w') as fp:
        json.dump(dump_testcases_with_latency(testcases), fp, indent=4)
    logging.keyinfo(f"Save the profiled testcases to {case_save_path}")
    return testcases


def init_data_sampler():
    pass


def regress_with_adaptive_sampler():
    pass