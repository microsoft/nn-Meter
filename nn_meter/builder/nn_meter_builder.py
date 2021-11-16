import os
import json
from .rule_tester import RuleTester
from .utils.utils import dump_testcases, read_testcases
import logging
from nn_meter.builder import builder_config as config

def create_testcases():
    """create testcases and save the testcase models in the workspace
    """
    tester = RuleTester()
    testcases = tester.generate()
    
    workspace_path = config.workspace_path
    case_save_path = os.path.join(workspace_path, "results", "origin_testcases.json")
    os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
    with open(case_save_path, 'w') as fp:
        json.dump(dump_testcases(testcases), fp, indent=4)
    logging.keyinfo(f"Save the original testcases to {case_save_path}")
    return testcases


def run_testcases(backend, testcases):
    """
    @params:

    backend: applied backend with its config, should be a subclass of BaseBackend

    testcases: the Dict of testcases or the path of the testcase json file

    workspace_path: workspace directory to save the testcase json file

    """
    if isinstance(testcases, str):
        with open(testcases, 'r') as fp:
            testcases = read_testcases(json.load(fp))

    for _, testcase in testcases.items():
        for _, model in testcase.items():
            model_path = model['model']
            model['latency'] = backend.profile_model_file(model_path, model['shapes'])

    workspace_path = config.workspace_path
    case_save_path = os.path.join(workspace_path, "results", "profiled_testcases.json")
    os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
    with open(case_save_path, 'w') as fp:
        json.dump(dump_testcases(testcases), fp, indent=4)
    logging.keyinfo(f"Save the profiled testcases to {case_save_path}")
    return testcases


def detect_fusionrule(testcases):
    """ detect fusion rule by testcases latency value
    @params:

    testcases: the Dict of testcases or the path of the testcase json file
    """
    if isinstance(testcases, str):
        with open(testcases, 'r') as fp:
            testcases = read_testcases(json.load(fp))

    tester = RuleTester()
    result = tester.analyze(testcases)
    
    workspace_path = config.workspace_path
    case_save_path = os.path.join(workspace_path, "results", "detected_testcases.json")
    os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
    with open(case_save_path, 'w') as fp:
        json.dump(testcases, fp, indent=4)
    logging.keyinfo(f"Save the testcases to {case_save_path}")
    return result
