import os
import json
import logging
from .rule_tester import config
from .rule_tester import RuleTester
from .utils.utils import dump_testcases, read_testcases

def create_testcases(workspace_path=""):
    """
    @params:

    workspace_path: workspace directory to save testcase models 

    """
    config.set('model_dir', os.path.join(workspace_path, "test_model"), 'ruletest')

    tester = RuleTester()
    testcases = tester.generate()
    
    if workspace_path:
        case_save_path = os.path.join(workspace_path, "results", "origin_testcases.json")
        os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
        with open(case_save_path, 'w') as fp:
            json.dump(dump_testcases(testcases), fp, indent=4)
        logging.keyinfo(f"Save the original testcases to {case_save_path}")
    return testcases


def run_testcases(backend, testcases, workspace_path=""):
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

    if workspace_path:
        case_save_path = os.path.join(workspace_path, "results", "profiled_testcases.json")
        os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
        with open(case_save_path, 'w') as fp:
            json.dump(dump_testcases(testcases), fp, indent=4)
        logging.keyinfo(f"Save the profiled testcases to {case_save_path}")
    return testcases


def detect_fusionrule(testcases, workspace_path=""):
    """
    @params:

    testcases: the Dict of testcases or the path of the testcase json file

    workspace_path: workspace directory to save the testcase json file

    """
    if isinstance(testcases, str):
        with open(testcases, 'r') as fp:
            testcases = read_testcases(json.load(fp))

    tester = RuleTester()
    result = tester.analyze(testcases)
    
    if workspace_path:
        case_save_path = os.path.join(workspace_path, "results", "detected_testcases.json")
        os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
        with open(case_save_path, 'w') as fp:
            json.dump(testcases, fp, indent=4)
        logging.keyinfo(f"Save the testcases to {case_save_path}")
    return result
