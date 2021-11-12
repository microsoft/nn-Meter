import os
import json
from .rule_tester import config
from .rule_tester import RuleTester
from .utils.utils import dump_testcases, read_testcases

def create_testcases(model_dir, case_save_path='./data/testcases.json'):
    """
    @params:

    model_dir: directory to save testcase models #TODO：refine interface
        

    case_save_path: path to save the testcase json file

    """
    config.set('model_dir', model_dir, 'ruletest')

    tester = RuleTester()
    testcases = tester.generate()
    
    if case_save_path:
        os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
        with open(case_save_path, 'w') as fp:
            json.dump(dump_testcases(testcases), fp, indent=4)

    return testcases


def run_testcases(backend, testcases, case_save_path='./data/profiled_testcases.json'):
    """
    @params:

    backend: applied backend with its config, should be a subclass of BaseBackend

    testcases: the Dict of testcases or the path of the testcase json file

    case_save_path: path to save the testcase json file

    """
    if isinstance(testcases, str):
        with open(testcases, 'r') as fp:
            testcases = read_testcases(json.load(fp))

    for _, testcase in testcases.items():
        for _, model in testcase.items():
            model_path = model['model']
            model['latency'] = backend.profile_model_file(model_path, model['shapes'])

    if case_save_path:
        os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
        with open(case_save_path, 'w') as fp:
            json.dump(dump_testcases(testcases), fp, indent=4)
    return testcases


def detect_fusionrule(testcases, case_save_path='./data/detected_testcases.json'):
    """
    @params:

    testcases: the Dict of testcases or the path of the testcase json file

    case_save_path: path to save the testcase json file

    """
    if isinstance(testcases, str):
        with open(testcases, 'r') as fp:
            testcases = read_testcases(json.load(fp))

    tester = RuleTester()
    result = tester.analyze(testcases)
    
    if case_save_path:
        os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
        with open(case_save_path, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result
