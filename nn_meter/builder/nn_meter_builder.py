import os
import json
from .rule_tester import config
from .rule_tester import RuleTester
from .utils.utils import dump_testcases, read_testcases

def get_testcases(model_dir, case_save_path='./data/testcases.json'):
    """
    @params:

    - model_dir: Directory to save testcase models

    - case_save_path: Path to save the testcase json file

    """
    config.set('model_dir', model_dir, 'ruletest')
    print(config.get_settings())

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

    - backend: Applied backend with its config, a subclass of BaseBackend

    - testcases: The TestCases class or the path of the testcase json file

    - case_save_path: Path to save the testcase json file

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


def get_fusionrule(testcases, case_save_path='./data/detected_testcases.json'):
    """
    @params:

    - testcases: The TestCases class or the path of the testcase json file

    - case_save_path: Path to save the testcase json file

    """
    if isinstance(testcases, str):
        with open(testcases, 'r') as fp:
            testcases = read_testcases(json.load(fp)) # TODO: if use json file, latency should be refine

    # assert testcases.profiled == False
    tester = RuleTester()
    testcases = tester.analyze(testcases)
    
    if case_save_path:
        os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
        with open(case_save_path, 'w') as fp:
            # json.dump(testcases._dump(), fp, indent=4)
            json.dump(dump_testcases(testcases), fp, indent=4)
    return testcases