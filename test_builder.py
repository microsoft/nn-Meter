import os
import json
import copy
from nn_meter.builder.rule_tester import config
from nn_meter.builder.backends import get_backend
from nn_meter.builder.rule_tester import RuleTester

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
            # json.dump(testcases._dump(), fp, indent=4)
            json.dump(testcases, fp, indent=4)

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
            testcases = json.load(fp)

    for _, testcase in testcases.items():
        for _, model in testcase.items():
            model_path = model['model']
            model['latency'] = backend.profile_model_file(model_path, model['shapes'])

    testcases_copy = copy.deepcopy(testcases)
    for _, testcase_copy in testcases_copy.items():
        for _, model in testcase_copy.items():
            model_path = model['model']
            model['latency'] = str(model['latency'])
    if case_save_path:
        os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
        with open(case_save_path, 'w') as fp:
            # json.dump(testcases._dump(), fp, indent=4)
            json.dump(testcases_copy, fp, indent=4)
    return testcases


def get_fusionrule(testcases, case_save_path='./data/detected_testcases.json'):
    """
    @params:

    - testcases: The TestCases class or the path of the testcase json file

    - case_save_path: Path to save the testcase json file

    """
    if isinstance(testcases, str):
        with open(testcases, 'r') as fp:
            testcases = json.load(fp) # TODO: if use json file, latency should be refine

    # assert testcases.profiled == False
    tester = RuleTester()
    testcases = tester.analyze(testcases)
    
    if case_save_path:
        os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
        with open(case_save_path, 'w') as fp:
            # json.dump(testcases._dump(), fp, indent=4)
            json.dump(testcases, fp, indent=4)
    return testcases


if __name__ == '__main__':    
    # initialize backend
    backend = get_backend(
        backend = 'tflite_cpu', 
        params = {
            'MODEL_DIR': '/data/jiahang/test_models',
            'REMOTE_MODEL_DIR': '/mnt/sdcard/tflite_bench',
            'KERNEL_PATH': '/mnt/sdcard/tflite_bench/kernel.cl',
            'BENCHMARK_MODEL_PATH': '/data/local/tmp/benchmark_model_fixed_group_size',
            'DEVICE_SERIAL': '0000028e2c780e4e',
        }
    )

    # generate testcases
    testcases = get_testcases(model_dir='/data/jiahang/test_models')

    # run testcases and collect profiling results
    profile_results = run_testcases(backend, testcases)
    
    # determine fusion rules from profiling results
    detect_results = get_fusionrule(profile_results)
   
