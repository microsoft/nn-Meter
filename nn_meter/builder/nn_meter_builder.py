import os
import json
import logging
from .utils import builder_config as config
from .utils.utils import dump_testcases_with_latency, read_testcases_with_latency


def create_testcases(save_as=None):
    """create testcases and save the testcase models and testcase json file in the workspace
    """
    from .rule_tester import RuleTester
    tester = RuleTester()
    testcases = tester.generate()
    
    workspace_path = config.workspace_path
    case_save_path = save_as or os.path.join(workspace_path, "results", "origin_testcases.json")
    os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
    with open(case_save_path, 'w') as fp:
        json.dump(testcases, fp, indent=4)
    logging.keyinfo(f"Save the original testcases to {case_save_path}")
    return testcases


def run_testcases(backend, testcases, metrics=["latency"], save_as=None):
    """ run testcases with given backend and return latency of testcase models
    @params:

    backend: applied backend with its config, should be a subclass of BaseBackend
    testcases: the Dict of testcases or the path of the testcase json file
    metrics: required metrics to report. We only support latency for metric by now.
    save_as: the customized path to save testcases dictionary. Set as None by default.

    """
    if isinstance(testcases, str):
        with open(testcases, 'r') as fp:
            testcases = json.load(fp)

    for _, testcase in testcases.items():
        for _, model in testcase.items():
            model_path = model['model']
            profiled_res = backend.profile_model_file(model_path, model['shapes'])
            for metric in metrics:
                model[metric] = profiled_res[metric]

    workspace_path = config.workspace_path
    case_save_path = os.path.join(workspace_path, "results", "profiled_testcases.json")
    os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
    with open(case_save_path, 'w') as fp:
        json.dump(dump_testcases_with_latency(testcases), fp, indent=4)
    logging.keyinfo(f"Save the profiled testcases to {case_save_path}")
    return testcases


def detect_fusionrule(testcases):
    """ detect fusion rule by testcases latency value
    @params:

    testcases: the Dict of testcases or the path of the testcase json file
    """
    from .rule_tester import RuleTester
    if isinstance(testcases, str):
        with open(testcases, 'r') as fp:
            testcases = read_testcases_with_latency(json.load(fp))

    tester = RuleTester()
    result = tester.analyze(testcases)
    
    workspace_path = config.workspace_path
    case_save_path = os.path.join(workspace_path, "results", "detected_testcases.json")
    os.makedirs(os.path.dirname(case_save_path), exist_ok=True)
    with open(case_save_path, 'w') as fp:
        json.dump(result, fp, indent=4)
    logging.keyinfo(f"Save the testcases to {case_save_path}")
    return result
