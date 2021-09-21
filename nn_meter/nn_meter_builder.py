from nn_meter.builder.ruletest.rules.tester import RuleTester
import copy
from nn_meter.builder.config import app_config


def get_testcases(model_dir):
    """
    @params:
    model_dir: directory to save testcase models
    """
    app_config.set('model_dir', model_dir, 'ruletest')

    tester = RuleTester()
    testcases = tester.gen()
    return testcases


def get_fusionrule(profile_results):
    tester = RuleTester()
    return tester.analyze(profile_results)


def run_testcases(backend, testcases):
    profile_results = copy.deepcopy(testcases)
    for _, profile_result in profile_results.items():
        for _, model in profile_result.items():
            model_path = model['model']
            model['latency'] = backend.profile_model_path(model_path, model['shapes'])
    return profile_results
