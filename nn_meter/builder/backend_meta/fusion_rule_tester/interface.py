# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import logging
from ..utils import read_profiled_results
from nn_meter.builder.utils import merge_info
from nn_meter.builder.backend_meta.utils import Latency


class BaseTestCase:
    name = ''
    cases = None
    true_case = ''
    deps = {}
    input_shape = None
    implement = None

    def __init__(self, config, **kwargs):
        self._kwargs = kwargs
        self.latency = {}
        self.config = config
        self.load_config()

    def generate_testcase(self):
        testcase = {}
        model, shapes = self._model_block()
        testcase['block'] = {
            'model': model,
            'shapes': shapes,
        }

        for _, ops in self.cases.items():
            for op in ops:
                try:
                    model, shapes = getattr(self, '_model_' + op)()
                    testcase[op] = {
                        'model': model,
                        'shapes': shapes
                    }
                except:
                    from .utils import generate_single_model
                    model, shapes = generate_single_model(op, self.input_shape, self.config, self.implement)
                    testcase[op] = {
                        'model': model,
                        'shapes': shapes
                    }
        return testcase

    def save_testcase(self):
        from .utils import save_model
        testcase = self.generate_testcase()

        for op, model in testcase.items():
            model_path = os.path.join(self.workspace_path, self.name + '_' + op)
            model_path = save_model(model, model_path, self.implement)
            testcase[op]['model'] = model_path

        return testcase

    def load_latency(self, testcase):
        self.latency['block'] = Latency(testcase['block']['latency'])

        for case, ops in self.cases.items():
            latency_sum = 0
            for op in ops:
                if op not in self.latency:
                    self.latency[op] = Latency(testcase[op]['latency'])
                latency_sum += self.latency[op]
            self.latency[case] = latency_sum

    def test(self):
        true_case_lat_diff = abs(self.latency[self.true_case].avg - self.latency['block'].avg)

        for case, _ in self.cases.items():
            if case != self.true_case and true_case_lat_diff > abs(self.latency[case].avg - self.latency['block'].avg):
                return case

        return self.true_case

    def load_config(self):
        config = self.config
        if not self.input_shape:
            self.input_shape = [config['HW'], config['HW'], config['CIN']]
        self.kernel_size = config['KERNEL_SIZE']
        self.cout = config['COUT']
        self.padding = config['PADDING']
        self.workspace_path = os.path.join(config['WORKSPACE'], 'testcases')
        os.makedirs(self.workspace_path, exist_ok=True)

    def _model_block(self):
        pass


def generate_testcases():
    """generate testcases and save the testcase models and testcase json file in the workspace
    Users could edit the configurations of testcases in <workspace-path>/configs/ruletest_config.yaml.
    The config will take effect after the the config file is saved and closed.
    """
    from nn_meter.builder import builder_config
    config = builder_config.get_module('ruletest')

    from .test_fusion_rule import FusionRuleTester
    tester = FusionRuleTester()
    testcases = tester.generate()

    # save information to json file
    workspace_path = config['WORKSPACE']
    info_save_path = os.path.join(workspace_path, "results", "origin_testcases.json")
    new_testcases = merge_info(new_info=testcases, info_save_path=info_save_path)
    os.makedirs(os.path.dirname(info_save_path), exist_ok=True)
    with open(info_save_path, 'w') as fp:
        json.dump(new_testcases, fp, indent=4)
    logging.keyinfo(f"Save the original testcases information to {info_save_path}")
    return testcases


def detect_fusion_rule(profiled_testcases):
    """ detect fusion rule by testcases latency value
    @params:

    testcases: the Dict of testcases or the path of the testcase json file
    """
    if isinstance(profiled_testcases, str):
        with open(profiled_testcases, 'r') as fp:
            profiled_testcases = read_profiled_results(json.load(fp))

    from .test_fusion_rule import FusionRuleTester
    tester = FusionRuleTester()
    result = tester.analyze(profiled_testcases)

    # save information to json file
    from nn_meter.builder import builder_config
    config = builder_config.get_module('ruletest')
    workspace_path = config['WORKSPACE']
    info_save_path = os.path.join(workspace_path, "results", "detected_fusion_rule.json")
    new_result = merge_info(new_info=result, info_save_path=info_save_path)
    os.makedirs(os.path.dirname(info_save_path), exist_ok=True)
    with open(info_save_path, 'w') as fp:
        json.dump(new_result, fp, indent=4)
    logging.keyinfo(f"Save the detected fusion rule information to {info_save_path}")
    return result
