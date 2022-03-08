# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import logging
from ..utils import read_profiled_results
from nn_meter.builder.utils import merge_prev_info


def generate_testcases():
    """generate testcases and save the testcase models and testcase json file in the workspace
    """
    from nn_meter.builder import builder_config
    config = builder_config.get_module('ruletest')

    from .test_fusion_rule import FusionRuleTester
    tester = FusionRuleTester()
    testcases = tester.generate()

    # save information to json file
    ws_path = config['MODEL_DIR']
    info_save_path = os.path.join(ws_path, "results", "origin_testcases.json")
    new_testcases = merge_prev_info(new_info=testcases, info_save_path=info_save_path)
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
    ws_path = config['MODEL_DIR']
    info_save_path = os.path.join(ws_path, "results", "detected_fusion_rule.json")
    new_result = merge_prev_info(new_info=result, info_save_path=info_save_path)
    os.makedirs(os.path.dirname(info_save_path), exist_ok=True)
    with open(info_save_path, 'w') as fp:
        json.dump(new_result, fp, indent=4)
    logging.keyinfo(f"Save the detected fusion rule information to {info_save_path}")
    return result
