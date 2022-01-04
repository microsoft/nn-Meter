# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import logging
from .detect_fusion_rule import FusionRuleTester
from ..utils import read_profiled_results
from nn_meter.builder.utils import builder_config as config


def generate_testcases():
    """generate testcases and save the testcase models and testcase json file in the workspace
    """
    tester = FusionRuleTester()
    testcases = tester.generate()

    # save information to json file
    ws_path = config.get('MODEL_DIR', 'ruletest')
    info_save_path = os.path.join(ws_path, "results", "origin_testcases.json")
    os.makedirs(os.path.dirname(info_save_path), exist_ok=True)
    with open(info_save_path, 'w') as fp:
        json.dump(testcases, fp, indent=4)
    logging.keyinfo(f"Save the original testcases information to {info_save_path}")
    return testcases


def detect_fusionrule(profiled_testcases):
    """ detect fusion rule by testcases latency value
    @params:

    testcases: the Dict of testcases or the path of the testcase json file
    """
    if isinstance(profiled_testcases, str):
        with open(profiled_testcases, 'r') as fp:
            profiled_testcases = read_profiled_results(json.load(fp))

    tester = FusionRuleTester()
    result = tester.analyze(profiled_testcases)

    # save information to json file
    ws_path = config.get('MODEL_DIR', 'ruletest')
    info_save_path = os.path.join(ws_path, "results", "detected_testcases.json")
    os.makedirs(os.path.dirname(info_save_path), exist_ok=True)
    with open(info_save_path, 'w') as fp:
        json.dump(result, fp, indent=4)
    logging.keyinfo(f"Save the detected fusion rule information to {info_save_path}")
    return result
