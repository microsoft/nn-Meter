# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .utils import *
from .interface import BaseTestCase
from nn_meter.builder.backend_meta.utils import Latency


class BasicFusion(BaseTestCase):
    name = ''
    cases = {
        'ops': ['', ''],
    }
    false_case = 'ops'

    def load_config(self):
        super().load_config()
        self.eps = self.config['EMP_ALPHA']

    def test(self):
        secondary_op_lat = min(lat for op, lat in self.latency.items() if op != 'block' or op != self.false_case)
        return self.latency[self.false_case].avg - self.latency['block'].avg > self.eps * secondary_op_lat.avg

    def load_latency(self, testcase):
        self.latency['block'] = Latency(testcase['block']['latency'])

        op1, op2 = self.cases['ops']
        op1_alias, op2_alias = op1, op2

        if op1_alias == op2_alias:
            op1_alias += '_1'
            op2_alias += '_2'
        
        self.latency[op1_alias] = Latency(testcase[op1_alias]['latency'])
        self.latency[op2_alias] = Latency(testcase[op2_alias]['latency'])
        self.latency['ops'] = self.latency[op1_alias] + self.latency[op2_alias]

    def generate_testcase(self):
        testcase = {}

        op1, op2 = self.cases['ops']
        op1_alias, op2_alias = op1, op2

        if op1_alias == op2_alias:
            op1_alias += '_1'
            op2_alias += '_2'

        op1_model, op2_model, block_model, op1_shapes, op2_shapes, block_shapes = \
            generate_models_for_testcase(op1, op2, self.input_shape, self.config, self.implement)
        testcase[op1_alias] = {
            'model': op1_model,
            'shapes': op1_shapes,
        }
        testcase[op2_alias] = {
            'model': op2_model,
            'shapes': op2_shapes,
        }
        testcase['block'] = {
            'model': block_model,
            'shapes': block_shapes,
        }
        return testcase


def generate_testcases():
    testcases_list = {}
    from nn_meter.builder import builder_config
    config =  builder_config.get_module('ruletest')
    implement = builder_config.get('IMPLEMENT', 'ruletest')

    if config['BASIC_TESTCASES'] != None:
        testcases = [case.split('_') for case in config['BASIC_TESTCASES']]
        d1_required_layers = config['LAYERS_1D']
        for op1, op2 in testcases:
            class_name = f'BasicFusion_{op1}_{op2}'
            name = f'BF_{op1}_{op2}'
            cases = {
                'ops': [op1, op2],
            }
            if op1 in d1_required_layers or op2 in d1_required_layers:
                input_shape = [config['SHAPE_1D']]
            else:
                input_shape = [config['HW'], config['HW'], config['CIN']]
            bf_cls = type(class_name, (BasicFusion,), {
                'name': name,
                'cases': cases,
                'input_shape': input_shape,
                'implement': implement
            })
            testcases_list[bf_cls.name] = bf_cls
            
    if config['OTHER_TESTCASES'] != None:
        for testcase in config['OTHER_TESTCASES']:
            testcases_list[testcase] = get_special_testcases_by_name(testcase, implement=implement)

    return testcases_list
