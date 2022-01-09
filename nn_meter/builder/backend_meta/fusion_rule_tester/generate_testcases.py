# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from tensorflow import keras
from .utils import generate_model_for_testcase
from nn_meter.builder.utils import get_tensor_by_shapes, builder_config
from nn_meter.builder.backend_meta.utils import Latency

config =  builder_config.get_module('ruletest')
rules = {}


class TestCasesGenerator:
    name = ''
    cases = None
    true_case = ''
    deps = {}
    input_shape = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._register()

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.latency = {}
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
                model, shapes = getattr(self, '_model_' + op)()
                testcase[op] = {
                    'model': model,
                    'shapes': shapes
                }

        return testcase

    def save_testcase(self):
        testcase = self.generate_testcase()

        for op, model in testcase.items():
            model_path = os.path.join(self.model_dir, self.name + '_' + op)
            model['model'](get_tensor_by_shapes(model['shapes']))
            keras.models.save_model(model['model'], model_path)
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
        if not self.input_shape:
            self.input_shape = [config['HW'], config['HW'], config['CIN']]
        self.kernel_size = config['KERNEL_SIZE']
        self.cout = config['COUT']
        self.padding = config['PADDING']
        self.model_dir = os.path.join(config['MODEL_DIR'], 'models')
        os.makedirs(self.model_dir, exist_ok=True)

    @classmethod
    def _register(cls):
        if (cls.name != '' and cls.name.startswith("BF")) or \
            (config['OTHER_TESTCASES'] != None and cls.name in config['OTHER_TESTCASES']):
                rules[cls.name] = cls

    def _model_block(self):
        pass


class BasicFusionImpl(TestCasesGenerator):
    name = ''
    cases = {
        'ops': ['', ''],
    }
    false_case = 'ops'

    def load_config(self):
        super().load_config()
        self.eps = config['EMP_ALPHA']

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
            generate_model_for_testcase(op1, op2, self.input_shape, config)
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


class BasicFusion(TestCasesGenerator):
    name = ''
    d1_required_layers = []

    @classmethod
    def _register(cls):
        if config['BASIC_TESTCASES'] == None: return
        testcases = [case.split('_') for case in config['BASIC_TESTCASES']]
        d1_required_layers = config['LAYERS_1D']
        for op1, op2 in testcases:
            classname = f'BasicFusion_{op1}_{op2}'
            name = f'BF_{op1}_{op2}'
            cases = {
                'ops': [op1, op2],
            }
            if op1 in d1_required_layers or op2 in d1_required_layers:
                input_shape = [config['SHAPE_1D']]
            else:
                input_shape = [config['HW'], config['HW'], config['CIN']]
            bf_cls = type(classname, (BasicFusionImpl,), {
                'name': name,
                'cases': cases,
                'input_shape': input_shape,
            })
            rules[bf_cls.name] = bf_cls


class MultipleOutNodes(TestCasesGenerator):
    name = 'MON'
    cases = {
        'case1': ['relu_relu', 'relu_dwconv', 'dwconv'],
        'case2': ['dwconv_relu_relu', 'relu_dwconv'],
        'case3': ['dwconv_relu', 'dwconv', 'relu_relu']
    }
    true_case = 'case1'
    deps = {
        'BF_dwconv_relu': True,
    }

    def _model_block(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)
        branch_1 = keras.layers.ReLU(negative_slope=0)(x)
        branch_1 = keras.layers.ReLU(negative_slope=0)(branch_1)
        branch_2 = keras.layers.ReLU(negative_slope=2)(x)
        branch_2 = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(branch_2)

        return keras.models.Model(input_layer, [branch_1, branch_2]), [self.input_shape]

    def _model_relu_relu(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.ReLU()(input_layer)
        x = keras.layers.ReLU()(x)

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_dwconv_relu_relu(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)
        x = keras.layers.ReLU()(x)
        x = keras.layers.ReLU()(x)

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_relu_dwconv(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.ReLU()(input_layer)
        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(x)

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_dwconv_relu(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)
        x = keras.layers.ReLU()(x)

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_dwconv(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)

        return keras.models.Model(input_layer, x), [self.input_shape]
