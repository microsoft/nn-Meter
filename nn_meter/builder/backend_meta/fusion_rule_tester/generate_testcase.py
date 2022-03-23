# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import yaml
import importlib
from tensorflow import keras
from .utils import get_operator_by_name, generate_model_for_testcase
from .build_models import SingleOpModel
from nn_meter.builder.backend_meta.utils import Latency

__BUILTIN_TESTCASES__ = {'MON'}

__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__registry_cfg_filename__ = 'registry.yaml'
__REG_TESTCASES__ = {}
if os.path.isfile(os.path.join(__user_config_folder__, __registry_cfg_filename__)):
    with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'r') as fp:
        registry_modules = yaml.load(fp, yaml.FullLoader)
    if "testcases" in registry_modules:
        __REG_TESTCASES__ = registry_modules["testcases"]


class BaseTestCase:
    name = ''
    cases = None
    true_case = ''
    deps = {}
    input_shape = None

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
                    layer, _, op1_is_two_inputs = get_operator_by_name(op, self.input_shape, self.config)
                    model = SingleOpModel(layer)
                    shapes = [self.input_shape] * (1 + op1_is_two_inputs)
                    testcase[op] = {
                        'model': model,
                        'shapes': shapes
                    }
        return testcase

    def save_testcase(self):
        from nn_meter.builder.nn_generator.tf_networks.utils import get_tensor_by_shapes
        testcase = self.generate_testcase()

        for op, model in testcase.items():
            model_path = os.path.join(self.model_dir, self.name + '_' + op)
            model['model'](get_tensor_by_shapes(model['shapes'], batch_size=self.config['BATCH_SIZE']))
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
        config = self.config
        if not self.input_shape:
            self.input_shape = [config['HW'], config['HW'], config['CIN']]
        self.kernel_size = config['KERNEL_SIZE']
        self.cout = config['COUT']
        self.padding = config['PADDING']
        self.model_dir = os.path.join(config['MODEL_DIR'], 'models')
        os.makedirs(self.model_dir, exist_ok=True)

    def _model_block(self):
        pass


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
            generate_model_for_testcase(op1, op2, self.input_shape, self.config)
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


class MultipleOutNodes(BaseTestCase):
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


def generate_testcases():
    testcases_list = {}
    from nn_meter.builder import builder_config
    config =  builder_config.get_module('ruletest')

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
            })
            testcases_list[bf_cls.name] = bf_cls
            
    if config['OTHER_TESTCASES'] != None:
        for testcase in config['OTHER_TESTCASES']:
            if testcase in __BUILTIN_TESTCASES__:
                testcases_list[testcase] = MultipleOutNodes
            else:
                try:
                    testcase_info = __REG_TESTCASES__[testcase]
                    sys.path.append(testcase_info["package_location"])
                    testcase_module_name = testcase_info["class_name"]
                    testcase_module = importlib.import_module(testcase_info["class_module"])
                    testcase_cls = getattr(testcase_module, testcase_module_name)
                    testcases_list[testcase] = testcase_cls
                except:
                    raise KeyError(f'Unsupported test case: {testcase}.')

    return testcases_list

def list_testcases():
    return __BUILTIN_TESTCASES__ + ["* " + item for item in list(__REG_TESTCASES__.keys())]
