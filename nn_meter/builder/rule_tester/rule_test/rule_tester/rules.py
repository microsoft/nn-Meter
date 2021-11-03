import os
from tensorflow import keras

from ...config_manager import config
from ..model_builder.utils import get_model_by_ops
from nn_meter.builder.utils import get_tensor_by_shapes
from nn_meter.builder.utils.latency import Latency


rules = {}


class RuleTestBase:
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

    def _gen_testcase(self):
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
        testcase = self._gen_testcase()

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
        self.input_shape = self.input_shape or config.get('default_input_shape', 'ruletest')
        self.kernel_size = config.get('kernel_size', 'ruletest')
        self.filters = config.get('filters', 'ruletest')
        self.enabled = self.name in config.get('enabled', 'ruletest')
        self.model_dir = config.get('model_dir', 'ruletest')

        for key, value in config.get('params', 'ruletest').get(self.name, {}).items():
            setattr(self, key, value)

    @classmethod
    def _register(cls):
        if cls.name != '':
            rules[cls.name] = cls

    def _model_relu(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.ReLU()(input_layer)

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_dwconv_relu(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding='same')(input_layer)
        x = keras.layers.ReLU()(x)

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_add(self):
        input_1 = keras.Input(shape=self.input_shape)
        input_2 = keras.Input(shape=self.input_shape)

        x = keras.layers.Add()([input_1, input_2])

        return keras.models.Model([input_1, input_2], x), [self.input_shape, self.input_shape]

    def _model_dwconv(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding='same')(input_layer)

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_conv(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.Conv2D(self.filters, self.kernel_size, padding='same')(input_layer)

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_block(self):
        pass


class SingleCaseBase(RuleTestBase):
    false_case = ''

    def test(self):
        latency_block = self.latency['block']
        latency_case = self.latency[self.false_case]

        if abs(latency_block.avg - latency_case.avg) < (latency_block.std + latency_case.std):
            return False
        else:
            return True


class BasicFusionImpl(RuleTestBase):
    name = ''
    cases = {
        'ops': ['dwconv', 'relu'],
    }
    false_case = 'ops'

    def load_config(self):
        super().load_config()
        self.enabled = 'BF' in config.get('enabled', 'ruletest')
        self.eps = config.get('params', 'ruletest')['BF']['eps']

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

    def _gen_testcase(self):
        testcase = {}

        op1, op2 = self.cases['ops']
        op1_alias, op2_alias = op1, op2

        if op1_alias == op2_alias:
            op1_alias += '_1'
            op2_alias += '_2'

        op1_model, op2_model, block_model, op1_shapes, op2_shapes, block_shapes = get_model_by_ops(op1, op2, self.input_shape)
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


class BasicFusion(RuleTestBase):
    name = 'BF'

    layers = [
        'reshape',
        'dwconv',
        'relu',
        'add',
        'conv',
        'concat',
        'convtrans',
        'dense',
        'pooling',
    ]

    disabled_combinations = [
        ('dense', 'pooling'),
        ('dense', 'dwconv'),
        ('dense', 'conv'),
        ('dense', 'convtrans'),
        ('dense', 'reshape'),
    ]

    d1_required_layers = [
        'dense',
    ]

    additional_combinations = [
        ('conv', 'hswish'),
        ('conv', 'se'),
        ('se', 'relu'),
    ]

    @classmethod
    def get_combinations(cls):
        combinations = []
        for op1 in cls.layers:
            for op2 in cls.layers:
                if (op1, op2) not in cls.disabled_combinations and (op2, op1) not in cls.disabled_combinations:
                    combinations.append((op1, op2))
        for combi in cls.additional_combinations:
            combinations.append(combi)
        return combinations

    @classmethod
    def _register(cls):
        for op1, op2 in cls.get_combinations():
            classname = f'BasicFusion_{op1}_{op2}'
            name = f'BF_{op1}_{op2}'
            cases = {
                'ops': [op1, op2],
            }
            if op1 in cls.d1_required_layers or op2 in cls.d1_required_layers:
                input_shape = config.get('d1_input_shape', 'ruletest')
            else:
                input_shape = config.get('default_input_shape', 'ruletest')
            bf_cls = type(classname, (BasicFusionImpl,), {
                'name': name,
                'cases': cases,
                'input_shape': input_shape,
            })
            rules[bf_cls.name] = bf_cls


class MultipleOutNodes(RuleTestBase):
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

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding='same')(input_layer)
        branch_1 = keras.layers.ReLU(negative_slope=0)(x)
        branch_1 = keras.layers.ReLU(negative_slope=0)(branch_1)
        branch_2 = keras.layers.ReLU(negative_slope=2)(x)
        branch_2 = keras.layers.DepthwiseConv2D(self.kernel_size, padding='same')(branch_2)

        return keras.models.Model(input_layer, [branch_1, branch_2]), [self.input_shape]

    def _model_relu_relu(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.ReLU()(input_layer)
        x = keras.layers.ReLU()(x)

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_dwconv_relu_relu(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding='same')(input_layer)
        x = keras.layers.ReLU()(x)
        x = keras.layers.ReLU()(x)

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_relu_dwconv(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.ReLU()(input_layer)
        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding='same')(x)

        return keras.models.Model(input_layer, x), [self.input_shape]


class ConvBranchContext(SingleCaseBase):
    name = 'CBC'
    cases = {
        'ops': ['conv', 'conv', 'conv', 'concat3'],
    }
    false_case = 'ops'

    def _model_block(self):
        input_layer = keras.Input(shape=self.input_shape)

        branch_1 = keras.layers.Conv2D(self.filters, self.kernel_size, padding='same')(input_layer)
        branch_2 = keras.layers.Conv2D(self.filters, self.kernel_size, padding='same')(input_layer)
        branch_3 = keras.layers.Conv2D(self.filters, self.kernel_size, padding='same')(input_layer)

        output_layer = keras.layers.Concatenate()([branch_1, branch_2, branch_3])

        return keras.models.Model(input_layer, output_layer), [self.input_shape]

    def _model_concat3(self):
        input_1 = keras.Input(shape=self.input_shape)
        input_2 = keras.Input(shape=self.input_shape)
        input_3 = keras.Input(shape=self.input_shape)

        output_layer = keras.layers.Concatenate()([input_1, input_2, input_3])

        return keras.models.Model([input_1, input_2, input_3], output_layer), [self.input_shape, self.input_shape, self.input_shape]


class ReLUBranchContext(SingleCaseBase):
    name = 'RBC'
    cases = {
        'ops': ['3xrelu', 'conv'],
    }
    false_case = 'ops'

    def _model_block(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.Conv2D(self.filters, self.kernel_size, padding='same')(input_layer)

        branch_1 = keras.layers.ReLU(negative_slope=0)(x)
        branch_2 = keras.layers.ReLU(negative_slope=0.5)(x)
        branch_3 = keras.layers.ReLU(negative_slope=2)(x)

        return keras.models.Model(input_layer, [branch_1, branch_2, branch_3]), [self.input_shape]

    def _model_3xrelu(self):
        input_layer = keras.Input(shape=self.input_shape)

        branch_1 = keras.layers.ReLU(negative_slope=0)(input_layer)
        branch_2 = keras.layers.ReLU(negative_slope=0.5)(input_layer)
        branch_3 = keras.layers.ReLU(negative_slope=2)(input_layer)

        return keras.models.Model(input_layer, [branch_1, branch_2, branch_3]), [self.input_shape]


class ReadyTensor(RuleTestBase):
    name = 'RT'
    cases = {
        'case1': ['dwconv_add', 'dwconv', 'dwconv', 'add', 'relu'],
        'case2': ['dwconv_add_add', 'dwconv', 'dwconv', 'relu'],
    }
    true_case = 'case1'
    deps = {
        'MON': True,
        'BF_dwconv_relu': True,
    }

    def _model_block(self):
        input_layer = keras.Input(shape=self.input_shape)

        branch_1 = keras.layers.DepthwiseConv2D(self.kernel_size, padding='same')(input_layer)
        branch_2 = keras.layers.DepthwiseConv2D(self.kernel_size, padding='same')(input_layer)
        output_1 = keras.layers.Add()([branch_1, branch_2])
        branch_3 = keras.layers.DepthwiseConv2D(self.kernel_size, padding='same')(input_layer)
        output_1 = keras.layers.Add()([branch_3, output_1])

        output_2 = keras.layers.ReLU()(branch_3)

        return keras.Model(input_layer, [output_1, output_2]), [self.input_shape]

    def _model_dwconv_add(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding='same')(input_layer)
        x = keras.layers.Add()([x, input_layer])

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_dwconv_add_add(self):
        input_1 = keras.Input(shape=self.input_shape)
        input_2 = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding='same')(input_1)
        x = keras.layers.Add()([x, input_1])
        x = keras.layers.Add()([x, input_2])

        return keras.models.Model([input_1, input_2], x), [self.input_shape, self.input_shape]
