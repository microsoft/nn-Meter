# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import yaml
import importlib

__BUILTIN_OPERATORS__ = {
    # builtin_name: module_name
    "conv": "Conv",
    "dwconv": "DwConv",
    "convtrans": "ConvTrans",
    "bn": "BN",
    "globalavgpool": "GlobalAvgpool",
    "maxpool": "MaxPool",
    "avgpool": "AvgPool",
    "se": "SE",
    "fc": "FC",
    "relu": "Relu",
    "relu6": "Relu6",
    "sigmoid": "Sigmoid",
    "hswish": "Hswish",
    "reshape": "Reshape",
    "add": "Add",
    "concat": "Concat",
    "flatten": "Flatten",
    "split": "Split"
}
__BUILTIN_TESTCASES__ = {'MON'}

__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__registry_cfg_filename__ = 'registry.yaml'
__REG_OPERATORS__, __REG_TESTCASES__ = {}, {}
if os.path.isfile(os.path.join(__user_config_folder__, __registry_cfg_filename__)):
    with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'r') as fp:
        registry_modules = yaml.load(fp, yaml.FullLoader)
    if "operators" in registry_modules:
        __REG_OPERATORS__ = registry_modules["operators"]
    if "testcases" in registry_modules:
        __REG_TESTCASES__ = registry_modules["testcases"]


def get_operator_by_name(operator_name, input_shape, config = None, implement = None):
    """ get operator information by builtin name
    """
    if operator_name in __REG_OPERATORS__ and implement in __REG_OPERATORS__[operator_name]:
        operator_info = __REG_OPERATORS__[operator_name][implement]
        sys.path.append(operator_info["package_location"])
        operator_module_name = operator_info["class_name"]
        operator_module = importlib.import_module(operator_info["class_module"])

    elif operator_name in __BUILTIN_OPERATORS__:
        operator_module_name = __BUILTIN_OPERATORS__[operator_name]
        if implement == 'tensorflow':
            from nn_meter.builder.nn_modules.tf_networks import operators
        elif implement == 'torch':
            from nn_meter.builder.nn_modules.torch_networks import operators
        else:
            raise NotImplementedError('You must choose one implementation of kernel from "tensorflow" or "pytorch"')
        operator_module = operators

    else:
        raise ValueError(f"Unsupported operator name: {operator_name}. Please register the operator first.")

    operator_cls = getattr(operator_module, operator_module_name)(input_shape, config)
    operator = operator_cls.get_model()
    output_shape = operator_cls.get_output_shape()
    op_is_two_inputs = operator_cls.get_is_two_inputs()
    
    return operator, output_shape, op_is_two_inputs


def get_special_testcases_by_name(testcase, implement=None):
    if testcase in __REG_TESTCASES__ and implement in __REG_TESTCASES__[testcase]:
        testcase_info = __REG_TESTCASES__[testcase][implement]
        sys.path.append(testcase_info["package_location"])
        testcase_module_name = testcase_info["class_name"]
        testcase_module = importlib.import_module(testcase_info["class_module"])
        return getattr(testcase_module, testcase_module_name)

    elif testcase in __BUILTIN_TESTCASES__:
        assert implement != None
        if implement == 'tensorflow':
            from .build_tf_models import MultipleOutNodes
            return MultipleOutNodes
        elif implement == 'torch':
            from .build_torch_models import MultipleOutNodes
            return MultipleOutNodes
        else:
            raise NotImplementedError('You must choose one implementation of kernel from "tensorflow" or "pytorch"')

    else:
        raise ValueError(f"Unsupported operator name: {testcase}. Please register the operator first.")


def generate_models_for_testcase(op1, op2, input_shape, config, implement):
    if implement == 'tensorflow':
        from .build_tf_models import SingleOpModel, TwoOpModel
        from nn_meter.builder.nn_modules.tf_networks.utils import get_inputs_by_shapes
    elif implement == 'torch':
        from .build_torch_models import SingleOpModel, TwoOpModel
        from nn_meter.builder.nn_modules.torch_networks.utils import get_inputs_by_shapes
    else:
        raise NotImplementedError('You must choose one implementation of kernel from "tensorflow" or "pytorch"')

    layer1, op1_output_shape, op1_is_two_inputs = get_operator_by_name(op1, input_shape, config, implement)
    layer2, _, op2_is_two_inputs = get_operator_by_name(op2, op1_output_shape, config, implement)

    op1_model = SingleOpModel(layer1)
    op1_shapes = [input_shape] * (1 + op1_is_two_inputs)
    op1_model(get_inputs_by_shapes(op1_shapes))

    op2_model = SingleOpModel(layer2)
    op2_shapes = [op1_output_shape] * (1 + op2_is_two_inputs)
    op2_model(get_inputs_by_shapes(op2_shapes))

    block_model = TwoOpModel(layer1, layer2, op1_is_two_inputs, op2_is_two_inputs)
    block_shapes = [input_shape] * (1 + op1_is_two_inputs) + [op1_output_shape] * op2_is_two_inputs
    block_model(get_inputs_by_shapes(block_shapes))

    return op1_model, op2_model, block_model, op1_shapes, op2_shapes, block_shapes


def generate_single_model(op, input_shape, config, implement):
    if implement == 'tensorflow':
        from .build_tf_models import SingleOpModel
        from nn_meter.builder.nn_modules.tf_networks.utils import get_inputs_by_shapes
    elif implement == 'torch':
        from .build_torch_models import SingleOpModel
        from nn_meter.builder.nn_modules.torch_networks.utils import get_inputs_by_shapes
    else:
        raise NotImplementedError('You must choose one implementation of kernel from "tensorflow" or "pytorch"')

    layer, _, is_two_inputs = get_operator_by_name(op, input_shape, config, implement)

    model = SingleOpModel(layer)
    shapes = [input_shape] * (1 + is_two_inputs)
    model(get_inputs_by_shapes(shapes))

    return model, shapes


def save_model(model, model_path, implement):
    if implement == 'tensorflow':
        from tensorflow import keras
        from nn_meter.builder.nn_modules.tf_networks.utils import get_tensor_by_shapes
        model['model'](get_tensor_by_shapes(model['shapes']))
        keras.models.save_model(model['model'], model_path)
        return model_path

    elif implement == 'torch':
        import torch
        from nn_meter.builder.nn_modules.torch_networks.utils import get_inputs_by_shapes
        torch.onnx.export(
            model['model'],
            get_inputs_by_shapes(model['shapes']),
            model_path + '.onnx',
            input_names=['input'],
            output_names=['output'],
            verbose=False,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
        )
        return model_path + '.onnx'

    else:
        import pdb; pdb.set_trace()
        raise NotImplementedError('You must choose one implementation of kernel from "tensorflow" or "pytorch"')


def list_operators():
    return list(__BUILTIN_OPERATORS__.keys()) + \
        [f"* {key} ({', '.join(list(info.keys()))})" for key, info in __REG_OPERATORS__.items()]


def list_testcases():
    return list(__BUILTIN_TESTCASES__) + \
        [f"* {key} ({', '.join(list(info.keys()))})" for key, info in __REG_TESTCASES__.items()]
