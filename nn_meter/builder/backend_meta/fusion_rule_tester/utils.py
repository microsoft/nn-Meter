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

__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__registry_cfg_filename__ = 'registry.yaml'
__REG_OPERATORS__ = {}
if os.path.isfile(os.path.join(__user_config_folder__, __registry_cfg_filename__)):
    with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'r') as fp:
        registry_modules = yaml.load(fp, yaml.FullLoader)
    if "operators" in registry_modules:
        __REG_OPERATORS__ = registry_modules["operators"]


def get_operator_by_name(operator_name, input_shape, config = None):
    """ get operator information by builtin name
    """
    if operator_name in __REG_OPERATORS__:
        operator_info = __REG_OPERATORS__[operator_name]
        sys.path.append(operator_info["package_location"])
        operator_module_name = operator_info["class_name"]
        operator_module = importlib.import_module(operator_info["class_module"])

    elif operator_name in __BUILTIN_OPERATORS__:
        operator_module_name = __BUILTIN_OPERATORS__[operator_name]
        from nn_meter.builder.nn_generator.tf_networks import operators
        operator_module = operators

    else:
        raise ValueError(f"Unsupported operator name: {operator_name}. Please register the operator first.")

    operator_cls = getattr(operator_module, operator_module_name)(input_shape, config)
    operator = operator_cls.get_model()
    output_shape = operator_cls.get_output_shape()
    op_is_two_inputs = operator_cls.get_is_two_inputs()
    
    return operator, output_shape, op_is_two_inputs


def generate_model_for_testcase(op1, op2, input_shape, config):
    from .build_models import SingleOpModel, TwoOpModel
    from nn_meter.builder.utils import get_inputs_by_shapes
    layer1, op1_output_shape, op1_is_two_inputs = get_operator_by_name(op1, input_shape, config)
    layer2, _, op2_is_two_inputs = get_operator_by_name(op2, op1_output_shape, config)

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


def list_operators():
    return __BUILTIN_OPERATORS__ + ["* " + item for item in list(__REG_OPERATORS__.keys())]
