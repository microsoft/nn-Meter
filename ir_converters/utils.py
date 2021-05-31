# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import onnx
import json
from .onnx_converter import OnnxConverter
from .frozenpb_converter import FrozenPbConverter


def model_to_graph(model, model_type):
    if model_type == 'onnx':
        converter = OnnxConverter(model)
        result = converter.convert()
    elif model_type == 'pb':
        raise NotImplementedError
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    return result


def model_file_to_graph(filename, model_type=None):
    if model_type is None:
        if filename.endswith('.onnx'):
            model_type = 'onnx'
        elif filename.endswith('.pb'):
            model_type = 'pb'
        elif filename.endswith('.json'):
            model_type = 'json'
        else:
            raise ValueError(f'Unknown file type: {filename}')

    if model_type == 'onnx':
        model = onnx.load(filename)
        return model_to_graph(model, model_type)
    elif model_type == 'pb':
        converter = FrozenPbConverter(filename)
        return converter.get_flatten_grapher()
    elif model_type == 'json':
        with open(filename, 'r') as fp:
            return json.load(fp)
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

